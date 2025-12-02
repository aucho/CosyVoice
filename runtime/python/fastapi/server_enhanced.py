# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
增强版 FastAPI 服务器，使用封装好的 AudioGenerator
"""
import os
import sys
import argparse
import logging
import threading
import tempfile
import shutil
import json
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Path
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import io
from scipy.io import wavfile
from pydub import AudioSegment

logging.getLogger("matplotlib").setLevel(logging.WARNING)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(ROOT_DIR)))
sys.path.append(project_root)

# 添加 Matcha-TTS 第三方模块路径（CosyVoice2 需要）
matcha_tts_path = os.path.join(project_root, "third_party", "Matcha-TTS")
if os.path.exists(matcha_tts_path):
    sys.path.append(matcha_tts_path)
    logging.debug(f"已添加 Matcha-TTS 路径: {matcha_tts_path}")
else:
    logging.warning(f"Matcha-TTS 模块路径不存在: {matcha_tts_path}")
    logging.warning(
        "如果遇到 'No module named matcha' 错误，请执行以下命令初始化 git submodule:"
    )
    logging.warning("  git submodule update --init --recursive")

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.api.generator import AudioGenerator

# 配置 Pydantic 以禁用受保护命名空间检查（解决 model_dir 警告）
import warnings

# 忽略 Pydantic 的受保护命名空间警告
warnings.filterwarnings("ignore", message=".*protected namespace.*")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Field.*has conflict with protected namespace.*",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logging.info("应用启动中...")
    # 加载持久化的任务状态
    load_task_status()
    # 不强制要求模型初始化，允许通过 /init 接口手动初始化
    yield
    # 关闭时
    logging.info("应用关闭中，清理资源...")
    # 保存任务状态
    save_task_status()
    global task_executor
    # 等待所有任务完成或超时
    task_executor.shutdown(wait=True, timeout=30)
    logging.info("任务执行器已关闭")


app = FastAPI(title="CosyVoice API", version="1.0.0", lifespan=lifespan)
# 设置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
audio_generator: Optional[AudioGenerator] = None
task_stop_events = {}
task_stop_lock = threading.Lock()

# 异步任务相关
task_status: Dict[str, Dict] = (
    {}
)  # stepId -> {status, result_path, error, created_at, updated_at}
task_status_lock = threading.Lock()
task_executor = ThreadPoolExecutor(max_workers=5)  # 最大并发任务数
temp_output_dir = os.path.join(tempfile.gettempdir(), "cosyvoice_async_results")
os.makedirs(temp_output_dir, exist_ok=True)

# 持久化文件路径
task_status_file = os.path.join(project_root, "task_status.json")


def save_task_status():
    """保存任务状态到文件"""
    try:
        with task_status_lock:
            # 只保存已完成或失败的任务（包含下载地址的）
            status_to_save = {}
            for step_id, info in task_status.items():
                if info.get("status") in ["completed", "failed"] and info.get(
                    "result_path"
                ):
                    status_to_save[step_id] = {
                        "status": info["status"],
                        "result_path": info["result_path"],
                        "error": info.get("error"),
                        "created_at": info.get("created_at"),
                        "updated_at": info.get("updated_at"),
                    }

            with open(task_status_file, "w", encoding="utf-8") as f:
                json.dump(status_to_save, f, ensure_ascii=False, indent=2)
            logging.debug(f"任务状态已保存到: {task_status_file}")
    except Exception as e:
        logging.warning(f"保存任务状态失败: {e}")


def load_task_status():
    """从文件加载任务状态"""
    global task_status
    if not os.path.exists(task_status_file):
        logging.info(f"任务状态文件不存在: {task_status_file}，使用空状态")
        return

    try:
        with open(task_status_file, "r", encoding="utf-8") as f:
            loaded_status = json.load(f)

        # 验证文件中的路径是否仍然存在
        valid_status = {}
        for step_id, info in loaded_status.items():
            result_path = info.get("result_path")
            if result_path and os.path.exists(result_path):
                valid_status[step_id] = info
            else:
                logging.debug(
                    f"任务 {step_id} 的结果文件不存在，跳过加载: {result_path}"
                )

        with task_status_lock:
            task_status.update(valid_status)

        logging.info(f"从文件加载了 {len(valid_status)} 个任务状态")
    except json.JSONDecodeError as e:
        logging.warning(f"任务状态文件格式错误: {e}，使用空状态")
    except Exception as e:
        logging.warning(f"加载任务状态失败: {e}，使用空状态")


# 持久化文件路径
task_status_file = os.path.join(project_root, "task_status.json")


def save_task_status():
    """保存任务状态到文件"""
    try:
        with task_status_lock:
            # 只保存已完成或失败的任务（包含下载地址的）
            status_to_save = {}
            for step_id, info in task_status.items():
                if info.get("status") in ["completed", "failed"] and info.get(
                    "result_path"
                ):
                    status_to_save[step_id] = {
                        "status": info["status"],
                        "result_path": info["result_path"],
                        "error": info.get("error"),
                        "created_at": info.get("created_at"),
                        "updated_at": info.get("updated_at"),
                    }

            with open(task_status_file, "w", encoding="utf-8") as f:
                json.dump(status_to_save, f, ensure_ascii=False, indent=2)
            logging.debug(f"任务状态已保存到: {task_status_file}")
    except Exception as e:
        logging.warning(f"保存任务状态失败: {e}")


def load_task_status():
    """从文件加载任务状态"""
    global task_status
    if not os.path.exists(task_status_file):
        logging.info(f"任务状态文件不存在: {task_status_file}，使用空状态")
        return

    try:
        with open(task_status_file, "r", encoding="utf-8") as f:
            loaded_status = json.load(f)

        # 验证文件中的路径是否仍然存在
        valid_status = {}
        for step_id, info in loaded_status.items():
            result_path = info.get("result_path")
            if result_path and os.path.exists(result_path):
                valid_status[step_id] = info
            else:
                logging.debug(
                    f"任务 {step_id} 的结果文件不存在，跳过加载: {result_path}"
                )

        with task_status_lock:
            task_status.update(valid_status)

        logging.info(f"从文件加载了 {len(valid_status)} 个任务状态")
    except json.JSONDecodeError as e:
        logging.warning(f"任务状态文件格式错误: {e}，使用空状态")
    except Exception as e:
        logging.warning(f"加载任务状态失败: {e}，使用空状态")


def generate_audio_bytes(model_output, sample_rate):
    """将音频生成器输出转换为字节流"""
    for i in model_output:
        if isinstance(i, tuple) and len(i) == 2:
            sr, audio_array = i
            # 转换为 int16 格式
            audio_bytes = (audio_array * (2**15)).astype(np.int16).tobytes()
            yield audio_bytes
        elif isinstance(i, dict) and "tts_speech" in i:
            tts_audio = (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
            yield tts_audio


@app.post("/init")
async def init_model(
    model_dir: str = Form(..., description="模型目录路径"),
    load_jit: bool = Form(False),
    load_trt: bool = Form(False),
    fp16: bool = Form(False),
    trt_concurrent: int = Form(1),
):
    """初始化模型"""
    global audio_generator

    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=400, detail=f"模型目录不存在: {model_dir}")

    try:
        logging.info(f"尝试使用 CosyVoice 初始化模型: {model_dir}")
        try:
            cosyvoice = CosyVoice(
                model_dir,
                load_jit=load_jit,
                load_trt=load_trt,
                fp16=fp16,
                trt_concurrent=trt_concurrent,
            )
            logging.info("成功使用 CosyVoice 初始化模型")
        except Exception as e1:
            logging.info(
                f"CosyVoice 初始化失败: {type(e1).__name__}: {str(e1)}，尝试 CosyVoice2..."
            )
            try:
                cosyvoice = CosyVoice2(
                    model_dir,
                    load_jit=load_jit,
                    load_trt=load_trt,
                    fp16=fp16,
                    trt_concurrent=trt_concurrent,
                )
                logging.info("成功使用 CosyVoice2 初始化模型")
            except Exception as e2:
                # 检查是否是 matcha 模块缺失的问题
                error_detail = str(e2)
                if (
                    "No module named 'matcha'" in error_detail
                    or "ModuleNotFoundError" in str(type(e2).__name__)
                ):
                    # 计算项目根目录
                    current_file = os.path.abspath(__file__)
                    project_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                    )
                    matcha_tts_path = os.path.join(
                        project_root, "third_party", "Matcha-TTS"
                    )
                    error_msg = (
                        f"CosyVoice2 初始化失败: 缺少 matcha 模块。\n"
                        f"CosyVoice错误: {type(e1).__name__}: {str(e1)}\n"
                        f"CosyVoice2错误: {type(e2).__name__}: {str(e2)}\n\n"
                        f"解决方案：\n"
                        f"1. 检查 Matcha-TTS 路径是否存在: {matcha_tts_path}\n"
                        f"2. 如果不存在，请执行: git submodule update --init --recursive\n"
                        f"3. 或者手动克隆: git clone https://github.com/shivammehta25/Matcha-TTS.git {matcha_tts_path}"
                    )
                else:
                    error_msg = f"无法使用 CosyVoice 或 CosyVoice2 初始化模型。CosyVoice错误: {type(e1).__name__}: {str(e1)}；CosyVoice2错误: {type(e2).__name__}: {str(e2)}"
                logging.error(error_msg, exc_info=True)
                raise HTTPException(status_code=500, detail=error_msg)

        audio_generator = AudioGenerator(cosyvoice)
        return JSONResponse(content={"status": "success", "message": "模型初始化成功"})
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"模型初始化失败: {type(e).__name__}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/generate_audio")
async def generate_audio_endpoint(
    tts_text: str = Form(),
    mode: str = Form(
        description="推理模式: 预训练音色/3s极速复刻/跨语种复刻/自然语言控制"
    ),
    sft_dropdown: str = Form(""),
    prompt_text: str = Form(""),
    prompt_wav: Optional[UploadFile] = File(None),
    instruct_text: str = Form(""),
    seed: int = Form(0),
    stream: bool = Form(False),
    speed: float = Form(1.0),
):
    """生成单个音频"""
    if audio_generator is None:
        raise HTTPException(status_code=500, detail="模型未初始化，请先调用 /init 接口")

    # 处理 prompt_wav
    prompt_wav_data = None
    if prompt_wav:
        prompt_wav_data = await prompt_wav.read()

    # 创建停止事件
    task_id = threading.current_thread().ident
    stop_event = threading.Event()
    with task_stop_lock:
        task_stop_events[task_id] = stop_event

    try:
        model_output = audio_generator.generate_audio(
            tts_text=tts_text,
            mode=mode,
            sft_dropdown=sft_dropdown,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav_data,
            instruct_text=instruct_text,
            seed=seed,
            stream=stream,
            speed=speed,
            stop_event=stop_event,
        )
        return StreamingResponse(
            generate_audio_bytes(model_output, audio_generator.sample_rate),
            media_type="audio/wav",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")
    finally:
        with task_stop_lock:
            task_stop_events.pop(task_id, None)


@app.post("/generate_batch_audio")
async def generate_batch_audio_endpoint(
    text_segments: str = Form(description="文本段列表，用换行符分隔"),
    output_dir: str = Form("./audios"),
    mode: str = Form(
        description="推理模式: 预训练音色/3s极速复刻/跨语种复刻/自然语言控制"
    ),
    sft_dropdown: str = Form(""),
    prompt_text: str = Form(""),
    prompt_wav: Optional[UploadFile] = File(None),
    instruct_text: str = Form(""),
    seed: int = Form(0),
    stream: bool = Form(False),
    speed: float = Form(1.0),
    enable_unstable_effects: bool = Form(False),
):
    """批量生成音频"""
    if audio_generator is None:
        raise HTTPException(status_code=500, detail="模型未初始化，请先调用 /init 接口")

    # 解析文本段
    segments = [s.strip() for s in text_segments.split("\n") if s.strip()]
    if not segments:
        raise HTTPException(status_code=400, detail="文本段列表不能为空")

    # 处理 prompt_wav
    prompt_wav_data = None
    if prompt_wav:
        prompt_wav_data = await prompt_wav.read()

    # 创建停止事件
    task_id = threading.current_thread().ident
    stop_event = threading.Event()
    with task_stop_lock:
        task_stop_events[task_id] = stop_event

    try:
        # 批量生成（返回状态消息）
        results = []
        for status, download_path in audio_generator.generate_batch_audio(
            text_segments=segments,
            output_dir=output_dir,
            mode=mode,
            sft_dropdown=sft_dropdown,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav_data,
            instruct_text=instruct_text,
            seed=seed,
            stream=stream,
            speed=speed,
            enable_unstable_effects=enable_unstable_effects,
            stop_event=stop_event,
        ):
            results.append({"status": status, "download_path": download_path})

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")
    finally:
        with task_stop_lock:
            task_stop_events.pop(task_id, None)


def save_audio_to_file(model_output, output_path: str, default_sample_rate: int):
    """将音频生成器输出保存为MP3文件"""
    audio_chunks = []
    actual_sample_rate = default_sample_rate

    for item in model_output:
        if isinstance(item, tuple) and len(item) == 2:
            sr, audio_array = item
            actual_sample_rate = sr  # 使用实际的采样率
            if isinstance(audio_array, np.ndarray):
                audio_chunks.append(audio_array)
            else:
                # 如果是 torch.Tensor，转换为 numpy
                if hasattr(audio_array, "numpy"):
                    audio_chunks.append(audio_array.numpy().flatten())
                elif hasattr(audio_array, "cpu"):
                    audio_chunks.append(audio_array.cpu().numpy().flatten())
                else:
                    audio_chunks.append(np.array(audio_array).flatten())
        elif isinstance(item, dict) and "tts_speech" in item:
            audio_data = item["tts_speech"]
            if hasattr(audio_data, "numpy"):
                audio_chunks.append(audio_data.numpy().flatten())
            elif hasattr(audio_data, "cpu"):
                audio_chunks.append(audio_data.cpu().numpy().flatten())
            else:
                audio_chunks.append(np.array(audio_data).flatten())

    if not audio_chunks:
        raise ValueError("没有生成任何音频数据")

    # 合并所有音频块
    full_audio = np.concatenate(audio_chunks)

    # 归一化到 [-1, 1] 范围
    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
        full_audio = np.clip(full_audio, -1.0, 1.0)

    # 转换为 int16 格式
    audio_int16 = (full_audio * 32767).astype(np.int16)

    # 先保存为临时WAV文件（pydub需要）
    temp_wav_path = output_path.replace('.mp3', '.wav')
    wavfile.write(temp_wav_path, actual_sample_rate, audio_int16)
    
    # 使用pydub转换为MP3
    try:
        audio_segment = AudioSegment.from_wav(temp_wav_path)
        audio_segment.export(output_path, format="mp3", bitrate="192k")
        # 删除临时WAV文件
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        logging.info(f"音频已保存为MP3: {output_path}, 采样率: {actual_sample_rate}Hz")
    except Exception as e:
        # 如果MP3转换失败，保留WAV文件并重命名
        logging.warning(f"MP3转换失败: {e}，保留WAV格式")
        if os.path.exists(temp_wav_path):
            os.rename(temp_wav_path, output_path.replace('.mp3', '.wav'))
        raise


def execute_async_generation(
    step_id: str,
    tts_text: str,
    mode: str,
    sft_dropdown: str,
    prompt_text: str,
    prompt_wav_data: Optional[bytes],
    instruct_text: str,
    seed: int,
    stream: bool,
    speed: float,
):
    """在后台线程中执行音频生成任务"""
    try:
        # 更新状态为运行中
        with task_status_lock:
            task_status[step_id]["status"] = "processing"
            task_status[step_id]["updated_at"] = datetime.now().isoformat()

        # 创建停止事件
        stop_event = threading.Event()
        with task_stop_lock:
            task_stop_events[step_id] = stop_event

        # 生成音频
        model_output = audio_generator.generate_audio(
            tts_text=tts_text,
            mode=mode,
            sft_dropdown=sft_dropdown,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav_data,
            instruct_text=instruct_text,
            seed=seed,
            stream=stream,
            speed=speed,
            stop_event=stop_event,
        )

        # 检查是否被停止
        if stop_event.is_set():
            with task_status_lock:
                task_status[step_id]["status"] = "cancelled"
                task_status[step_id]["updated_at"] = datetime.now().isoformat()
            with task_stop_lock:
                task_stop_events.pop(step_id, None)
            return

        # 保存到临时文件（MP3格式）
        output_filename = f"{step_id}.mp3"
        output_path = os.path.join(temp_output_dir, output_filename)
        save_audio_to_file(model_output, output_path, audio_generator.sample_rate)

        # 再次检查是否被停止（保存过程中可能被停止）
        if stop_event.is_set():
            # 删除已保存的文件
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            with task_status_lock:
                task_status[step_id]["status"] = "cancelled"
                task_status[step_id]["updated_at"] = datetime.now().isoformat()
            with task_stop_lock:
                task_stop_events.pop(step_id, None)
            return

        # 更新状态为完成
        with task_status_lock:
            task_status[step_id]["status"] = "completed"
            task_status[step_id]["result_path"] = output_path
            task_status[step_id]["updated_at"] = datetime.now().isoformat()

        # 保存任务状态到文件
        save_task_status()

        # 清理停止事件
        with task_stop_lock:
            task_stop_events.pop(step_id, None)

    except Exception as e:
        logging.error(f"异步生成任务 {step_id} 失败: {str(e)}", exc_info=True)
        # 更新状态为失败
        with task_status_lock:
            task_status[step_id]["status"] = "failed"
            task_status[step_id]["error"] = str(e)
            task_status[step_id]["updated_at"] = datetime.now().isoformat()

        # 保存任务状态到文件
        save_task_status()

        # 清理停止事件
        with task_stop_lock:
            task_stop_events.pop(step_id, None)


@app.post("/generate_audio_enhanced_async")
async def generate_audio_enhanced_async(
    step_id: str = Form(description="任务ID，用于查询状态和下载结果"),
    tts_text: str = Form(),
    mode: str = Form(
        description="推理模式: 预训练音色/3s极速复刻/跨语种复刻/自然语言控制"
    ),
    sft_dropdown: str = Form(""),
    prompt_text: str = Form(""),
    prompt_wav: Optional[UploadFile] = File(None),
    instruct_text: str = Form(""),
    seed: int = Form(0),
    stream: bool = Form(False),
    speed: float = Form(1.0),
):
    """异步生成音频 - 立即返回，后台执行生成任务"""
    if audio_generator is None:
        raise HTTPException(status_code=500, detail="模型未初始化，请先调用 /init 接口")

    # 检查 step_id 是否已存在
    with task_status_lock:
        if step_id in task_status:
            current_status = task_status[step_id]["status"]
            if current_status in ["pending", "processing"]:
                raise HTTPException(
                    status_code=400, detail=f"任务 {step_id} 已存在且正在处理中"
                )

    # 处理 prompt_wav
    prompt_wav_data = None
    if prompt_wav:
        prompt_wav_data = await prompt_wav.read()

    # 创建任务状态记录
    with task_status_lock:
        task_status[step_id] = {
            "status": "pending",
            "result_path": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    # 提交后台任务
    task_executor.submit(
        execute_async_generation,
        step_id,
        tts_text,
        mode,
        sft_dropdown,
        prompt_text,
        prompt_wav_data,
        instruct_text,
        seed,
        stream,
        speed,
    )

    return JSONResponse(
        content={
            "status": "pending",
            "message": "任务已提交，正在后台生成",
            "step_id": step_id,
            "is_success": True,
        }
    )


@app.get("/get_task_status/{step_id}")
async def get_task_status(step_id: str = Path(description="任务ID")):
    """查询任务状态"""
    with task_status_lock:
        if step_id not in task_status:
            raise HTTPException(status_code=404, detail=f"任务 {step_id} 不存在")

        task_info = task_status[step_id].copy()
        status = task_info["status"]

        response = {
            "step_id": step_id,
            "status": status,
            "created_at": task_info["created_at"],
            "updated_at": task_info["updated_at"],
        }

        if status == "completed":
            # 提供下载地址
            result_path = task_info["result_path"]
            if result_path and os.path.exists(result_path):
                response["download_url"] = f"/download_result/{step_id}"
                response["result_path"] = result_path
            else:
                response["status"] = "failed"
                response["error"] = "结果文件不存在"
        elif status == "failed":
            response["error"] = task_info.get("error", "未知错误")

        return JSONResponse(content=response)


@app.get("/download_result/{step_id}")
async def download_result(step_id: str = Path(description="任务ID")):
    """下载生成结果"""
    with task_status_lock:
        if step_id not in task_status:
            raise HTTPException(status_code=404, detail=f"任务 {step_id} 不存在")

        task_info = task_status[step_id]
        status = task_info["status"]

        if status != "completed":
            raise HTTPException(
                status_code=400, detail=f"任务 {step_id} 尚未完成，当前状态: {status}"
            )

        result_path = task_info.get("result_path")
        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="结果文件不存在")

    # 根据文件扩展名确定媒体类型
    if result_path.endswith('.mp3'):
        media_type = "audio/mpeg"
        filename = f"{step_id}.mp3"
    else:
        media_type = "audio/wav"
        filename = f"{step_id}.wav"
    
    return FileResponse(
        path=result_path, media_type=media_type, filename=filename
    )


@app.post("/stop_generation")
async def stop_generation_endpoint(task_id: Optional[int] = Form(None)):
    """停止生成任务（旧接口，兼容性保留）"""
    with task_stop_lock:
        if task_id is not None:
            if task_id in task_stop_events:
                task_stop_events[task_id].set()
                return JSONResponse(
                    content={"status": "success", "message": f"任务 {task_id} 已停止"}
                )
            else:
                return JSONResponse(
                    content={"status": "error", "message": f"任务 {task_id} 不存在"}
                )
        else:
            # 停止所有任务
            for event in task_stop_events.values():
                event.set()
            return JSONResponse(
                content={"status": "success", "message": "所有任务已停止"}
            )


@app.post("/stop_async_task/{step_id}")
async def stop_async_task(step_id: str = Path(description="任务ID")):
    """停止异步生成任务"""
    with task_status_lock:
        if step_id not in task_status:
            raise HTTPException(status_code=404, detail=f"任务 {step_id} 不存在")

        current_status = task_status[step_id]["status"]
        if current_status not in ["pending", "processing"]:
            raise HTTPException(
                status_code=400,
                detail=f"任务 {step_id} 当前状态为 {current_status}，无法停止",
            )

    # 设置停止事件
    with task_stop_lock:
        if step_id in task_stop_events:
            task_stop_events[step_id].set()

    # 更新状态
    with task_status_lock:
        task_status[step_id]["status"] = "cancelled"
        task_status[step_id]["updated_at"] = datetime.now().isoformat()

    return JSONResponse(
        content={
            "is_success": True,
            "message": f"任务 {step_id} 已停止",
            "step_id": step_id,
        }
    )


@app.get("/list_speakers")
async def list_speakers():
    """获取可用的预训练音色列表"""
    if audio_generator is None:
        raise HTTPException(status_code=500, detail="模型未初始化，请先调用 /init 接口")

    try:
        speakers = audio_generator.cosyvoice.list_available_spks()
        return JSONResponse(content={"speakers": speakers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取音色列表失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return JSONResponse(
        content={"status": "ok", "model_loaded": audio_generator is not None}
    )


if __name__ == "__main__":
    # 获取项目根目录（从 runtime/python/fastapi 向上三级）
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    default_model_dir = os.path.join(
        project_root, "pretrained_models", "CosyVoice2-0.5B"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help="local path or modelscope repo id",
    )
    parser.add_argument("--load_jit", action="store_true", help="load jit model")
    parser.add_argument("--load_trt", action="store_true", help="load tensorrt model")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument(
        "--trt_concurrent", type=int, default=1, help="tensorrt concurrent number"
    )
    args = parser.parse_args()

    # 如果提供了模型路径，自动初始化
    if args.model_dir:
        # 检查模型目录是否存在
        if not os.path.exists(args.model_dir):
            logging.warning(
                f"模型目录不存在: {args.model_dir}，请使用 /init 接口手动初始化"
            )
        else:
            try:
                logging.info(f"尝试初始化 CosyVoice 模型: {args.model_dir}")
                try:
                    cosyvoice = CosyVoice(
                        args.model_dir,
                        load_jit=args.load_jit,
                        load_trt=args.load_trt,
                        fp16=args.fp16,
                        trt_concurrent=args.trt_concurrent,
                    )
                    logging.info("成功使用 CosyVoice 初始化模型")
                except Exception as e1:
                    logging.debug(f"CosyVoice 初始化失败: {e1}，尝试 CosyVoice2...")
                    try:
                        cosyvoice = CosyVoice2(
                            args.model_dir,
                            load_jit=args.load_jit,
                            load_trt=args.load_trt,
                            fp16=args.fp16,
                            trt_concurrent=args.trt_concurrent,
                        )
                        logging.info("成功使用 CosyVoice2 初始化模型")
                    except Exception as e2:
                        # 检查是否是 matcha 模块缺失的问题
                        error_detail = str(e2)
                        if (
                            "No module named 'matcha'" in error_detail
                            or "ModuleNotFoundError" in str(type(e2).__name__)
                        ):
                            matcha_tts_path = os.path.join(
                                project_root, "third_party", "Matcha-TTS"
                            )
                            error_msg = (
                                f"CosyVoice2 初始化失败: 缺少 matcha 模块。\n"
                                f"CosyVoice错误: {type(e1).__name__}: {str(e1)}\n"
                                f"CosyVoice2错误: {type(e2).__name__}: {str(e2)}\n\n"
                                f"解决方案：\n"
                                f"1. 检查 Matcha-TTS 路径是否存在: {matcha_tts_path}\n"
                                f"2. 如果不存在，请执行: git submodule update --init --recursive\n"
                                f"3. 或者手动克隆: git clone https://github.com/shivammehta25/Matcha-TTS.git {matcha_tts_path}"
                            )
                        else:
                            error_msg = f"无法使用 CosyVoice 或 CosyVoice2 初始化模型。CosyVoice错误: {type(e1).__name__}: {str(e1)}，CosyVoice2错误: {type(e2).__name__}: {str(e2)}"
                        logging.error(error_msg)
                        logging.error("详细错误堆栈:", exc_info=True)
                        raise TypeError(error_msg)
                audio_generator = AudioGenerator(cosyvoice)
                logging.info(f"模型已自动初始化: {args.model_dir}")
            except Exception as e:
                logging.warning(
                    f"自动初始化模型失败: {type(e).__name__}: {str(e)}，请使用 /init 接口手动初始化"
                )
                logging.debug("详细错误堆栈:", exc_info=True)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
