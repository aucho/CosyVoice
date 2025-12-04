# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
音频生成器模块，封装了 webui 中的生成方法，方便 FastAPI 等接口调用
"""
import os
import re
import random
import threading
from pathlib import Path
from datetime import datetime
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# 默认参数
max_val = 0.8
prompt_sr = 16000


def postprocess(speech, sample_rate=16000, top_db=60, hop_length=220, win_length=440):
    """音频后处理：去除静音、归一化、添加尾部静音"""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(sample_rate * 0.2))], dim=1)
    return speech


def sample_speed_with_variation(base_speed, enable_variation):
    """根据需求随机生成语速波动"""
    if not enable_variation:
        return base_speed
    # 6%-14% 波动，正负随机
    variation = random.uniform(0.06, 0.14)
    direction = random.choice([-1, 1])
    varied_speed = base_speed * (1 + direction * variation)
    # 保证仍在可用范围
    return max(0.5, min(2.0, varied_speed))


def apply_tail_pitch_shift(waveform, sample_rate):
    """在句尾随机上扬/下沉语调"""
    if waveform.numel() == 0:
        return waveform
    tail_duration = min(0.4, max(0.1, waveform.shape[-1] / sample_rate * 0.2))
    tail_samples = max(1, int(sample_rate * tail_duration))
    tail = waveform[-tail_samples:].unsqueeze(0)
    semitone_shift = random.uniform(0.2, 0.4)
    if random.choice([True, False]):
        semitone_shift = -semitone_shift
    try:
        shifted = torchaudio.functional.pitch_shift(
            tail,
            sample_rate=sample_rate,
            n_steps=semitone_shift,
            bins_per_octave=12
        ).squeeze(0)
        if shifted.shape[-1] != tail_samples:
            if shifted.shape[-1] > tail_samples:
                shifted = shifted[:tail_samples]
            else:
                shifted = F.pad(shifted, (0, tail_samples - shifted.shape[-1]))
        waveform[-tail_samples:] = shifted
    except Exception as exc:
        logging.warning(f'尾部语调调整失败: {exc}')
    return waveform


def create_breath_segment(sample_rate):
    """生成轻微呼吸音"""
    duration = random.uniform(0.12, 0.25)
    intensity = random.uniform(0.15, 0.35)
    breath_len = max(1, int(sample_rate * duration))
    breath = torch.randn(breath_len) * 0.015 * intensity
    window = torch.hann_window(breath_len)
    return breath * window


def append_breath_and_pause(waveform, sample_rate, add_breath=True):
    """在段落末尾加入呼吸与停顿"""
    pause_duration = random.uniform(0.2, 0.5)
    pause = torch.zeros(max(1, int(sample_rate * pause_duration)))
    segments = [waveform]
    if add_breath:
        breath = create_breath_segment(sample_rate)
        segments.append(breath)
    segments.append(pause)
    return torch.cat(segments)


def add_environment_noise(waveform):
    """添加轻微环境噪声"""
    if waveform.numel() == 0:
        return waveform
    noise_level = random.uniform(0.1, 0.3)
    env_noise = torch.randn_like(waveform) * 0.02
    mixed = waveform * (1 - noise_level) + env_noise * noise_level
    max_val = mixed.abs().max()
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed


def apply_unstable_speech_effects(audio_tensor, sample_rate, add_trailing_pause):
    """应用真人随机讲话效果"""
    if audio_tensor.dim() == 2:
        waveform = audio_tensor.squeeze(0).detach().cpu()
    else:
        waveform = audio_tensor.detach().cpu()
    waveform = apply_tail_pitch_shift(waveform, sample_rate)
    waveform = add_environment_noise(waveform)
    if add_trailing_pause:
        waveform = append_breath_and_pause(waveform, sample_rate, add_breath=True)
    return waveform.unsqueeze(0)


class AudioGenerator:
    """音频生成器类，封装了各种生成方法"""
    
    def __init__(self, cosyvoice: Union[CosyVoice, CosyVoice2]):
        """
        初始化音频生成器
        
        Args:
            cosyvoice: CosyVoice 或 CosyVoice2 实例
        """
        self.cosyvoice = cosyvoice
        self.sample_rate = cosyvoice.sample_rate
        self.task_stop_flags = {}
        self.task_stop_lock = threading.Lock()
    
    def generate_audio(
        self,
        tts_text: str,
        mode: str,
        sft_dropdown: str = '',
        prompt_text: str = '',
        prompt_wav: Optional[Union[str, bytes]] = None,
        instruct_text: str = '',
        seed: int = 0,
        stream: bool = False,
        speed: float = 1.0,
        stop_event: Optional[threading.Event] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        生成单个音频（流式或非流式）
        
        Args:
            tts_text: 要合成的文本
            mode: 推理模式 ('预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制')
            sft_dropdown: 预训练音色ID（预训练音色模式需要）
            prompt_text: prompt文本（3s极速复刻模式需要）
            prompt_wav: prompt音频文件路径或字节数据
            instruct_text: instruct文本（自然语言控制模式需要）
            seed: 随机种子
            stream: 是否流式推理
            speed: 语速（0.5-2.0，仅非流式推理支持）
            stop_event: 停止事件，用于中断生成
        
        Yields:
            (sample_rate, audio_array): 音频采样率和音频数据（numpy数组）
        """
        # 为当前任务创建独立的停止标志
        task_id = threading.current_thread().ident
        task_stop_event = stop_event or threading.Event()
        with self.task_stop_lock:
            self.task_stop_flags[task_id] = task_stop_event
        
        try:
            # 处理prompt音频
            prompt_speech_16k = None
            if prompt_wav is not None:
                if isinstance(prompt_wav, str):
                    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr), sample_rate=prompt_sr)
                elif isinstance(prompt_wav, bytes):
                    # 处理字节数据（需要先保存为临时文件或使用其他方法）
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file.write(prompt_wav)
                        tmp_path = tmp_file.name
                    try:
                        prompt_speech_16k = postprocess(load_wav(tmp_path, prompt_sr), sample_rate=prompt_sr)
                    finally:
                        os.unlink(tmp_path)
            
            set_all_random_seed(seed)
            
            # 根据模式调用不同的推理方法
            if mode == '预训练音色':
                logging.info('get sft inference request')
                for i in self.cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                    if task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    audio_data = i['tts_speech']
                    if isinstance(audio_data, torch.Tensor):
                        audio_array = audio_data.numpy().flatten()
                    else:
                        audio_array = audio_data.flatten()
                    yield (self.sample_rate, audio_array)
                    
            elif mode == '3s极速复刻':
                logging.info('get zero_shot inference request')
                if prompt_speech_16k is None:
                    raise ValueError('3s极速复刻模式需要提供prompt音频')
                for i in self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
                    if task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    audio_data = i['tts_speech']
                    if isinstance(audio_data, torch.Tensor):
                        audio_array = audio_data.numpy().flatten()
                    else:
                        audio_array = audio_data.flatten()
                    yield (self.sample_rate, audio_array)
                    
            elif mode == '跨语种复刻':
                logging.info('get cross_lingual inference request')
                if prompt_speech_16k is None:
                    raise ValueError('跨语种复刻模式需要提供prompt音频')
                for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                    if task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    audio_data = i['tts_speech']
                    if isinstance(audio_data, torch.Tensor):
                        audio_array = audio_data.numpy().flatten()
                    else:
                        audio_array = audio_data.flatten()
                    yield (self.sample_rate, audio_array)
                    
            else:  # 自然语言控制
                logging.info('get instruct inference request')
                for i in self.cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
                    if task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    audio_data = i['tts_speech']
                    if isinstance(audio_data, torch.Tensor):
                        audio_array = audio_data.numpy().flatten()
                    else:
                        audio_array = audio_data.flatten()
                    yield (self.sample_rate, audio_array)
                    
        except Exception as e:
            logging.error(f'生成过程中出现错误: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 返回空音频
            default_data = np.zeros(self.sample_rate)
            yield (self.sample_rate, default_data)
        finally:
            # 清理当前任务的停止标志
            with self.task_stop_lock:
                self.task_stop_flags.pop(task_id, None)
    
    def generate_batch_audio(
        self,
        text_segments: List[str],
        output_dir: str,
        mode: str,
        sft_dropdown: str = '',
        prompt_text: str = '',
        prompt_wav: Optional[Union[str, bytes]] = None,
        instruct_text: str = '',
        seed: int = 0,
        stream: bool = False,
        speed: float = 1.0,
        enable_unstable_effects: bool = False,
        stop_event: Optional[threading.Event] = None
    ) -> Generator[Tuple[str, Optional[str]], None, None]:
        """
        批量生成音频并保存到指定目录
        
        Args:
            text_segments: 文本段列表
            output_dir: 输出目录路径
            mode: 推理模式
            sft_dropdown: 预训练音色ID
            prompt_text: prompt文本
            prompt_wav: prompt音频文件路径或字节数据
            instruct_text: instruct文本
            seed: 随机种子
            stream: 是否流式推理（批量生成建议设为False）
            speed: 语速
            enable_unstable_effects: 是否启用真人效果
            stop_event: 停止事件
        
        Yields:
            (status_message, download_file_path): 状态消息和合并音频文件路径
        """
        # 为当前任务创建独立的停止标志
        task_id = threading.current_thread().ident
        task_stop_event = stop_event or threading.Event()
        with self.task_stop_lock:
            self.task_stop_flags[task_id] = task_stop_event
        
        if not text_segments or len(text_segments) == 0:
            yield "没有文本需要生成", None
            with self.task_stop_lock:
                self.task_stop_flags.pop(task_id, None)
            return
        
        # 规范化输出目录路径
        output_dir = str(output_dir).strip()
        if not output_dir:
            output_dir = "./audios"
        
        if os.name == 'nt':  # Windows
            if len(output_dir) >= 2 and output_dir[1] == ':':
                output_path = Path(output_dir)
            else:
                output_path = Path(output_dir).resolve()
        else:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                output_path = output_path.resolve()
            else:
                output_path = Path(output_dir)
        
        # 创建时间命名的文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = output_path / timestamp
        
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'创建保存目录: {save_dir}')
            yield f"保存目录已创建: {save_dir}", None
        except Exception as e:
            error_msg = f"无法创建保存目录 {save_dir}: {str(e)}"
            logging.error(error_msg)
            yield error_msg, None
            return
        
        total_segments = len(text_segments)
        
        # 处理prompt音频
        prompt_speech_16k = None
        if prompt_wav is not None:
            if isinstance(prompt_wav, str):
                prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr), sample_rate=prompt_sr)
            elif isinstance(prompt_wav, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(prompt_wav)
                    tmp_path = tmp_file.name
                try:
                    prompt_speech_16k = postprocess(load_wav(tmp_path, prompt_sr), sample_rate=prompt_sr)
                finally:
                    os.unlink(tmp_path)
        
        task_stop_event.clear()
        saved_count = 0
        download_file_path = None
        
        try:
            for idx, tts_text in enumerate(text_segments, 1):
                if task_stop_event.is_set():
                    yield f"生成已停止（已生成 {saved_count}/{total_segments} 个音频）", download_file_path
                    break
                
                if not tts_text or not tts_text.strip():
                    continue
                
                yield f"正在生成第 {idx}/{total_segments} 个音频...", download_file_path
                
                try:
                    current_speed = sample_speed_with_variation(speed, enable_unstable_effects)
                    audio_data = None
                    audio_chunks = []
                    
                    set_all_random_seed(seed)
                    
                    if mode == '预训练音色':
                        for i in self.cosyvoice.inference_sft(tts_text, sft_dropdown, stream=False, speed=current_speed):
                            if 'tts_speech' in i:
                                audio_chunks.append(i['tts_speech'])
                    elif mode == '3s极速复刻':
                        if prompt_speech_16k is None:
                            continue
                        for i in self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=current_speed):
                            if 'tts_speech' in i:
                                audio_chunks.append(i['tts_speech'])
                    elif mode == '跨语种复刻':
                        if prompt_speech_16k is None:
                            continue
                        for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False, speed=current_speed):
                            if 'tts_speech' in i:
                                audio_chunks.append(i['tts_speech'])
                    else:  # 自然语言控制
                        for i in self.cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=False, speed=current_speed):
                            if 'tts_speech' in i:
                                audio_chunks.append(i['tts_speech'])
                    
                    # 拼接所有音频片段
                    if audio_chunks:
                        processed_chunks = []
                        for chunk in audio_chunks:
                            if chunk.dim() == 1:
                                chunk = chunk.unsqueeze(0)
                            elif chunk.dim() == 2 and chunk.shape[0] > 1:
                                chunk = chunk[0:1]
                            processed_chunks.append(chunk)
                        
                        audio_data = torch.cat(processed_chunks, dim=1)
                        logging.info(f'收集到 {len(audio_chunks)} 个音频片段，拼接后总长度: {audio_data.shape[1]} 采样点')
                    
                    if audio_data is not None:
                        # 确保音频数据格式正确
                        if audio_data.dim() == 1:
                            audio_data = audio_data.unsqueeze(0)
                        elif audio_data.dim() == 2 and audio_data.shape[0] > 1:
                            audio_data = audio_data[0:1]
                        
                        # 应用真人随机讲话效果
                        if enable_unstable_effects:
                            audio_data = apply_unstable_speech_effects(
                                audio_data,
                                self.sample_rate,
                                add_trailing_pause=(idx != total_segments)
                            )
                        
                        # 保存音频
                        audio_path = save_dir / f"{idx:04d}.wav"
                        try:
                            audio_path_str = str(audio_path.resolve())
                            audio_path_str = os.path.normpath(audio_path_str)
                            if os.name == 'nt':
                                audio_path_str = audio_path_str.replace('/', '\\')
                            torchaudio.save(audio_path_str, audio_data, self.sample_rate)
                            
                            if audio_path.exists() and audio_path.stat().st_size > 0:
                                saved_count += 1
                                logging.info(f'成功保存音频: {audio_path.resolve()}')
                                yield f"已生成第 {idx}/{total_segments} 个音频，保存到: {audio_path}", download_file_path
                                
                                if audio_data.is_cuda:
                                    audio_data = audio_data.cpu()
                                del audio_data
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                if audio_data.is_cuda:
                                    audio_data = audio_data.cpu()
                                del audio_data
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                yield f"文件保存失败: {audio_path}", download_file_path
                        except Exception as save_error:
                            logging.error(f"保存音频文件时出错: {str(save_error)}")
                            if audio_data is not None:
                                if audio_data.is_cuda:
                                    audio_data = audio_data.cpu()
                                del audio_data
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            yield f"第 {idx}/{total_segments} 个音频保存失败: {str(save_error)}", download_file_path
                    else:
                        yield f"第 {idx}/{total_segments} 个音频生成失败", download_file_path
                    
                    del audio_chunks
                    if 'processed_chunks' in locals():
                        del processed_chunks
                        
                except Exception as e:
                    logging.error(f'生成第 {idx} 个音频时出现错误: {e}')
                    if 'audio_chunks' in locals():
                        del audio_chunks
                    if 'audio_data' in locals():
                        if audio_data is not None and audio_data.is_cuda:
                            audio_data = audio_data.cpu()
                        del audio_data
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    yield f"第 {idx}/{total_segments} 个音频生成出错: {str(e)}", download_file_path
            
            if not task_stop_event.is_set():
                # 合并所有生成的音频文件
                if saved_count > 0:
                    try:
                        audio_files = []
                        for idx in range(1, total_segments + 1):
                            audio_file = save_dir / f"{idx:04d}.wav"
                            if audio_file.exists() and audio_file.stat().st_size > 0:
                                audio_files.append(audio_file)
                        
                        if len(audio_files) > 0:
                            merged_audio_chunks = []
                            for audio_file in audio_files:
                                waveform, sample_rate = torchaudio.load(str(audio_file))
                                if waveform.dim() == 1:
                                    waveform = waveform.unsqueeze(0)
                                elif waveform.dim() == 2 and waveform.shape[0] > 1:
                                    waveform = waveform[0:1]
                                merged_audio_chunks.append(waveform)
                            
                            if merged_audio_chunks:
                                merged_audio = torch.cat(merged_audio_chunks, dim=1)
                                merged_audio_path = save_dir / "合并的音频.wav"
                                merged_audio_path_str = str(merged_audio_path.resolve())
                                merged_audio_path_str = os.path.normpath(merged_audio_path_str)
                                if os.name == 'nt':
                                    merged_audio_path_str = merged_audio_path_str.replace('/', '\\')
                                
                                torchaudio.save(merged_audio_path_str, merged_audio, self.sample_rate)
                                
                                if merged_audio_path.exists() and merged_audio_path.stat().st_size > 0:
                                    logging.info(f'成功合并音频: {merged_audio_path.resolve()}')
                                    final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {save_dir}\n合并音频已保存: {merged_audio_path}"
                                    download_file_path = str(merged_audio_path.resolve())
                                else:
                                    final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {save_dir}\n合并音频保存失败"
                            else:
                                final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {save_dir}"
                        else:
                            final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {save_dir}"
                    except Exception as merge_error:
                        logging.error(f'合并音频时出错: {merge_error}')
                        final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {save_dir}\n合并音频出错: {str(merge_error)}"
                else:
                    final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {save_dir}"
                
                logging.info(final_msg)
                yield final_msg, download_file_path
                
        except Exception as e:
            logging.error(f'批量生成过程中出现错误: {e}')
            yield f"批量生成出错: {str(e)}", download_file_path
        finally:
            with self.task_stop_lock:
                self.task_stop_flags.pop(task_id, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

