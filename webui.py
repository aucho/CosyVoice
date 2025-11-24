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
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import random
import librosa
import threading
import re
from datetime import datetime
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

# 全局停止标志
stop_generation = threading.Event()
# 任务停止标志字典，用于管理多个并发任务
task_stop_flags = {}
task_stop_lock = threading.Lock()

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def stop_generation_func():
    """停止生成函数"""
    stop_generation.set()
    # 停止所有任务
    with task_stop_lock:
        for task_id in task_stop_flags:
            task_stop_flags[task_id].set()
    return "生成已停止"

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def count_words(text):
    """计算文本的单词数：中文按字符数，英文按单词数"""
    if not text or not text.strip():
        return 0
    # 分离中文字符和英文单词
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 英文单词：按空格分割，过滤空字符串
    english_words = len([w for w in re.findall(r'[a-zA-Z]+', text)])
    # 其他字符（标点、数字等）计入中文字符
    return chinese_chars + english_words

def split_text_by_words(text, words_per_segment):
    """根据单词数分割文本"""
    if not text or not text.strip():
        return []
    
    segments = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # 跳过空白字符
        while current_pos < text_length and text[current_pos].isspace():
            current_pos += 1
        
        if current_pos >= text_length:
            break
        
        # 找到当前段的结束位置
        segment_start = current_pos
        word_count = 0
        segment_end = current_pos
        
        # 寻找分割点
        while segment_end < text_length and word_count < words_per_segment:
            char = text[segment_end]
            
            # 中文字符直接计数
            if re.match(r'[\u4e00-\u9fff]', char):
                word_count += 1
                segment_end += 1
            # 英文单词：找到单词边界
            elif re.match(r'[a-zA-Z]', char):
                # 找到整个单词
                word_match = re.match(r'[a-zA-Z]+', text[segment_end:])
                if word_match:
                    word_count += 1
                    segment_end += len(word_match.group())
                else:
                    segment_end += 1
            else:
                # 其他字符（标点、空格等）继续
                segment_end += 1
        
        # 如果还没达到单词数限制就已到文本末尾，直接取剩余部分
        if segment_end >= text_length:
            segment = text[segment_start:].strip()
            if segment:
                segments.append(segment)
            break
        
        # 尝试在合适的位置分割（避免截断单词）
        # 向后查找空格、标点等分隔符
        best_split = segment_end
        search_start = max(segment_start, segment_end - 50)  # 在最后50个字符内查找
        
        for i in range(segment_end - 1, search_start, -1):
            if text[i] in [' ', '\n', '\t', '。', '，', '.', ',', ';', ':', '!', '?', '！', '？']:
                best_split = i + 1
                break
        
        segment = text[segment_start:best_split].strip()
        if segment:
            segments.append(segment)
        
        current_pos = best_split
    
    return segments if segments else [text.strip()] if text.strip() else []

def process_text_split(input_text, words_per_segment):
    """处理文本分割，返回分割后的文本列表和分段数量"""
    if not input_text or not input_text.strip():
        return [], 0
    
    segments = split_text_by_words(input_text, words_per_segment)
    return segments, len(segments)


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
    noise_level = random.uniform(0.04, 0.06)
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

def generate_batch_audio(text_segments, output_dir, mode_checkbox_group, sft_dropdown, prompt_text, 
                         prompt_wav_upload, prompt_wav_record, instruct_text, seed, stream, speed,
                         enable_unstable_effects=False):
    """批量生成音频并保存到指定目录"""
    # 为当前任务创建独立的停止标志
    task_id = threading.current_thread().ident
    task_stop_event = threading.Event()
    with task_stop_lock:
        task_stop_flags[task_id] = task_stop_event
    
    if not text_segments or len(text_segments) == 0:
        empty_audio = np.zeros(1000, dtype=np.float32)
        yield "没有文本需要生成", (cosyvoice.sample_rate, empty_audio), None
        with task_stop_lock:
            task_stop_flags.pop(task_id, None)
        return
    
    # 规范化输出目录路径（处理Windows路径）
    output_dir = str(output_dir).strip()
    if not output_dir:
        # WSL/Linux环境下的默认路径：使用当前工作目录下的audios文件夹
        output_dir = "./audios"
    
    # 确保路径正确处理（支持相对路径和绝对路径）
    # 在Windows上，检查路径是否以驱动器字母开头（如 c:/ 或 c:\）
    if os.name == 'nt':  # Windows
        # Windows 路径处理：确保识别 c:/ 或 c:\ 为绝对路径
        if len(output_dir) >= 2 and output_dir[1] == ':':
            # 路径以驱动器字母开头，已经是绝对路径
            output_path = Path(output_dir)
        else:
            # 相对路径，转换为绝对路径（相对于当前工作目录）
            output_path = Path(output_dir).resolve()
    else:
        # Linux/WSL系统：支持相对路径和绝对路径
        output_path = Path(output_dir)
        # 如果是相对路径（如 ./audios），resolve() 会将其转换为绝对路径
        # 如果是绝对路径（如 /home/user/audios），保持原样
        if not output_path.is_absolute():
            output_path = output_path.resolve()
        else:
            # 已经是绝对路径，直接使用
            output_path = Path(output_dir)
    
    # 创建时间命名的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = output_path / timestamp
    
    # 确保父目录存在
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'创建保存目录: {save_dir}')
        empty_audio = np.zeros(1000, dtype=np.float32)
        display_path = format_path_for_display(save_dir.resolve())
        yield f"保存目录已创建: {display_path}", (cosyvoice.sample_rate, empty_audio), None
    except Exception as e:
        error_msg = f"无法创建保存目录 {save_dir}: {str(e)}"
        logging.error(error_msg)
        empty_audio = np.zeros(1000, dtype=np.float32)
        yield error_msg, (cosyvoice.sample_rate, empty_audio), None
        return
    
    total_segments = len(text_segments)
    
    # 处理prompt音频
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    
    # 重置全局停止标志（仅用于兼容旧代码）
    stop_generation.clear()
    # 重置当前任务的停止标志
    task_stop_event.clear()
    
    saved_count = 0
    empty_audio = np.zeros(1000, dtype=np.float32)
    download_file_path = None
    
    try:
        for idx, tts_text in enumerate(text_segments, 1):
            # 检查全局停止标志或当前任务停止标志
            if stop_generation.is_set() or task_stop_event.is_set():
                yield f"生成已停止（已生成 {saved_count}/{total_segments} 个音频）", (cosyvoice.sample_rate, empty_audio), download_file_path
                break
            
            if not tts_text or not tts_text.strip():
                continue
            
            yield f"正在生成第 {idx}/{total_segments} 个音频...", (cosyvoice.sample_rate, empty_audio), download_file_path
            
            try:
                current_speed = sample_speed_with_variation(speed, enable_unstable_effects)
                # 生成音频
                audio_data = None
                audio_chunks = []  # 收集所有音频片段
                
                if mode_checkbox_group == '预训练音色':
                    set_all_random_seed(seed)
                    for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=False, speed=current_speed):
                        # 收集所有音频片段，不要只取第一个
                        if 'tts_speech' in i:
                            audio_chunks.append(i['tts_speech'])
                elif mode_checkbox_group == '3s极速复刻':
                    if prompt_wav is None:
                        continue
                    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
                    set_all_random_seed(seed)
                    for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=current_speed):
                        if 'tts_speech' in i:
                            audio_chunks.append(i['tts_speech'])
                elif mode_checkbox_group == '跨语种复刻':
                    if prompt_wav is None:
                        continue
                    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
                    set_all_random_seed(seed)
                    for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False, speed=current_speed):
                        if 'tts_speech' in i:
                            audio_chunks.append(i['tts_speech'])
                else:  # 自然语言控制
                    set_all_random_seed(seed)
                    for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=False, speed=current_speed):
                        if 'tts_speech' in i:
                            audio_chunks.append(i['tts_speech'])
                
                # 拼接所有音频片段
                if audio_chunks:
                    # 将所有片段拼接在一起
                    # 首先统一所有片段的维度
                    processed_chunks = []
                    for chunk in audio_chunks:
                        # 确保每个片段都是 (channels, length) 格式
                        if chunk.dim() == 1:
                            chunk = chunk.unsqueeze(0)  # (length,) -> (1, length)
                        elif chunk.dim() == 2 and chunk.shape[0] > 1:
                            chunk = chunk[0:1]  # (batch, length) -> (1, length)
                        processed_chunks.append(chunk)
                    
                    # 在时间维度（dim=1）上拼接所有片段
                    audio_data = torch.cat(processed_chunks, dim=1)
                    logging.info(f'收集到 {len(audio_chunks)} 个音频片段，拼接后总长度: {audio_data.shape[1]} 采样点')
                else:
                    audio_data = None
                
                if audio_data is not None:
                    # 确保音频数据格式正确 (channels, length)
                    if audio_data.dim() == 1:
                        audio_data = audio_data.unsqueeze(0)  # (length,) -> (1, length)
                    elif audio_data.dim() == 2 and audio_data.shape[0] > 1:
                        # 如果是(batch, length)，取第一个
                        audio_data = audio_data[0:1]  # (batch, length) -> (1, length)
                    elif audio_data.dim() == 2 and audio_data.shape[0] == 1:
                        # 已经是(1, length)，保持不变
                        pass
                    else:
                        # 其他情况，尝试flatten并unsqueeze
                        audio_data = audio_data.flatten().unsqueeze(0)
                    
                    # 应用真人随机讲话效果
                    if enable_unstable_effects:
                        audio_data = apply_unstable_speech_effects(
                            audio_data,
                            cosyvoice.sample_rate,
                            add_trailing_pause=(idx != total_segments)
                        )

                    # 保存音频
                    audio_path = save_dir / f"{idx:04d}.wav"
                    try:
                        # 规范化路径，避免被识别为URI协议
                        audio_path_str = str(audio_path.resolve())
                        audio_path_str = os.path.normpath(audio_path_str)
                        if os.name == 'nt':  # Windows
                            audio_path_str = audio_path_str.replace('/', '\\')
                        torchaudio.save(audio_path_str, audio_data, cosyvoice.sample_rate)
                        
                        # 验证文件是否保存成功
                        if audio_path.exists() and audio_path.stat().st_size > 0:
                            saved_count += 1
                            abs_path = audio_path.resolve()
                            logging.info(f'成功保存音频: {abs_path} (大小: {audio_path.stat().st_size} 字节)')
                            
                            # 转换为 float32 格式的 numpy 数组，确保值在 -1.0 到 1.0 之间
                            audio_numpy = audio_data.squeeze().cpu().numpy().astype(np.float32)
                            # 确保值在有效范围内
                            if audio_numpy.max() > 1.0 or audio_numpy.min() < -1.0:
                                audio_numpy = np.clip(audio_numpy, -1.0, 1.0)
                            
                            yield f"已生成第 {idx}/{total_segments} 个音频，保存到: {format_path_for_display(abs_path)}", (cosyvoice.sample_rate, audio_numpy), download_file_path
                            
                            # 每段生成完后清显存
                            # 将张量移到CPU并删除
                            if audio_data.is_cuda:
                                audio_data = audio_data.cpu()
                            del audio_data
                            del audio_numpy
                            # 清空GPU缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            error_msg = f"文件保存失败: {audio_path} (文件不存在或大小为0)"
                            logging.error(error_msg)
                            # 即使保存失败也要清显存
                            if audio_data.is_cuda:
                                audio_data = audio_data.cpu()
                            del audio_data
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            yield error_msg, (cosyvoice.sample_rate, empty_audio), download_file_path
                    except Exception as save_error:
                        error_msg = f"保存音频文件时出错: {str(save_error)}"
                        logging.error(error_msg)
                        # 出错时也要清显存
                        if audio_data is not None:
                            if audio_data.is_cuda:
                                audio_data = audio_data.cpu()
                            del audio_data
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        yield f"第 {idx}/{total_segments} 个音频保存失败: {str(save_error)}", (cosyvoice.sample_rate, empty_audio), download_file_path
                else:
                    yield f"第 {idx}/{total_segments} 个音频生成失败", (cosyvoice.sample_rate, empty_audio), download_file_path
                
                # 清理音频片段列表
                del audio_chunks
                del processed_chunks
                    
            except Exception as e:
                logging.error(f'生成第 {idx} 个音频时出现错误: {e}')
                # 出错时也要清显存
                if 'audio_chunks' in locals():
                    del audio_chunks
                if 'audio_data' in locals():
                    if audio_data is not None and audio_data.is_cuda:
                        audio_data = audio_data.cpu()
                    del audio_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                yield f"第 {idx}/{total_segments} 个音频生成出错: {str(e)}", (cosyvoice.sample_rate, empty_audio), download_file_path
        
        if not stop_generation.is_set():
            # 合并所有生成的音频文件
            if saved_count > 0:
                try:
                    # 按序号顺序读取所有音频文件
                    audio_files = []
                    for idx in range(1, total_segments + 1):
                        audio_file = save_dir / f"{idx:04d}.wav"
                        if audio_file.exists() and audio_file.stat().st_size > 0:
                            audio_files.append(audio_file)
                    
                    if len(audio_files) > 0:
                        # 读取所有音频文件并拼接
                        merged_audio_chunks = []
                        for audio_file in audio_files:
                            waveform, sample_rate = torchaudio.load(str(audio_file))
                            # 确保格式统一：转换为 (channels, length) 格式
                            if waveform.dim() == 1:
                                waveform = waveform.unsqueeze(0)
                            elif waveform.dim() == 2 and waveform.shape[0] > 1:
                                waveform = waveform[0:1]  # 取第一个声道
                            merged_audio_chunks.append(waveform)
                        
                        # 在时间维度上拼接所有音频
                        if merged_audio_chunks:
                            merged_audio = torch.cat(merged_audio_chunks, dim=1)
                            
                            # 保存合并的音频
                            merged_audio_path = save_dir / "合并的音频.wav"
                            merged_audio_path_str = str(merged_audio_path.resolve())
                            merged_audio_path_str = os.path.normpath(merged_audio_path_str)
                            if os.name == 'nt':  # Windows
                                merged_audio_path_str = merged_audio_path_str.replace('/', '\\')
                            
                            torchaudio.save(merged_audio_path_str, merged_audio, cosyvoice.sample_rate)
                            
                            if merged_audio_path.exists() and merged_audio_path.stat().st_size > 0:
                                merged_display_path = format_path_for_display(merged_audio_path.resolve())
                                logging.info(f'成功合并音频: {merged_display_path}')
                                display_path = format_path_for_display(save_dir.resolve())
                                final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {display_path}\n合并音频已保存: {merged_display_path}"
                                download_file_path = str(merged_audio_path.resolve())
                            else:
                                display_path = format_path_for_display(save_dir.resolve())
                                final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {display_path}\n合并音频保存失败"
                        else:
                            display_path = format_path_for_display(save_dir.resolve())
                            final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {display_path}"
                    else:
                        display_path = format_path_for_display(save_dir.resolve())
                        final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {display_path}"
                except Exception as merge_error:
                    logging.error(f'合并音频时出错: {merge_error}')
                    display_path = format_path_for_display(save_dir.resolve())
                    final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {display_path}\n合并音频出错: {str(merge_error)}"
            else:
                display_path = format_path_for_display(save_dir.resolve())
                final_msg = f"全部完成！共生成 {saved_count}/{total_segments} 个音频，保存目录: {display_path}"
            
            logging.info(final_msg)
            yield final_msg, (cosyvoice.sample_rate, empty_audio), download_file_path
            
    except Exception as e:
        logging.error(f'批量生成过程中出现错误: {e}')
        yield f"批量生成出错: {str(e)}", (cosyvoice.sample_rate, empty_audio), download_file_path
    finally:
        # 清理当前任务的停止标志
        with task_stop_lock:
            task_stop_flags.pop(task_id, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    # 为当前任务创建独立的停止标志
    task_id = threading.current_thread().ident
    task_stop_event = threading.Event()
    with task_stop_lock:
        task_stop_flags[task_id] = task_stop_event
    
    try:
        # 重置停止标志
        stop_generation.clear()
        task_stop_event.clear()
        
        if prompt_wav_upload is not None:
            prompt_wav = prompt_wav_upload
        elif prompt_wav_record is not None:
            prompt_wav = prompt_wav_record
        else:
            prompt_wav = None
        # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
        if mode_checkbox_group in ['自然语言控制']:
            if cosyvoice.instruct is False:
                gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir))
                yield (cosyvoice.sample_rate, default_data)
                return
            if instruct_text == '':
                gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
                yield (cosyvoice.sample_rate, default_data)
                return
            if prompt_wav is not None or prompt_text != '':
                gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
        # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
        if mode_checkbox_group in ['跨语种复刻']:
            if cosyvoice.instruct is True:
                gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
                yield (cosyvoice.sample_rate, default_data)
                return
            if instruct_text != '':
                gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
            if prompt_wav is None:
                gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
                yield (cosyvoice.sample_rate, default_data)
                return
            gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
        # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
        if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
            if prompt_wav is None:
                gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
                yield (cosyvoice.sample_rate, default_data)
                return
            if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
                gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
                yield (cosyvoice.sample_rate, default_data)
                return
        # sft mode only use sft_dropdown
        if mode_checkbox_group in ['预训练音色']:
            if instruct_text != '' or prompt_wav is not None or prompt_text != '':
                gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
            if sft_dropdown == '':
                gr.Warning('没有可用的预训练音色！')
                yield (cosyvoice.sample_rate, default_data)
                return
        # zero_shot mode only use prompt_wav prompt text
        if mode_checkbox_group in ['3s极速复刻']:
            if prompt_text == '':
                gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
                yield (cosyvoice.sample_rate, default_data)
                return
            if instruct_text != '':
                gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

        try:
            if mode_checkbox_group == '预训练音色':
                logging.info('get sft inference request')
                set_all_random_seed(seed)
                for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                    if stop_generation.is_set() or task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
            elif mode_checkbox_group == '3s极速复刻':
                logging.info('get zero_shot inference request')
                prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
                set_all_random_seed(seed)
                for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
                    if stop_generation.is_set() or task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
            elif mode_checkbox_group == '跨语种复刻':
                logging.info('get cross_lingual inference request')
                prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
                set_all_random_seed(seed)
                for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                    if stop_generation.is_set() or task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
            else:
                logging.info('get instruct inference request')
                set_all_random_seed(seed)
                for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
                    if stop_generation.is_set() or task_stop_event.is_set():
                        logging.info('生成被用户停止')
                        break
                    yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        except Exception as e:
            logging.error(f'生成过程中出现错误: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield (cosyvoice.sample_rate, default_data)
    finally:
        # 清理当前任务的停止标志
        with task_stop_lock:
            task_stop_flags.pop(task_id, None)

def format_path_for_display(path):
    """将路径格式化显示，将 /workspace 替换为 C:\\repo\\CosyVoice"""
    path_str = str(path)
    # 将 /workspace 替换为 C:\repo\CosyVoice
    if path_str.startswith('/workspace'):
        path_str = path_str.replace('/workspace', 'C:\\repo\\CosyVoice', 1)
    return path_str

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        with gr.Row():
            tts_text = gr.Textbox(label="输入合成文本", lines=5, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
            with gr.Column(scale=0.3):
                words_per_segment = gr.Number(value=10000, label="单段单词数", minimum=50, maximum=99999, step=100)
                split_button = gr.Button("分割文本", variant="secondary")
                segment_count_display = gr.Textbox(label="分段数量", value="0", interactive=False)
        
        # 分割后的文本框区域
        with gr.Column(visible=True) as segments_column:
            gr.Markdown("### 分割后的文本段（可编辑）")
            text_segments_list = []
            for i in range(100):  # 最多支持100个分段
                text_segments_list.append(gr.Textbox(label=f"分段 {i+1}", lines=3, interactive=True, visible=False))
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[1])  # 改为 [1] 表示 '3s极速复刻'
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[1]], scale=0.5)  # 改为 [1] 对应 '3s极速复刻' 的操作步骤
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        # 目录选择
        output_dir = gr.Textbox(label="输出目录", value="./audios", interactive=True)
        unstable_speech = gr.Checkbox(
            label="真人随机语气增强",
            value=False,
            info="开启后将自动加入语速波动、呼吸、环境噪声、随机停顿等效果"
        )
        
        with gr.Row():
            generate_button = gr.Button("开始生成", variant="primary")
            stop_button = gr.Button("停止生成", variant="stop")

        audio_output = gr.Audio(label="合成音频预览", autoplay=True, streaming=True)
        status_text = gr.Textbox(label="状态", value="就绪", interactive=False)
        download_file = gr.File(label="合并音频下载", interactive=False)

        # 更新分割功能
        def update_segments(input_text, words_per_seg):
            if not input_text or not input_text.strip():
                return [gr.update(visible=False, value="")] * 100 + [gr.update(value="0")]
            
            segments, segment_count = process_text_split(input_text, words_per_seg)
            
            updates = []
            for i in range(100):
                if i < len(segments):
                    updates.append(gr.update(visible=True, value=segments[i], label=f"分段 {i+1}"))
                else:
                    updates.append(gr.update(visible=False, value=""))
            
            updates.append(gr.update(value=str(segment_count)))
            return updates
        
        split_button.click(
            fn=update_segments,
            inputs=[tts_text, words_per_segment],
            outputs=text_segments_list + [segment_count_display]
        )
        
        # 批量生成函数
        def batch_generate_wrapper(*args):
            # 从文本框中提取所有文本段
            text_segments = [tb for tb in args[:100] if tb and tb.strip()]
            
            if not text_segments:
                yield "没有文本需要生成", (cosyvoice.sample_rate, default_data), None
                return
            
            # 获取其他参数
            output_dir_path = args[100]
            mode_checkbox_group = args[101]
            sft_dropdown = args[102]
            prompt_text = args[103]
            prompt_wav_upload = args[104]
            prompt_wav_record = args[105]
            instruct_text = args[106]
            seed = args[107]
            stream = args[108]
            speed = args[109]
            unstable_mode = args[110]
            
            # 调用批量生成函数
            for status, audio, download_path in generate_batch_audio(
                text_segments, output_dir_path, mode_checkbox_group, sft_dropdown,
                prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                seed, stream, speed, enable_unstable_effects=unstable_mode
            ):
                yield status, audio, download_path
        
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            batch_generate_wrapper,
            inputs=[*text_segments_list, output_dir, mode_checkbox_group, sft_dropdown, prompt_text, 
                   prompt_wav_upload, prompt_wav_record, instruct_text, seed, stream, speed, unstable_speech],
            outputs=[status_text, audio_output, download_file]
        )
        stop_button.click(stop_generation_func, inputs=[], outputs=[status_text])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    # 增加并发限制以支持多个网页同时运行多个任务
    # max_size: 队列最大长度，default_concurrency_limit: 默认并发数
    # 可以根据GPU显存和性能调整这些值
    demo.queue(max_size=args.max_queue_size, default_concurrency_limit=args.max_concurrent)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--max_concurrent',
                        type=int,
                        default=8,
                        help='最大并发任务数（默认8，可根据GPU显存调整）')
    parser.add_argument('--max_queue_size',
                        type=int,
                        default=20,
                        help='任务队列最大长度（默认20）')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
