# CosyVoice API 生成器模块

这个模块封装了 webui 中的音频生成方法，方便在 FastAPI 等接口中调用。

## 主要功能

- **AudioGenerator 类**：封装了所有音频生成功能
  - `generate_audio()`: 单个音频生成（支持流式）
  - `generate_batch_audio()`: 批量音频生成并保存

## 使用示例

### 基本使用

```python
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.api.generator import AudioGenerator

# 初始化模型
cosyvoice = CosyVoice("iic/CosyVoice-300M")
# 或
# cosyvoice = CosyVoice2("pretrained_models/CosyVoice2-0.5B")

# 创建生成器
generator = AudioGenerator(cosyvoice)

# 生成单个音频（流式）
for sample_rate, audio_array in generator.generate_audio(
    tts_text="你好，世界",
    mode="预训练音色",
    sft_dropdown="中文女",
    seed=42,
    stream=True
):
    # 处理音频数据
    print(f"采样率: {sample_rate}, 音频长度: {len(audio_array)}")

# 批量生成音频
text_segments = ["第一段文本", "第二段文本", "第三段文本"]
for status, download_path in generator.generate_batch_audio(
    text_segments=text_segments,
    output_dir="./output",
    mode="预训练音色",
    sft_dropdown="中文女",
    seed=42,
    speed=1.0
):
    print(status)
```

### 在 FastAPI 中使用

参考 `runtime/python/fastapi/server_enhanced.py` 文件，该文件展示了如何在 FastAPI 中使用 AudioGenerator。

主要接口：
- `POST /init`: 初始化模型
- `POST /generate_audio`: 生成单个音频
- `POST /generate_batch_audio`: 批量生成音频
- `POST /stop_generation`: 停止生成任务
- `GET /list_speakers`: 获取可用音色列表
- `GET /health`: 健康检查

## 支持的推理模式

- `预训练音色`: 使用预训练音色进行合成
- `3s极速复刻`: 使用 prompt 音频和文本进行零样本复刻
- `跨语种复刻`: 跨语种音色复刻
- `自然语言控制`: 使用自然语言指令控制语音风格

## 注意事项

1. 批量生成时建议将 `stream` 设为 `False`
2. `prompt_wav` 可以是文件路径（str）或字节数据（bytes）
3. 批量生成会自动创建时间戳命名的子目录
4. 支持通过 `stop_event` 参数中断生成任务



