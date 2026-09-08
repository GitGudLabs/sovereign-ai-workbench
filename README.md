# Sovereign AI Workbench

This project is a fully local Speech-to-Speech (S2S) assistant. Microphone
audio is processed on the user's machine through local Speech-to-Text (STT),
a quantized local 3B-4B LLM, and local Piper Text-to-Speech (TTS). Runtime
code has no cloud API, cloud storage, remote inference, telemetry, or network
dependency.

## Local S2S stages

The first runnable slice is in `backend/speech/s2s.py`:

1. Capture a WAV file with `record_wav`.
2. Transcribe it with CPU `faster-whisper` using `compute_type="int8"`.
3. Generate a short response with a local GGUF model through `llama-cpp-python`.
4. Synthesize the response with a local Piper model and executable.

The module never downloads models. Place model files outside version control,
then construct `S2SConfig` with their local paths:

```python
from pathlib import Path
from backend.speech.s2s import LocalS2S, S2SConfig

assistant = LocalS2S(S2SConfig(
		stt_model=Path("models/whisper-small-int8"),
		llm_model=Path("models/qwen2.5-3b-instruct-q4_k_m.gguf"),
		piper_model=Path("models/en_US-lessac-medium.onnx"),
))
transcript, response, output_wav = assistant.process_wav(Path("input.wav"))
```

Install the optional local runtime packages according to the platform and
hardware:

```text
pip install -r requirements.txt
```

The `piper-tts` package provides the local Piper executable. Keep model binaries
out of Git. On Windows, `llama-cpp-python` may require its prebuilt CPU wheel
from the llama.cpp Python wheel index. The default configuration is intended
for an Intel i7-13620H with 16 GB RAM and integrated graphics: CPU inference,
int8 STT, small context, and short responses.

## Evaluation plan

Record these metrics for each stage and for the complete turn:

- STT: Word Error Rate (WER), STT latency, and Real-Time Factor (RTF)
- LLM: tokens/second and response latency
- TTS: synthesis latency, RTF, and perceived naturalness
- End-to-end: latency, peak RAM, CPU utilization, response accuracy
- Privacy: network-disabled execution and verification that no audio, text, or
	telemetry leaves the machine

Future stages will add voice-activity detection, interruption handling,
streaming responses, and durable opt-in local conversation memory.
