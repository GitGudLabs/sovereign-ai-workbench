"""Local-only speech-to-speech pipeline.

The adapters are deliberately lazy: installing this module does not download
models or require heavyweight ML packages until a pipeline stage is used.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class LocalOnlyError(RuntimeError):
    """Raised when a stage would use a non-local resource."""


@dataclass(frozen=True)
class S2SConfig:
    """Paths and limits for a CPU-first local deployment."""

    stt_model: Path
    llm_model: Path
    piper_model: Path
    piper_executable: str = "piper"
    sample_rate: int = 16_000
    max_new_tokens: int = 128
    temperature: float = 0.2
    system_prompt: str = (
        "You are a concise helpful assistant. Answer in plain text suitable "
        "for speech. Do not use markdown, emojis, or long lists."
    )

    def validate(self) -> None:
        for name, path in (
            ("STT model", self.stt_model),
            ("LLM model", self.llm_model),
            ("Piper model", self.piper_model),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")


@dataclass
class ConversationMemory:
    """Small in-process memory; nothing is persisted by default."""

    messages: list[dict[str, str]] = field(default_factory=list)
    max_messages: int = 10

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        del self.messages[:-self.max_messages]


class LocalS2S:
    """Compose local STT, LLM, and Piper TTS stages."""

    def __init__(self, config: S2SConfig, memory: ConversationMemory | None = None):
        config.validate()
        self.config = config
        self.memory = memory or ConversationMemory()
        self._stt: Any = None
        self._llm: Any = None

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV file with faster-whisper running locally."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to enable local STT") from exc

        if self._stt is None:
            self._stt = WhisperModel(
                str(self.config.stt_model), device="cpu", compute_type="int8"
            )
        segments, _ = self._stt.transcribe(str(wav_path), vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def respond(self, text: str) -> str:
        """Generate a response with a local GGUF model via llama.cpp."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install llama-cpp-python to enable the local LLM") from exc

        if self._llm is None:
            self._llm = Llama(
                model_path=str(self.config.llm_model),
                n_ctx=2048,
                n_threads=max(1, (os.cpu_count() or 2) - 1),
                verbose=False,
            )
        self.memory.add("user", text)
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.memory.messages)
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        response = result["choices"][0]["message"]["content"].strip()
        self.memory.add("assistant", response)
        return response

    def synthesize(self, text: str, wav_path: Path) -> Path:
        """Synthesize speech with a local Piper executable and model."""
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.piper_executable,
            "--model",
            str(self.config.piper_model),
            "--output_file",
            str(wav_path),
        ]
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                capture_output=True,
                env={**os.environ, "PIPER_NO_UPDATE": "1"},
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Piper executable was not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or "Piper synthesis failed") from exc
        return wav_path

    def process_wav(self, input_wav: Path, output_wav: Path | None = None) -> tuple[str, str, Path]:
        """Run one complete offline turn from WAV input to WAV output."""
        transcript = self.transcribe(input_wav)
        response = self.respond(transcript)
        if output_wav is None:
            handle, output_name = tempfile.mkstemp(suffix=".wav")
            os.close(handle)
            output_wav = Path(output_name)
        return transcript, response, self.synthesize(response, output_wav)


def record_wav(output: Path, seconds: float, sample_rate: int = 16_000) -> Path:
    """Capture microphone audio locally using sounddevice."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Install sounddevice and soundfile to capture audio") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(output), audio, sample_rate, subtype="PCM_16")
    return output