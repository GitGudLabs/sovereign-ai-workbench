from pathlib import Path

import pytest

from backend.speech.s2s import ConversationMemory, S2SConfig


def test_config_rejects_missing_local_models(tmp_path: Path) -> None:
    config = S2SConfig(
        stt_model=tmp_path / "stt",
        llm_model=tmp_path / "llm.gguf",
        piper_model=tmp_path / "voice.onnx",
    )

    with pytest.raises(FileNotFoundError, match="STT model"):
        config.validate()


def test_config_accepts_existing_local_models(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("stt", "llm.gguf", "voice.onnx")]
    for path in paths:
        path.touch()

    S2SConfig(*paths).validate()


def test_memory_is_bounded_and_in_process() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.add("user", "first")
    memory.add("assistant", "second")
    memory.add("user", "third")

    assert memory.messages == [
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]