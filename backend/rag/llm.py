import logging
from backend.rag.interfaces import BaseLLM

logger = logging.getLogger(__name__)

class OllamaLLM(BaseLLM):
    """
    Local LLM response generator using Ollama Python client.
    """
    def __init__(self, model_name: str = "qwen2.5vl:3b"):
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            import ollama
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(model=self.model_name, messages=messages)
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error generating LLM response via Ollama: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")
