from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .ollama import OllamaProvider

def get_llm_provider(config: dict, api_key: str = None):
    provider = config.get("llm_provider", "gemini").lower()
    if provider == "openai":
        return OpenAIProvider(api_key=api_key)
    elif provider == "ollama":
        return OllamaProvider()
    else:
        return GeminiProvider(api_key=api_key)
