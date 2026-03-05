from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any

class LLMProvider(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_text_stream(self, prompt: str, system_instruction: str = None) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def generate_json_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def generate_json(self, prompt: str, schema_class: Any) -> Any:
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        pass
