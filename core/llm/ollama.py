import math
import json
import httpx
from core.llm.base import LLMProvider
from config import config

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.base_url = config.get("ollama_base_url", "http://localhost:11434/api")
        self.model = config.get("primary_model", "llama3")
        self.json_model = config.get("json_model", self.model)
        self.embed_model = config.get("embedding_model", "nomic-embed-text")
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def generate_text(self, prompt: str) -> str:
        response = await self.client.post(f"{self.base_url}/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "")

    async def generate_text_stream(self, prompt: str, system_instruction: str = None):
        system_prompt = f"{system_instruction}\n\n" if system_instruction else ""
        async with self.client.stream("POST", f"{self.base_url}/generate", json={
            "model": self.model,
            "prompt": system_prompt + prompt,
            "stream": True
        }) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")

    async def generate_json_stream(self, prompt: str):
        async with self.client.stream("POST", f"{self.base_url}/generate", json={
            "model": self.json_model,
            "prompt": prompt,
            "format": "json",
            "stream": True
        }) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")

    async def generate_json(self, prompt: str, schema_class) -> any:
        response = await self.client.post(f"{self.base_url}/generate", json={
            "model": self.json_model,
            "prompt": prompt,
            "format": "json",
            "stream": False
        })
        data = response.json().get("response", "{}").strip()
        
        # Strip markdown formatting if the local model hallucinates it
        if data.startswith("```json"):
            data = data[7:]
        elif data.startswith("```"):
            data = data[3:]
        if data.endswith("```"):
            data = data[:-3]
            
        data = data.strip()
        return schema_class.model_validate_json(data)

    async def generate_embedding(self, text: str) -> list[float]:
        response = await self.client.post(f"{self.base_url}/embeddings", json={
            "model": self.embed_model,
            "prompt": text
        })
        vec = response.json().get("embedding", [])
        norm = math.sqrt(sum(x * x for x in vec)) if vec else 0
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
