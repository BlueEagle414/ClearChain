import math
from google import genai
from google.genai import types
from core.llm.base import LLMProvider
from config import config

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    async def generate_text(self, prompt: str) -> str:
        response = await self.client.aio.models.generate_content(
            model=config["primary_model"],
            contents=prompt
        )
        return response.text

    async def generate_text_stream(self, prompt: str, system_instruction: str = None):
        config_args = {}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
            
        response = await self.client.aio.models.generate_content_stream(
            model=config["primary_model"],
            contents=prompt,
            config=types.GenerateContentConfig(**config_args) if config_args else None
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def generate_json_stream(self, prompt: str):
        from core.llm_service import CoVeVerificationResult
        response = await self.client.aio.models.generate_content_stream(
            model=config["json_model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CoVeVerificationResult,
                temperature=0.0
            )
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def generate_json(self, prompt: str, schema_class) -> any:
        response = await self.client.aio.models.generate_content(
            model=config["json_model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_class,
                temperature=0.0
            )
        )
        return response.parsed

    async def generate_embedding(self, text: str) -> list[float]:
        result = await self.client.aio.models.embed_content(
            model=config["embedding_model"], 
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        vec = result.embeddings[0].values
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
