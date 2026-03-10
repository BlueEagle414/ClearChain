import math
from openai import AsyncOpenAI
from core.llm.base import LLMProvider
from config import config

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        # Allow custom base_url for LM Studio, vLLM, or custom model servers
        self.base_url = config.get("openai_base_url")
        
        # If using a local server like LM Studio, API key can be a dummy string
        if self.base_url and not api_key:
            api_key = "lm-studio"
            
        self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        self.model = config.get("primary_model", "gpt-4o-mini")
        self.json_model = config.get("json_model", self.model)
        self.embed_model = config.get("embedding_model", "text-embedding-3-small")
        
    async def generate_text(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def generate_text_stream(self, prompt: str, system_instruction: str = None):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_json_stream(self, prompt: str):
        stream = await self.client.chat.completions.create(
            model=self.json_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=True,
            temperature=0.0
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_json(self, prompt: str, schema_class) -> any:
        # Local servers often don't support the beta parse endpoint, so we fallback to standard JSON mode
        if self.base_url:
            response = await self.client.chat.completions.create(
                model=self.json_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            
            # Strip markdown formatting if the local model hallucinates it
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            content = content.strip()
            return schema_class.model_validate_json(content)
        else:
            response = await self.client.beta.chat.completions.parse(
                model=self.json_model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema_class,
                temperature=0.0
            )
            return response.choices[0].message.parsed

    async def generate_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.embed_model,
            input=text
        )
        vec = response.data[0].embedding
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
