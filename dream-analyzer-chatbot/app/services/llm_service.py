from openai import AsyncOpenAI
from typing import List, AsyncGenerator
from app.config import settings
from app.models.schemas import ChatMessage

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class LLMService:
    def __init__(self):
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        
    async def get_completion(
        self, 
        messages: List[ChatMessage],
        system_prompt: str = None
    ) -> str:
        """Get a single completion from OpenAI"""
        
        # Build messages for OpenAI
        openai_messages = []
        
        # Add system prompt
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Call OpenAI
        response = await client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def get_streaming_completion(
        self,
        messages: List[ChatMessage],
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """Get a streaming completion from OpenAI"""
        
        # Build messages for OpenAI
        openai_messages = []
        
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Call OpenAI with streaming
        stream = await client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

llm_service = LLMService()
