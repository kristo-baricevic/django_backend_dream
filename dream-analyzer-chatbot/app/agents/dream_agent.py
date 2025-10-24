from typing import List
from app.models.schemas import ChatMessage
from app.services.llm_service import llm_service
from app.services.drf_client import drf_client

class DreamAgent:
    """
    Agentic orchestrator for dream analysis conversations.
    Will use tools to interact with DRF backend.
    """
    
    def __init__(self):
        self.system_prompt = """You are a helpful dream analysis assistant. 
You help users track, analyze, and understand their dreams.

You can:
- Help users record their dreams
- Retrieve past dreams
- Analyze dream patterns and symbolism
- Search through dream history
- Provide insights based on dream psychology

Be conversational, empathetic, and insightful. Ask clarifying questions when needed.
"""
    
    async def process_message(
        self, 
        user_message: str,
        conversation_history: List[ChatMessage]
    ) -> str:
        """
        Process user message and determine actions.
        
        Future: This will use function calling to decide which tools to use
        (save dream, retrieve dreams, analyze, etc.)
        """
        
        # Add user message to history
        messages = conversation_history + [
            ChatMessage(role="user", content=user_message)
        ]
        
        # For now, just get LLM response
        # TODO: Add tool/function calling logic here
        response = await llm_service.get_completion(
            messages=messages,
            system_prompt=self.system_prompt
        )
        
        return response
    
    async def stream_response(
        self,
        user_message: str,
        conversation_history: List[ChatMessage]
    ):
        """Stream response for real-time updates"""
        
        messages = conversation_history + [
            ChatMessage(role="user", content=user_message)
        ]
        
        async for chunk in llm_service.get_streaming_completion(
            messages=messages,
            system_prompt=self.system_prompt
        ):
            yield chunk
    
    # Tool methods (to be implemented)
    
    async def _save_dream(self, dream_content: str, metadata: dict = None):
        """Tool: Save a dream to the backend"""
        # Will call drf_client.create_dream()
        pass
    
    async def _get_recent_dreams(self, limit: int = 5):
        """Tool: Get recent dreams"""
        # Will call drf_client.get_dreams()
        pass
    
    async def _analyze_dream_patterns(self):
        """Tool: Analyze patterns across dreams"""
        # Will call drf_client or do analysis
        pass

dream_agent = DreamAgent()
