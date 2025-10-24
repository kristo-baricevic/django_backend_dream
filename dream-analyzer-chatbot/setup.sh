#!/bin/bash

echo "🚀 Setting up Dream Analyzer Chatbot..."

# Create directory structure
echo "📁 Creating directories..."
mkdir -p app/agents
mkdir -p app/models
mkdir -p app/services
mkdir -p app/routes

# Create __init__.py files
echo "📝 Creating __init__.py files..."
cat > app/__init__.py << 'EOF'
"""Dream Analyzer Chatbot Application"""
EOF

cat > app/models/__init__.py << 'EOF'
"""Data models and schemas"""
EOF

cat > app/services/__init__.py << 'EOF'
"""External services and clients"""
EOF

cat > app/agents/__init__.py << 'EOF'
"""Agent logic and tools"""
EOF

cat > app/routes/__init__.py << 'EOF'
"""API routes"""
EOF

# Create main.py
echo "📝 Creating app/main.py..."
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from app.config import settings

app = FastAPI(
    title="Dream Analyzer Chatbot",
    description="Agentic chatbot for dream analysis",
    version="1.0.0"
)

# CORS - adjust origins for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Dream Analyzer Chatbot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
EOF

# Create config.py
echo "📝 Creating app/config.py..."
cat > app/config.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    
    # Backend API
    DRF_BACKEND_URL: str = "http://localhost:8000"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # OpenAI Settings
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    
    # Chat Settings
    MAX_CONVERSATION_HISTORY: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
EOF

# Create schemas.py
echo "📝 Creating app/models/schemas.py..."
cat > app/models/schemas.py << 'EOF'
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    message: str
    timestamp: datetime
    conversation_id: Optional[str] = None

class StreamChunk(BaseModel):
    content: str
    is_complete: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
EOF

# Create llm_service.py
echo "📝 Creating app/services/llm_service.py..."
cat > app/services/llm_service.py << 'EOF'
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
EOF

# Create drf_client.py
echo "📝 Creating app/services/drf_client.py..."
cat > app/services/drf_client.py << 'EOF'
import httpx
from typing import Dict, List, Optional
from app.config import settings

class DRFClient:
    """Client for communicating with Django DRF backend"""
    
    def __init__(self):
        self.base_url = settings.DRF_BACKEND_URL
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    # Placeholder methods - implement based on your DRF API
    
    async def get_dreams(self, limit: int = 10) -> List[Dict]:
        """Get user's dreams from DRF backend"""
        # TODO: Implement when DRF endpoints are ready
        # response = await self.client.get("/api/dreams/", params={"limit": limit})
        # response.raise_for_status()
        # return response.json()
        return []
    
    async def create_dream(self, dream_data: Dict) -> Dict:
        """Create a new dream entry"""
        # TODO: Implement when DRF endpoints are ready
        # response = await self.client.post("/api/dreams/", json=dream_data)
        # response.raise_for_status()
        # return response.json()
        return {}
    
    async def analyze_dream(self, dream_id: int) -> Dict:
        """Get dream analysis from backend"""
        # TODO: Implement when DRF endpoints are ready
        # response = await self.client.get(f"/api/dreams/{dream_id}/analyze/")
        # response.raise_for_status()
        # return response.json()
        return {}
    
    async def search_dreams(self, query: str) -> List[Dict]:
        """Search dreams by keywords"""
        # TODO: Implement when DRF endpoints are ready
        # response = await self.client.get("/api/dreams/search/", params={"q": query})
        # response.raise_for_status()
        # return response.json()
        return []

drf_client = DRFClient()
EOF

# Create dream_agent.py
echo "📝 Creating app/agents/dream_agent.py..."
cat > app/agents/dream_agent.py << 'EOF'
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
EOF

# Create chat.py
echo "📝 Creating app/routes/chat.py..."
cat > app/routes/chat.py << 'EOF'
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
import json
from typing import List

from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.agents.dream_agent import dream_agent

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message and get a response (non-streaming)
    """
    try:
        response_text = await dream_agent.process_message(
            user_message=request.message,
            conversation_history=request.conversation_history or []
        )
        
        return ChatResponse(
            message=response_text,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/message/stream")
async def send_message_stream(request: ChatRequest):
    """
    Send a message and get a streaming response
    """
    async def generate():
        try:
            async for chunk in dream_agent.stream_response(
                user_message=request.message,
                conversation_history=request.conversation_history or []
            ):
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    conversation_history: List[ChatMessage] = []
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                continue
            
            # Add user message to history
            conversation_history.append(
                ChatMessage(role="user", content=user_message)
            )
            
            # Stream response back
            full_response = ""
            async for chunk in dream_agent.stream_response(
                user_message=user_message,
                conversation_history=conversation_history[:-1]  # Exclude current message
            ):
                full_response += chunk
                await websocket.send_json({
                    "type": "stream",
                    "content": chunk,
                    "done": False
                })
            
            # Send completion
            await websocket.send_json({
                "type": "stream",
                "content": "",
                "done": True
            })
            
            # Add assistant response to history
            conversation_history.append(
                ChatMessage(role="assistant", content=full_response)
            )
            
            # Trim history if too long
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
EOF

# Create requirements.txt
echo "📝 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
openai==1.54.3
pydantic==2.9.2
pydantic-settings==2.6.0
httpx==0.27.2
python-dotenv==1.0.1
websockets==13.1
EOF

# Create .env.example
echo "📝 Creating .env.example..."
cat > .env.example << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
MAX_TOKENS=1000

# Django DRF Backend
DRF_BACKEND_URL=http://localhost:8000

# CORS Origins (comma-separated)
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Chat Settings
MAX_CONVERSATION_HISTORY=10
EOF

# Create test_client.py
echo "📝 Creating test_client.py..."
cat > test_client.py << 'EOF'
"""
Simple test script to verify the chatbot is working.
Run this after starting the server.
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8001"

async def test_health():
    """Test health endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"✓ Health check: {response.json()}")

async def test_chat():
    """Test chat message endpoint"""
    async with httpx.AsyncClient() as client:
        data = {
            "message": "Hello! Can you help me understand my dreams?",
            "conversation_history": []
        }
        
        response = await client.post(
            f"{BASE_URL}/api/chat/message",
            json=data,
            timeout=30.0
        )
        
        result = response.json()
        print(f"\n✓ Chat Response:")
        print(f"  {result['message'][:100]}...")

async def test_streaming():
    """Test streaming endpoint"""
    async with httpx.AsyncClient() as client:
        data = {
            "message": "What are common dream symbols?",
            "conversation_history": []
        }
        
        print(f"\n✓ Streaming Response:")
        print("  ", end="")
        
        async with client.stream(
            'POST',
            f"{BASE_URL}/api/chat/message/stream",
            json=data,
            timeout=30.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if not data.get('done'):
                        print(data.get('content', ''), end='', flush=True)
        
        print("\n")

async def main():
    """Run all tests"""
    print("Testing Dream Analyzer Chatbot API\n")
    print("=" * 50)
    
    try:
        await test_health()
        await test_chat()
        await test_streaming()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the server is running:")
        print("  uvicorn app.main:app --reload --port 8001")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Environment
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
EOF

# Create README.md
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
# Dream Analyzer Chatbot

Agentic FastAPI chatbot for conversational dream analysis.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Run the server:**
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

4. **Test it:**
   ```bash
   python test_client.py
   ```

## API Documentation

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Endpoints

- `POST /api/chat/message` - Send message (non-streaming)
- `POST /api/chat/message/stream` - Send message (streaming)
- `WS /api/chat/ws` - WebSocket connection

See full documentation at http://localhost:8001/docs
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create .env file:     cp .env.example .env"
echo "2. Add your OpenAI key to .env"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Run the server:       uvicorn app.main:app --reload --port 8001"
echo "5. Test it:              python test_client.py"
echo ""
echo "📚 API docs will be at: http://localhost:8001/docs"
EOF

chmod +x setup.sh

echo "✅ Setup script created!"
echo ""
echo "Run it with:"
echo "  ./setup.sh"