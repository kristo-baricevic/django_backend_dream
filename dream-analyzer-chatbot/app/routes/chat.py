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
    # ... rest of your code ...

@router.post("/message/stream")
async def send_message_stream(request: ChatRequest):
    # ... rest of your code ...

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
