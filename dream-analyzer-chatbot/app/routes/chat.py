import json
from datetime import datetime
from fastapi import APIRouter, Request, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.drf_client import APIClient
from app.agents.dream_agent import DreamAgent

router = APIRouter()

@router.post('/message', response_model=ChatResponse)
async def message(payload: ChatRequest, request: Request):
    api = APIClient(request.headers)
    try:
        await api.client.validate()
        result = await DreamAgent(api).process_message(payload.message, payload.conversation_history or [])
        await api.client.validate()  # discard a response after logout/expiry during generation
        return ChatResponse(message=result['text'], timestamp=datetime.now(), ui_events=result.get('ui_events', []))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Chat unavailable.')
    finally:
        await api.close()

@router.post('/message/stream')
async def stream(payload: ChatRequest, request: Request):
    api = APIClient(request.headers)
    try:
        await api.client.validate()
    except Exception:
        await api.close()
        raise
    async def generate():
        try:
            async for chunk in DreamAgent(api).stream_response(payload.message, payload.conversation_history or []):
                await api.client.validate()
                yield 'data: ' + json.dumps({'content': chunk, 'done': False}) + '\n\n'
            yield 'data: ' + json.dumps({'content': '', 'done': True}) + '\n\n'
        except Exception:
            yield 'data: ' + json.dumps({'error': 'Chat unavailable.', 'done': True}) + '\n\n'
        finally:
            await api.close()
    return StreamingResponse(generate(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-store, private', 'Vary': 'Cookie'})

@router.websocket('/ws')
async def websocket(websocket: WebSocket):
    # Browser uses HTTP. Keep unused legacy transport closed until authenticated protocol exists.
    await websocket.close(code=1008)
