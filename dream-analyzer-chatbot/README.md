# Dream Analyzer Chatbot

Agentic FastAPI chatbot for conversational dream analysis.

## Quick Start

1. **Install dependencies:**
   pip install -r requirements.txt

2. **Configure environment:**
   cp .env.example .env

3. **Run the server:**
   uvicorn app.main:app --reload --port 8001

4. **Test it:**
   python test_client.py

## API Documentation

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Endpoints

- `POST /api/chat/message` - Send message (non-streaming)
- `POST /api/chat/message/stream` - Send message (streaming)
- `WS /api/chat/ws` - WebSocket connection

See full documentation at http://localhost:8001/docs
