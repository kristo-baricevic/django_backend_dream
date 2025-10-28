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
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Dream Analyzer Chatbot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
