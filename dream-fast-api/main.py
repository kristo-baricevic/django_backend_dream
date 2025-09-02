from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from contextlib import asynccontextmanager

# Import your journal analyzer
from core.dream_analyzer import (
    DreamJournalService, 
    JournalEntry, 
    JournalAnalysis,
    EmotionType
)

from dotenv import load_dotenv

load_dotenv()

# Global service instance
service: Optional[DreamJournalService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    service = DreamJournalService(openai_api_key)
    await service.initialize()  # Add this line
    yield
    service = None

# Initialize FastAPI app
app = FastAPI(
    title="Dream Journal Analysis API",
    description="API for analyzing dream journal entries using LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalyzeEntryRequest(BaseModel):
    content: str
    personality_type: Optional[str] = "empathetic"

class QARequest(BaseModel):
    question: str
    entries: List[JournalEntry]

class GenerateDreamRequest(BaseModel):
    theme: Optional[str] = "flying"

class BatchAnalyzeRequest(BaseModel):
    entries: List[JournalEntry]
    personality_type: Optional[str] = "empathetic"

# Dependency to get the service
def get_service() -> DreamJournalService:
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return service

# Routes
@app.get("/")
async def root():
    return {"message": "Dream Journal Analysis API", "version": "1.0.0"}

@app.post("/analyze", response_model=JournalAnalysis)
async def analyze_entry(
    request: AnalyzeEntryRequest,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Analyze a single journal entry and return structured analysis."""
    try:
        result = await journal_service.analyze_single_entry(
            content=request.content,
            personality=request.personality_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/qa")
async def qa_analysis(
    request: QARequest,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Perform Q&A analysis over multiple journal entries."""
    try:
        result = await journal_service.get_cumulative_analysis(request.entries)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A analysis failed: {str(e)}")

@app.post("/generate-dream")
async def generate_dream(
    request: GenerateDreamRequest,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Generate a sample dream based on a theme."""
    try:
        result = await journal_service.generate_sample_dream(request.theme)
        return {"dream": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dream generation failed: {str(e)}")

@app.post("/batch-analyze", response_model=List[JournalAnalysis])
async def batch_analyze(
    request: BatchAnalyzeRequest,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Analyze multiple journal entries in batch."""
    try:
        results = await journal_service.analyzer.batch_analyze_entries(
            entries=request.entries,
            personality_type=request.personality_type
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/emotions", response_model=List[str])
async def get_emotions():
    """Get list of available emotion types."""
    return [emotion.value for emotion in EmotionType]

@app.get("/personalities")
async def get_personalities():
    """Get list of available personality types."""
    return {
        "empathetic": "Empathetic and compassionate analysis",
        "analytical": "Logical and systematic analysis", 
        "mystical": "Mystical and spiritual interpretation",
        "practical": "Practical and solution-oriented analysis"
    }

@app.post("/custom-question")
async def custom_question(
    request: QARequest,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Handle custom questions about dreams."""
    try:
        result = await journal_service.ask_custom_question(request.question, request.entries)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom question failed: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )