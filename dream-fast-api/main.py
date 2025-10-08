import django_setup
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import BackgroundTasks
from core.workflow_tracker import WorkflowTracker

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
    entries: List[JournalEntry]
    personality: Optional[str] = None
    settings: Dict[str, Any] = None

class QARequestCustom(BaseModel):
    question: str
    entries: List[JournalEntry]
    personality: Optional[str] = None
    settings: Dict[str, Any] = None


class GenerateDreamRequest(BaseModel):
    theme: Optional[str] = "flying"

class BatchAnalyzeRequest(BaseModel):
    entries: List[JournalEntry]
    personality_type: Optional[str] = "empathetic"
    settings: Dict[str, Any] = None

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
        result = await journal_service.get_cumulative_analysis(request.entries, request.personality, request.settings)
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
    request: QARequestCustom,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Handle custom questions about dreams."""
    try:
        result = await journal_service.ask_custom_question(request.question, request.entries, request.personality, request.settings)
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

@app.post("/qa-with-workflow")
async def qa_with_workflow(
    request: QARequest,
    background_tasks: BackgroundTasks,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Start workflow and return immediately"""
    
    # Create workflow ID first
    tracker = WorkflowTracker(
        workflow_type="cumulative_analysis",
        routine_name="Dream Analysis",
        user_id=request.settings.get('user_id') if request.settings else None
    )
    workflow_id = await tracker.start_workflow()
    
    print(f"🔵 CREATED NEW WORKFLOW: {workflow_id}")  # ADD THIS
    
    # Process in background
    background_tasks.add_task(
        process_analysis_workflow,
        workflow_id,
        request.entries,
        request.personality,
        request.settings,
        journal_service
    )
    
    response_data = {
        "workflow_id": workflow_id,
        "status": "processing"
    }
    
    print(f"🔵 RETURNING TO FRONTEND: {response_data}")  # ADD THIS
    
    # Return immediately
    return response_data


@app.post("/custom-question-with-workflow")
async def custom_question_with_workflow(
    request: QARequestCustom,
    background_tasks: BackgroundTasks,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Start workflow and return immediately"""
    
    # Create workflow ID first
    tracker = WorkflowTracker(
        workflow_type="custom-question-with-workflow",
        routine_name="Custom Question",
        user_id=request.settings.get('user_id') if request.settings else None
    )
    workflow_id = await tracker.start_workflow()
    
    print(f"🔵 CREATED NEW WORKFLOW: {workflow_id}")  # ADD THIS
    
    # Process in background
    background_tasks.add_task(
        process_custom_question_workflow,
        workflow_id,
        request.question,
        request.entries,
        request.personality,
        request.settings,
        journal_service
    )
    
    response_data = {
        "workflow_id": workflow_id,
        "status": "processing"
    }
    
    print(f"🔵 RETURNING TO FRONTEND: {response_data}")  # ADD THIS
    
    # Return immediately
    return response_data

async def process_analysis_workflow(
    workflow_id: str,
    entries: List[Dict],
    personality: str,
    settings: Dict,
    journal_service: DreamJournalService
):
    """Background task to process the workflow"""
    try:
        result, _ = await journal_service.analyzer.qa_analysis_with_workflow(
            entries, 
            personality, 
            settings,
            existing_workflow_id=workflow_id  # Pass existing workflow_id
        )
    except Exception as e:
        print(f"❌ Background task failed: {e}")

async def process_custom_question_workflow(
    workflow_id: str,
    question: str,
    entries: List[Dict],
    personality: str,
    settings: Dict,
    journal_service: DreamJournalService,
    existing_workflow_id: str = None
):
    """Background task to process the workflow"""
    try:
        result, _ = await journal_service.analyzer.custom_question_with_workflow(
            question,
            entries, 
            personality, 
            settings,
            existing_workflow_id=workflow_id
        )
    except Exception as e:
        print(f"❌ Background task failed: {e}")

@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow execution details"""
    from myapp.models import WorkflowExecution
    from asgiref.sync import sync_to_async
    
    try:
        execution = await sync_to_async(
            WorkflowExecution.objects.prefetch_related('steps__citations').get
        )(id=workflow_id)
        
        steps = await sync_to_async(list)(execution.steps.all())
        
        return {
            "id": str(execution.id),
            "workflow_type": execution.workflow_type,
            "routine_name": execution.routine_name,
            "status": execution.status,
            # ... rest of response
        }
    except WorkflowExecution.DoesNotExist:
        raise HTTPException(status_code=404, detail="Workflow not found")
