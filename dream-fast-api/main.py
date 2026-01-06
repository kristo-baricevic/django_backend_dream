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
from asgiref.sync import sync_to_async
from myapp.models import WorkflowExecution
from django.utils import timezone

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
    settings: Optional[Dict[str, Any]] = None

class QARequest(BaseModel):
    entries: List[JournalEntry]
    settings: Dict[str, Any] = None

class QARequestCustom(BaseModel):
    question: str
    entries: List[JournalEntry]
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

# fastapi: endpoint
@app.post("/analyze", response_model=JournalAnalysis)
async def analyze_entry(
    request: AnalyzeEntryRequest,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Analyze a single journal entry using doctor personality from settings."""
    print(f"request ==00== settings {request.settings}")
    try:
        result = await journal_service.analyze_single_entry(
            content=request.content,
            settings=request.settings
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
    print(f"qa_with_workflow start == {workflow_id}")

    # Create workflow ID first
    tracker = WorkflowTracker(
        workflow_type="cumulative_analysis",
        routine_name="Dream Analysis",
        user_id=request.settings.get('user_id') if request.settings else None
    )
    workflow_id = await tracker.start_workflow()
    
    print(f"🔵 CREATED NEW WORKFLOW: {workflow_id}")
    
    # Process in background
    background_tasks.add_task(
        process_analysis_workflow,
        workflow_id,
        request.entries,
        request.settings,
        journal_service
    )
    
    response_data = {
        "workflow_id": workflow_id,
        "status": "processing"
    }
    
    print(f"🔵 RETURNING TO FRONTEND: {response_data}") 
    
    return response_data


@app.post("/custom-question-with-workflow")
async def custom_question_with_workflow(
    request: QARequestCustom,
    background_tasks: BackgroundTasks,
    journal_service: DreamJournalService = Depends(get_service)
):
    """Start workflow and return immediately"""
    
    tracker = WorkflowTracker(
        workflow_type="custom-question",
        routine_name="Custom Question",
        user_id=request.settings.get('user_id') if request.settings else None
    )
    workflow_id = await tracker.start_workflow()
    
    print(f"🔵 CREATED NEW WORKFLOW: {workflow_id}")
    
    # Generate a temporary analysis ID (will be created properly in background)
    import uuid
    temp_analysis_id = str(uuid.uuid4())
    
    # Add to background tasks WITHOUT waiting for it
    background_tasks.add_task(
        process_custom_question_workflow,
        workflow_id,
        request.question,
        request.entries,
        request.settings,
        journal_service
    )

    print(f"🔵 RETURNING TO FRONTEND IMMEDIATELY")

    return {
        "workflow_id": workflow_id,
        # "analysis_id": temp_analysis_id,  # Temporary ID
        "status": "processing"
    }

async def process_analysis_workflow(
    workflow_id: str,
    entries: List[Dict],
    settings: Dict,
    journal_service: DreamJournalService
):
    """Background task to process the workflow"""
    try:
        result, _ = await journal_service.analyzer.qa_analysis_with_workflow(
            entries, 
            settings,
            existing_workflow_id=workflow_id  # Pass existing workflow_id
        )
    except Exception as e:
        print(f"❌ Background task failed: {e}")

async def process_custom_question_workflow(
    workflow_id: str,
    question: str,
    entries: List[Dict],
    settings: Dict,
    journal_service: DreamJournalService,
    existing_workflow_id: str = None,
    custom_question_id: str = None
):
    """Background task to process the workflow"""
    try:
        # Run analysis and get the final result + created CustomQuestion ID
        result, custom_question_id, _ = await journal_service.analyzer.custom_question_with_workflow(
            question,
            entries, 
            settings,
            existing_workflow_id=workflow_id
        )

        print(f"✅ Created CustomQuestion ID: {custom_question_id}")

        # ✅ Attach workflow → custom question
        from myapp.models import CustomQuestion

        custom_question = await sync_to_async(CustomQuestion.objects.get)(id=custom_question_id)
        custom_question.workflow_execution_id = workflow_id
        await sync_to_async(custom_question.save)(update_fields=["workflow_execution"])

        # ✅ Persist analysis_id and result to the workflow record
        execution = await sync_to_async(WorkflowExecution.objects.get)(id=workflow_id)
        execution.analysis_id = custom_question_id
        execution.final_result = result
        execution.status = "completed"
        await sync_to_async(execution.save)(update_fields=["analysis_id", "final_result", "status"])

        print(f"💾 Updated WorkflowExecution {workflow_id} with analysis_id {custom_question_id}")

    except Exception as e:
        print(f"❌ Background task failed: {e}")
        # Optional: mark workflow as failed
        try:
            execution = await sync_to_async(WorkflowExecution.objects.get)(id=workflow_id)
            execution.status = "failed"
            execution.final_result = str(e)
            await sync_to_async(execution.save)()
        except Exception as inner_e:
            print(f"⚠️ Failed to update workflow status after error: {inner_e}")


@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow execution details"""
    from myapp.models import WorkflowExecution
    from asgiref.sync import sync_to_async
    print("🔍 get_workflow called")
    try:
        execution = await sync_to_async(
            WorkflowExecution.objects.prefetch_related('steps__citations').get
        )(id=workflow_id)

        steps = await sync_to_async(list)(execution.steps.all())
        print(f"execution.analysis_id = {execution.analysis_id}")
        return {
            "id": str(execution.id),
            "workflow_type": execution.workflow_type,
            "routine_name": execution.routine_name,
            "status": execution.status,
            "analysis_id": execution.analysis_id,
            "final_result": execution.final_result,
        }
    except WorkflowExecution.DoesNotExist:
        raise HTTPException(status_code=404, detail="Workflow not found")
