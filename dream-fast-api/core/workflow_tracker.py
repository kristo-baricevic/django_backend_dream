# dream-fast-api/core/workflow_tracker.py
from datetime import datetime
from typing import Optional, Dict, Any, List
import sys
sys.path.append('..')  # Add parent directory to path
from django.utils import timezone

from myapp.models import WorkflowExecution, WorkflowStep, StepCitation
from asgiref.sync import sync_to_async
import uuid

class WorkflowTracker:
    """Helper class to track workflow execution and steps"""
    
    def __init__(self, workflow_type: str, routine_name: str, user_id: Optional[int] = None):
        self.workflow_type = workflow_type
        self.routine_name = routine_name
        self.user_id = user_id
        self.execution = None
        self.current_step_number = 0
        
    async def start_workflow(self) -> str:
        """Start tracking a new workflow execution"""
        self.execution = await sync_to_async(WorkflowExecution.objects.create)(
            user_id=self.user_id,
            workflow_type=self.workflow_type,
            routine_name=self.routine_name,
            status='running'
        )
        print(f"📊 Started workflow: {self.execution.id}")
        return str(self.execution.id)
    
    async def start_step(self, name: str, step_type: str, input_data: Any = None) -> 'StepTracker':
        """Start tracking a new step"""
        self.current_step_number += 1
        
        step = await sync_to_async(WorkflowStep.objects.create)(
            execution=self.execution,
            step_number=self.current_step_number,
            name=name,
            step_type=step_type,
            status='running',
            start_time=datetime.utcnow(),
            input_data=input_data
        )
        
        print(f"  ▶️  Step {self.current_step_number}: {name}")
        return StepTracker(step)
    

    async def complete_workflow(self, result: str, confidence: Optional[float] = None, total_citations: int = 0):
        """Mark workflow as completed"""
        self.execution.status = 'completed'
        self.execution.end_time = timezone.now()  # 🔥 Changed from datetime.utcnow()
        self.execution.final_result = result
        self.execution.overall_confidence = confidence
        self.execution.total_citations = total_citations
        await sync_to_async(self.execution.save)()
        
        print(f"✅ Workflow completed: {self.execution.id}")

    async def fail_workflow(self, error: str):
        """Mark workflow as failed"""
        self.execution.status = 'failed'
        self.execution.end_time = datetime.utcnow()
        self.execution.error_message = error
        await sync_to_async(self.execution.save, thread_sensitive=True)()
        
        print(f"❌ Workflow failed: {error}")

class StepTracker:
    """Helper class to track individual step execution"""
    
    def __init__(self, step: WorkflowStep):
        self.step = step
        
    async def complete(
        self, 
        output: Any, 
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
        citations: Optional[List[Dict]] = None
    ):
        """Mark step as completed"""
        self.step.status = 'completed'
        self.step.end_time = datetime.utcnow()
        self.step.duration_ms = int((self.step.end_time - self.step.start_time).total_seconds() * 1000)
        self.step.output_data = output
        self.step.confidence = confidence
        self.step.reasoning = reasoning
        self.step.model_used = model
        self.step.tokens_used = tokens
        
        await sync_to_async(self.step.save, thread_sensitive=True)()
        
        # Add citations if provided
        if citations:
            for citation in citations:
                await sync_to_async(StepCitation.objects.create)(
                    step=self.step,
                    source=citation.get('source', 'unknown'),
                    content=citation.get('content', ''),
                    confidence=citation.get('confidence', 0.0),
                    reference=citation.get('reference')
                )
        
        print(f"    ✓ Completed in {self.step.duration_ms}ms (confidence: {confidence})")
    
    async def fail(self, error: str):
        """Mark step as failed"""
        self.step.status = 'failed'
        self.step.end_time = datetime.utcnow()
        self.step.error = error
        await sync_to_async(self.step.save, thread_sensitive=True)()

        print(f"    ✗ Failed: {error}")