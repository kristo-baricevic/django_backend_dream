import django_setup
import os
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import UUID
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from asgiref.sync import sync_to_async
from django.conf import settings as django_settings
from django.http import Http404
from rest_framework.exceptions import APIException
from myapp.models import WorkflowExecution, CustomQuestion
from myapp.security.service_auth import authenticate_service_request
from myapp.security.analysis_input import analysis_input
from myapp.security.owner import require_owned_object
from myapp.security.jobs import JobContext, current_job, async_job_write
from myapp.security.owner import owned_atomic_write
from myapp.security.freshness import lock_state, result_current
from myapp.security.owner import validate_request_session

service = None

@asynccontextmanager
async def lifespan(app):
    global service
    if os.environ.get('DREAM_DISABLE_PROVIDER_STARTUP') == '1':
        if django_settings.DREAM_ENV not in {'local', 'test'}:
            raise RuntimeError('Offline startup is local/test only')
    else:
        from core.dream_analyzer import DreamJournalService
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('Configured provider credentials required')
        service = DreamJournalService(key)
        await service.initialize()
    yield
    service = None

app = FastAPI(title='Dream Journal Analysis API', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=django_settings.CORS_ALLOWED_ORIGINS,
                   allow_credentials=True, allow_methods=['GET', 'POST'],
                   allow_headers=['Content-Type', 'X-CSRFToken', 'X-Dream-Principal', 'X-Dream-Session'])

@app.middleware('http')
async def private_cache(request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store, private'
    response.headers['Vary'] = 'Cookie'
    return response

def get_service():
    if service is None:
        raise HTTPException(503, 'Analysis service unavailable.')
    return service

class EntrySelection(BaseModel):
    id: UUID

class Selection(BaseModel):
    entries: list[EntrySelection] = Field(default_factory=list, max_length=500)
    settings: Optional[dict[str, Any]] = None

class Question(Selection):
    question: str = Field(min_length=1, max_length=4000)

class Analyze(BaseModel):
    content: str = Field(min_length=1, max_length=100000)
    entry_id: Optional[UUID] = None
    settings: Optional[dict[str, Any]] = None

class Generate(BaseModel):
    theme: str = Field(default='flying', max_length=4000)

async def inputs(owner, ids):
    try:
        rows, config, snapshot = await sync_to_async(analysis_input, thread_sensitive=True)(owner, ids, with_snapshot=True)
    except Http404:
        raise HTTPException(404, 'Entry not found.')
    except APIException as error:
        raise HTTPException(error.status_code, 'Invalid analysis selection.')
    if sum(len(row['content']) for row in rows) > 200000:
        raise HTTPException(400, 'Selected source text exceeds the analysis budget.')
    from core.dream_analyzer import JournalEntry
    return [JournalEntry(**row) for row in rows], config, JobContext(owner, tuple(ids), snapshot, True)

@app.get('/health')
async def health():
    return {'status': 'healthy'}

@app.get('/')
async def root(owner=Depends(authenticate_service_request)):
    return {'message': 'Dream Journal Analysis API'}

@app.post('/analyze')
async def analyze(request: Analyze, owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    ids = (request.entry_id,) if request.entry_id else ()
    entries, config, job = await inputs(owner, ids)
    token = current_job.set(job)
    try:
        result = await svc.analyze_single_entry(entries[0].content if entries else request.content, config)
        await async_job_write(lambda profile: None)
        payload = result.model_dump(mode='json') if hasattr(result, 'model_dump') else dict(result)
        payload['source_snapshot'] = job.source_metadata
        return payload
    except APIException:
        raise HTTPException(403, 'Analysis owner unavailable.')
    except Exception:
        raise HTTPException(503, 'Analysis unavailable.')
    finally:
        current_job.reset(token)

@app.post('/generate-dream')
async def generate(request: Generate, owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    _, _, job = await inputs(owner, ())
    token = current_job.set(job)
    try:
        await async_job_write(lambda profile: None)
        result = await svc.generate_sample_dream(request.theme)
        await async_job_write(lambda profile: None)
        return {'dream': result}
    finally:
        current_job.reset(token)

@app.get('/emotions')
async def emotions(owner=Depends(authenticate_service_request)):
    from core.dream_analyzer import EmotionType
    return [emotion.value for emotion in EmotionType]

@app.get('/personalities')
async def personalities(owner=Depends(authenticate_service_request)):
    return {'empathetic': 'Empathetic analysis', 'analytical': 'Logical analysis',
            'mystical': 'Mystical interpretation', 'practical': 'Practical analysis'}

async def run_workflow(job, config, entries, svc, workflow_id, question=None):
    token = current_job.set(job)
    try:
        if question is None:
            await svc.analyzer.qa_analysis_with_workflow(entries, config, existing_workflow_id=workflow_id)
        else:
            # The analyzer atomically publishes result, source dependencies and
            # workflow status under the accepted owner/workflow locks.
            await svc.analyzer.custom_question_with_workflow(
                question, entries, config, existing_workflow_id=workflow_id)
    except Exception:
        try:
            from core.workflow_tracker import _fail_owned_workflow
            await _fail_owned_workflow(workflow_id)
        except Exception:
            pass
    finally:
        current_job.reset(token)

async def start(request, background_tasks, owner, svc, question=None):
    ids = tuple(row.id for row in request.entries)
    entries, config, request_job = await inputs(owner, ids)
    if not entries:
        raise HTTPException(400, 'Select at least one entry.')
    token = current_job.set(request_job)
    try:
        workflow = await async_job_write(lambda profile: WorkflowExecution.objects.create(
            user=profile, accepted_auth_user_id=owner.auth_user_id, status='running',
            source_metadata=request_job.source_metadata,
            workflow_type='custom_question' if question else 'cumulative_analysis',
            routine_name='Custom Question' if question else 'Dream Analysis'))
    finally:
        current_job.reset(token)
    # Accepted durable jobs use the same captured input and account mapping;
    # they never retain or require the initiating session credential.
    durable_owner = type(owner)(owner.domain_user_id, owner.auth_user_id, owner.is_staff, owner.login_deadline)
    durable_job = JobContext(durable_owner, ids, request_job.source_metadata)
    background_tasks.add_task(run_workflow, durable_job, config, entries, svc, str(workflow.pk), question)
    return {'workflow_id': str(workflow.pk), 'status': 'processing'}

@app.post('/qa-with-workflow')
async def qa_workflow(request: Selection, background_tasks: BackgroundTasks,
                      owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    return await start(request, background_tasks, owner, svc)

@app.post('/custom-question-with-workflow')
async def custom_workflow(request: Question, background_tasks: BackgroundTasks,
                          owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    return await start(request, background_tasks, owner, svc, request.question)

@app.post('/qa')
async def qa(request: Selection, owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    entries, config, job = await inputs(owner, tuple(row.id for row in request.entries))
    token = current_job.set(job)
    try:
        answer = await svc.get_cumulative_analysis(entries, settings=config)
        await async_job_write(lambda profile: None)
        return {'answer': answer}
    finally:
        current_job.reset(token)

@app.post('/custom-question')
async def custom(request: Question, owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    entries, config, job = await inputs(owner, tuple(row.id for row in request.entries))
    token = current_job.set(job)
    try:
        answer = await svc.ask_custom_question(request.question, entries, settings=config)
        await async_job_write(lambda profile: None)
        return {'answer': answer}
    finally:
        current_job.reset(token)

@app.post('/batch-analyze')
async def batch(request: Selection, owner=Depends(authenticate_service_request), svc=Depends(get_service)):
    entries, config, job = await inputs(owner, tuple(row.id for row in request.entries))
    token = current_job.set(job)
    try:
        result = await svc.analyzer.batch_analyze_entries(entries, config)
        await async_job_write(lambda profile: None)
        return result
    finally:
        current_job.reset(token)

@app.get('/workflows/{workflow_id}')
async def workflow(workflow_id: UUID, owner=Depends(authenticate_service_request)):
    def read(profile):
        validate_request_session(owner)
        row = require_owned_object(WorkflowExecution, owner, workflow_id)
        if not result_current(row, owner):
            raise Http404
        return {'id': str(row.pk), 'workflow_type': row.workflow_type, 'routine_name': row.routine_name,
                'status': row.status, 'analysis_id': row.analysis_id, 'final_result': row.final_result}
    try:
        return await sync_to_async(owned_atomic_write, thread_sensitive=True)(
            owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
            workflow_id=None, lock_state=lock_state, write=read)
    except Http404:
        raise HTTPException(404, 'Workflow not found.')
