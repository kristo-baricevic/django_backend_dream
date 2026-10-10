from django.utils import timezone
from asgiref.sync import sync_to_async
from myapp.models import WorkflowExecution, WorkflowStep
from myapp.security.jobs import current_job, async_job_write
from myapp.security.owner import owned_atomic_write


async def _fail_owned_workflow(workflow_id, step_id=None):
    """Redacted terminal update; source deletion must not block failure status.

    Account, owner and workflow authority are still locked and revalidated. This
    narrow operation cannot publish result content or accept a caller callback.
    """
    job = current_job.get()
    if job is None:
        raise ValueError('analysis_owner_unavailable')
    def write(profile):
        if step_id is not None:
            WorkflowStep.objects.filter(pk=step_id, execution_id=workflow_id, execution__user=profile,
                                        status='running').update(
                status='failed', end_time=timezone.now(), error='Analysis unavailable.')
        WorkflowExecution.objects.filter(pk=workflow_id, user=profile, status='running').update(
            status='failed', end_time=timezone.now(), error_message='Analysis unavailable.')
    await sync_to_async(owned_atomic_write, thread_sensitive=True)(
        owner_id=job.owner.domain_user_id, expected_auth_user_id=job.owner.auth_user_id,
        workflow_id=workflow_id, write=write)


class WorkflowTracker:
    def __init__(self, workflow_type, routine_name, user_id=None):
        self.workflow_type, self.routine_name = workflow_type, routine_name
        self.execution, self.current_step_number = None, 0

    async def start_workflow(self):
        job = current_job.get()
        self.execution = await async_job_write(lambda profile: WorkflowExecution.objects.create(
            user=profile, accepted_auth_user_id=job.owner.auth_user_id,
            source_metadata=job.source_metadata or {},
            workflow_type=self.workflow_type, routine_name=self.routine_name, status='running'))
        return str(self.execution.pk)

    async def start_step(self, name, step_type, input_data=None):
        self.current_step_number += 1
        step = await async_job_write(lambda profile: WorkflowStep.objects.create(
            execution_id=self.execution.pk, step_number=self.current_step_number, name=name,
            step_type=step_type, status='running', start_time=timezone.now(), input_data=input_data), self.execution.pk)
        return StepTracker(step)

    async def complete_workflow(self, result, confidence=None, total_citations=0):
        await async_job_write(lambda profile: WorkflowExecution.objects.filter(pk=self.execution.pk, user=profile).update(
            status='completed', end_time=timezone.now(), final_result=result,
            overall_confidence=None, total_citations=total_citations), self.execution.pk)

    async def fail_workflow(self, error, step_id=None):
        await _fail_owned_workflow(self.execution.pk, step_id)


class StepTracker:
    def __init__(self, step):
        self.step = step

    async def complete(self, output, confidence=None, reasoning=None, model=None, tokens=None, citations=None):
        def write(profile):
            step = WorkflowStep.objects.get(pk=self.step.pk, execution__user=profile)
            step.status, step.end_time = 'completed', timezone.now()
            step.duration_ms = int((step.end_time - step.start_time).total_seconds() * 1000)
            data = dict(output) if isinstance(output, dict) else {'result': output}
            data['calibration_status'] = 'unavailable'
            if citations:
                data['references'] = [{key: value for key, value in citation.items() if key != 'confidence'} for citation in citations]
            step.output_data, step.confidence, step.reasoning = data, None, reasoning
            step.model_used, step.tokens_used = model, tokens
            step.save()
            # Legacy StepCitation requires a numeric confidence. New uncalibrated
            # references stay in JSON output_data, without invented score values.
        await async_job_write(write, self.step.execution_id)

    async def fail(self, error):
        await _fail_owned_workflow(self.step.execution_id, self.step.pk)
