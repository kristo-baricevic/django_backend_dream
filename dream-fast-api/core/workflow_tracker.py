from django.utils import timezone
from myapp.models import WorkflowExecution, WorkflowStep, StepCitation
from myapp.security.jobs import current_job, async_job_write


class WorkflowTracker:
    def __init__(self, workflow_type, routine_name, user_id=None):
        self.workflow_type, self.routine_name = workflow_type, routine_name
        self.execution, self.current_step_number = None, 0

    async def start_workflow(self):
        job = current_job.get()
        self.execution = await async_job_write(lambda profile: WorkflowExecution.objects.create(
            user=profile, accepted_auth_user_id=job.owner.auth_user_id,
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
            overall_confidence=confidence, total_citations=total_citations), self.execution.pk)

    async def fail_workflow(self, error):
        await async_job_write(lambda profile: WorkflowExecution.objects.filter(pk=self.execution.pk, user=profile).update(
            status='failed', end_time=timezone.now(), error_message='Analysis unavailable.'), self.execution.pk)


class StepTracker:
    def __init__(self, step):
        self.step = step

    async def complete(self, output, confidence=None, reasoning=None, model=None, tokens=None, citations=None):
        def write(profile):
            step = WorkflowStep.objects.get(pk=self.step.pk, execution__user=profile)
            step.status, step.end_time = 'completed', timezone.now()
            step.duration_ms = int((step.end_time - step.start_time).total_seconds() * 1000)
            step.output_data, step.confidence, step.reasoning = output, confidence, reasoning
            step.model_used, step.tokens_used = model, tokens
            step.save()
            for citation in citations or []:
                StepCitation.objects.create(step=step, source=citation.get('source', 'unknown'),
                    content=citation.get('content', ''), confidence=citation.get('confidence', 0.0),
                    reference=citation.get('reference'))
        await async_job_write(write, self.step.execution_id)

    async def fail(self, error):
        await async_job_write(lambda profile: WorkflowStep.objects.filter(pk=self.step.pk, execution__user=profile).update(
            status='failed', end_time=timezone.now(), error='Analysis unavailable.'), self.step.execution_id)
