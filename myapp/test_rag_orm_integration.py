"""Real analyzer, ORM publication and cookie routes with synthetic providers.

Only embeddings/reference retrieval/model dispatch are replaced. Job guards,
tracker transactions, selected-source hashes and serving are production code.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.sessions.models import Session
from django.db import close_old_connections
from django.test import TransactionTestCase
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied
from myapp.models import JournalEntry, CumulativeAnalysis, WorkflowExecution
from myapp import test_auth_security as auth_helpers
from myapp.security.analysis_input import analysis_input
from myapp.security.contracts import OwnerContext
from myapp.security.jobs import JobContext, current_job
from myapp.security.freshness import StaleSource
from myapp.longitudinal import services


class RagOrmIntegrationTests(TransactionTestCase):
    make_account = auth_helpers.AccountIsolationTests.make_account

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dream-fast-api'))
        from core import dream_analyzer
        self.module = dream_analyzer
        self.auth, self.profile, self.client, self.csrf = self.make_account('rag-orm')
        data = self.client.get('/api/auth/session/').json()
        self.owner = OwnerContext(self.profile.pk, self.auth.pk, False,
            timezone.datetime.fromisoformat(data['loginDeadline']), self.client.session.session_key)
        self.entries = [JournalEntry.objects.create(user=self.profile, content=text)
            for text in ('Synthetic bridge record.', 'Synthetic river record.')]
        rows, self.config, self.capture = analysis_input(self.owner, [e.pk for e in self.entries], with_snapshot=True)
        self.inputs = [dream_analyzer.JournalEntry(**row) for row in rows]
        self.analyzer = object.__new__(dream_analyzer.DreamJournalAnalyzer)
        self.analyzer.embeddings = object()
        self.dispatches = []
        outer = self
        class FakeFAISS:
            @staticmethod
            def from_documents(documents, embeddings):
                outer.dispatches.append('embeddings')
                return SimpleNamespace(similarity_search=lambda question, k: documents[:1])
        self.faiss = FakeFAISS
        def invoke(messages, **kwargs):
            close_old_connections()
            self.dispatches.append('model')
            payload = json.loads(messages[1].content.split('\n', 1)[1])
            text = 'The selected record describes a bridge.'
            item = payload['evidence'][0]
            return SimpleNamespace(content=json.dumps({'answer': text,
                'claims': [{'kind': 'observation', 'text': text, 'evidence_ids': [item['evidence_id']]}]}))
        self.analyzer.llm = SimpleNamespace(invoke=invoke, get_num_tokens_from_messages=lambda _: 500)
        async def retrieve(*args, **kwargs):
            return {'docs': [], 'retrieval': {}, 'dream_elements': {}}
        self.analyzer.comprehensive_knowledge_retrieval = retrieve

    def execute(self, *, durable=False, existing_workflow_id=None):
        owner = OwnerContext(self.profile.pk, self.auth.pk, False, self.owner.login_deadline) if durable else self.owner
        job = JobContext(owner, tuple(e.pk for e in self.entries), self.capture, not durable)
        async def run():
            token = current_job.set(job)
            try:
                return await self.analyzer.qa_analysis_with_workflow(
                    self.inputs, self.config, existing_workflow_id=existing_workflow_id)
            finally:
                current_job.reset(token)
        with patch.object(self.module, 'FAISS', self.faiss):
            return async_to_sync(run)()

    def accept_workflow(self):
        from core.workflow_tracker import WorkflowTracker
        async def accept():
            token = current_job.set(JobContext(self.owner,
                tuple(e.pk for e in self.entries), self.capture, True))
            try:
                return await WorkflowTracker('cumulative_analysis', 'Synthetic accepted job').start_workflow()
            finally:
                current_job.reset(token)
        return async_to_sync(accept)()

    def test_real_publication_and_omitted_source_revocation_with_two_owners(self):
        answer, workflow_id = self.execute()
        self.assertEqual(self.dispatches, ['embeddings', 'model'])
        workflow = WorkflowExecution.objects.get(pk=workflow_id)
        result = CumulativeAnalysis.objects.get(workflow_execution=workflow)
        self.assertFalse(result.is_historical)
        self.assertEqual(workflow.source_metadata, self.capture)
        dependencies = workflow.steps.get(step_type='answer_generation').output_data['coverage']['source_dependencies']
        self.assertEqual(len(dependencies), 2, 'omitted retrieval input is still a dependency')
        self.assertIn(answer, self.client.get(f'/api/workflows/{workflow_id}/').content.decode())
        _, other, other_client, _ = self.make_account('rag-orm-other')
        self.assertEqual(other_client.get(f'/api/workflows/{workflow_id}/').status_code, 404)
        self.assertEqual(other_client.get('/api/cumulative-analyses/').json()['count'], 0)
        JournalEntry.objects.filter(pk=self.entries[1].pk).update(content='synthetic edited omitted source')
        self.assertEqual(self.client.get(f'/api/workflows/{workflow_id}/').status_code, 404)
        self.assertEqual(self.client.get('/api/cumulative-analyses/').json()['count'], 0)

    def test_privacy_change_between_retrieval_and_model_stops_dispatch(self):
        async def withdraw(*args, **kwargs):
            await sync_to_async(services.set_preferences, thread_sensitive=True)(self.owner, enabled=True)
            return {'docs': [], 'retrieval': {}, 'dream_elements': {}}
        self.analyzer.comprehensive_knowledge_retrieval = withdraw
        with self.assertRaises(StaleSource): self.execute()
        self.assertEqual(self.dispatches, ['embeddings'])
        self.assertFalse(CumulativeAnalysis.objects.exists())
        self.assertEqual(WorkflowExecution.objects.get().status, 'failed')

    def test_logout_during_provider_blocks_publication_but_durable_job_outlives_logout(self):
        accepted_id = self.accept_workflow()
        revoked_id = self.accept_workflow()
        original = self.analyzer.llm.invoke
        def logout(messages, **kwargs):
            close_old_connections()
            Session.objects.filter(session_key=self.owner.session_key).delete()
            return original(messages, **kwargs)
        self.analyzer.llm.invoke = logout
        with self.assertRaises(PermissionDenied): self.execute()
        self.assertFalse(CumulativeAnalysis.objects.exists())
        self.analyzer.llm.invoke = original
        answer, workflow_id = self.execute(durable=True, existing_workflow_id=accepted_id)
        self.assertEqual(workflow_id, accepted_id)
        self.assertEqual(CumulativeAnalysis.objects.filter(workflow_execution_id=workflow_id).count(), 1)
        def deactivate(messages, **kwargs):
            close_old_connections()
            self.auth.is_active = False
            self.auth.save(update_fields=['is_active'])
            return original(messages, **kwargs)
        self.analyzer.llm.invoke = deactivate
        with self.assertRaises(PermissionDenied):
            self.execute(durable=True, existing_workflow_id=revoked_id)
        self.assertEqual(CumulativeAnalysis.objects.count(), 1)
        self.assertFalse(CumulativeAnalysis.objects.filter(workflow_execution_id=revoked_id).exists())

    def test_changed_sources_before_embedding_never_dispatch_or_create_workflow(self):
        JournalEntry.objects.filter(pk=self.entries[0].pk).update(content='synthetic source changed before work')
        with self.assertRaises(StaleSource): self.execute()
        self.assertEqual(self.dispatches, [])
        self.assertFalse(WorkflowExecution.objects.exists())
