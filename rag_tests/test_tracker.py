import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrackerTest(unittest.TestCase):
    def test_redacted_failure_revalidates_owner_without_deleted_source(self):
        source = ROOT / 'dream-fast-api/core/workflow_tracker.py'
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        writes, updates = [], []
        owner = SimpleNamespace(domain_user_id='accepted-domain', auth_user_id=7)
        job = SimpleNamespace(owner=owner, entry_ids=('deleted-source',))
        def sync_to_async(fn, **kwargs):
            async def call(*args, **kwargs): return fn(*args, **kwargs)
            return call
        def owned_write(**kwargs):
            writes.append({key: value for key, value in kwargs.items() if key != 'write'})
            return kwargs['write']('locked-owner')
        async def source_guarded_write(*args, **kwargs):
            raise RuntimeError('Deleted source cannot authorize publication')
        def model(name):
            return SimpleNamespace(objects=SimpleNamespace(filter=lambda **scope: SimpleNamespace(update=lambda **data: updates.append((name, scope, data)))))
        namespace = {'current_job': SimpleNamespace(get=lambda: job), 'sync_to_async': sync_to_async,
                     'owned_atomic_write': owned_write, 'async_job_write': source_guarded_write,
                     'timezone': SimpleNamespace(now=lambda: datetime.now(timezone.utc)),
                     'WorkflowStep': model('step'), 'WorkflowExecution': model('workflow')}
        exec(compile(tree, str(source), 'exec'), namespace)
        tracker = namespace['WorkflowTracker']('test', 'Synthetic', user_id='forged-body-owner')
        tracker.execution = SimpleNamespace(pk='accepted-workflow')
        asyncio.run(tracker.fail_workflow('private provider exception', step_id='accepted-step'))
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0], {'owner_id': 'accepted-domain', 'expected_auth_user_id': 7, 'workflow_id': 'accepted-workflow'})
        self.assertEqual([item[0] for item in updates], ['step', 'workflow'])
        self.assertTrue(all(item[2]['status'] == 'failed' for item in updates))
        self.assertNotIn('private provider exception', str(updates))
        self.assertEqual(updates[0][1]['execution__user'], 'locked-owner')
        self.assertEqual(updates[1][1]['user'], 'locked-owner')
        job.owner = SimpleNamespace(domain_user_id='revoked', auth_user_id=7)
        def denied_write(**kwargs): raise PermissionError('Owner revoked')
        namespace['owned_atomic_write'] = denied_write
        with self.assertRaises(PermissionError):
            asyncio.run(tracker.fail_workflow('private exception'))
        self.assertEqual(len(updates), 2, 'revoked owner cannot receive a failure write')

    def test_legacy_confidence_arguments_are_never_presented_as_measurements(self):
        source = ROOT / 'dream-fast-api/core/workflow_tracker.py'
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        records, citations = [], []
        def sync_to_async(fn, **kwargs):
            async def call(*args, **kwargs):
                return fn(*args, **kwargs)
            return call
        async def write(fn, workflow_id=None):
            return fn('owner')
        step = SimpleNamespace(pk='step', execution_id='workflow', start_time=datetime.now(timezone.utc), save=lambda: None)
        namespace = {'datetime': datetime, 'Optional': __import__('typing').Optional, 'Dict': dict, 'Any': object, 'List': list,
                     'timezone': SimpleNamespace(now=lambda: datetime.now(timezone.utc)), 'sync_to_async': sync_to_async,
                     'async_job_write': write, 'WorkflowStep': SimpleNamespace(objects=SimpleNamespace(get=lambda **kw: step)),
                     'StepCitation': SimpleNamespace(objects=SimpleNamespace(create=lambda **kw: citations.append(kw))),
                     'WorkflowExecution': SimpleNamespace(objects=SimpleNamespace(filter=lambda **kw: SimpleNamespace(update=lambda **data: records.append(data))))}
        exec(compile(tree, str(source), 'exec'), namespace)
        tracker = namespace['WorkflowTracker']('test', 'Synthetic')
        tracker.execution = SimpleNamespace(pk='workflow', id='workflow', save=lambda: None)
        asyncio.run(tracker.complete_workflow('Synthetic answer', confidence=.87))
        self.assertIsNone(records[0]['overall_confidence'] if records else tracker.execution.overall_confidence)
        asyncio.run(namespace['StepTracker'](step).complete({'count': 2}, confidence=.85, citations=[{'source': 'synthetic-paper', 'content': 'Synthetic quote.', 'confidence': .9}]))
        self.assertIsNone(step.confidence)
        self.assertEqual(step.output_data['calibration_status'], 'unavailable')
        self.assertFalse(citations, 'nonnullable legacy citation confidence must not get fake sentinel scores')


if __name__ == '__main__':
    unittest.main()
