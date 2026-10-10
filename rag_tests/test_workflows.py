"""Mock provider/index/ORM tests execute real analyzer workflow methods."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import ast
from types import SimpleNamespace
import unittest
from test_analyzer_adapters import analyzer_class


class WorkflowTests(unittest.TestCase):
    def test_single_saved_and_draft_sources_keep_honest_identity_and_time(self):
        from core.rag_contracts import entry_evidence
        for saved in (True, False):
            row = SimpleNamespace(id='saved-source', content='I saw a bridge.', created_at=datetime(2026, 1, 2).astimezone())
            job = SimpleNamespace(owner=object(), entry_ids=(row.id,) if saved else ())
            inputs, revalidations = [], []
            def invoke(messages, **kwargs):
                data = json.loads(messages[1].content.split('\n', 1)[1])
                inputs.append(data)
                evidence = next(item for item in data['evidence'] if item['kind'] in {'user_entry', 'user_draft'})
                text = 'A bridge appears in this record.'
                return SimpleNamespace(content=json.dumps({'answer': text, 'claims': [{'kind': 'observation', 'text': text, 'evidence_ids': [evidence['evidence_id']]}],
                    'presentation': {'mood': 'interest', 'summary': text, 'subject': 'Bridge', 'negative': False, 'sentiment_score': 0, 'symbols': ['bridge']}}))
            async def write(fn, workflow_id=None):
                revalidations.append(job.entry_ids)
                return fn(job.owner)
            cls = analyzer_class({'analyze_entry', '_grounded_response'}, current_job=SimpleNamespace(get=lambda: job), load_owned_entries=lambda owner, ids: [row], async_job_write=write,
                JournalAnalysis=lambda **kw: kw, EmotionType=lambda value: value, get_emotion_color=lambda mood: '#000000', json=json)
            analyzer = cls()
            analyzer.llm = SimpleNamespace(invoke=invoke, get_num_tokens_from_messages=lambda messages: 1000)
            async def resolve(settings): return {}, SimpleNamespace(name='Academic', tone='Gentle'), {'theory': 1}, {}
            async def retrieve(*args, **kw): return {'docs': [], 'dream_elements': {}, 'retrieval': {}}
            analyzer._resolve_analysis_profile, analyzer.comprehensive_knowledge_retrieval = resolve, retrieve
            analyzer.normalize_emotion = lambda value: value
            result = asyncio.run(analyzer.analyze_entry(row.content, {}))
            self.assertEqual(result['evidence'][0]['kind'], 'user_entry' if saved else 'user_draft')
            self.assertEqual(result['evidence'][0]['entry_id'], row.id if saved else None)
            self.assertEqual(result['evidence'][0]['recorded_at'], row.created_at.isoformat() if saved else None)
            self.assertEqual(revalidations, [job.entry_ids, job.entry_ids], 'saved and draft requests must fence provider dispatch and response release')
            self.assertEqual([item['entry_id'] for item in result['coverage']['source_dependencies']],
                             [row.id] if saved else [], 'drafts must not invent saved-source dependencies')

    def test_batch_sets_and_restores_each_authorized_entry_context(self):
        from contextvars import ContextVar
        context = ContextVar('synthetic-job', default=None)
        owner = object()
        make_job = lambda owner, ids, metadata=None, bound=False: SimpleNamespace(owner=owner, entry_ids=ids, source_metadata=metadata, request_bound=bound)
        cls = analyzer_class({'batch_analyze_entries'}, current_job=context, JobContext=make_job)
        analyzer = cls()
        a, b = SimpleNamespace(id='a', content='bridge'), SimpleNamespace(id='b', content='tram')
        async def analyze(content, settings):
            self.assertEqual(context.get().source_metadata, {'synthetic_snapshot': True})
            self.assertTrue(context.get().request_bound)
            return (context.get().entry_ids[0], content)
        analyzer.analyze_entry = analyze
        original = make_job(owner, ('a', 'b'), {'synthetic_snapshot': True}, True)
        token = context.set(original)
        try:
            self.assertEqual(asyncio.run(analyzer.batch_analyze_entries([a, b], {})), [('a', 'bridge'), ('b', 'tram')])
            self.assertIs(context.get(), original)
            with self.assertRaisesRegex(ValueError, 'analysis_owner_unavailable'):
                asyncio.run(analyzer.batch_analyze_entries([a], {}))
        finally:
            context.reset(token)

    def test_workflow_atomic_publication_rechecks_sources_and_ignores_body_owner(self):
        for custom in (False, True):
            for edited in (False, True):
                with self.subTest(custom=custom, edited=edited):
                    writes, finishes = [], []
                    owner = object()
                    entry = SimpleNamespace(id='entry', content='Synthetic bridge', created_at=datetime(2026, 10, 10))
                    fresh = SimpleNamespace(id='entry', content='Synthetic tram' if edited else entry.content)
                    async def write(fn, workflow_id=None):
                        return fn(owner)
                    class Step:
                        def __init__(self):
                            self.step = SimpleNamespace(pk='step', start_time=datetime.now().astimezone(), save=lambda: None)
                        async def complete(self, **kw): pass
                        async def fail(self, error): finishes.append('failed-step')
                    class Tracker:
                        def __init__(self, **kw): self.execution = None
                        async def start_workflow(self): return 'workflow'
                        async def start_step(self, *args, **kwargs): return Step()
                        async def fail_workflow(self, error, step_id=None): finishes.append('failed-workflow')
                    saved = SimpleNamespace(id='saved', pk='saved')
                    model = SimpleNamespace(objects=SimpleNamespace(create=lambda **kw: writes.append(kw) or saved))
                    step_row = SimpleNamespace(start_time=datetime.now().astimezone(), save=lambda: None)
                    cls = analyzer_class({'_run_grounded_workflow'}, WorkflowTracker=Tracker, current_job=SimpleNamespace(get=lambda: SimpleNamespace(owner=owner, entry_ids=('entry',))),
                        async_job_write=write, load_owned_entries=lambda current_owner, ids: [fresh],
                        CumulativeAnalysis=model, CustomQuestion=model, WorkflowStep=SimpleNamespace(objects=SimpleNamespace(get=lambda **kw: step_row)),
                        WorkflowExecution=SimpleNamespace(objects=SimpleNamespace(filter=lambda **kw: SimpleNamespace(update=lambda **data: finishes.append(data)))),
                        timezone=SimpleNamespace(now=lambda: datetime.now().astimezone()))
                    analyzer = cls()
                    async def resolve(settings): return settings, SimpleNamespace(name='Academic'), {'theory': 1}, {}
                    from core.rag_contracts import entry_evidence
                    async def generate(*args, **kw):
                        return {'answer': 'Synthetic reflection.', 'evidence': [], 'claims': [], 'semantic_support': 'not_evaluated', 'span_status': 'validated_at_pack_time'}, {'source_dependencies': [{'entry_id': 'entry', 'content_hash': entry_evidence('entry', entry.content)['content_hash']}]}
                    analyzer._resolve_analysis_profile, analyzer._grounded_response = resolve, generate
                    call = analyzer._run_grounded_workflow([entry], {'user_id': 'forged-body-owner'}, None, question='Where?' if custom else None)
                    if edited:
                        with self.assertRaisesRegex(ValueError, 'analysis_sources_changed'): asyncio.run(call)
                        self.assertFalse(writes)
                    else:
                        asyncio.run(call)
                        self.assertIs(writes[0]['user'], owner)
                        self.assertEqual(finishes[0]['analysis_id'], 'saved')
                        self.assertIsNone(finishes[0]['overall_confidence'])

    def test_http_service_wrappers_call_the_implemented_workflows(self):
        source = Path(__file__).resolve().parents[1] / 'dream-fast-api/core/dream_analyzer.py'
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DreamJournalService')
        cls.body = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name in {'get_cumulative_analysis', 'ask_custom_question'}]
        tree.body = [cls]
        namespace = {'List': list, 'Dict': dict, 'Any': object, 'JournalEntry': object}
        exec(compile(tree, str(source), 'exec', flags=__import__('__future__').annotations.compiler_flag), namespace)
        class Analyzer:
            async def qa_analysis_with_workflow(self, entries, settings):
                return 'Synthetic cumulative answer', 'workflow'
            async def custom_question_with_workflow(self, question, entries, settings):
                return 'Synthetic question answer', 'question', 'workflow'
        service = namespace['DreamJournalService']()
        service.analyzer = Analyzer()
        self.assertEqual(asyncio.run(service.get_cumulative_analysis([], settings={})), 'Synthetic cumulative answer')
        self.assertEqual(asyncio.run(service.ask_custom_question('Where?', [], settings={})), 'Synthetic question answer')

    def test_single_entry_uses_shared_resolver_and_contract(self):
        async def write(fn, workflow_id=None): return fn('owner')
        cls = analyzer_class({'analyze_entry'}, JournalAnalysis=lambda **kw: kw, EmotionType=lambda value: value, get_emotion_color=lambda mood: '#000000', async_job_write=write, current_job=SimpleNamespace(get=lambda: SimpleNamespace(entry_ids=(), owner=object())))
        analyzer = cls()
        calls = []
        async def resolve(settings):
            calls.append(('resolve', settings))
            return {'doctorPersonality': 'Academic', 'doctor_influence': 0}, SimpleNamespace(name='Academic'), {'theory': 1}, {'clinical_authority': 'none'}
        async def generate(question, entries, settings, profile, weights, **kwargs):
            calls.append(('generate', settings))
            return {'answer': 'One possibility is change.', 'claims': [], 'evidence': [], 'calibration_status': 'unavailable', 'presentation': {'mood': 'interest', 'summary': 'Bridge.', 'negative': False, 'subject': 'Bridge', 'sentiment_score': 0, 'symbols': ['bridge']}}, {'selected_entries': 1}
        analyzer._resolve_analysis_profile = resolve
        analyzer._grounded_response = generate
        analyzer.normalize_emotion = lambda value: value
        try:
            result = asyncio.run(analyzer.analyze_entry('Synthetic bridge.', {'doctor_influence': 0}))
        except (ImportError, AttributeError, TypeError):
            self.fail('single-entry route does not use the shared resolver and grounded generator')
        self.assertEqual(calls[0][0], 'resolve')
        self.assertEqual(calls[1][1]['doctor_influence'], 0)
        self.assertEqual(result['provenance']['clinical_authority'], 'none')
        self.assertEqual(result['calibration_status'], 'unavailable')

    def test_grounded_generator_uses_actual_question_and_checks_every_claim(self):
        doc = SimpleNamespace(page_content='I saw a bridge.', metadata={'source': 'synthetic-entry', 'date': '2026-10-10T00:00:00Z'})
        class FakeFAISS:
            @staticmethod
            def from_documents(docs, embeddings):
                return SimpleNamespace(similarity_search=lambda query, k: docs)
        prompts = []
        def invoke(prompt, **kwargs):
            prompts.append('\n'.join(message.content for message in prompt))
            return SimpleNamespace(content=json.dumps({'answer': 'What might the bridge mean to you?', 'claims': [{'kind': 'reflective_prompt', 'text': 'What might the bridge mean to you?', 'evidence_ids': []}]}))
        cls = analyzer_class({'_grounded_response'}, FAISS=FakeFAISS, json=json)
        self.assertTrue(hasattr(cls, '_grounded_response'), 'missing bounded shared generation pipeline')
        analyzer = cls()
        analyzer.embeddings = object()
        analyzer.llm = SimpleNamespace(invoke=invoke, get_num_tokens_from_messages=lambda prompt: 1000)
        async def retrieve(entries, settings, **kw):
            self.assertEqual(kw['question'], 'Where was the bridge?')
            return {'docs': [], 'dream_elements': {}, 'retrieval': {'queries_executed': 1}}
        analyzer.comprehensive_knowledge_retrieval = retrieve
        entry = SimpleNamespace(id='synthetic-entry', content=doc.page_content, created_at=datetime(2026, 10, 10))
        profile = SimpleNamespace(name='Academic', tone='Gentle', prompt_style='Ask questions.')
        answer, coverage = asyncio.run(analyzer._grounded_response('Where was the bridge?', [entry], {'medicalHistory': {'psychological': ['synthetic-sensitive-value']}}, profile, {'theory': 1}))
        self.assertEqual(answer['calibration_status'], 'unavailable')
        self.assertEqual(coverage['selected_entries'], 1)
        self.assertIn('fictional response style', prompts[0])
        self.assertIn('untrusted data', prompts[0])
        self.assertNotIn('synthetic-sensitive-value', prompts[0])
        analyzer.llm = SimpleNamespace(invoke=lambda prompt, **kw: SimpleNamespace(content=json.dumps({'answer': 'Undeclared medical claim.', 'claims': [{'kind': 'hypothesis', 'text': 'Different claim.', 'evidence_ids': []}]})), get_num_tokens_from_messages=lambda prompt: 1000)
        with self.assertRaises(ValueError):
            asyncio.run(analyzer._grounded_response('Where was the bridge?', [entry], {}, profile, {'theory': 1}))
        large = SimpleNamespace(id='too-large', content='x' * 200001, created_at=entry.created_at)
        with self.assertRaisesRegex(ValueError, 'analysis_source_budget_exceeded'):
            asyncio.run(analyzer._grounded_response('Where?', [large], {}, profile, {'theory': 1}))
        analyzer.llm.get_num_tokens_from_messages = lambda prompt: 10001
        with self.assertRaisesRegex(ValueError, 'analysis_prompt_budget_exceeded'):
            asyncio.run(analyzer._grounded_response('Where was the bridge?', [entry], {}, profile, {'theory': 1}))


if __name__ == '__main__':
    unittest.main()
