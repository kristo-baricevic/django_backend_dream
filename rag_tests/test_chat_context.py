import asyncio
import ast
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace


class ChatProfileTest(unittest.TestCase):
    def test_failed_workflow_stops_polling_and_returns_safe_failure(self):
        from unittest.mock import patch
        path = Path(__file__).resolve().parents[1] / 'dream-analyzer-chatbot/app/agents/dream_agent.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DreamAgent')
        cls.body = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == '_execute_tool']
        tree.body = [cls]
        polls = []
        async def run_analysis(*args, **kwargs): return {'workflow_id': 'synthetic-workflow'}
        async def entries(**kwargs): return {'results': [{'id': 'synthetic-record'}]}
        async def status(workflow_id):
            polls.append(workflow_id)
            return {'status': 'failed', 'error_message': 'Private provider detail'}
        async def no_wait(*args): pass
        namespace = {'Dict': dict, 'Any': object, 'run_analysis': run_analysis}
        exec(compile(tree, str(path), 'exec'), namespace)
        agent = namespace['DreamAgent']()
        agent.api_client = SimpleNamespace(get_journal_entries=entries, get_workflow_status=status)
        with patch('asyncio.sleep', no_wait):
            for tool in ('get_cumulative_analysis', 'ask_custom_question'):
                result = asyncio.run(agent._execute_tool(tool, {'question': 'Synthetic question'}))
                self.assertFalse(result['success'])
                self.assertEqual(result.get('error'), 'Analysis unavailable.')
                self.assertNotIn('Private provider detail', str(result))
        self.assertEqual(polls, ['synthetic-workflow', 'synthetic-workflow'])

    def test_model_selected_save_cannot_write_without_explicit_user_command(self):
        path = Path(__file__).resolve().parents[1] / 'dream-analyzer-chatbot/app/agents/dream_agent.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DreamAgent')
        cls.body = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == '_execute_tool']
        tree.body = [cls]
        namespace = {'Dict': dict, 'Any': object}
        exec(compile(tree, str(path), 'exec'), namespace)
        agent = namespace['DreamAgent']()
        agent.api_client = SimpleNamespace()
        result = asyncio.run(agent._execute_tool('save_dream', {'content': 'Synthetic dream'}))
        self.assertFalse(result['success'])
        self.assertIn('journal editor', result['error'])
        writes = []
        async def create(content): writes.append(content); return {'id': 'synthetic-id'}
        async def update(entry_id, content): return {'id': entry_id, 'analysis': None}
        agent.api_client = SimpleNamespace(create_journal_entry=create, update_journal_entry=update)
        agent.allow_save = True
        self.assertFalse(asyncio.run(agent._execute_tool('save_dream', {'content': 'Synthetic dream'}))['success'])
        self.assertFalse(asyncio.run(agent._execute_tool('save_dream', {'content': 'Duplicate synthetic dream'}))['success'])
        self.assertEqual(writes, [], 'stateless chat must never persist model-invented source text')

    def test_analyzer_answer_is_displayed_without_an_unchecked_rewrite(self):
        path = Path(__file__).resolve().parents[1] / 'dream-analyzer-chatbot/app/agents/dream_agent.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DreamAgent')
        cls.body = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'process_message']
        tree.body = [cls]
        namespace = {'Dict': dict, 'Any': object, 'List': list, 'ChatMessage': object, 'UIEvent': object, 'json': __import__('json'), 'settings': SimpleNamespace(OPENAI_MODEL='existing-model')}
        namespace['format_grounded_text'] = lambda answer, metadata: answer
        exec(compile(tree, str(path), 'exec'), namespace)
        count = []
        async def create(**kwargs):
            count.append(kwargs)
            if len(count) > 1: self.fail('validated analysis was rewritten by an unchecked model call')
            call = SimpleNamespace(id='tool', function=SimpleNamespace(name='analyze_dream', arguments='{}'))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
        agent = namespace['DreamAgent']()
        agent.system_prompt, agent.tools = 'Synthetic instruction', []
        validations = []
        async def validate(): validations.append('current')
        agent.api_client = SimpleNamespace(client=SimpleNamespace(validate=validate))
        agent._provider = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        async def execute(name, args): return {'success': True, 'analysis': {'interpretation': 'Validated reflective answer.'}}
        agent._execute_tool = execute
        self.assertEqual(asyncio.run(agent.process_message('Synthetic question', []))['text'], 'Validated reflective answer.')
        self.assertEqual(len(count), 1)
        self.assertEqual(validations, ['current'])

    def test_display_formatter_maps_only_known_sources_to_claims(self):
        path = Path(__file__).resolve().parents[1] / 'dream-analyzer-chatbot/app/services/analysis_context.py'
        spec = importlib.util.spec_from_file_location('analysis_context', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        meta = {'claims': [{'kind': 'observation', 'text': 'Bridge appears.', 'evidence_ids': ['entry-1']}], 'evidence': [{'evidence_id': 'entry-1', 'kind': 'user_entry', 'entry_id': 'synthetic-record', 'start_char': 0, 'end_char': 6}]}
        text = module.format_grounded_text('Bridge appears.', meta)
        self.assertIn('Bridge appears. [1]', text)
        self.assertIn('[1] Saved record synthetic-record', text)
        self.assertIn('Confidence unavailable', text)
        with self.assertRaises(ValueError): module.format_grounded_text('Uninventoried assertion.', meta)

    def test_real_agent_single_tool_uses_the_current_request_client(self):
        path = Path(__file__).resolve().parents[1] / 'dream-analyzer-chatbot/app/agents/dream_agent.py'
        tree = ast.parse(path.read_text())
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == 'app.services.analysis_context' and any(alias.name == 'run_analysis' for alias in node.names) for node in tree.body), 'production adapter import is missing')
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DreamAgent')
        method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == '_execute_tool')
        cls.body = [method]
        tree.body = [cls]
        calls = []
        async def run_analysis(client, operation, **kwargs):
            calls.append((client, operation, kwargs))
            return {'interpretation': 'Synthetic reflection.', 'calibration_status': 'unavailable'}
        namespace = {'Dict': dict, 'Any': object, 'run_analysis': run_analysis}
        exec(compile(tree, str(path), 'exec'), namespace)
        agent = namespace['DreamAgent']()
        agent.api_client = object()
        response = asyncio.run(agent._execute_tool('analyze_dream', {'content': 'Synthetic bridge'}))
        self.assertTrue(response['success'])
        self.assertEqual(calls[0][:2], (agent.api_client, 'single'))
        self.assertEqual(response['analysis']['calibration_status'], 'unavailable')

    def test_every_analysis_fetches_fresh_settings_from_its_request_client(self):
        path = Path(__file__).resolve().parents[1] / 'dream-analyzer-chatbot/app/services/analysis_context.py'
        spec = importlib.util.spec_from_file_location('analysis_context', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Client:
            def __init__(self, persona):
                self.persona, self.fetches, self.calls = persona, 0, []
            async def get_settings(self):
                self.fetches += 1
                return {'doctorPersonality': self.persona, 'doctor_influence': 0}
            async def analyze_dream(self, content, settings):
                self.calls.append(settings)
                return content
            async def get_cumulative_analysis_workflow(self, entries, settings):
                self.calls.append(settings)
                return entries
            async def ask_custom_question_workflow(self, question, entries, settings):
                self.calls.append(settings)
                return question
        a, b = Client('Academic'), Client('Artistic')
        for operation in ('single', 'cumulative', 'question'):
            asyncio.run(module.run_analysis(a, operation, content='Synthetic bridge', question='Where?', entries=[]))
        asyncio.run(module.run_analysis(b, 'single', content='Synthetic tram'))
        self.assertEqual(a.fetches, 3)
        self.assertEqual(a.calls[0]['doctor_influence'], 0)
        self.assertEqual(b.calls[0]['doctorPersonality'], 'Artistic')
        self.assertEqual(a.calls[-1]['doctorPersonality'], 'Academic')
        with self.assertRaises(ValueError):
            asyncio.run(module.run_analysis(a, 'unknown'))


if __name__ == '__main__':
    unittest.main()
