from myapp import test_auth_security
from myapp.models import Analysis, JournalEntry, CumulativeAnalysis, CustomQuestion, WorkflowExecution, WorkflowStep, AnalysisFeedback
from django.test import TestCase
from django.utils import timezone
from myapp.security.contracts import OwnerContext
from myapp.security.analysis_input import analysis_input


class PrivateRouteTests(TestCase):
    make_account = test_auth_security.AccountIsolationTests.make_account
    def setUp(self):
        _, self.mine, self.client, self.csrf = self.make_account('routeowner')
        _, self.foreign, _, _ = self.make_account('routeforeign')
        self.entry = JournalEntry.objects.create(user=self.foreign, content='synthetic foreign text')
        self.analysis = Analysis.objects.create(user=self.foreign, entry=self.entry, mood='foreign-mood',
            color='foreign-color', summary='synthetic foreign summary', interpretation='synthetic foreign interpretation',
            subject='foreign-subject', symbols=['foreign-symbol'])
        self.entry.analysis = self.analysis; self.entry.save()
        self.cumulative = CumulativeAnalysis.objects.create(user=self.foreign, analysis='synthetic foreign cumulative')
        self.question = CustomQuestion.objects.create(user=self.foreign, question='synthetic foreign question', answer='foreign answer')
        self.workflow = WorkflowExecution.objects.create(user=self.foreign, routine_name='synthetic foreign workflow', workflow_type='test')
        AnalysisFeedback.objects.create(user=self.foreign, analysis=self.analysis, rating='bad', comment='synthetic private comment')

    def test_search_statistics_and_workflow_hide_foreign_records(self):
        for path in ['/api/entries/?entries=foreign', '/api/data/', '/api/symbols/', '/api/entries/moods/',
            '/api/cumulative-analyses/?search=foreign', '/api/custom-questions/?search=foreign', '/api/feedback/stats/']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn(b'foreign', response.content, path)
        self.assertEqual(self.client.get(f'/api/workflows/{self.workflow.pk}/').status_code,404)

    def test_foreign_mutations_and_feedback_are_rejected(self):
        self.assertEqual(self.client.patch(f'/api/entries/{self.entry.pk}/update/', {'content':'synthetic forged'},
            content_type='application/json', HTTP_X_CSRFTOKEN=self.csrf).status_code,404)
        self.assertEqual(self.client.delete(f'/api/entries/{self.entry.pk}/delete/',HTTP_X_CSRFTOKEN=self.csrf).status_code,404)
        for label,row in [('analysis',self.analysis),('cumulative',self.cumulative),('custom_question',self.question)]:
            response=self.client.post('/api/feedback/submit/',{'analysis_type':label,'analysis_id':str(row.pk),'rating':'good'},
                content_type='application/json',HTTP_X_CSRFTOKEN=self.csrf)
            self.assertEqual(response.status_code,404)
        self.entry.refresh_from_db();self.assertEqual(self.entry.content,'synthetic foreign text')

    def test_unsafe_private_requests_require_csrf_and_file_settings_are_closed(self):
        response=self.client.post('/api/entries/create/',{'content':'synthetic'},content_type='application/json')
        self.assertEqual(response.status_code,403)
        for settings in [{'doctorPersonality':'../../private'}, {'personality':'../../private'}, {'astrology':{'sun':'../../private'}}]:
            response=self.client.post('/api/settings/update/',settings,content_type='application/json',HTTP_X_CSRFTOKEN=self.csrf)
            self.assertEqual(response.status_code,400)

    def test_feedback_preferences_handle_all_targets_and_ignore_invalid_legacy_links(self):
        session = self.client.get('/api/auth/session/').json()
        owner = OwnerContext(self.mine.pk, self.mine.auth_user_id, False,
            timezone.datetime.fromisoformat(session['loginDeadline']), self.client.session.session_key)
        entry = JournalEntry.objects.create(user=self.mine, content='synthetic current owned input')
        _, _, capture = analysis_input(owner, [entry.pk], with_snapshot=True)
        analysis = Analysis.objects.create(user=self.mine, entry=entry, mood='joy', color='pink',
            summary='synthetic current summary', interpretation='synthetic current interpretation', subject='synthetic',
            doctor_personality='own-style', grounding_metadata={'source_snapshot': capture})
        entry.analysis = analysis; entry.save(update_fields=['analysis'])
        cumulative = CumulativeAnalysis.objects.create(user=self.mine, analysis='synthetic own cumulative',
            doctor_personality='own-style', weights={'dream': 0.7})
        question = CustomQuestion.objects.create(user=self.mine, question='synthetic own question',
            answer='synthetic own answer', doctor_personality='own-style', weights={'dream': 0.6})
        for target in (cumulative, question):
            workflow = WorkflowExecution.objects.create(user=self.mine, accepted_auth_user_id=self.mine.auth_user_id,
                status='completed', workflow_type='synthetic', routine_name='synthetic current workflow',
                source_metadata=capture, analysis_id=str(target.pk))
            WorkflowStep.objects.create(execution=workflow, step_number=1, name='synthetic final step',
                step_type='answer_generation', status='completed', output_data={'coverage': {'source_dependencies': capture['sources']}})
            target.workflow_execution = workflow; target.save(update_fields=['workflow_execution'])
        for relation, target in [('analysis', analysis), ('cumulative_analysis', cumulative), ('custom_question', question)]:
            AnalysisFeedback.objects.create(user=self.mine, **{relation: target}, rating='good')
        historical = CustomQuestion.objects.create(user=self.mine, is_historical=True,
            question='synthetic old question', answer='synthetic old answer', doctor_personality='historical-style')
        AnalysisFeedback.objects.create(user=self.mine, custom_question=historical, rating='good')
        AnalysisFeedback.objects.create(user=self.mine, rating='bad', comment='synthetic orphan')
        AnalysisFeedback.objects.create(user=self.mine, analysis=self.analysis, rating='bad', comment='synthetic invalid link')
        for path in ['/api/feedback/preferences/', '/api/feedback/recommendations/', '/api/feedback/stats/']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn(b'foreign', response.content, path)
        preferences = self.client.get('/api/feedback/preferences/').json()
        self.assertEqual(preferences['preferred_doctors'], {'own-style': {'liked': 3, 'disliked': 0}})
        self.assertEqual(self.client.get('/api/feedback/stats/').json()['total_feedback'], 3)
        self.assertEqual(self.client.get('/api/feedback/recommendations/').json()['suggested_doctor'], 'own-style')
