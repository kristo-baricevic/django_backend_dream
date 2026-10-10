"""Security regressions use synthetic text and the disposable test database."""
from django.test import Client, TestCase, override_settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from myapp.models import JournalEntry, User, WorkflowExecution, Analysis, CumulativeAnalysis, CustomQuestion
from datetime import datetime, timezone
from unittest.mock import patch


class AnonymousIsolationTests(TestCase):
    def test_anonymous_journal_list_cannot_read_unassigned_records(self):
        JournalEntry.objects.create(content='synthetic private record')
        response = Client(enforce_csrf_checks=True).get('/api/entries/')
        self.assertIn(response.status_code, (401, 403))
        self.assertNotIn(b'synthetic private record', response.content)

    def test_anonymous_object_lookup_cannot_read_unassigned_record(self):
        entry = JournalEntry.objects.create(content='synthetic object record')
        response = Client(enforce_csrf_checks=True).get(f'/api/entries/{entry.id}/')
        self.assertIn(response.status_code, (401, 403))
        self.assertNotIn(b'synthetic object record', response.content)


class AccountIsolationTests(TestCase):
    def make_account(self, name):
        auth = get_user_model().objects.create_user(name, password='synthetic-password')
        owner = User.objects.create(auth_user=auth, email=name+'@example.invalid')
        client = Client(enforce_csrf_checks=True)
        csrf = client.get('/api/auth/csrf/').json()['csrfToken']
        response = client.post('/api/auth/login/', {'username':name,'password':'synthetic-password'},
            content_type='application/json', HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(response.status_code, 200)
        return auth, owner, client, response.json()['csrfToken']

    def test_django_auth_id_is_not_used_as_domain_owner(self):
        auth = get_user_model().objects.create_user('alice', password='synthetic-password')
        self.assertIn('auth_user', [f.name for f in User._meta.fields])
        owner = User.objects.create(auth_user=auth, email='alice@example.invalid')
        other = User.objects.create(email='unassigned@example.invalid')
        mine = JournalEntry.objects.create(user=owner, content='synthetic mine')
        foreign = JournalEntry.objects.create(user=other, content='synthetic foreign')
        client = Client(enforce_csrf_checks=True)
        csrf = client.get('/api/auth/csrf/').json()['csrfToken']
        login = client.post('/api/auth/login/', {'username':'alice','password':'synthetic-password'},
                            content_type='application/json', HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()['user']['id'], str(owner.id))
        records = client.get('/api/entries/').json()['results']
        self.assertEqual([x['id'] for x in records], [str(mine.id)])
        self.assertEqual(client.get(f'/api/entries/{foreign.id}/').status_code, 404)

    def test_login_requires_csrf_before_authentication(self):
        response = Client(enforce_csrf_checks=True).post('/api/auth/login/',
            {'username':'alice','password':'synthetic-password'}, content_type='application/json')
        self.assertEqual(response.status_code, 403)

    def test_session_expiry_and_logout_close_private_routes(self):
        auth, owner, client, csrf = self.make_account('expiry')
        self.assertEqual(client.get('/api/auth/session/').status_code, 200)
        session = client.session
        session['dream_login_deadline'] = datetime.now(timezone.utc).timestamp() - 1
        session.save()
        self.assertEqual(client.get('/api/entries/').status_code, 401)
        self.assertEqual(client.post('/api/auth/logout/', HTTP_X_CSRFTOKEN=csrf).status_code, 200)
        self.assertEqual(client.get('/api/auth/session/').status_code, 401)

    def test_deactivation_revokes_existing_session(self):
        auth, owner, client, csrf = self.make_account('deactivate')
        auth.is_active = False
        auth.save(update_fields=['is_active'])
        self.assertEqual(client.get('/api/entries/').status_code, 401)

    def test_unlinked_account_has_no_legacy_data_fallback(self):
        get_user_model().objects.create_user('unlinked', password='synthetic-password')
        client = Client(enforce_csrf_checks=True)
        csrf = client.get('/api/auth/csrf/').json()['csrfToken']
        response = client.post('/api/auth/login/', {'username':'unlinked','password':'synthetic-password'},
            content_type='application/json', HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(response.status_code, 401)

    def test_forged_owner_is_ignored_when_creating_entry(self):
        auth, owner, client, csrf = self.make_account('creator')
        other = User.objects.create(email='other-owner@example.invalid')
        response = client.post('/api/entries/create/', {'content':'synthetic new','user':str(other.id)},
            content_type='application/json', HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(JournalEntry.objects.get(pk=response.json()['id']).user_id, owner.id)

    def test_all_private_routes_reject_anonymous_access(self):
        for path in ['data/','entries/','symbols/','entries/moods/','cumulative-analyses/',
                     'custom-questions/','settings/','feedback/stats/','feedback/preferences/',
                     'feedback/recommendations/','workflows/00000000-0000-0000-0000-000000000001/']:
            with self.subTest(path=path):
                self.assertEqual(Client(enforce_csrf_checks=True).get('/api/'+path).status_code,401)

    def test_local_fixture_is_rejected_outside_local_mode(self):
        auth, owner, client, csrf = self.make_account('fixture')
        owner.is_local_fixture = True
        owner.save(update_fields=['is_local_fixture'])
        self.assertEqual(client.get('/api/entries/').status_code,401)

    def test_staff_domain_admin_does_not_show_foreign_profile(self):
        auth, owner, client, csrf = self.make_account('staff')
        auth.is_staff = True
        auth.save(update_fields=['is_staff'])
        auth.user_permissions.add(Permission.objects.get(codename='view_user',content_type__app_label='myapp'))
        other = User.objects.create(email='foreign-staff-visible@example.invalid')
        response = client.get('/admin/myapp/user/')
        self.assertEqual(response.status_code,200)
        self.assertNotIn(b'foreign-staff-visible@example.invalid',response.content)

    def test_authenticator_error_does_not_hold_future_login_capacity(self):
        auth, owner, client, csrf = self.make_account('recovery')
        client.post('/api/auth/logout/',HTTP_X_CSRFTOKEN=csrf)
        client.raise_request_exception=False
        payload={'username':'recovery','password':'synthetic-password'}
        with patch('myapp.auth_views.authenticate',side_effect=RuntimeError('synthetic failure')):
            for _ in range(10):
                response=client.post('/api/auth/login/',payload,content_type='application/json',HTTP_X_CSRFTOKEN=csrf)
                self.assertEqual(response.status_code,503)
        response=client.post('/api/auth/login/',payload,content_type='application/json',HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(response.status_code,200)

    def test_expired_staff_session_can_reach_login_and_logout(self):
        auth, owner, client, csrf = self.make_account('expiredstaff')
        auth.is_staff=True;auth.save(update_fields=['is_staff'])
        session=client.session;session['dream_login_deadline']=1;session.save()
        response=client.get('/admin/login/')
        self.assertEqual(response.status_code,200)
        self.assertEqual(client.get('/api/auth/session/').status_code,401)
        self.assertIn(client.post('/admin/logout/',HTTP_X_CSRFTOKEN=csrf).status_code,(200,302))
