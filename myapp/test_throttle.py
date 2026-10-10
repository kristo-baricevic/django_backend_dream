from datetime import timedelta
from unittest.mock import patch
from django.test import TestCase
from django.utils import timezone
from django.contrib.auth import get_user_model
from myapp.models import User
from myapp.security.throttle import reserve, finish, LoginLimited


class ThrottleTests(TestCase):
    def test_pair_limit_uses_server_address_and_success_recovers(self):
        for _ in range(5): finish(reserve('synthetic', '127.0.0.1'), success=False)
        with self.assertRaises(LoginLimited): reserve('synthetic', '127.0.0.1')
        other = reserve('other', '127.0.0.1'); self.assertTrue(finish(other, success=True))

    def test_distributed_failures_trigger_bounded_account_cooldown(self):
        for number in range(10): finish(reserve('synthetic', '10.0.0.'+str(number)), success=False)
        with self.assertRaises(LoginLimited): reserve('synthetic', '10.0.1.1')
        with patch('myapp.security.throttle.timezone.now', return_value=timezone.now()+timedelta(seconds=31)):
            self.assertTrue(finish(reserve('synthetic', '10.0.1.1'), success=True))

    def test_abandoned_leases_and_late_settlement_recover_within_30_seconds(self):
        start = timezone.now()
        attempts = [reserve('synthetic', '10.0.0.'+str(number)) for number in range(10)]
        with self.assertRaises(LoginLimited): reserve('synthetic', '10.0.1.1')
        with patch('myapp.security.throttle.timezone.now', return_value=start+timedelta(seconds=31)):
            self.assertFalse(finish(attempts[0], success=True))
            self.assertTrue(finish(reserve('synthetic', '10.0.1.1'), success=True))

    def test_admin_does_not_keep_session_after_late_reservation(self):
        auth = get_user_model().objects.create_user('slowstaff', password='synthetic-password', is_staff=True)
        User.objects.create(auth_user=auth, email='slowstaff@example.invalid')
        self.client.get('/admin/login/')
        start = timezone.now()
        calls = iter([start, start+timedelta(seconds=31)])
        with patch('myapp.security.throttle._now', side_effect=lambda: next(calls)):
            response = self.client.post('/admin/login/', {'username': 'slowstaff', 'password': 'synthetic-password'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get('/api/auth/session/').status_code, 401)
