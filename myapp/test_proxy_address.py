from django.test import TestCase, RequestFactory, override_settings
from myapp.security.client_address import login_address
from myapp.security.throttle import reserve, finish

class ProxyAddressTests(TestCase):
    @override_settings(DREAM_LOGIN_PROXY_CIDRS=['127.0.0.0/8'], DREAM_ACCEPT_VERIFIED_CLIENT_IP=True)
    def test_only_trusted_peer_can_attest_address(self):
        factory=RequestFactory()
        direct=factory.post('/',REMOTE_ADDR='192.0.2.5',HTTP_X_DREAM_CLIENT_IP='198.51.100.6',HTTP_X_FORWARDED_FOR='198.51.100.7')
        self.assertEqual(login_address(direct),'192.0.2.5')
        trusted=factory.post('/',REMOTE_ADDR='127.0.0.1',HTTP_X_DREAM_CLIENT_IP='198.51.100.6')
        self.assertEqual(login_address(trusted),'198.51.100.6')

    @override_settings(DREAM_LOGIN_PROXY_CIDRS=['127.0.0.0/8'], DREAM_ACCEPT_VERIFIED_CLIENT_IP=False)
    def test_shared_proxy_does_not_share_ip_cap_between_accounts(self):
        request=RequestFactory().post('/',REMOTE_ADDR='127.0.0.1',HTTP_X_DREAM_CLIENT_IP='198.51.100.6')
        address=login_address(request);self.assertIsNone(address)
        for number in range(40): self.assertTrue(finish(reserve('account-'+str(number),address),success=False))
        self.assertTrue(finish(reserve('legitimate',address),success=True))
