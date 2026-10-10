from uuid import uuid4
from rest_framework.test import APIClient
from .test_services import SyntheticCase


class ApiTests(SyntheticCase):
    def test_authentication_body_owner_and_revision_boundaries(self):
        client = APIClient()
        self.assertIn(client.get("/longitudinal/preferences/").status_code, (401, 403))
        from django.contrib.auth import get_user_model
        client.force_authenticate(get_user_model().objects.get(pk=self.owner.auth_user_id))
        client.credentials(HTTP_X_DREAM_PRINCIPAL=str(self.user.pk))
        response = client.get("/longitudinal/preferences/")
        self.assertEqual(response.status_code, 200)
        revision = response.json()["revision"]
        bad = client.patch("/longitudinal/preferences/", {"enabled": True, "user_id": str(self.other.pk), "expected_revision": revision}, format="json")
        self.assertEqual(bad.status_code, 400)
        good = client.patch("/longitudinal/preferences/", {"enabled": True, "expected_revision": revision}, format="json")
        self.assertEqual(good.status_code, 200)
        stale = client.patch("/longitudinal/preferences/", {"enabled": False, "expected_revision": revision}, format="json")
        self.assertEqual(stale.status_code, 409)
        self.assertIn("no-store", good["Cache-Control"])
        self.assertEqual(client.get("/longitudinal/context/").json()["status"], "disabled")

    def test_owned_source_api_never_exposes_other_owner(self):
        client = APIClient()
        from django.contrib.auth import get_user_model
        client.force_authenticate(get_user_model().objects.get(pk=self.owner.auth_user_id))
        client.credentials(HTTP_X_DREAM_PRINCIPAL=str(self.user.pk))
        other = self.entry("Private synthetic record.", self.other)
        self.assertEqual(client.get(f"/entries/{other.pk}/audit/").status_code, 404)
        self.assertEqual(client.get(f"/longitudinal/audits/{uuid4()}/").status_code, 404)

    def test_account_intent_fence_rejects_changed_account_without_writing(self):
        from django.contrib.auth import get_user_model
        from myapp.longitudinal_models import DreamReflection
        client = APIClient()
        client.force_authenticate(get_user_model().objects.get(pk=self.other_owner.auth_user_id))
        client.credentials(HTTP_X_DREAM_PRINCIPAL=str(self.user.pk))
        response = client.post('/longitudinal/reflections/', {'text': 'Synthetic account A draft'}, format='json')
        self.assertEqual(response.status_code, 403)
        self.assertFalse(DreamReflection.objects.exists())
        self.assertEqual(client.get('/longitudinal/profile/').status_code, 403)

    def test_malformed_inputs_are_generic_400_without_private_input_echo(self):
        from django.contrib.auth import get_user_model
        client = APIClient()
        client.force_authenticate(get_user_model().objects.get(pk=self.owner.auth_user_id))
        client.credentials(HTTP_X_DREAM_PRINCIPAL=str(self.user.pk))
        revision = client.get('/longitudinal/preferences/').json()['revision']
        phrase = 'Synthetic private concern that is not an ID'
        response = client.post('/longitudinal/reflections/', {'text': 'Synthetic', 'entry_id': phrase}, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(phrase, response.content.decode())
        entry = self.entry('A river.')
        for field in ('origin', 'language'):
            response = client.patch(f'/entries/{entry.pk}/audit-preferences/', {field: [], 'expected_revision': revision}, format='json')
            self.assertEqual(response.status_code, 400)
        response = client.post('/longitudinal/audits/preview/', {'entry_ids': [str(entry.pk)], 'mode': {}}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_authored_unicode_validation_and_server_entity_normalization(self):
        from django.contrib.auth import get_user_model
        from myapp.longitudinal import services
        client = APIClient()
        client.force_authenticate(get_user_model().objects.get(pk=self.owner.auth_user_id))
        client.credentials(HTTP_X_DREAM_PRINCIPAL=str(self.user.pk))
        import json
        for text in ('Synthetic\x00secret', 'Synthetic\ud800secret'):
            response = client.post('/longitudinal/reflections/', json.dumps({'text': text}, ensure_ascii=True), content_type='application/json')
            self.assertEqual(response.status_code, 400)
            self.assertNotIn('secret', response.content.decode())
        services.set_preferences(self.owner, enabled=True)
        for quote in ('Straße', '  Mira  Smith ', 'Σς', '😀Mira'):
            entry = self.entry('I saw ' + quote + '.')
            services.set_entry_preferences(self.owner, entry.pk, included=True, origin='user_reported', language='en')
            revision = services.preferences(self.owner)['revision']
            candidate = {'kind': 'entity', 'key': 'entity:auto', 'quote': quote, 'start': 6, 'end': 6 + len(quote), 'polarity': 'affirmed', 'experiencer': 'unknown'}
            response = client.post(f'/entries/{entry.pk}/observations/', {'observation': candidate, 'expected_revision': revision}, format='json')
            self.assertEqual(response.status_code, 201)
