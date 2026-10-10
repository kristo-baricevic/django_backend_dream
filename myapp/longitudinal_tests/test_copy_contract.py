from django.test import TestCase
from django.utils import timezone
from django.core.serializers.json import DjangoJSONEncoder
from hashlib import sha256
import json
from uuid import uuid4
from myapp.longitudinal.copy_contract import COPY_POLICY, inspect_private_inventory, selected_reflection_plan
from myapp.longitudinal.core import InvalidEvidence


class CopyContractTests(TestCase):
    def test_all_private_models_are_inventoried_and_unknown_namespaces_stop_copy(self):
        self.assertEqual(len(COPY_POLICY), 14)
        result = inspect_private_inventory(list(COPY_POLICY), [])
        self.assertEqual(result['indexes'], [])
        with self.assertRaises(InvalidEvidence):
            inspect_private_inventory(['myapp.UnknownProfile'], [])
        with self.assertRaises(InvalidEvidence):
            inspect_private_inventory([], ['shared-faiss-private-dreams'])

    def test_reflection_copy_requires_selection_owner_and_source_expansion(self):
        source, target, id, entry, run, mapped_entry = [uuid4() for _ in range(6)]
        row = {'id': str(id), 'user_id': str(source), 'entry_id': str(entry), 'text': 'Synthetic authored note', 'use_for_profile': True, 'use_for_chat': True, 'created_at': timezone.now(), 'updated_at': timezone.now()}
        options = dict(selected_ids=[id], source_owner_id=source, target_owner_id=target, copy_run_id=run, entry_map={})
        with self.assertRaises(InvalidEvidence):
            selected_reflection_plan([row], **options)
        options['entry_map'] = {str(entry): str(mapped_entry)}
        first = selected_reflection_plan([row], **options)
        self.assertEqual(first, selected_reflection_plan([row], **options))
        result = first[0]['destination']
        self.assertNotEqual(result['id'], str(id))
        self.assertEqual(result['user_id'], str(target))
        self.assertEqual(result['entry_id'], str(mapped_entry))
        self.assertFalse(result['use_for_profile'] or result['use_for_chat'])
        self.assertEqual(selected_reflection_plan([row], **(options | {'selected_ids': []})), [])
        with self.assertRaises(InvalidEvidence):
            selected_reflection_plan([row | {'user_id': str(target)}], **options)

    def test_real_timestamp_row_plan_matches_auth_ledger_persisted_digest(self):
        from myapp.models import User
        from myapp.longitudinal_models import DreamReflection
        source = User.objects.create(email='copy-source@example.invalid')
        target = User.objects.create(email='copy-target@example.invalid')
        original = DreamReflection.objects.create(user=source, text='Synthetic explicit global reflection', use_for_profile=True)
        row = DreamReflection.objects.filter(pk=original.pk).values().get()
        plan = selected_reflection_plan([row], selected_ids=[original.pk], source_owner_id=source.pk, target_owner_id=target.pk, copy_run_id=uuid4(), entry_map={})[0]
        destination = plan['destination']
        copied = DreamReflection.objects.create(**destination)
        DreamReflection.objects.filter(pk=copied.pk).update(created_at=destination['created_at'], updated_at=destination['updated_at'])
        stored = DreamReflection.objects.filter(pk=copied.pk).values().get()
        digest = sha256(json.dumps(stored, sort_keys=True, separators=(',', ':'), cls=DjangoJSONEncoder).encode()).hexdigest()
        self.assertEqual(digest, plan['digest'])
