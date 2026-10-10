from datetime import timedelta
from uuid import UUID, uuid5
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.test import TestCase
from django.utils import timezone

from myapp.models import Analysis, JournalEntry, User
from myapp.longitudinal import core
from .security_double import MAPPINGS, OwnerContext

try:
    service = importlib.import_module("myapp.longitudinal.services")
except ModuleNotFoundError:
    service = None


class SyntheticCase(TestCase):
    def setUp(self):
        self.assertIsNotNone(service, "Longitudinal services are not implemented")
        contract_patch = patch.dict(sys.modules, {"myapp.security.contracts": SimpleNamespace(OwnerContext=OwnerContext)})
        contract_patch.start()
        self.addCleanup(contract_patch.stop)
        self.user = User.objects.create(email="synthetic-a@example.invalid")
        auth = get_user_model().objects.create_user(username="synthetic-a")
        MAPPINGS[self.user.pk] = auth.pk
        self.owner = OwnerContext(self.user.pk, auth.pk, False, timezone.now() + timedelta(hours=1))
        self.other = User.objects.create(email="synthetic-b@example.invalid")
        other_auth = get_user_model().objects.create_user(username="synthetic-b")
        MAPPINGS[self.other.pk] = other_auth.pk
        self.other_owner = OwnerContext(self.other.pk, other_auth.pk, False, timezone.now() + timedelta(hours=1))

    def tearDown(self):
        MAPPINGS.clear()

    def entry(self, content, owner=None):
        return JournalEntry.objects.create(user=owner or self.user, content=content)

    def include(self, entry):
        service.set_entry_preferences(self.owner, entry.pk, included=True, origin="user_reported", language="en")

    def enabled(self):
        service.set_preferences(self.owner, enabled=True)


class ServiceTests(SyntheticCase):

    def test_old_reflection_is_reachable_and_deletable_through_bounded_pages(self):
        from myapp.longitudinal_models import DreamReflection
        first = service.create_reflection(self.owner, 'Oldest synthetic note')
        for index in range(100):
            service.create_reflection(self.owner, f'Synthetic note {index}')
        result = service.list_reflections(self.owner)
        self.assertEqual(len(result['reflections']), 100)
        older = service.list_reflections(self.owner, offset=result['next_offset'])
        self.assertEqual(older['reflections'][0]['id'], str(first.pk))
        self.assertEqual(older['revision'], result['revision'])
        newest = DreamReflection.objects.get(pk=result['reflections'][0]['id'])
        service.delete_reflection(self.owner, newest.pk, expected_revision=newest.revision)
        changed = service.list_reflections(self.owner, offset=result['next_offset'])
        self.assertNotEqual(changed['revision'], result['revision'])
        service.delete_reflection(self.owner, first.pk, expected_revision=first.revision)
        self.assertFalse(DreamReflection.objects.filter(pk=first.pk).exists())

    def test_private_rejected_decisions_remain_reachable_beyond_public_and_page_limits(self):
        from myapp.longitudinal_models import ThemeRecord, ThemeReview
        self.enabled()
        for key in core.CODEBOOK:
            theme = ThemeRecord.objects.create(user=self.user, kind='topic', key=key, namespace=core.CODEBOOK_VERSION)
            ThemeReview.objects.create(user=self.user, theme=theme, status='rejected')
        private = []
        for index in range(101):
            theme = ThemeRecord.objects.create(user=self.user, kind='topic', key=f'custom:{UUID(int=index + 1)}', namespace=core.CODEBOOK_VERSION)
            ThemeReview.objects.create(user=self.user, theme=theme, status='rejected', label=f'Private annotation {index}')
            private.append(theme)
        page = service.rejected_decisions(self.owner)
        self.assertEqual(len(page['decisions']), 100)
        self.assertTrue(all(item['key'].startswith('custom:') for item in page['decisions']))
        last = service.rejected_decisions(self.owner, offset=page['next_offset'])
        self.assertEqual(last['decisions'][0]['id'], str(private[-1].pk))
        self.assertIsNone(last['next_offset'])
        service.review_theme(self.owner, private[-1].pk, status='unreviewed', expected_revision=last['revision'])
        self.assertIsNone(service.rejected_decisions(self.owner)['next_offset'])

    def test_public_rejection_can_be_undone_after_private_source_reset(self):
        from uuid import uuid4
        self.enabled()
        a, b = self.entry('A river.'), self.entry('The river.')
        for entry in (a, b):
            self.include(entry)
        service.process_run(service.queue_audit(self.owner, [a.pk, b.pk], str(uuid4())).pk)
        theme = service.profile(self.owner)['themes'][0]
        service.review_theme(self.owner, theme['id'], status='rejected', label='Private old label')
        service.invalidate_source(self.owner, a.pk)
        self.include(a)
        service.process_run(service.queue_audit(self.owner, [a.pk, b.pk], str(uuid4())).pk)
        self.assertEqual(service.profile(self.owner)['themes'], [])
        service.undo_public_rejection(self.owner, theme['key'], expected_revision=service.preferences(self.owner)['revision'])
        result = service.profile(self.owner)['themes'][0]
        self.assertEqual(result['review'], 'unreviewed')
        self.assertNotEqual(result['label'], 'Private old label')

    def test_plain_confirmation_preserves_literal_provenance_and_rename_is_authored(self):
        from uuid import uuid4
        self.enabled()
        a, b = self.entry('A river.'), self.entry('The river.')
        for entry in (a, b):
            self.include(entry)
        service.process_run(service.queue_audit(self.owner, [a.pk, b.pk], str(uuid4())).pk)
        theme = service.profile(self.owner)['themes'][0]
        service.review_theme(self.owner, theme['id'], status='confirmed')
        self.assertEqual(service.profile(self.owner)['themes'][0]['provenance'], 'literal_content')
        service.review_theme(self.owner, theme['id'], status='confirmed', label='My travel concern')
        self.assertEqual(service.profile(self.owner)['themes'][0]['provenance'], 'user_interpretation')

    def test_public_rejection_undo_preserves_a_retained_user_label(self):
        from uuid import uuid4
        self.enabled()
        a, b = self.entry('A river.'), self.entry('The river.')
        for entry in (a, b):
            self.include(entry)
        service.process_run(service.queue_audit(self.owner, [a.pk, b.pk], str(uuid4())).pk)
        theme = service.profile(self.owner)['themes'][0]
        service.review_theme(self.owner, theme['id'], status='confirmed', label='My travel concern')
        service.review_theme(self.owner, theme['id'], status='rejected')
        service.undo_public_rejection(self.owner, theme['key'], expected_revision=service.preferences(self.owner)['revision'])
        result = service.profile(self.owner)['themes'][0]
        self.assertEqual((result['label'], result['review'], result['provenance']), ('My travel concern', 'unreviewed', 'user_interpretation'))

    def test_removed_literal_candidate_stays_removed_across_retries(self):
        from uuid import uuid4
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid4()))
        service.process_run(run.pk)
        evidence = service.entry_audit(self.owner, a.pk)
        service.delete_observation(self.owner, a.pk, evidence["observations"][0]["id"], expected_revision=evidence["revision"])
        self.assertEqual(service.profile(self.owner)["themes"], [])
        for _ in range(2):
            retry = service.queue_audit(self.owner, [a.pk], str(uuid4()))
            service.process_run(retry.pk)
            service.process_run(retry.pk)
            self.assertEqual(service.entry_audit(self.owner, a.pk)["observations"], [])
            self.assertEqual(service.profile(self.owner)["themes"], [])

    def test_disabled_and_unknown_sources_are_private_empty_and_no_implicit_work(self):
        entry = self.entry("A river.")
        self.assertEqual(service.profile(self.owner)["status"], "disabled")
        self.enabled()
        self.assertEqual(service.preview(self.owner, [entry.pk])["new_count"], 0)
        self.assertEqual(service.profile(self.owner)["status"], "empty")

    def test_owned_lookup_fails_before_any_cross_owner_mutation(self):
        entry = self.entry("Other person's river.", self.other)
        with self.assertRaises(JournalEntry.DoesNotExist):
            self.include(entry)
        self.assertEqual(service.profile(self.other_owner)["status"], "disabled")
        forged = OwnerContext(self.other.pk, self.owner.auth_user_id, False, self.owner.login_deadline)
        with self.assertRaises(PermissionDenied):
            service.set_preferences(forged, enabled=True)

    def test_bounded_local_run_is_idempotent_and_different_key_reuses_extraction(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        first = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "synthetic-key")))
        self.assertEqual(first.pk, service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "synthetic-key"))).pk)
        service.process_run(first.pk)
        result = service.profile(self.owner)
        self.assertEqual(result["themes"][0]["support"], 2)
        second = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "another-key")))
        service.process_run(second.pk)
        from myapp.longitudinal_models import EntryAudit
        self.assertEqual(EntryAudit.objects.count(), 2)
        with self.assertRaises(service.Conflict):
            service.queue_audit(self.owner, [a.pk], str(uuid5(UUID(int=0), "synthetic-key")))

    def test_privacy_reset_preserves_raw_analysis_and_authored_global_reflection(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        analysis = Analysis.objects.create(entry=a, user=self.user, mood="neutral", summary="Synthetic",
            color="pink", interpretation="Synthetic hypothesis", subject="water")
        a.analysis = analysis
        a.save(update_fields=["analysis"])
        for entry in (a, b):
            self.include(entry)
        reflection = service.create_reflection(self.owner, "I connect this to a planned trip.", use_for_profile=True)
        linked = service.create_reflection(self.owner, "A source-specific thought.", entry_id=a.pk)
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        theme = service.profile(self.owner)["themes"][0]
        service.review_theme(self.owner, theme["id"], status="confirmed", label="Private synthetic name")
        self.assertEqual(service.source_reset_preview(self.owner)["private_reviews_reset"], 1)
        service.invalidate_source(self.owner, a.pk)
        from myapp.longitudinal_models import DreamReflection, EntryAudit, ThemeRecord, ProfileSnapshot
        self.assertTrue(JournalEntry.objects.filter(pk=a.pk).exists())
        self.assertTrue(Analysis.objects.filter(pk=analysis.pk).exists())
        self.assertTrue(DreamReflection.objects.filter(pk=reflection.pk).exists())
        self.assertFalse(DreamReflection.objects.filter(pk=linked.pk).exists())
        self.assertEqual(ThemeRecord.objects.count(), 0)
        self.assertEqual(ProfileSnapshot.objects.count(), 0)
        self.assertFalse(EntryAudit.objects.filter(entry=a).exists())
        self.assertTrue(EntryAudit.objects.filter(entry=b).exists())

    def test_review_reaggregation_preserves_extraction_and_rejection(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        theme = service.profile(self.owner)["themes"][0]
        service.review_theme(self.owner, theme["id"], status="rejected")
        self.assertEqual(service.profile(self.owner)["themes"], [])
        from myapp.longitudinal_models import EntryAudit
        self.assertEqual(EntryAudit.objects.filter(status="complete").count(), 2)
        service.invalidate_source(self.owner, a.pk)
        self.assertEqual(service.preferences(self.owner)["suppressed_public_keys"], ["topic:water"])

    def test_late_run_cannot_republish_deleted_source_and_context_is_disabled(self):
        self.enabled()
        a = self.entry("A river.")
        self.include(a)
        run = service.queue_audit(self.owner, [a.pk], str(uuid5(UUID(int=0), "initial")))
        service.invalidate_source(self.owner, a.pk)
        service.process_run(run.pk)
        self.assertEqual(service.profile(self.owner)["status"], "empty")
        self.assertEqual(service.confirmed_context(self.owner)["items"], [])

    def test_missing_security_contract_fails_closed(self):
        with self.settings(LONGITUDINAL_SECURITY_MODULE=""):
            with self.assertRaises(service.ConfigurationRequired):
                service.set_preferences(self.owner, enabled=True)

    def test_rollback_restores_privacy_state_and_evidence(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        reflection = service.create_reflection(self.owner, "An independent global thought.")
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        theme = service.profile(self.owner)["themes"][0]
        service.review_theme(self.owner, theme["id"], status="confirmed", label="Private synthetic rename")
        service.profile(self.owner)
        from myapp import longitudinal_models as models
        models.ObservationSuppression.objects.create(user=self.user, entry=a, source_hash=core.raw_identity(a.content), key='topic:water', start=2, end=7)
        merge = models.ThemeMerge.objects.create(user=self.user, label="An explicit synthetic merge")
        models.ThemeMergeMember.objects.create(user=self.user, merge=merge, theme_id=theme["id"])
        tables = (models.EntryAudit, models.DreamObservation, models.ObservationSuppression, models.ProfileSnapshot, models.SnapshotSource,
                  models.ThemeRecord, models.ThemeReview, models.ThemeMerge, models.ThemeMergeMember,
                  models.AuditRun, models.AuditRunEntry, models.EntryAuditPreference, models.DreamReflection)
        contents = {model.__name__: list(model.objects.order_by("id").values()) for model in tables}
        before = service.preferences(self.owner)
        actual = service._invalidate_locked
        def fail_after_cleanup(*args, **kwargs):
            actual(*args, **kwargs)
            raise RuntimeError("synthetic failure after destructive work")
        with patch("myapp.longitudinal.services._invalidate_locked", side_effect=fail_after_cleanup):
            with self.assertRaises(RuntimeError):
                service.invalidate_source(self.owner, a.pk)
        self.assertEqual(service.preferences(self.owner), before)
        self.assertEqual(contents, {model.__name__: list(model.objects.order_by("id").values()) for model in tables})
        self.assertTrue(JournalEntry.objects.filter(pk=a.pk).exists())

    def test_deleted_last_sources_and_edits_without_snapshots_scrub_private_reviews(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        theme = service.profile(self.owner)["themes"][0]
        service.review_theme(self.owner, theme["id"], status="confirmed", label="Private source name")
        from myapp.longitudinal_models import ThemeReview, EntryAudit, ProfileSnapshot
        self.assertEqual(ProfileSnapshot.objects.count(), 0)
        JournalEntry.objects.filter(pk=a.pk).update(content="An unhooked correction.")
        self.assertEqual(service.profile(self.owner)["status"], "stale")
        self.assertEqual(ThemeReview.objects.count(), 0)
        self.assertFalse(EntryAudit.objects.filter(entry=a).exists())
        service.profile(self.owner)
        b.delete()
        self.assertEqual(service.profile(self.owner)["status"], "stale")
        self.assertEqual(service.profile(self.owner)["status"], "empty")
        run.refresh_from_db()
        self.assertEqual((run.manifest_digest, run.request_fingerprint), ("", ""))

    def test_manual_then_local_audit_preserves_bounded_authored_evidence(self):
        self.enabled()
        entry = self.entry("A river beside a school.")
        self.include(entry)
        for index in range(20):
            service.add_observation(self.owner, entry.pk, {"kind": "topic", "key": "custom:" + str(UUID(int=index + 1)),
                "start": 2, "end": 7, "quote": "river", "polarity": "affirmed", "experiencer": "unknown"})
        run = service.queue_audit(self.owner, [entry.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        from myapp.longitudinal_models import DreamObservation, EntryAudit
        self.assertEqual(DreamObservation.objects.count(), 20)
        self.assertEqual(EntryAudit.objects.get(entry=entry).status, "partial")
        self.assertEqual(DreamObservation.objects.filter(origin="user_authored").count(), 20)

    def test_reflection_edit_delete_and_manual_annotation_keep_provenance(self):
        self.enabled()
        entry = self.entry("I met Mira beside a river.")
        self.include(entry)
        reflection = service.create_reflection(self.owner, "A thought.")
        updated = service.edit_reflection(self.owner, reflection.pk, text="My corrected thought.", expected_revision=1)
        self.assertEqual((updated.text, updated.revision), ("My corrected thought.", 2))
        with self.assertRaises(service.Conflict):
            service.edit_reflection(self.owner, reflection.pk, text="Stale edit", expected_revision=1)
        observation = service.add_observation(self.owner, entry.pk, {
            "kind": "entity", "key": "entity:mira", "start": 6, "end": 10,
            "quote": "Mira", "polarity": "affirmed", "experiencer": "unknown"})
        self.assertEqual(observation.origin, "user_authored")
        service.delete_reflection(self.owner, reflection.pk, expected_revision=2)
        from myapp.longitudinal_models import DreamReflection
        self.assertFalse(DreamReflection.objects.filter(pk=reflection.pk).exists())
        with self.assertRaises(DreamReflection.DoesNotExist):
            service.edit_reflection(self.other_owner, reflection.pk, text="Stolen edit", expected_revision=2)

    def test_source_edit_and_generated_origin_are_one_atomic_write(self):
        self.enabled()
        entry = self.entry("A river.")
        self.include(entry)
        service.save_source(self.owner, entry.pk, content="A generated lake.",
            audit_preferences={"origin": "synthetic", "included": False})
        entry.refresh_from_db()
        self.assertEqual(entry.content, "A generated lake.")
        from myapp.longitudinal_models import EntryAuditPreference
        pref = EntryAuditPreference.objects.get(entry=entry)
        self.assertEqual((pref.origin, pref.included), ("synthetic", False))
        with self.assertRaises(core.InvalidEvidence):
            service.save_source(self.owner, entry.pk, content="Must roll back", audit_preferences={"origin": "synthetic", "included": True})
        entry.refresh_from_db()
        self.assertEqual(entry.content, "A generated lake.")

    def test_duplicate_preview_and_live_manifest_guard(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("A river.")
        for entry in (a, b):
            self.include(entry)
        preview = service.preview(self.owner, [a.pk, b.pk])
        self.assertEqual(preview["new_count"], 1)
        self.assertEqual(sum(s["skip"] == "duplicate_occurrence" for s in preview["sources"]), 1)
        run = service.queue_audit(self.owner, [a.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        # Deliberate bypass reproduces a missing integration hook. Reader must fail closed and scrub.
        JournalEntry.objects.filter(pk=a.pk).update(content="A private revised name by a lake.")
        self.assertEqual(service.profile(self.owner)["status"], "stale")
        from myapp.longitudinal_models import EntryAudit, EntryAuditPreference
        self.assertFalse(EntryAudit.objects.filter(entry=a).exists())
        self.assertFalse(EntryAuditPreference.objects.get(entry=a).included)

    def test_source_preferences_preserve_explicit_language_and_empty_patch_is_noop(self):
        self.enabled()
        a = self.entry("Vi un río.")
        service.set_entry_preferences(self.owner, a.pk, included=True, origin="user_reported", language="unsupported")
        before = service.preferences(self.owner)
        service.set_entry_preferences(self.owner, a.pk)
        self.assertEqual(service.preferences(self.owner), before)
        service.set_entry_preferences(self.owner, a.pk, independent_occurrence=True)
        from myapp.longitudinal_models import EntryAuditPreference
        self.assertEqual(EntryAuditPreference.objects.get(entry=a).language, "unsupported")

    def test_excluded_content_never_contributes_to_new_manifest_or_snapshot_lineage(self):
        self.enabled()
        excluded = self.entry("A private excluded name beside water.")
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        digest = service._manifest(self.owner)
        JournalEntry.objects.filter(pk=excluded.pk).update(content="Changed private excluded content.")
        self.assertEqual(service._manifest(self.owner), digest)
        from myapp.longitudinal_models import SnapshotSource
        self.assertFalse(SnapshotSource.objects.filter(audit__entry=excluded).exists())

    def test_worker_manifest_mismatch_scrubs_old_audit_hashes_and_run_links(self):
        self.enabled()
        entry = self.entry("A river.")
        self.include(entry)
        run = service.queue_audit(self.owner, [entry.pk], str(uuid5(UUID(int=0), "initial")))
        JournalEntry.objects.filter(pk=entry.pk).update(content="A corrected private name.")
        service.process_run(run.pk)
        from myapp.longitudinal_models import EntryAudit, AuditRunEntry
        self.assertEqual(EntryAudit.objects.count(), 0)
        self.assertEqual(AuditRunEntry.objects.count(), 0)
        run.refresh_from_db()
        self.assertEqual((run.manifest_digest, run.request_fingerprint), ("", ""))

    def test_locked_profile_rechecks_withdrawal_before_materialization(self):
        self.enabled()
        entry = self.entry("A river.")
        self.include(entry)
        original, raced = service.owner_write, False
        def withdraw_first(owner, write):
            nonlocal raced
            if not raced:
                raced = True
                service.set_preferences(owner, enabled=False)
            return original(owner, write)
        with patch("myapp.longitudinal.services.owner_write", side_effect=withdraw_first):
            self.assertEqual(service.profile(self.owner)["status"], "disabled")
        from myapp.longitudinal_models import ProfileSnapshot
        self.assertEqual(ProfileSnapshot.objects.count(), 0)

    def test_noop_save_preserves_private_review_and_review_only_run_changes_terminate(self):
        self.enabled()
        a, b = self.entry("A river."), self.entry("The river.")
        for entry in (a, b):
            self.include(entry)
        run = service.queue_audit(self.owner, [a.pk, b.pk], str(uuid5(UUID(int=0), "initial")))
        service.process_run(run.pk)
        theme = service.profile(self.owner)["themes"][0]
        service.review_theme(self.owner, theme["id"], status="confirmed", label="A user label")
        service.save_source(self.owner, a.pk, content=a.content)
        self.assertEqual(service.profile(self.owner)["themes"][0]["label"], "A user label")
        queued = service.queue_audit(self.owner, [a.pk], str(uuid5(UUID(int=0), "later")))
        service.create_reflection(self.owner, "A later thought.")
        service.process_run(queued.pk)
        queued.refresh_from_db()
        self.assertEqual(queued.status, "stale")
        self.assertEqual(service.profile(self.owner)["themes"][0]["label"], "A user label")
