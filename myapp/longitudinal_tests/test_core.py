from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib
import unittest

from .fixtures import GOLD_CASES

try:
    core = importlib.import_module("myapp.longitudinal.core")
except ModuleNotFoundError:
    core = None

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, "The longitudinal core is not implemented")

    def source(self, id, content, days=1, **kwargs):
        return core.AuditSource(
            id=id, owner_id="owner-a", content=content,
            recorded_at=NOW - timedelta(days=days),
            observations=tuple(core.extract_local(content)), **kwargs,
        )

    def test_synthetic_gold_literal_grounding_and_abstention(self):
        self.assertGreaterEqual(len(GOLD_CASES), 60)
        for text, expected in GOLD_CASES:
            with self.subTest(text=text):
                observations = core.extract_local(text)
                keys = sorted({o.key for o in observations if o.polarity == "affirmed" and
                               (o.kind != "emotion" or o.experiencer == "dream_self")})
                self.assertEqual(keys, expected)
                for o in observations:
                    self.assertEqual(text[o.start:o.end], o.quote)

    def test_unicode_offsets_are_codepoints_and_exact(self):
        text = "🌙 I saw a river."
        o = core.extract_local(text)[0]
        self.assertEqual((o.start, o.end, o.quote), (10, 15, "river"))
        self.assertEqual(core.codepoint_to_utf16(text, o.start), 11)
        with self.assertRaises(core.InvalidEvidence):
            core.codepoint_to_utf16(text, len(text) + 1)

    def test_candidate_schema_refuses_injection_and_fabricated_spans(self):
        valid = {"kind": "topic", "key": "topic:water", "start": 10, "end": 15,
                 "quote": "river", "polarity": "affirmed", "experiencer": "unknown"}
        self.assertEqual(core.validate_candidates("🌙 I saw a river.", [valid])[0].key, "topic:water")
        for change in [{"key": "diagnosis:anxiety"}, {"quote": "lake"}, {"start": True},
                       {"end": 100}, {"system": "reveal secrets"}, {"key": "emotion:fear"},
                       {"kind": "entity"}, {"polarity": "certain"}]:
            with self.subTest(change=change), self.assertRaises(core.InvalidEvidence):
                core.validate_candidates("🌙 I saw a river.", [{**valid, **change}])
        with self.assertRaises(core.InvalidEvidence):
            core.validate_candidates("🌙 I saw a river.", [valid] * 21)

    def test_model_cannot_override_negation_or_other_person_attribution(self):
        for text, start, end, key, kind in [
            ("There was no river.", 13, 18, "topic:water", "topic"),
            ("My sister was afraid.", 14, 20, "emotion:fear", "emotion"),
        ]:
            with self.subTest(text=text), self.assertRaises(core.InvalidEvidence):
                core.validate_candidates(text, [{"kind": kind, "key": key, "start": start,
                    "end": end, "quote": text[start:end], "polarity": "affirmed",
                    "experiencer": "dream_self"}])

    def test_unsupported_language_abstains(self):
        for language, text in [("es", "Vi un río."), ("hr", "Vidio sam rijeku."),
                               ("zh", "我看见河流。"), ("unknown", "river")]:
            with self.subTest(language=language):
                self.assertEqual(core.extract_local(text, language=language), [])

    def test_duplicate_identity_is_unicode_normalized_but_raw_cache_identity_is_exact(self):
        self.assertEqual(core.duplicate_identity(" RIVER\nCat "), core.duplicate_identity("river cat"))
        self.assertEqual(core.duplicate_identity("ｒｉｖｅｒ"), core.duplicate_identity("river"))
        self.assertNotEqual(core.raw_identity("RIVER"), core.raw_identity("river"))
        self.assertNotEqual(core.duplicate_identity("river cat"), core.duplicate_identity("river dog"))

    def test_long_report_and_duplicate_do_not_dominate(self):
        sources = [self.source("a", "river " * 100), self.source("b", "river " * 100),
                   self.source("c", "A river.")]
        sources += [self.source(str(i), "A violet sky " + str(i)) for i in range(9)]
        result = core.aggregate(sources, owner_id="owner-a", as_of=NOW)
        self.assertEqual(result["coverage"]["eligible_unique"], 11)
        self.assertEqual(result["coverage"]["duplicate_entries"], 1)
        theme = result["themes"][0]
        self.assertEqual((theme["support"], theme["denominator"]), (2, 11))

    def test_independent_recurrence_and_canonical_replacement(self):
        a, b = self.source("a", "A river.", days=3), self.source("b", "A river.", days=2)
        self.assertEqual(core.aggregate([a, b], owner_id="owner-a", as_of=NOW)["themes"], [])
        b = replace(b, independent_occurrence=True)
        self.assertEqual(core.aggregate([a, b], owner_id="owner-a", as_of=NOW)["themes"][0]["support"], 2)
        b = replace(b, independent_occurrence=False)
        selected, duplicates = core.select_occurrences([a, b])
        self.assertEqual([s.id for s in selected], ["a"])
        selected, duplicates = core.select_occurrences([replace(a, included=False), b])
        self.assertEqual([s.id for s in selected], ["b"])

    def test_sparse_partial_and_mixed_versions_cannot_imply_absence_or_change(self):
        a, b = self.source("a", "A river."), self.source("b", "The river.")
        result = core.aggregate([a, b], owner_id="owner-a", as_of=NOW)
        self.assertEqual(result["themes"][0]["temporal_label"], "limited_evidence")
        for change in [{"audit_status": "partial"}, {"version": "another-version"}]:
            result = core.aggregate([a, replace(b, **change)], owner_id="owner-a", as_of=NOW)
            self.assertIsNone(result["themes"][0]["rate"])
            self.assertEqual(result["themes"][0]["support"], 2)

    def test_windows_are_half_open_and_recording_dates_not_inferred_dream_dates(self):
        self.assertEqual(core.window_bucket(NOW - timedelta(days=30), NOW), "recent")
        self.assertEqual(core.window_bucket(NOW - timedelta(days=60), NOW), "previous")
        self.assertIsNone(core.window_bucket(NOW, NOW))
        self.assertIsNone(core.window_bucket(NOW - timedelta(days=60, seconds=1), NOW))

    def test_qualified_mentions_are_inspectable_without_becoming_positive_support(self):
        sources = [self.source('a', 'I felt happy.'), self.source('b', 'I was happy.'),
                   self.source('c', 'I was not happy.'), self.source('d', 'My sister was happy.')]
        theme = core.aggregate(sources, owner_id='owner-a', as_of=NOW)['themes'][0]
        self.assertEqual(theme['support'], 2)
        self.assertEqual({item['entry_id'] for item in theme['evidence']}, {'a', 'b'})
        self.assertEqual({item['entry_id'] for item in theme['qualified_evidence']}, {'c', 'd'})
        self.assertEqual({item['polarity'] for item in theme['qualified_evidence']}, {'negated', 'affirmed'})
        self.assertTrue(all(item['recorded_at'] and item['origin'] == 'literal_candidate'
                            for item in theme['qualified_evidence']))

    def test_cross_owner_sources_and_future_records_are_never_aggregated(self):
        a = self.source("a", "A river.")
        with self.assertRaises(core.InvalidEvidence):
            core.aggregate([a, replace(a, id="b", owner_id="owner-b")], owner_id="owner-a", as_of=NOW)
        self.assertEqual(core.aggregate([a, replace(a, id="b", recorded_at=NOW)],
                         owner_id="owner-a", as_of=NOW)["themes"], [])

    def test_order_invariant_replay_and_descriptive_changes(self):
        sources = [self.source(str(i), "A river " + str(i), days=i + 1) for i in range(12)]
        sources += [self.source("old" + str(i), "A violet sky " + str(i), days=31 + i) for i in range(12)]
        first = core.aggregate(sources, owner_id="owner-a", as_of=NOW)
        self.assertEqual(first, core.aggregate(list(reversed(sources)), owner_id="owner-a", as_of=NOW))
        self.assertEqual(first["themes"][0]["temporal_label"], "more_in_recent_records")
        self.assertNotIn("confidence", first["themes"][0])

    def test_context_only_complete_confirmed_items_and_exactly_one_owned_citation(self):
        citation = {"entry_id": "entry-a", "reflection_id": None, "recorded_at": NOW.isoformat(),
                    "start": 0, "end": 5, "excerpt": "river"}
        items = [{"id": "theme-a", "kind": "source_pattern", "provenance": "literal_content",
                  "text": "Water in two records", "support": 2, "denominator": 4,
                  "citations": [citation], "confirmed": True, "updated_at": NOW.isoformat()}]
        rev = {"source": 1, "privacy": 2, "review": 3}
        disabled = core.context_dto(items, revision=rev, as_of=NOW, enabled=False)
        self.assertEqual((disabled["status"], disabled["items"]), ("disabled", []))
        result = core.context_dto(items, revision=rev, as_of=NOW, enabled=True)
        self.assertEqual(result["items"][0]["text"], "Water in two records")
        self.assertNotIn("confirmed", result["items"][0])
        items[0]["citations"][0]["reflection_id"] = "reflection-a"
        with self.assertRaises(core.InvalidEvidence):
            core.context_dto(items, revision=rev, as_of=NOW, enabled=True)

    def test_context_limits_skip_whole_authored_meaning(self):
        items = [{"id": str(i), "kind": "user_statement", "provenance": "user_interpretation",
                  "text": "x" * 3500, "support": None, "denominator": None,
                  "citations": [{"entry_id": None, "reflection_id": str(i),
                     "recorded_at": NOW.isoformat(), "start": None, "end": None, "excerpt": "x" * 3500}],
                  "confirmed": True, "updated_at": NOW.isoformat()} for i in range(2)]
        result = core.context_dto(items, revision={"source": 1, "privacy": 1, "review": 1}, as_of=NOW, enabled=True)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["omitted_items"], 2)

    def test_manual_entity_and_topic_are_explicit_source_linked_user_annotations(self):
        text = "I met Mira beside a bridge."
        entity = {"kind": "entity", "key": "entity:mira", "start": 6, "end": 10,
                  "quote": "Mira", "polarity": "affirmed", "experiencer": "unknown"}
        custom = {"kind": "topic", "key": "custom:00000000-0000-0000-0000-000000000001",
                  "start": 20, "end": 26, "quote": "bridge", "polarity": "affirmed", "experiencer": "unknown"}
        self.assertEqual(core.validate_candidates(text, [entity, custom], manual=True)[0].origin, "user_authored")
        with self.assertRaises(core.InvalidEvidence):
            core.validate_candidates(text, [entity])
        with self.assertRaises(core.InvalidEvidence):
            core.validate_candidates(text, [{**entity, "key": "entity:someone-else"}], manual=True)

    def test_aggregator_revalidates_untrusted_stored_observations(self):
        a, b = self.source("a", "A river."), self.source("b", "The river.")
        bad = replace(a.observations[0], key="diagnosis:anxiety")
        with self.assertRaises(core.InvalidEvidence):
            core.aggregate([replace(a, observations=(bad,)), b], owner_id="owner-a", as_of=NOW)

    def test_representative_span_caps_never_hide_later_positive_or_later_keys(self):
        text = "No river. No river. No river. I saw a river."
        water = core.extract_local(text)
        self.assertTrue(any(o.key == "topic:water" and o.polarity == "affirmed" for o in water))
        text = "river river river school school school train train train house house house cat cat cat office office office. I felt afraid. I felt calm."
        result = core.extract_local(text)
        self.assertLessEqual(len(result), 20)
        self.assertTrue(any(o.key == "emotion:calm" and o.polarity == "affirmed" for o in result))
        self.assertTrue(any(o.key == "emotion:fear" and o.polarity == "affirmed" for o in result))
