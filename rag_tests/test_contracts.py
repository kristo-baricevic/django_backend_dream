"""Offline behavior tests. Application startup, databases and providers are absent."""
import asyncio
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class ContractsTest(unittest.TestCase):
    def contracts(self):
        path = ROOT / "dream-fast-api/core/rag_contracts.py"
        self.assertTrue(path.exists(), "missing pure RAG contracts")
        spec = importlib.util.spec_from_file_location("rag_contracts", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_persona_sections_do_not_bleed_or_disappear(self):
        c = self.contracts()
        profile = c.parse_persona_text("Name: Example\nArchetype: Reader\nTone: Gentle\nBackground:\nFirst paragraph.\nSecond paragraph.\n\nPersonality Style:\nCalm reflection.\nPrompt Style:\nAsk an open question.\nWeights:\n  theory: 0.7\n  astrology: 0.1\n  personality: 0.1\n  medicalHistory: 0.1\n")
        self.assertEqual(profile["name"], "Example")
        self.assertEqual(profile["background"], "First paragraph.\nSecond paragraph.")
        self.assertEqual(profile["personality_style"], "Calm reflection.")
        self.assertEqual(profile["prompt_style"], "Ask an open question.")
        self.assertEqual(profile["weights"]["medicalHistory"], .1)

    def test_canonical_default_and_legacy_alias_preserve_explicit_zero(self):
        c = self.contracts()
        self.assertEqual(c.canonical_settings({})["doctor_influence"], .7)
        profile = c.canonical_settings({"influence": {"doctor": 0}, "occupation": "synthetic illustrator"})
        self.assertEqual(profile["doctor_influence"], 0)
        self.assertEqual(profile["occupation"], "synthetic illustrator")
        self.assertEqual(profile["profile_provenance"]["doctor_influence"], "legacy_influence.doctor")

    def test_conflicting_aliases_are_rejected_instead_of_silent_override(self):
        c = self.contracts()
        with self.assertRaises(c.ContractError):
            c.canonical_settings({"doctor_influence": .7, "influence": {"doctor": .5}})

    def test_normalization_is_idempotent_and_profile_paths_are_allowlisted(self):
        c = self.contracts()
        first = c.canonical_settings({"doctor_personality": " academic ", "weights": {"doctor": .5}, "personality_type": "intj", "astrology": {"sun": "aries"}})
        self.assertEqual(c.canonical_settings(first), first)
        for value in ({"personality": "../files/10000dreams"}, {"astrology": {"sun": "../../files/10000dreams"}}, {"astrology": {"sun": 1}}, {"astrology": {"other": "Aries"}}):
            with self.subTest(value=value), self.assertRaises(c.ContractError):
                c.canonical_settings(value)

    def test_evidence_budgets_and_query_truncation_are_safe(self):
        c = self.contracts()
        doc = SimpleNamespace(page_content="a" * 100, metadata={})
        for args in ({"per_document": -1}, {"max_chars": True}, {"max_chars": 12001}):
            with self.subTest(args=args), self.assertRaises(c.ContractError):
                c.reference_evidence([doc], **args)
        self.assertEqual(len(c.query_plan(None, ["a" * 4000 + "b", "a" * 4000 + "c"])), 1)

    def test_forged_evidence_does_not_claim_validated_span(self):
        c = self.contracts()
        ev = c.entry_evidence("synthetic-entry", "bridge")
        payload = {"answer": "Bridge.", "claims": [{"kind": "observation", "text": "Bridge.", "evidence_ids": [ev["evidence_id"]]}]}
        with self.assertRaises(c.ContractError):
            c.validate_answer(payload, [{**ev, "quote": "tram"}])

    def test_source_snapshot_rejects_edit_delete_and_foreign_replacement(self):
        c = self.contracts()
        expected = [{"entry_id": "a", "content_hash": c.entry_evidence("a", "bridge")["content_hash"]}]
        c.validate_source_dependencies(expected, [SimpleNamespace(id="a", content="bridge")])
        for rows in ([], [SimpleNamespace(id="a", content="tram")], [SimpleNamespace(id="b", content="bridge")]):
            with self.subTest(rows=rows), self.assertRaisesRegex(c.ContractError, "analysis_sources_changed"):
                c.validate_source_dependencies(expected, rows)

    def test_nonfinite_out_of_range_and_boolean_weights_fail_closed(self):
        c = self.contracts()
        for bad in (float("nan"), float("inf"), -1, 1.1, True, "garbage"):
            with self.subTest(bad=bad), self.assertRaises(c.ContractError):
                c.canonical_settings({"doctor_influence": bad})

    def test_model_fields_and_serialized_settings_resolve_identically(self):
        c = self.contracts()
        snake = {"doctor_personality": "Academic", "personality_type": "INTJ", "medical_history": {"physical": []}, "weights": {"astrology": .2}, "doctor_influence": .7}
        camel = {"doctorPersonality": "Academic", "personality": "INTJ", "medicalHistory": {"physical": []}, "influence": {"astrology": .2}, "doctor_influence": .7}
        a, b = c.canonical_settings(snake), c.canonical_settings(camel)
        for key in ("doctorPersonality", "personality", "medicalHistory", "influence", "doctor_influence"):
            self.assertEqual(a[key], b[key])

    def test_question_is_first_and_expansions_are_unique_bounded(self):
        c = self.contracts()
        plan = c.query_plan("Where was the violet heron?", ["bridge", "bridge", "tram", "lake"])
        self.assertEqual(plan, ["Where was the violet heron?", "bridge", "tram"])

    def test_stable_document_identity_uses_full_content_and_source_location(self):
        c = self.contracts()
        prefix = "x" * 200
        a = SimpleNamespace(page_content=prefix + "bridge", metadata={"source": "paper-a", "page": 1})
        b = SimpleNamespace(page_content=prefix + "tram", metadata={"source": "paper-a", "page": 1})
        d = SimpleNamespace(page_content=a.page_content, metadata={"source": "paper-b", "page": 1})
        self.assertNotEqual(c.document_id(a), c.document_id(b))
        self.assertNotEqual(c.document_id(a), c.document_id(d))
        self.assertEqual(c.document_id(a), c.document_id(a))

    def test_all_queries_contribute_before_final_cap_and_duplicates_are_shared(self):
        c = self.contracts()
        a = SimpleNamespace(page_content="bridge", metadata={"source": "a"})
        b = SimpleNamespace(page_content="tram", metadata={"source": "b"})
        extra = [SimpleNamespace(page_content=f"bridge filler {i}", metadata={"source": str(i)}) for i in range(20)]
        async def retrieve(query, k):
            return ([a] + extra)[:k] if query == "bridge" else [a, b]
        result = asyncio.run(c.collect_candidates(["bridge", "tram"], retrieve, total_k=2))
        self.assertEqual([d.page_content for d in result["docs"]], ["bridge", "tram"])
        self.assertEqual(result["queries_executed"], 2)

    def test_reference_spans_and_unicode_user_spans_validate_original_text(self):
        c = self.contracts()
        doc = SimpleNamespace(page_content="First sentence. Second sentence.", metadata={"source": "paper", "page": 2})
        refs = c.reference_evidence([doc], max_chars=15)
        self.assertEqual(refs[0]["quote"], "First sentence.")
        self.assertEqual(refs[0]["start_char"], 0)
        self.assertEqual(refs[0]["end_char"], 15)
        ev = c.entry_evidence("synthetic-entry", "I saw 🌉 then a bridge.", 13, 21)
        self.assertEqual(ev["quote"], "a bridge")
        self.assertEqual(ev["offset_unit"], "unicode_code_point")

    def test_claim_inventory_rejects_undeclared_prose_and_unknown_evidence(self):
        c = self.contracts()
        ev = c.entry_evidence("synthetic-entry", "I saw a bridge.", 6, 14)
        valid = {"answer": "A bridge appears in this record.", "claims": [{"kind": "observation", "text": "A bridge appears in this record.", "evidence_ids": [ev["evidence_id"]]}]}
        answer = c.validate_answer(valid, [ev])
        self.assertEqual(answer["calibration_status"], "unavailable")
        self.assertEqual(answer["semantic_support"], "not_evaluated")
        for bad in ({**valid, "answer": valid["answer"] + " You have a disorder."}, {"answer": "A bridge appears.", "claims": [{"kind": "observation", "text": "A bridge appears.", "evidence_ids": ["missing"]}]}):
            with self.subTest(bad=bad), self.assertRaises(c.ContractError):
                c.validate_answer(bad, [ev])

    def test_hypothesis_is_not_promoted_by_confirmation(self):
        c = self.contracts()
        item = {"status": "ready", "revision": {"source": 1, "privacy": 1, "review": 1}, "as_of": "2026-10-10T00:00:00Z", "items": [{"id": "synthetic-hypothesis", "kind": "user_statement", "provenance": "assistant_hypothesis", "text": "fictional hypothesis", "support": None, "denominator": None, "citations": [{"entry_id": "entry", "reflection_id": None, "recorded_at": "2026-10-10T00:00:00Z", "start": 0, "end": 6, "excerpt": "bridge"}]}], "omitted_items": 0}
        with self.assertRaisesRegex(c.ContractError, "invalid_context_provenance"):
            c.validate_confirmed_context(item)

    def test_ready_context_and_invalid_citation_identity_or_time(self):
        c = self.contracts()
        citation = {"entry_id": "entry", "reflection_id": None, "recorded_at": "2026-10-10T00:00:00Z", "start": 0, "end": 6, "excerpt": "bridge"}
        item = {"id": "pattern", "kind": "source_pattern", "provenance": "literal_content", "text": "Bridge occurs.", "support": 1, "denominator": 2, "citations": [citation]}
        dto = {"status": "ready", "revision": {"source": 1, "privacy": 1, "review": 1}, "as_of": "2026-10-10T00:00:00Z", "items": [item], "omitted_items": 0}
        self.assertEqual(c.validate_confirmed_context(dto), dto)
        reflection = {**citation, "entry_id": None, "reflection_id": "reflection", "start": None, "end": None}
        self.assertEqual(c.validate_confirmed_context({**dto, "items": [{**item, "citations": [reflection]}]})["status"], "ready")
        for change in ({"entry_id": True}, {"recorded_at": {}}, {"recorded_at": "2026-10-10"}):
            with self.subTest(change=change), self.assertRaises(c.ContractError):
                c.validate_confirmed_context({**dto, "items": [{**item, "citations": [{**citation, **change}]}]})

    def test_disabled_context_never_returns_a_previous_item(self):
        c = self.contracts()
        dto = {"status": "disabled", "revision": {"source": 1, "privacy": 2, "review": 1}, "as_of": "2026-10-10T00:00:00Z", "items": [], "omitted_items": 0}
        self.assertEqual(c.validate_confirmed_context(dto)["items"], [])
        with self.assertRaises(c.ContractError):
            c.validate_confirmed_context({**dto, "items": [{"text": "stale private text"}]})


if __name__ == "__main__":
    unittest.main()
