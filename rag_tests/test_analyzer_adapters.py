"""Execute original analyzer methods without importing startup/model dependencies."""
import ast
import asyncio
from dataclasses import dataclass
from datetime import datetime
import io
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dream-fast-api"))
from core import rag_contracts


def analyzer_class(names, **extra):
    file = ROOT / "dream-fast-api/core/dream_analyzer.py"
    tree = ast.parse(file.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DreamJournalAnalyzer")
    methods = [n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    isolated = ast.Module(body=[ast.ClassDef(name="Analyzer", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(isolated)
    async def wrap_result(fn, *args, **kwargs):
        return fn(None)
    def sync_to_async(fn, **kwargs):
        async def call(*args, **kwargs):
            return fn(*args, **kwargs)
        return call
    namespace = {**vars(rag_contracts), "__file__": str(file), "os": os, "Path": Path, "re": re,
                 "datetime": datetime, "List": list, "Dict": dict, "Any": object, "Optional": object,
                 "Document": SimpleNamespace, "JournalEntry": SimpleNamespace,
                 "SystemMessage": SimpleNamespace, "HumanMessage": SimpleNamespace,
                 "DoctorProfile": SimpleNamespace, "sync_to_async": sync_to_async,
                 "async_job_write": wrap_result, **extra}
    # Forward annotations are data; external dependency imports never execute.
    exec(compile(isolated, str(file), "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)
    return namespace["Analyzer"]


class AnalyzerAdaptersTest(unittest.TestCase):
    def test_query_expansions_use_actual_extractor_fields_and_entities(self):
        cls = analyzer_class({"_build_queries_from_elements"})
        result = cls()._build_queries_from_elements({"primary_symbols": ["bridge"], "key_phrases": ["violet heron"], "entities": [("tram station", "LOC")], "emotions": []})
        self.assertEqual(result, ["bridge", "violet heron", "tram station"])

    def test_file_persona_name_and_sections_are_complete(self):
        cls = analyzer_class({"_get_doctor_profile", "_parse_weights_from_text"})
        analyzer = cls()
        analyzer.DoctorProfile = lambda **kw: SimpleNamespace(**kw)
        text = "Name: Academic\nArchetype: Reader\nTone: Gentle\nBackground:\nSynthetic background.\nPersonality Style:\nSynthetic style.\nPrompt Style:\nSynthetic prompt.\nWeights:\n  theory: 0.7\n  astrology: 0.1\n  personality: 0.1\n  medicalHistory: 0.1\n"
        with patch("builtins.open", return_value=io.StringIO(text)):
            profile = asyncio.run(analyzer._get_doctor_profile("Academic"))
        self.assertEqual(profile.name, "Academic")
        self.assertEqual(profile.background, "Synthetic background.")
        self.assertEqual(profile.prompt_style, "Synthetic prompt.")

    def test_custom_question_retrieves_both_concepts_without_first_query_exhaustion(self):
        cls = analyzer_class({"comprehensive_knowledge_retrieval", "_is_relevant_interpretation_content"},
                             PersonalityKnowledgeBase=lambda: None, AstrologyKnowledgeBase=lambda: None)
        analyzer = cls()
        async def extract(entries):
            return {"symbols": ["bridge", "tram"]}
        analyzer.symbol_extractor = SimpleNamespace(extract_dream_elements=extract)
        analyzer._build_queries_from_elements = lambda elements: ["bridge", "tram", "extra"]
        bridge = SimpleNamespace(page_content="bridge symbolism", metadata={"source": "paper-bridge"})
        tram = SimpleNamespace(page_content="tram symbolism", metadata={"source": "paper-tram"})
        async def search(query, k):
            return [tram] if "tram" in query else [bridge] * 20
        analyzer.knowledge_base = SimpleNamespace(search_relevant_knowledge=search)
        try:
            result = asyncio.run(analyzer.comprehensive_knowledge_retrieval([], {}, k=20, total_k=2,
                                                                           question="Compare bridge and tram imagery"))
        except TypeError:
            self.fail("actual question is not accepted by reference retrieval")
        self.assertEqual({doc.page_content for doc in result["docs"]}, {"bridge symbolism", "tram symbolism"})
        self.assertEqual(result["retrieval"]["queries"][0], "Compare bridge and tram imagery")
        self.assertLessEqual(result["retrieval"]["queries_executed"], 3)

    def test_resolver_uses_db_consistently_and_records_file_conflict(self):
        db = SimpleNamespace(name="Academic", archetype="Reader", tone="Gentle", background="DB background",
                             personality_style="DB style", prompt_style="DB prompt", full_prompt="",
                             weights={"theory": .6, "astrology": .1, "personality": .2, "medicalHistory": .1},
                             id="synthetic-persona", updated_at=datetime(2026, 10, 10))
        model = SimpleNamespace(objects=SimpleNamespace(filter=lambda **kw: SimpleNamespace(first=lambda: db)))
        cls = analyzer_class({"_resolve_analysis_profile", "_compute_final_weights", "_normalize_weights", "_blend"}, DoctorProfile=model)
        analyzer = cls()
        analyzer.DoctorProfile = lambda **kw: SimpleNamespace(**kw)
        async def file_profile(name):
            return SimpleNamespace(name="Academic", archetype="Reader", tone="Gentle", background="File background",
                                   personality_style="File style", prompt_style="File prompt", raw_text="",
                                   weights={"theory": .7, "astrology": .1, "personality": .1, "medicalHistory": .1})
        analyzer._get_doctor_profile = file_profile
        self.assertTrue(hasattr(analyzer, "_resolve_analysis_profile"), "routes have no canonical persona resolver")
        settings, profile, weights, provenance = asyncio.run(analyzer._resolve_analysis_profile({"influence": {"doctor": .7}}))
        self.assertEqual(profile.background, "DB background")
        self.assertEqual(settings["doctor_influence"], .7)
        self.assertEqual(provenance["persona_source"], "database")
        self.assertTrue(provenance["file_database_conflict"])
        self.assertAlmostEqual(sum(weights.values()), 1)


if __name__ == "__main__":
    unittest.main()
