"""Pure analysis contracts; no ORM, provider, index or private cache access.

Authentication and longitudinal provenance remain their producers' authority.
These functions validate/pack already authorized, request-local inputs.
"""
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import math
import re

PERSONAS = frozenset({"academic", "scientific", "mystical", "artistic", "compassionate"})
WEIGHT_KEYS = ("theory", "astrology", "personality", "medicalHistory")
PROFILE_SCHEMA = "analysis-profile-1"
MBTI_TYPES = frozenset({"INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"})
SIGNS = frozenset({"aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"})


class ContractError(ValueError):
    """Safe error code; never include private source text in errors."""


class PackedEvidence(dict):
    """Request-local registry marker with the source used to check exact spans.

    JSON transport loses this marker: re-pack against authorized current sources
    before validation. Never persist the private source_text attribute.
    """
    def __init__(self, data, source_text):
        super().__init__(data)
        self.source_text = source_text


def _unit(value):
    if isinstance(value, bool):
        raise ContractError("invalid_influence")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ContractError("invalid_influence") from None
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ContractError("invalid_influence")
    return number


def canonical_settings(settings):
    """Normalize a server-resolved snapshot, never authenticate a body owner.

    Explicit legacy values survive. Missing influence uses the existing model's
    .7 default. The existing numeric blend and historical analyses are unchanged.
    """
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ContractError("invalid_settings_snapshot")
    out = deepcopy(settings)
    aliases = {"doctorPersonality": "doctor_personality", "personality": "personality_type",
               "medicalHistory": "medical_history", "influence": "weights"}
    for name, alias in aliases.items():
        if name in out and alias in out and out[name] != out[alias]:
            raise ContractError("conflicting_settings_alias")
        if name not in out and alias in out:
            out[name] = out[alias]
        out.pop(alias, None)
    name = out.get("doctorPersonality") or "Academic"
    if not isinstance(name, str) or name.strip().casefold() not in PERSONAS:
        raise ContractError("invalid_persona")
    out["doctorPersonality"] = name.strip().capitalize()
    influences = out.get("influence") or {}
    if not isinstance(influences, dict):
        raise ContractError("invalid_influence")
    influences = deepcopy(influences)
    explicit = "doctor_influence" in out
    legacy = "doctor" in influences
    if explicit and legacy and _unit(out["doctor_influence"]) != _unit(influences["doctor"]):
        raise ContractError("conflicting_doctor_influence")
    value = out["doctor_influence"] if explicit else influences.get("doctor", .7)
    out["doctor_influence"] = _unit(value)
    influences.pop("doctor", None)
    for key, value in influences.items():
        if key not in WEIGHT_KEYS:
            raise ContractError("invalid_influence_key")
        influences[key] = _unit(value)
    out["influence"] = influences
    for key in ("personality", "occupation"):
        out.setdefault(key, "")
        if out[key] is not None and not isinstance(out[key], str):
            raise ContractError("invalid_profile_field")
        if out[key] is not None and len(out[key]) > 1000:
            raise ContractError("profile_field_too_long")
    for key in ("medicalHistory", "astrology"):
        out.setdefault(key, {})
        if out[key] is not None and not isinstance(out[key], dict):
            raise ContractError("invalid_profile_field")
    personality = (out["personality"] or "").strip().upper()
    if personality and personality not in MBTI_TYPES:
        raise ContractError("invalid_personality_type")
    out["personality"] = personality
    astrology = out["astrology"] or {}
    if set(astrology) - {"sun", "moon", "rising"}:
        raise ContractError("invalid_astrology_role")
    for role, sign in astrology.items():
        if sign is None or sign == "":
            astrology[role] = None
        elif not isinstance(sign, str) or sign.strip().casefold() not in SIGNS:
            raise ContractError("invalid_astrology_sign")
        else:
            astrology[role] = sign.strip().capitalize()
    out["astrology"] = astrology
    previous = out.get("profile_provenance", {})
    origin = "explicit" if explicit else "legacy_influence.doctor" if legacy else "model_default"
    if isinstance(previous, dict) and previous.get("schema") == PROFILE_SCHEMA and previous.get("doctor_influence") in {"explicit", "legacy_influence.doctor", "model_default"}:
        origin = previous["doctor_influence"]
    out["profile_provenance"] = {
        "schema": PROFILE_SCHEMA,
        "doctor_influence": origin,
        "medical_slider_effect": "inert_in_legacy_blend",
    }
    return out


def validate_persona_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_KEYS):
        raise ContractError("invalid_persona_weights")
    return {key: _unit(weights[key]) for key in WEIGHT_KEYS}


def parse_persona_text(text):
    headers = list(re.finditer(r"^(Name|Archetype|Tone|Background|Personality Style|Prompt Style|Weights)[ \t]*:[ \t]*(.*)$", text, re.M | re.I))
    sections = {}
    for i, match in enumerate(headers):
        key = match.group(1).casefold().replace(" ", "_")
        if key in sections:
            raise ContractError("duplicate_persona_section")
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[key] = (match.group(2) + text[match.end():end]).strip()
    if not sections.get("name") or not sections.get("weights"):
        raise ContractError("incomplete_persona")
    weights = {}
    for line in sections["weights"].splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*(theory|astrology|personality|medicalHistory)\s*:\s*(\S+)\s*", line)
        if not match or match.group(1) in weights:
            raise ContractError("invalid_persona_weights")
        weights[match.group(1)] = _unit(match.group(2))
    return {key: sections.get(key, "") for key in
            ("name", "archetype", "tone", "background", "personality_style", "prompt_style")} | {"raw_text": text, "weights": validate_persona_weights(weights)}


def profile_version(profile):
    return sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def query_plan(question, expansions, *, max_queries=3):
    if not isinstance(max_queries, int) or not 1 <= max_queries <= 3:
        raise ContractError("invalid_query_budget")
    if question is not None and (not isinstance(question, str) or len(question) > 4000):
        raise ContractError("invalid_question")
    result, seen = [], set()
    for value in ([question] if question else []) + list(expansions):
        if not isinstance(value, str):
            raise ContractError("invalid_query")
        value = value.strip()[:4000]
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value[:4000])
        if len(result) == max_queries:
            break
    return result


def document_id(doc):
    meta = doc.metadata
    identity = {key: meta.get(key) for key in ("source", "page", "start_index", "chunk_id")}
    identity["chunk_content_hash"] = sha256(doc.page_content.encode("utf-8")).hexdigest()
    return "ref-" + sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


async def collect_candidates(queries, retrieve, *, total_k=20, per_query=12,
                             seeds=(), accept=lambda doc: True):
    """Collect every bounded query before final selection; fair coverage + RRF.

    MMR remains inside the existing retriever. Rank fusion is deterministic and
    introduces no new embedding/index/model dependency.
    """
    if type(total_k) is not int or not 0 <= total_k <= 20 or type(per_query) is not int or not 1 <= per_query <= 12:
        raise ContractError("invalid_candidate_budget")
    if not total_k:
        return {"docs": [], "queries_executed": 0, "candidate_count": 0, "query_coverage": []}
    queries = query_plan(None, queries)
    pool, ranks, groups = {}, {}, []
    seed_ids = []
    for doc in list(seeds)[:4]:
        if accept(doc):
            key = document_id(doc)
            pool[key] = doc
            if key not in seed_ids:
                seed_ids.append(key)
    for query in queries:
        group = []
        for rank, doc in enumerate((await retrieve(query, per_query))[:per_query], 1):
            if not accept(doc):
                continue
            key = document_id(doc)
            pool[key] = doc
            if key in group:
                continue
            group.append(key)
            ranks[key] = ranks.get(key, 0) + 1 / (60 + rank)
        groups.append(group)
    chosen = []
    # Each query gets a distinct eligible candidate before repeated matches or seeds.
    for group in groups:
        key = next((key for key in group if key not in chosen), None)
        if key is not None and len(chosen) < total_k:
            chosen.append(key)
    for key in seed_ids + sorted(pool, key=lambda key: (-ranks.get(key, 0), key)):
        if key not in chosen and len(chosen) < total_k:
            chosen.append(key)
    return {"docs": [pool[key] for key in chosen], "queries_executed": len(queries),
            "candidate_count": len(pool), "query_coverage": [len(set(group) & set(chosen)) for group in groups]}


def source_type(source):
    path = str(source).replace("\\", "/").casefold()
    if "doctor_profiles/" in path:
        return "persona_template"
    if "astrology/" in path:
        return "astrology"
    if "personality/" in path:
        return "mbti_interpretive"
    return "reference_unclassified"


def reference_evidence(docs, *, max_chars=12000, per_document=1200):
    if type(max_chars) is not int or not 1 <= max_chars <= 12000 or type(per_document) is not int or not 1 <= per_document <= 1200:
        raise ContractError("invalid_evidence_budget")
    evidence, remaining, seen = [], max_chars, set()
    for doc in docs:
        key = document_id(doc)
        if key in seen or remaining <= 0:
            continue
        seen.add(key)
        quote = doc.page_content[:min(per_document, remaining)]
        if not quote:
            continue
        remaining -= len(quote)
        meta = doc.metadata
        evidence.append(PackedEvidence({"evidence_id": key, "kind": "public_reference", "source": str(meta.get("source", "unknown")),
                         "source_type": source_type(meta.get("source", "")), "page": meta.get("page"),
                         "url": meta.get("url"), "title": meta.get("title"), "author": meta.get("author"),
                         "chunk_content_hash": sha256(doc.page_content.encode("utf-8")).hexdigest(),
                         "offset_unit": "unicode_code_point", "start_char": 0, "end_char": len(quote), "quote": quote,
                         "calibration_status": "unavailable", "attribution_status": "metadata_only"}, doc.page_content))
    return evidence


def entry_evidence(entry_id, content, start=0, end=None, *, draft=False):
    end = len(content) if end is None else end
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content):
        raise ContractError("invalid_evidence_span")
    digest = sha256(content.encode("utf-8")).hexdigest()
    key = sha256(f"{entry_id}:{digest}:{start}:{end}".encode("utf-8")).hexdigest()
    return PackedEvidence({"evidence_id": ("draft-" if draft else "entry-") + key, "kind": "user_draft" if draft else "user_entry", "entry_id": None if draft else str(entry_id),
            "content_hash": digest, "offset_unit": "unicode_code_point", "start_char": start,
            "end_char": end, "quote": content[start:end]}, content)


def validate_source_dependencies(expected, entries):
    """Compare exact current owned sources under the auth writer's locks."""
    actual = {str(entry.id): sha256(entry.content.encode("utf-8")).hexdigest() for entry in entries}
    wanted = {item["entry_id"]: item["content_hash"] for item in expected}
    if len(wanted) != len(expected) or actual != wanted:
        raise ContractError("analysis_sources_changed")


def validate_answer(payload, evidence):
    for item in evidence:
        if not isinstance(item, PackedEvidence):
            raise ContractError("unvalidated_evidence_registry")
        start, end = item.get("start_char"), item.get("end_char")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(item.source_text) or item.get("quote") != item.source_text[start:end]:
            raise ContractError("invalid_evidence_span")
        digest_key = "content_hash" if item.get("kind") in {"user_entry", "user_draft"} else "chunk_content_hash"
        if item.get(digest_key) != sha256(item.source_text.encode("utf-8")).hexdigest():
            raise ContractError("invalid_evidence_hash")
    if not isinstance(payload, dict) or set(payload) != {"answer", "claims"} or not isinstance(payload["answer"], str):
        raise ContractError("invalid_answer_contract")
    claims = payload["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= 20:
        raise ContractError("invalid_claims")
    registry = {item["evidence_id"]: item for item in evidence}
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"kind", "text", "evidence_ids"}:
            raise ContractError("invalid_claim")
        if claim["kind"] not in {"observation", "source_report", "hypothesis", "reflective_prompt", "style_suggestion"}:
            raise ContractError("invalid_claim_kind")
        if not isinstance(claim["text"], str) or not claim["text"].strip() or len(claim["text"]) > 4000:
            raise ContractError("invalid_claim_text")
        ids = claim["evidence_ids"]
        if not isinstance(ids, list) or len(ids) > 8 or any(not isinstance(key, str) or key not in registry for key in ids):
            raise ContractError("unknown_evidence")
        if claim["kind"] in {"observation", "source_report"} and not ids:
            raise ContractError("missing_evidence")
        if claim["kind"] == "observation" and any(registry[key]["kind"] not in {"user_entry", "user_draft"} for key in ids):
            raise ContractError("wrong_observation_source")
        if claim["kind"] == "source_report" and any(registry[key]["kind"] != "public_reference" for key in ids):
            raise ContractError("wrong_reference_source")
    if " ".join(payload["answer"].split()) != " ".join(" ".join(claim["text"] for claim in claims).split()):
        raise ContractError("incomplete_claim_inventory")
    if len(payload["answer"]) > 20000:
        raise ContractError("answer_too_long")
    used = {key for claim in claims for key in claim["evidence_ids"]}
    return {"answer": payload["answer"], "claims": deepcopy(claims), "evidence": [deepcopy(dict(item)) for item in evidence if item["evidence_id"] in used],
            "calibration_status": "unavailable", "semantic_support": "not_evaluated", "span_status": "validated_at_pack_time"}


def validate_confirmed_context(dto):
    if not isinstance(dto, dict) or set(dto) != {"status", "revision", "as_of", "items", "omitted_items"}:
        raise ContractError("invalid_context")
    revision = dto["revision"]
    if not isinstance(revision, dict) or set(revision) != {"source", "privacy", "review"} or any(type(value) is not int or value < 0 for value in revision.values()):
        raise ContractError("invalid_context_revision")
    try:
        date = datetime.fromisoformat(dto["as_of"].replace("Z", "+00:00"))
        if date.utcoffset() is None:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ContractError("invalid_context_time") from None
    if dto["status"] not in {"ready", "disabled", "stale"} or not isinstance(dto["items"], list) or len(dto["items"]) > 8:
        raise ContractError("invalid_context_status")
    if type(dto["omitted_items"]) is not int or dto["omitted_items"] < 0:
        raise ContractError("invalid_context_omission_count")
    if dto["status"] != "ready" and dto["items"]:
        raise ContractError("stale_context_content")
    chars = 0
    for item in dto["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "kind", "provenance", "text", "support", "denominator", "citations"}:
            raise ContractError("invalid_context_item")
        if item["kind"] not in {"source_pattern", "user_statement"} or item["provenance"] not in {"literal_content", "user_interpretation"}:
            raise ContractError("invalid_context_provenance")
        if not isinstance(item["id"], str) or not item["id"] or not isinstance(item["text"], str) or not item["text"]:
            raise ContractError("invalid_context_text")
        for key in ("support", "denominator"):
            if item[key] is not None and (type(item[key]) is not int or item[key] < 0):
                raise ContractError("invalid_context_count")
        if item["support"] is not None and item["denominator"] is not None and item["support"] > item["denominator"]:
            raise ContractError("invalid_context_count")
        citations = item["citations"]
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise ContractError("invalid_context_citations")
        chars += len(item["text"])
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {"entry_id", "reflection_id", "recorded_at", "start", "end", "excerpt"}:
                raise ContractError("invalid_context_citation")
            if bool(citation["entry_id"]) == bool(citation["reflection_id"]) or not isinstance(citation["excerpt"], str):
                raise ContractError("invalid_context_citation")
            if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in (citation["entry_id"], citation["reflection_id"])):
                raise ContractError("invalid_context_citation")
            try:
                recorded = datetime.fromisoformat(citation["recorded_at"].replace("Z", "+00:00"))
                if recorded.utcoffset() is None:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError):
                raise ContractError("invalid_context_time") from None
            start, end = citation["start"], citation["end"]
            if start is None or end is None:
                if start is not None or end is not None or citation["reflection_id"] is None:
                    raise ContractError("invalid_context_span")
            elif type(start) is not int or type(end) is not int or not 0 <= start < end or end - start != len(citation["excerpt"]):
                raise ContractError("invalid_context_span")
            chars += len(citation["excerpt"])
    if chars > 6000:
        raise ContractError("context_too_large")
    return deepcopy(dto)
