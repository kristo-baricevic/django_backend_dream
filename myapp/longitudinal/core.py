"""Pure local evidence validation and descriptive aggregation.

Dream reports are untrusted reported content, never proof of waking events.
No settings, interpretations or previous profile enter extraction. Statistical
windows are display heuristics; correlated reports are not independent samples.
"""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
import re
import unicodedata
from uuid import UUID

CODEBOOK_VERSION = "literal-en-1"
EXTRACTOR_VERSION = "local-en-1"
AGGREGATOR_VERSION = "presence-1"
SCHEMA_VERSION = "evidence-1"

CODEBOOK = {
    "topic:water": ("Water in dream scenes", r"water|river|ocean|lake|stream"),
    "topic:school": ("School in dream scenes", r"school|classroom|campus"),
    "topic:travel": ("Transport in dream scenes", r"train|airport|bus"),
    "topic:home": ("Home in dream scenes", r"house|home|bedroom"),
    "topic:animals": ("Animals in dream scenes", r"cats?|dogs?|birds?|horses?"),
    "topic:work": ("Workplaces in dream scenes", r"office|workplace"),
    "emotion:fear": ("Reported fear in dreams", r"afraid|frightened|scared"),
    "emotion:joy": ("Reported joy in dreams", r"happy|joyful|delighted"),
    "emotion:sadness": ("Reported sadness in dreams", r"sad|sorrowful"),
    "emotion:anger": ("Reported anger in dreams", r"angry|furious"),
    "emotion:calm": ("Reported calm in dreams", r"calm|peaceful"),
}
PATTERNS = {key: re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE)
            for key, (_, pattern) in CODEBOOK.items()}
NEGATION = re.compile(r"\b(?:no|not|never|without|wasn't|weren't|didn't|isn't|don't|cannot)\b", re.I)
INJECTION = re.compile(r"ignore\s+(?:all|previous|the|system)|system\s+prompt|"
                       r"reveal.*(?:user|profile)|confirm.*theme|grant.*consent", re.I)
SELF_EMOTION = re.compile(r"\bI\s+(?:felt|feel|was|am|became)\s+(?:very\s+|so\s+)?$", re.I)


class InvalidEvidence(ValueError):
    """Safe validation failure; never include private source text in its message."""


@dataclass(frozen=True)
class Observation:
    kind: str
    key: str
    start: int
    end: int
    quote: str
    polarity: str
    experiencer: str
    origin: str = "literal_candidate"


@dataclass(frozen=True)
class AuditSource:
    id: str
    owner_id: str
    content: str
    recorded_at: datetime
    observations: tuple[Observation, ...] = ()
    audit_status: str = "complete"
    version: str = EXTRACTOR_VERSION
    included: bool = True
    origin: str = "user_reported"
    language: str = "en"
    independent_occurrence: bool = False


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise InvalidEvidence("timezone_required")
    return value.astimezone(timezone.utc)


def raw_identity(content: str) -> str:
    if not isinstance(content, str) or "\x00" in content:
        raise InvalidEvidence("invalid_source_encoding")
    try:
        return sha256(content.encode("utf-8", errors="strict")).hexdigest()
    except (AttributeError, UnicodeError) as exc:
        raise InvalidEvidence("invalid_source_encoding") from exc


def duplicate_identity(content: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", content).casefold().split())
    return raw_identity(normalized)


def codepoint_to_utf16(content: str, offset: int) -> int:
    if type(offset) is not int or not 0 <= offset <= len(content):
        raise InvalidEvidence("invalid_source_offset")
    try:
        return len(content[:offset].encode("utf-16-le", errors="strict")) // 2
    except UnicodeError as exc:
        raise InvalidEvidence("invalid_source_encoding") from exc


class _ContextIndex:
    """Per-call index; never cached across private sources or users."""
    def __init__(self, content):
        self.starts, self.sentences = [], []
        left = 0
        ends = [match.end() for match in re.finditer(r"[.!?\n]", content)]
        if not ends or ends[-1] != len(content):
            ends.append(len(content))
        for right in ends:
            sentence = content[left:right]
            uncertain = bool(INJECTION.search(sentence) or re.search(r"[\"“”]|\bnot\s+only\b", sentence, re.I) or "?" in sentence)
            clause_starts, clauses = [], []
            clause_left = 0
            for boundary in re.finditer(r"\b(?:and|but|while|although)\b|;", sentence, re.I):
                clause_starts.append(left + clause_left)
                clauses.append(bool(NEGATION.search(sentence[clause_left:boundary.start()])))
                clause_left = boundary.end()
            clause_starts.append(left + clause_left)
            clauses.append(bool(NEGATION.search(sentence[clause_left:])))
            self.starts.append(left)
            self.sentences.append((uncertain, clause_starts, clauses))
            left = right


def _context_flags(content: str, start: int, end: int, kind: str, index=None) -> tuple[str, str]:
    index = index or _ContextIndex(content)
    sentence_id = max(0, bisect_right(index.starts, start) - 1)
    uncertain, clause_starts, clauses = index.sentences[sentence_id]
    prefix = content[max(index.starts[sentence_id], start - 100):start]
    if uncertain:
        return "uncertain", "unknown"
    if content[start:end].casefold() in {"train", "stream"} and not re.search(r"\b(?:a|the|my|our|this|that)\s+(?:\w+\s+)?$", prefix, re.I) and not re.match(r"\s+(?:station|carriage|platform)\b", content[end:end + 100], re.I):
        return "uncertain", "unknown"
    clause_id = max(0, bisect_right(clause_starts, start) - 1)
    polarity = "negated" if clauses[clause_id] else "affirmed"
    experiencer = "unknown"
    if kind == "emotion":
        experiencer = "dream_self" if SELF_EMOTION.search(prefix) else "other"
    return polarity, experiencer


def extract_local(content: str, *, language: str = "en") -> list[Observation]:
    raw_identity(content)
    if language != "en":
        return []
    index = _ContextIndex(content)
    by_key = {}
    for key, pattern in PATTERNS.items():
        kind = key.split(":", 1)[0]
        buckets = defaultdict(list)
        for match in pattern.finditer(content):
            polarity, experiencer = _context_flags(content, match.start(), match.end(), kind, index)
            rank = 0 if polarity == "affirmed" and (kind != "emotion" or experiencer == "dream_self") else 1 if polarity == "negated" else 2
            if len(buckets[rank]) < 3:
                buckets[rank].append(Observation(kind, key, match.start(), match.end(), match.group(), polarity, experiencer))
        candidates = [o for rank in sorted(buckets) for o in buckets[rank]][:3]
        if candidates:
            by_key[key] = candidates
    # Every key gets its best representative before any key gets extra spans.
    # All matches were examined, so the evidence cap does not imply missing presence.
    result = [items[0] for items in by_key.values()]
    for index in (1, 2):
        for items in by_key.values():
            if index < len(items) and len(result) < 20:
                result.append(items[index])
    return sorted(result, key=lambda o: (o.start, o.end, o.key))


def validate_candidates(content: str, payload: object, *, manual: bool = False) -> list[Observation]:
    raw_identity(content)
    if not isinstance(payload, list) or len(payload) > 20:
        raise InvalidEvidence("invalid_observation_count")
    required = {"kind", "key", "start", "end", "quote", "polarity", "experiencer"}
    result, counts = [], defaultdict(int)
    for value in payload:
        if not isinstance(value, dict) or set(value) != required:
            raise InvalidEvidence("invalid_observation_schema")
        if any(not isinstance(value[name], str) for name in required - {"start", "end"}):
            raise InvalidEvidence("invalid_observation_type")
        kind, key = value["kind"], value["key"]
        if len(key) > 240:
            raise InvalidEvidence("observation_key_too_long")
        start, end = value["start"], value["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content):
            raise InvalidEvidence("invalid_source_offset")
        if end - start > 200 or value["quote"] != content[start:end]:
            raise InvalidEvidence("source_span_mismatch")
        if not isinstance(value["polarity"], str) or not isinstance(value["experiencer"], str) or value["polarity"] not in {"affirmed", "negated", "uncertain"} or value["experiencer"] not in {"dream_self", "other", "unknown"}:
            raise InvalidEvidence("invalid_observation_flags")
        if manual and kind == "entity":
            expected_key = "entity:" + " ".join(unicodedata.normalize("NFKC", value["quote"]).casefold().split())
            if key != expected_key:
                raise InvalidEvidence("entity_source_mismatch")
        elif manual and kind == "topic" and key.startswith("custom:"):
            try:
                UUID(key[7:])
            except ValueError as exc:
                raise InvalidEvidence("invalid_custom_topic_identity") from exc
        elif key not in CODEBOOK or kind != key.split(":", 1)[0]:
            raise InvalidEvidence("unsupported_observation_key")
        elif not PATTERNS[key].fullmatch(value["quote"]):
            raise InvalidEvidence("unsupported_source_association")
        polarity, experiencer = _context_flags(content, start, end, kind)
        if not manual and (value["polarity"] != polarity or value["experiencer"] != experiencer):
            raise InvalidEvidence("source_attribution_mismatch")
        counts[key] += 1
        if counts[key] > 3:
            raise InvalidEvidence("too_many_key_spans")
        result.append(Observation(kind, key, start, end, value["quote"], value["polarity"],
                                  value["experiencer"], "user_authored" if manual else "literal_candidate"))
    return sorted(result, key=lambda o: (o.start, o.end, o.key))


def select_occurrences(sources: list[AuditSource]) -> tuple[list[AuditSource], dict[str, str]]:
    selected, duplicates, canonical = [], {}, {}
    for source in sorted(sources, key=lambda s: (_aware(s.recorded_at), s.id)):
        if not source.included or source.origin != "user_reported":
            continue
        key = duplicate_identity(source.content)
        if key in canonical and not source.independent_occurrence:
            duplicates[source.id] = canonical[key]
            continue
        canonical.setdefault(key, source.id)
        selected.append(source)
    return selected, duplicates


def window_bucket(recorded_at: datetime, as_of: datetime) -> str | None:
    date, anchor = _aware(recorded_at), _aware(as_of)
    if anchor - timedelta(days=30) <= date < anchor:
        return "recent"
    if anchor - timedelta(days=60) <= date < anchor - timedelta(days=30):
        return "previous"
    return None


def _wilson(support: int, total: int) -> tuple[float, float]:
    # A descriptive display gate, not a calibrated probability about a person.
    z, rate = 1.959963984540054, support / total
    divisor = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / divisor
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / divisor
    return center - margin, center + margin


def aggregate(sources: list[AuditSource], *, owner_id: str, as_of: datetime,
              window: str = "all", suppressed_public_keys: tuple[str, ...] = ()) -> dict:
    anchor = _aware(as_of)
    if window not in {"all", "30d", "90d"}:
        raise InvalidEvidence("invalid_window")
    if any(source.owner_id != owner_id for source in sources) or len({s.id for s in sources}) != len(sources):
        raise InvalidEvidence("source_owner_or_identity_mismatch")
    before = [s for s in sources if _aware(s.recorded_at) < anchor]
    selected, duplicates = select_occurrences(before)
    days = {"30d": 30, "90d": 90}.get(window)
    visible = [s for s in selected if days is None or s.recorded_at >= anchor - timedelta(days=days)]
    complete = [s for s in visible if s.audit_status == "complete" and s.language == "en"]
    keys, evidence, kinds, manual_keys = defaultdict(set), defaultdict(list), {}, set()
    qualified = defaultdict(list)
    for source in visible:
        if source.audit_status not in {"complete", "partial"}:
            continue
        for o in source.observations:
            validate_candidates(source.content, [{key: value for key, value in asdict(o).items() if key != "origin"}],
                                manual=o.origin == "user_authored")
            if not 0 <= o.start < o.end <= len(source.content) or source.content[o.start:o.end] != o.quote:
                raise InvalidEvidence("source_span_mismatch")
            if o.polarity != "affirmed" or (o.kind == "emotion" and o.experiencer != "dream_self"):
                qualified[o.key].append({"entry_id": source.id, "recorded_at": _aware(source.recorded_at).isoformat(),
                    "start": o.start, "end": o.end, "polarity": o.polarity, "experiencer": o.experiencer, "origin": o.origin})
                continue
            if o.key in suppressed_public_keys:
                continue
            keys[o.key].add(source.id)
            kinds[o.key] = o.kind
            if o.origin == "user_authored":
                manual_keys.add(o.key)
            evidence[o.key].append({"entry_id": source.id, "recorded_at": _aware(source.recorded_at).isoformat(),
                                    "start": o.start, "end": o.end})
    themes = []
    for key in sorted(keys):
        if len(keys[key]) < 2:
            continue
        compatible = len({s.version for s in complete}) == 1
        full_support = len(keys[key] & {s.id for s in complete})
        if key in manual_keys:
            full_support = 0
        partial_support = len(keys[key]) - full_support
        rate = full_support / len(complete) if complete and compatible and not partial_support and key not in manual_keys else None
        windows = {}
        for bucket in ("recent", "previous"):
            eligible = [s for s in selected if window_bucket(s.recorded_at, anchor) == bucket]
            audited = [s for s in eligible if s.audit_status == "complete" and s.language == "en"]
            hits = {s.id for s in audited if any(o.key == key and o.polarity == "affirmed" and
                    (o.kind != "emotion" or o.experiencer == "dream_self") for o in s.observations)}
            windows[bucket] = {"support": len(hits), "audited": len(audited), "eligible": len(eligible)}
        temporal = "limited_evidence"
        r, p = windows["recent"], windows["previous"]
        comparable = len({s.version for s in selected if window_bucket(s.recorded_at, anchor) and
                          s.audit_status == "complete"}) == 1
        if rate is not None and comparable and all(w["audited"] >= 5 and w["audited"] / w["eligible"] >= .8 for w in (r, p)):
            delta = r["support"] / r["audited"] - p["support"] / p["audited"]
            ri, pi = _wilson(r["support"], r["audited"]), _wilson(p["support"], p["audited"])
            temporal = "no_clear_change_in_records"
            if abs(delta) >= .2 and (ri[0] > pi[1] or pi[0] > ri[1]):
                temporal = "more_in_recent_records" if delta > 0 else "less_in_recent_records"
        themes.append({"key": key, "label": CODEBOOK.get(key, ("User annotation", ""))[0],
                       "kind": kinds[key], "support": len(keys[key]),
                       "complete_support": full_support, "partial_support": partial_support,
                       "denominator": len(complete) if compatible and key not in manual_keys else None, "rate": rate,
                       "provenance": "user_interpretation" if key in manual_keys else "literal_content",
                       "temporal_label": temporal, "windows": windows,
                       "qualified_evidence": sorted(qualified[key], key=lambda e: (e["recorded_at"], e["entry_id"], e["start"])),
                       "evidence": sorted(evidence[key], key=lambda e: (e["recorded_at"], e["entry_id"], e["start"]))})
    return {"status": "current" if len(complete) == len(visible) else "partial", "as_of": anchor.isoformat(),
            "window": window, "aggregator_version": AGGREGATOR_VERSION,
            "coverage": {"eligible_unique": len(visible), "audited_unique": len(complete),
                         "excluded": sum(not s.included or s.origin != "user_reported" for s in before),
                         "duplicate_entries": len(duplicates),
                         "unsupported": sum(s.language != "en" for s in visible),
                         "pending": sum(s.audit_status not in {"complete", "partial", "unsupported"} for s in visible)},
            "duplicates": duplicates, "themes": themes}


def context_dto(items: list[dict], *, revision: dict, as_of: datetime,
                enabled: bool, stale: bool = False) -> dict:
    if set(revision) != {"source", "privacy", "review"} or any(type(v) is not int or v < 0 for v in revision.values()):
        raise InvalidEvidence("invalid_revision")
    result = {"status": "stale" if stale else "ready" if enabled else "disabled", "revision": dict(revision),
              "as_of": _aware(as_of).isoformat(), "items": [], "omitted_items": 0}
    if not enabled or stale:
        return result
    budget = 6000
    for item in sorted(items, key=lambda i: (i["updated_at"], i["id"]), reverse=True):
        if not item.get("confirmed"):
            continue
        if item["kind"] not in {"source_pattern", "user_statement"} or item["provenance"] not in {"literal_content", "user_interpretation"}:
            raise InvalidEvidence("invalid_context_provenance")
        if not isinstance(item["text"], str) or not item["text"]:
            raise InvalidEvidence("invalid_context_text")
        citations = item["citations"]
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise InvalidEvidence("invalid_context_citations")
        cost = len(item["text"])
        for citation in citations:
            required = {"entry_id", "reflection_id", "recorded_at", "start", "end", "excerpt"}
            if set(citation) != required or bool(citation["entry_id"]) == bool(citation["reflection_id"]):
                raise InvalidEvidence("invalid_context_citation_identity")
            if not isinstance(citation["excerpt"], str):
                raise InvalidEvidence("invalid_context_excerpt")
            cost += len(citation["excerpt"])
        if cost > budget or len(result["items"]) == 8:
            result["omitted_items"] += 1
            continue
        budget -= cost
        result["items"].append({key: item[key] for key in
            ("id", "kind", "provenance", "text", "support", "denominator", "citations")})
    return result
