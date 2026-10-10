"""Owner-scoped evidence lifecycle, with conservative privacy invalidation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from importlib import import_module
import json
from uuid import UUID, uuid4

from django.conf import settings
from django.db import transaction
from django.db.models import F
from django.utils import timezone

from myapp.models import JournalEntry
from myapp.longitudinal_models import (
    LongitudinalState, EntryAuditPreference, DreamReflection, EntryAudit, DreamObservation,
    ThemeRecord, ThemeReview, ThemeMerge, ProfileSnapshot, SnapshotSource, AuditRun, AuditRunEntry, ObservationSuppression,
)
from . import core
from .ownership import ConfigurationRequired, owner_write, security


class Conflict(ValueError):
    pass


def _state(owner, *, lock=False):
    # Only call inside owner_write: task-12 validates mapping/active state there.
    state, _ = LongitudinalState.objects.get_or_create(user_id=owner.domain_user_id)
    return LongitudinalState.objects.select_for_update().get(pk=state.pk) if lock else state


def _revision(state):
    return {"source": state.source_revision, "privacy": state.privacy_revision, "review": state.review_revision}


def _check_revision(state, expected):
    if expected is not None and expected != _revision(state):
        raise Conflict("revision_changed")


def preferences(owner):
    return owner_write(owner, lambda: _preferences(_state(owner, lock=True)))


def _preferences(state):
    return {name: getattr(state, name) for name in ("enabled", "provider_processing", "provider_context", "use_confirmed_context", "suppressed_public_keys")} | {"revision": _revision(state)}


def _purge_profiles(state, *, private_reviews=False):
    ProfileSnapshot.objects.filter(user_id=state.user_id).delete()
    if private_reviews:
        ThemeRecord.objects.filter(user_id=state.user_id).delete()
        ThemeMerge.objects.filter(user_id=state.user_id).delete()


def _scrub_runs(state):
    runs = AuditRun.objects.filter(user_id=state.user_id)
    AuditRunEntry.objects.filter(run__in=runs).delete()
    runs.update(status="cancelled", error_code="privacy_or_source_changed", request_fingerprint="",
                manifest_digest="", lease_token=None, lease_expires_at=None)


def _invalidate_locked(state, entry_id=None, *, privacy=False, clear_inclusion=True):
    state.source_revision += 1
    if privacy:
        state.privacy_revision += 1
    state.review_revision += 1
    state.source_manifest_digest = ""
    state.save()
    _purge_profiles(state, private_reviews=True)
    _scrub_runs(state)
    audits = EntryAudit.objects.filter(user_id=state.user_id)
    suppressions = ObservationSuppression.objects.filter(user_id=state.user_id)
    if entry_id is not None:
        audits = audits.filter(entry_id=entry_id)
        suppressions = suppressions.filter(entry_id=entry_id)
        DreamReflection.objects.filter(user_id=state.user_id, entry_id=entry_id).delete()
        if clear_inclusion:
            EntryAuditPreference.objects.filter(user_id=state.user_id, entry_id=entry_id).update(included=False)
    audits.delete()
    suppressions.delete()


def invalidate_source(owner, entry_id, *, expected_revision=None):
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        security().require_owned_object(JournalEntry, owner, entry_id, for_update=True)
        _invalidate_locked(state, entry_id)
    return owner_write(owner, write)


def source_reset_preview(owner):
    def read():
        state = _state(owner, lock=True)
        return {"private_reviews_reset": ThemeReview.objects.filter(user_id=state.user_id).count(),
                "enabled": state.enabled,
                "linked_reflections": DreamReflection.objects.filter(user_id=state.user_id, entry__isnull=False).count(),
                "merges_reset": ThemeMerge.objects.filter(user_id=state.user_id).count(),
                "global_reflections_preserved": DreamReflection.objects.filter(user_id=state.user_id, entry__isnull=True).count(),
                "revision": _revision(state), "warning": "Editing or excluding a source removes its annotations, corrections and linked reflections, and resets all private theme reviews and merges. Independently authored global reflections are preserved."}
    return owner_write(owner, read)


def set_preferences(owner, *, expected_revision=None, **changes):
    allowed = {"enabled", "provider_processing", "provider_context", "use_confirmed_context"}
    if not changes or set(changes) - allowed or any(type(v) is not bool for v in changes.values()):
        raise core.InvalidEvidence("invalid_consent")
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        withdrawn = any(getattr(state, key) and not value for key, value in changes.items())
        for key, value in changes.items():
            setattr(state, key, value)
        if withdrawn:
            _invalidate_locked(state, privacy=True, clear_inclusion=False)
        else:
            state.privacy_revision += 1
            state.save()
            _purge_profiles(state)
            _scrub_runs(state)
        return _preferences(state)
    return owner_write(owner, write)


def set_entry_preferences(owner, entry_id, *, expected_revision=None, **changes):
    allowed = {"included", "origin", "language", "independent_occurrence"}
    if set(changes) - allowed or any(type(changes[k]) is not bool for k in {"included", "independent_occurrence"} & set(changes)):
        raise core.InvalidEvidence("invalid_source_preferences")
    if not isinstance(changes.get("origin", "unknown"), str) or changes.get("origin", "unknown") not in {"unknown", "synthetic", "user_reported"}:
        raise core.InvalidEvidence("invalid_source_origin")
    if not isinstance(changes.get("language", "unknown"), str) or changes.get("language", "unknown") not in {"en", "unknown", "unsupported"}:
        raise core.InvalidEvidence("unsupported_language")
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        entry = security().require_owned_object(JournalEntry, owner, entry_id, for_update=True)
        preference, _ = EntryAuditPreference.objects.get_or_create(entry=entry, defaults={"user_id": state.user_id})
        if preference.user_id != state.user_id:
            raise core.InvalidEvidence("source_owner_mismatch")
        if not changes or all(getattr(preference, key) == value for key, value in changes.items()):
            return {key: getattr(preference, key) for key in allowed} | {"revision": _revision(state)}
        _invalidate_locked(state, entry.pk, clear_inclusion=False)
        for key, value in changes.items():
            setattr(preference, key, value)
        if preference.included and preference.origin != "user_reported":
            raise core.InvalidEvidence("explicit_personal_origin_required")
        preference.save()
        return {key: getattr(preference, key) for key in allowed} | {"revision": _revision(state)}
    return owner_write(owner, write)


def create_reflection(owner, text, *, entry_id=None, use_for_profile=False, use_for_chat=False):
    if not isinstance(text, str) or not text.strip() or len(text) > 4000 or type(use_for_profile) is not bool or type(use_for_chat) is not bool:
        raise core.InvalidEvidence("invalid_reflection")
    core.raw_identity(text)
    def write():
        state = _state(owner, lock=True)
        entry = security().require_owned_object(JournalEntry, owner, entry_id) if entry_id else None
        result = DreamReflection.objects.create(user_id=state.user_id, entry=entry, text=text,
            use_for_profile=use_for_profile, use_for_chat=use_for_chat)
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return result
    return owner_write(owner, write)


def edit_reflection(owner, id, *, text, expected_revision, use_for_profile=None, use_for_chat=None):
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise core.InvalidEvidence("invalid_reflection")
    core.raw_identity(text)
    if any(value is not None and type(value) is not bool for value in (use_for_profile, use_for_chat)):
        raise core.InvalidEvidence("invalid_reflection_consent")
    def write():
        state = _state(owner, lock=True)
        reflection = security().require_owned_object(DreamReflection, owner, id, for_update=True)
        if reflection.revision != expected_revision:
            raise Conflict("reflection_revision_changed")
        reflection.text = text
        if use_for_profile is not None:
            reflection.use_for_profile = use_for_profile
        if use_for_chat is not None:
            reflection.use_for_chat = use_for_chat
        reflection.revision += 1
        reflection.save()
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return reflection
    return owner_write(owner, write)


def delete_reflection(owner, id, *, expected_revision):
    def write():
        state = _state(owner, lock=True)
        reflection = security().require_owned_object(DreamReflection, owner, id, for_update=True)
        if reflection.revision != expected_revision:
            raise Conflict("reflection_revision_changed")
        reflection.delete()
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
    return owner_write(owner, write)


def save_source(owner, entry_id, *, content, audit_preferences=None, expected_revision=None):
    """Atomic content/origin write; legacy auth handlers can call the invalidation hook instead."""
    if not isinstance(content, str) or len(content) > 100000:
        raise core.InvalidEvidence("invalid_source_content")
    core.raw_identity(content)
    prefs = audit_preferences or {}
    allowed = {"included", "origin", "language", "independent_occurrence"}
    if set(prefs) - allowed or any(type(prefs[k]) is not bool for k in {"included", "independent_occurrence"} & set(prefs)):
        raise core.InvalidEvidence("invalid_source_preferences")
    if not isinstance(prefs.get("origin", "unknown"), str) or prefs.get("origin", "unknown") not in {"unknown", "user_reported", "synthetic"}:
        raise core.InvalidEvidence("invalid_source_origin")
    if not isinstance(prefs.get("language", "unknown"), str) or prefs.get("language", "unknown") not in {"unknown", "en", "unsupported"}:
        raise core.InvalidEvidence("unsupported_language")
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        entry = security().require_owned_object(JournalEntry, owner, entry_id, for_update=True)
        preference, _ = EntryAuditPreference.objects.get_or_create(entry=entry, defaults={"user_id": state.user_id})
        effective_changes = any(getattr(preference, key) != value for key, value in prefs.items())
        if content == entry.content and not effective_changes:
            return entry
        if content != entry.content or effective_changes:
            _invalidate_locked(state, entry.pk)
            preference.included = False
            for key, value in prefs.items():
                setattr(preference, key, value)
            if preference.included and preference.origin != "user_reported":
                raise core.InvalidEvidence("explicit_personal_origin_required")
            preference.save()
        entry.content = content
        entry.save(update_fields=["content", "updated_at"])
        return entry
    return owner_write(owner, write)


def invalidate_longitudinal(owner_id, reason, entry_id=None):
    """Trusted internal hook inside task-12's authenticated atomic write.

    Caller must already hold auth-user then domain-owner locks and invoke this
    before a workflow lock. This is not an authentication or public API helper.
    Bulk/admin/transfer paths must use the same contract; integration is required.
    """
    from django.db import connection
    if not connection.in_atomic_block:
        raise ConfigurationRequired("authenticated_atomic_write_required")
    if reason not in {"source_saved", "source_deleted", "source_excluded", "privacy_withdrawn"}:
        raise core.InvalidEvidence("invalid_invalidation_reason")
    if entry_id is not None and not JournalEntry.objects.filter(pk=entry_id, user_id=owner_id).exists():
        raise JournalEntry.DoesNotExist
    state, _ = LongitudinalState.objects.get_or_create(user_id=owner_id)
    state = LongitudinalState.objects.select_for_update().get(pk=state.pk)
    _invalidate_locked(state, entry_id, privacy=reason == "privacy_withdrawn")


def add_observation(owner, entry_id, payload, *, expected_revision=None):
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        entry = security().require_owned_object(JournalEntry, owner, entry_id, for_update=True)
        pref = EntryAuditPreference.objects.filter(entry=entry, user_id=state.user_id).first()
        if not state.enabled or not pref or not pref.included or pref.origin != "user_reported":
            raise core.InvalidEvidence("source_not_eligible")
        observation = core.validate_candidates(entry.content, [payload], manual=True)[0]
        audit, _ = EntryAudit.objects.get_or_create(user_id=state.user_id, entry=entry,
            source_hash=core.raw_identity(entry.content), extraction_fingerprint=_fingerprint("local", pref.language),
            defaults={"source_updated_at": entry.updated_at, "schema_version": core.SCHEMA_VERSION,
                "codebook_version": core.CODEBOOK_VERSION, "extractor_version": core.EXTRACTOR_VERSION,
                "language": pref.language, "status": "partial"})
        if audit.observations.count() >= 20 or audit.observations.filter(key=observation.key).count() >= 3:
            raise core.InvalidEvidence("observation_limit_reached")
        result = DreamObservation.objects.create(user_id=state.user_id, audit=audit,
            **{key: value for key, value in asdict(observation).items() if key != "quote"})
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return result
    return owner_write(owner, write)


def _manifest(owner):
    entries = list(security().owned_queryset(JournalEntry, owner).order_by("id"))
    prefs = {p.entry_id: p for p in EntryAuditPreference.objects.filter(user_id=owner.domain_user_id)}
    rows = []
    for entry in entries:
        pref = prefs.get(entry.pk)
        if not pref or not pref.included or pref.origin != "user_reported":
            continue
        rows.append([str(entry.pk), core.raw_identity(entry.content), entry.created_at.isoformat(), entry.updated_at.isoformat(),
                     pref.language, pref.independent_occurrence])
    return core.raw_identity(json.dumps(rows, separators=(",", ":")))


def _fingerprint(mode, language):
    return core.raw_identity("|".join([core.SCHEMA_VERSION, core.CODEBOOK_VERSION,
        core.EXTRACTOR_VERSION, mode, language]))


def _preview_locked(owner, state, ids, mode):
    if not isinstance(mode, str) or mode not in {"local", "provider"}:
        raise core.InvalidEvidence("unsupported_mode")
    if mode == "provider" and (not state.provider_processing or not getattr(settings, "LONGITUDINAL_NETWORK_ENABLED", False)):
        raise core.InvalidEvidence("provider_processing_disabled")
    if len(ids) > 100 or len({str(id) for id in ids}) != len(ids):
        raise core.InvalidEvidence("invalid_source_selection")
    sources, _ = _sources(owner)
    _, duplicates = core.select_occurrences(sources)
    details = []
    for id in ids:
        entry = security().require_owned_object(JournalEntry, owner, id)
        pref = EntryAuditPreference.objects.filter(user_id=state.user_id, entry=entry).first()
        reason = None
        if not state.enabled:
            reason = "disabled"
        elif not pref or not pref.included or pref.origin != "user_reported":
            reason = "excluded_or_unknown_origin"
        elif pref.language != "en":
            reason = "unsupported_language"
        elif str(entry.pk) in duplicates:
            reason = "duplicate_occurrence"
        elif len(entry.content) > 100000:
            reason = "source_too_long"
        audit = None if reason else EntryAudit.objects.filter(user_id=state.user_id, entry=entry, source_hash=core.raw_identity(entry.content),
            extraction_fingerprint=_fingerprint(mode, pref.language)).first()
        details.append({"entry": entry, "preference": pref, "audit": audit,
                        "skip": reason, "cached": bool(audit and (audit.status == "complete" or audit.status == "partial" and audit.attempts > 0))})
    return details


def preview(owner, ids, *, mode="local"):
    def read():
        state = _state(owner, lock=True)
        details = _preview_locked(owner, state, ids, mode)
        return {"revision": _revision(state), "mode": mode, "new_count": sum(not d["skip"] and not d["cached"] for d in details),
            "cached_count": sum(d["cached"] and not d["skip"] for d in details),
            "max_input_tokens": 0 if mode == "local" else 6000 * sum(not d["skip"] and not d["cached"] for d in details),
            "sources": [{"entry_id": str(d["entry"].pk), "skip": d["skip"], "cached": d["cached"]} for d in details]}
    return owner_write(owner, read)


def queue_audit(owner, ids, key, *, expected_revision=None, mode="local"):
    try:
        if not isinstance(key, str) or str(UUID(key)) != key:
            raise ValueError
    except ValueError:
        raise core.InvalidEvidence("invalid_run_key")
    request_hash = core.raw_identity(json.dumps({"ids": sorted(str(id) for id in ids), "mode": mode}, sort_keys=True))
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        existing = AuditRun.objects.filter(user_id=state.user_id, idempotency_key=key).first()
        if existing:
            if existing.request_fingerprint != request_hash:
                raise Conflict("idempotency_key_collision")
            return existing
        details = _preview_locked(owner, state, ids, mode)
        if sum(not d["skip"] and not d["cached"] for d in details) > 10:
            raise core.InvalidEvidence("source_budget_exceeded")
        run = AuditRun.objects.create(user_id=state.user_id, auth_user_id=owner.auth_user_id,
            login_deadline=owner.login_deadline, idempotency_key=key, request_fingerprint=request_hash,
            manifest_digest=_manifest(owner), source_revision=state.source_revision,
            privacy_revision=state.privacy_revision, review_revision=state.review_revision,
            as_of=timezone.now(), mode=mode)
        for detail in details:
            if detail["skip"]:
                continue
            entry, pref = detail["entry"], detail["preference"]
            audit, _ = EntryAudit.objects.get_or_create(user_id=state.user_id, entry=entry,
                source_hash=core.raw_identity(entry.content), extraction_fingerprint=_fingerprint(mode, pref.language),
                defaults={"source_updated_at": entry.updated_at, "schema_version": core.SCHEMA_VERSION,
                    "codebook_version": core.CODEBOOK_VERSION, "extractor_version": core.EXTRACTOR_VERSION,
                    "language": pref.language, "consent_mode": mode})
            AuditRunEntry.objects.create(user_id=state.user_id, run=run, entry=entry, audit=audit,
                status=audit.status if detail["cached"] else "queued")
        return run
    return owner_write(owner, write)


def _worker_owner(run):
    contracts = import_module("myapp.security.contracts")
    return contracts.OwnerContext(domain_user_id=run.user_id, auth_user_id=run.auth_user_id,
        is_staff=False, login_deadline=run.login_deadline)


def process_run(run_id):
    run = AuditRun.objects.get(pk=run_id)
    owner = _worker_owner(run)
    def write():
        state = _state(owner, lock=True)
        current = AuditRun.objects.select_for_update().get(pk=run_id, user_id=state.user_id)
        if current.status in {"cancelled", "stale", "complete", "partial", "failed"}:
            return current
        if not state.enabled or state.source_revision != current.source_revision or state.privacy_revision != current.privacy_revision or _manifest(owner) != current.manifest_digest:
            _cleanup_obsolete(owner, state, force=True)
            current.refresh_from_db()
            return current
        if state.review_revision != current.review_revision:
            current.entries.all().delete()
            current.status, current.error_code = "stale", "review_changed"
            current.request_fingerprint = current.manifest_digest = ""
            current.save()
            return current
        if current.mode != "local":
            current.status, current.error_code = "failed", "provider_adapter_not_enabled"
            current.save()
            return current
        current.status = "running"
        current.save()
        for item in current.entries.select_related("entry", "audit").order_by("entry_id"):
            if item.entry.user_id != state.user_id or item.audit.user_id != state.user_id:
                raise core.InvalidEvidence("source_owner_mismatch")
            pref = EntryAuditPreference.objects.get(entry=item.entry, user_id=state.user_id)
            if not pref.included or pref.origin != "user_reported" or core.raw_identity(item.entry.content) != item.audit.source_hash:
                raise Conflict("source_changed")
            if item.audit.status != "complete" and not (item.audit.status == "partial" and item.audit.attempts > 0):
                observations = core.extract_local(item.entry.content, language=pref.language)
                suppressed = set(ObservationSuppression.objects.filter(user_id=state.user_id, entry=item.entry, source_hash=item.audit.source_hash).values_list("key", "start", "end"))
                item.audit.observations.filter(origin="literal_candidate").delete()
                manual = list(item.audit.observations.filter(origin="user_authored"))
                counts = {}
                for observation in manual:
                    counts[observation.key] = counts.get(observation.key, 0) + 1
                retained = []
                for observation in observations:
                    if (observation.key, observation.start, observation.end) in suppressed:
                        continue
                    if len(manual) + len(retained) >= 20 or counts.get(observation.key, 0) >= 3:
                        continue
                    if any(o.key == observation.key and o.start == observation.start and o.end == observation.end for o in manual):
                        continue
                    counts[observation.key] = counts.get(observation.key, 0) + 1
                    retained.append(observation)
                DreamObservation.objects.bulk_create([DreamObservation(user_id=state.user_id, audit=item.audit,
                    **{key: value for key, value in asdict(o).items() if key != "quote"}) for o in retained])
                item.audit.status = "unsupported" if pref.language != "en" else "partial" if len(retained) < len(observations) else "complete"
                item.audit.attempts += 1
                item.audit.save()
            item.status = item.audit.status
            item.save()
        current.status = "partial" if current.entries.exclude(status="complete").exists() else "complete"
        current.save()
        _materialize(owner, state, current.as_of, "all")
        return current
    return owner_write(owner, write)


def _sources(owner):
    result, audits = [], []
    for entry in security().owned_queryset(JournalEntry, owner).order_by("id"):
        pref = EntryAuditPreference.objects.filter(entry=entry, user_id=owner.domain_user_id).first()
        audit = None
        if pref and pref.included and pref.origin == "user_reported":
            audit = EntryAudit.objects.filter(entry=entry, user_id=owner.domain_user_id,
                source_hash=core.raw_identity(entry.content), extraction_fingerprint=_fingerprint("local", pref.language)).first()
        observations = []
        if audit:
            audits.append(audit)
            for o in audit.observations.filter(user_id=owner.domain_user_id):
                observations.append(core.Observation(o.kind, o.key, o.start, o.end, entry.content[o.start:o.end], o.polarity, o.experiencer, o.origin))
        result.append(core.AuditSource(str(entry.pk), str(entry.user_id), entry.content, entry.created_at,
            tuple(observations), audit.status if audit else "pending", audit.extractor_version if audit else core.EXTRACTOR_VERSION,
            bool(pref and pref.included), pref.origin if pref else "unknown", pref.language if pref else "unknown",
            bool(pref and pref.independent_occurrence)))
    return result, audits


def _cleanup_obsolete(owner, state, *, force=False):
    changed = []
    for audit in EntryAudit.objects.filter(user_id=state.user_id):
        entry = security().owned_queryset(JournalEntry, owner).filter(pk=audit.entry_id).first()
        pref = EntryAuditPreference.objects.filter(user_id=state.user_id, entry_id=audit.entry_id).first()
        if not entry or not pref or not pref.included or pref.origin != "user_reported" or core.raw_identity(entry.content) != audit.source_hash:
            changed.append(audit.entry_id)
    if changed:
        for entry_id in sorted(set(changed), key=str):
            _invalidate_locked(state, entry_id)
    elif force:
        state.source_revision += 1
        state.review_revision += 1
        state.save()
        _purge_profiles(state, private_reviews=True)
        _scrub_runs(state)
        state.source_manifest_digest = ""
        state.save()
    return bool(changed) or force


def _materialize(owner, state, as_of, window):
    sources, audits = _sources(owner)
    result = core.aggregate(sources, owner_id=str(state.user_id), as_of=as_of, window=window,
                            suppressed_public_keys=tuple(state.suppressed_public_keys))
    result["revision"] = _revision(state)
    for theme in result["themes"]:
        record, _ = ThemeRecord.objects.get_or_create(user_id=state.user_id, kind=theme["kind"], key=theme["key"], namespace=core.CODEBOOK_VERSION)
        review = ThemeReview.objects.filter(theme=record, user_id=state.user_id).first()
        theme["id"] = str(record.pk)
        theme["review"] = review.status if review else "unreviewed"
        if review and review.label:
            theme["provenance"] = "user_interpretation"
        if review and review.label:
            theme["label"] = review.label
    result["themes"] = [t for t in result["themes"] if t["review"] != "rejected"]
    digest = _manifest(owner)
    state.source_manifest_digest = digest
    state.save(update_fields=["source_manifest_digest", "updated_at"])
    snapshot = ProfileSnapshot.objects.create(user_id=state.user_id, as_of=as_of,
        source_revision=state.source_revision, privacy_revision=state.privacy_revision, review_revision=state.review_revision,
        schema_version=core.SCHEMA_VERSION, codebook_version=core.CODEBOOK_VERSION, aggregator_version=core.AGGREGATOR_VERSION,
        manifest_digest=digest, window=window, status=result["status"], payload=result)
    SnapshotSource.objects.bulk_create([SnapshotSource(user_id=state.user_id, snapshot=snapshot, audit=a) for a in audits])
    # Bound retained reproducibility metadata; source/privacy invalidation deletes all.
    retain = list(ProfileSnapshot.objects.filter(user_id=state.user_id, window=window).order_by("-created_at").values_list("pk", flat=True)[:5])
    ProfileSnapshot.objects.filter(user_id=state.user_id, window=window).exclude(pk__in=retain).delete()
    return result


def profile(owner, *, window="all"):
    def write():
        locked = _state(owner, lock=True)
        if not locked.enabled:
            return {"status": "disabled", "revision": _revision(locked), "themes": []}
        current_digest = _manifest(owner)
        if _cleanup_obsolete(owner, locked, force=bool(locked.source_manifest_digest and locked.source_manifest_digest != current_digest)):
            return {"status": "stale", "revision": _revision(locked), "themes": []}
        sources, _ = _sources(owner)
        if not any(s.included and s.origin == "user_reported" for s in sources):
            return {"status": "empty", "revision": _revision(locked), "themes": []}
        snapshots = ProfileSnapshot.objects.filter(user_id=locked.user_id, window=window).order_by("-created_at")
        snapshot = snapshots.first()
        if snapshot and (snapshot.manifest_digest != _manifest(owner) or
            {"source": snapshot.source_revision, "privacy": snapshot.privacy_revision, "review": snapshot.review_revision} != _revision(locked)):
            # An unhooked source mutation must not leave private stale audit keys
            # at rest. Preserve unchanged sources; exclude changed sources.
            _cleanup_obsolete(owner, locked, force=True)
            return {"status": "stale", "revision": _revision(locked), "themes": []}
        result = _materialize(owner, locked, timezone.now(), window)
        # User-authored concerns are displayed separately, never treated as dream events.
        result["reflections"] = [reflection_dto(r) for r in security().owned_queryset(DreamReflection, owner).filter(use_for_profile=True).order_by("-created_at")[:50]]
        result["merges"] = [{"id": str(m.pk), "label": m.label, "theme_ids": [str(t) for t in m.members.filter(user_id=locked.user_id).values_list("theme_id", flat=True)]} for m in security().owned_queryset(ThemeMerge, owner).order_by("created_at")[:50]]
        decisions = _private_decisions(owner)
        result["rejected_decisions"] = [_decision_dto(r) for r in decisions[:100]]
        result["rejected_decisions_next_offset"] = 100 if decisions.count() > 100 else None
        for theme in result["themes"]:
            for evidence in theme["evidence"] + theme.get("qualified_evidence", []):
                entry = security().require_owned_object(JournalEntry, owner, evidence["entry_id"])
                evidence["excerpt"] = entry.content[evidence["start"]:evidence["end"]]
        return result
    return owner_write(owner, write)


def review_theme(owner, id, *, status, label=None, expected_revision=None):
    if not isinstance(status, str) or status not in {"confirmed", "rejected", "unreviewed"} or label is not None and (not isinstance(label, str) or len(label) > 200):
        raise core.InvalidEvidence("invalid_review")
    if label is not None:
        core.raw_identity(label)
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        theme = security().require_owned_object(ThemeRecord, owner, id, for_update=True)
        review, _ = ThemeReview.objects.get_or_create(user_id=state.user_id, theme=theme)
        review.status = status
        if label is not None:
            review.label = label
        review.revision += 1
        review.save()
        if theme.key in core.CODEBOOK:
            keys = set(state.suppressed_public_keys)
            keys.add(theme.key) if status == "rejected" else keys.discard(theme.key)
            state.suppressed_public_keys = sorted(keys)
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return {"id": str(theme.pk), "status": review.status, "label": review.label, "revision": _revision(state)}
    return owner_write(owner, write)


def confirmed_context(owner):
    def read():
        state = _state(owner, lock=True)
        # Fail closed until auth + task-13's privacy/retention integration is verified.
        return core.context_dto([], revision=_revision(state), as_of=timezone.now(), enabled=False)
    return owner_write(owner, read)


def reflection_dto(reflection):
    return {"id": str(reflection.pk), "entry_id": str(reflection.entry_id) if reflection.entry_id else None,
            "text": reflection.text, "revision": reflection.revision, "use_for_profile": reflection.use_for_profile,
            "use_for_chat": reflection.use_for_chat, "created_at": reflection.created_at.isoformat(),
            "provenance": "user_reflection"}


def sources(owner, *, offset=0):
    def read():
        state = _state(owner, lock=True)
        entries = security().owned_queryset(JournalEntry, owner).order_by("-created_at", "id")
        rows = []
        for entry in entries[offset:offset + 100]:
            pref = EntryAuditPreference.objects.filter(user_id=state.user_id, entry=entry).first()
            rows.append({"id": str(entry.pk), "recorded_at": entry.created_at.isoformat(),
                "included": bool(pref and pref.included), "origin": pref.origin if pref else "unknown",
                "language": pref.language if pref else "unknown", "independent_occurrence": bool(pref and pref.independent_occurrence)})
        return {"sources": rows, "next_offset": offset + 100 if entries.count() > offset + 100 else None, "revision": _revision(state)}
    return owner_write(owner, read)


def entry_audit(owner, entry_id):
    def read():
        state = _state(owner, lock=True)
        entry = security().require_owned_object(JournalEntry, owner, entry_id)
        _cleanup_obsolete(owner, state)
        pref = EntryAuditPreference.objects.filter(user_id=state.user_id, entry=entry).first()
        audit = None
        if state.enabled and pref and pref.included and pref.origin == "user_reported":
            audit = EntryAudit.objects.filter(user_id=state.user_id, entry=entry, source_hash=core.raw_identity(entry.content), extraction_fingerprint=_fingerprint("local", pref.language)).first()
        observations = []
        if audit:
            for o in audit.observations.filter(user_id=state.user_id).order_by("start", "id"):
                observations.append({"id": str(o.pk), "kind": o.kind, "key": o.key, "start": o.start, "end": o.end,
                    "quote": entry.content[o.start:o.end], "polarity": o.polarity, "experiencer": o.experiencer, "origin": o.origin})
        return {"entry_id": str(entry.pk), "content": entry.content if state.enabled and pref and pref.included and pref.origin == "user_reported" else None,
            "status": audit.status if audit else "not_audited", "revision": _revision(state),
            "recorded_at": entry.created_at.isoformat(), "observations": observations,
            "provenance": {"extractor": audit.extractor_version, "codebook": audit.codebook_version, "schema": audit.schema_version, "model": audit.model} if audit else None}
    return owner_write(owner, read)


def delete_observation(owner, entry_id, id, *, expected_revision):
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        entry = security().require_owned_object(JournalEntry, owner, entry_id, for_update=True)
        observation = security().require_owned_object(DreamObservation, owner, id, for_update=True)
        if observation.audit.entry_id != entry.pk or observation.audit.user_id != state.user_id:
            raise DreamObservation.DoesNotExist
        # Removal is a user correction. Mark partial so absence is not inferred
        # from a codebook scan that the user has overridden.
        audit = observation.audit
        if observation.origin == "literal_candidate" and observation.key in core.CODEBOOK:
            ObservationSuppression.objects.get_or_create(user_id=state.user_id, entry=entry, source_hash=audit.source_hash,
                key=observation.key, start=observation.start, end=observation.end)
        observation.delete()
        if observation.key not in core.CODEBOOK and not DreamObservation.objects.filter(user_id=state.user_id, key=observation.key).exists():
            records = ThemeRecord.objects.filter(user_id=state.user_id, key=observation.key)
            ThemeMerge.objects.filter(user_id=state.user_id, members__theme__in=records).delete()
            records.delete()
        audit.status = "partial"
        audit.attempts = 0
        audit.save()
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return {"revision": _revision(state)}
    return owner_write(owner, write)


def list_reflections(owner, *, offset=0):
    def read():
        state = _state(owner, lock=True)
        rows = security().owned_queryset(DreamReflection, owner).order_by("-created_at", "id")
        return {"reflections": [reflection_dto(r) for r in rows[offset:offset + 100]], "next_offset": offset + 100 if rows.count() > offset + 100 else None, "revision": _revision(state)}
    return owner_write(owner, read)


def _private_decisions(owner):
    return security().owned_queryset(ThemeReview, owner).filter(status="rejected", theme__user_id=owner.domain_user_id).exclude(theme__key__in=core.CODEBOOK).select_related("theme").order_by("created_at", "id")


def _decision_dto(review):
    return {"id": str(review.theme_id), "label": review.label or "Rejected user annotation", "key": review.theme.key}


def rejected_decisions(owner, *, offset=0):
    def read():
        state = _state(owner, lock=True)
        if not state.enabled:
            return {"decisions": [], "next_offset": None, "revision": _revision(state)}
        _cleanup_obsolete(owner, state, force=bool(state.source_manifest_digest and state.source_manifest_digest != _manifest(owner)))
        rows = _private_decisions(owner)
        return {"decisions": [_decision_dto(r) for r in rows[offset:offset + 100]], "next_offset": offset + 100 if rows.count() > offset + 100 else None, "revision": _revision(state)}
    return owner_write(owner, read)


def undo_public_rejection(owner, key, *, expected_revision):
    if not isinstance(key, str) or key not in core.CODEBOOK:
        raise core.InvalidEvidence("invalid_public_theme")
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        state.suppressed_public_keys = [k for k in state.suppressed_public_keys if k != key]
        ThemeReview.objects.filter(user_id=state.user_id, theme__user_id=state.user_id, theme__key=key).update(status='unreviewed', revision=F('revision') + 1, updated_at=timezone.now())
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return {"revision": _revision(state)}
    return owner_write(owner, write)


def merge_themes(owner, ids, label, *, expected_revision):
    from myapp.longitudinal_models import ThemeMergeMember
    if not isinstance(label, str) or not label.strip() or len(label) > 200 or not isinstance(ids, list) or not 2 <= len(ids) <= 8 or len(set(ids)) != len(ids):
        raise core.InvalidEvidence("invalid_merge")
    core.raw_identity(label)
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        themes = [security().require_owned_object(ThemeRecord, owner, id, for_update=True) for id in ids]
        if security().owned_queryset(ThemeMerge, owner).count() >= 50:
            raise core.InvalidEvidence("merge_limit_reached")
        result = ThemeMerge.objects.create(user_id=state.user_id, label=label)
        ThemeMergeMember.objects.bulk_create([ThemeMergeMember(user_id=state.user_id, merge=result, theme=t) for t in themes])
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return {"id": str(result.pk), "label": label, "theme_ids": ids, "revision": _revision(state)}
    return owner_write(owner, write)


def delete_merge(owner, id, *, expected_revision):
    def write():
        state = _state(owner, lock=True)
        _check_revision(state, expected_revision)
        security().require_owned_object(ThemeMerge, owner, id, for_update=True).delete()
        state.review_revision += 1
        state.save()
        _purge_profiles(state)
        return {"revision": _revision(state)}
    return owner_write(owner, write)


def run_status(owner, id, *, cancel=False, expected_revision=None):
    def read():
        state = _state(owner, lock=True)
        if cancel:
            _check_revision(state, expected_revision)
        run = security().require_owned_object(AuditRun, owner, id, for_update=cancel)
        if cancel and run.status == "queued":
            run.status = "cancelled"
            run.request_fingerprint = run.manifest_digest = ""
            run.save()
            run.entries.all().delete()
        return {"id": str(run.pk), "status": run.status, "error_code": run.error_code, "mode": run.mode,
            "source_count": run.entries.count(), "revision": _revision(state), "created_at": run.created_at.isoformat()}
    return owner_write(owner, read)
