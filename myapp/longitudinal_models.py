"""Additive, owner-scoped evidence tables. Never change or delete legacy Analysis rows."""
from uuid import uuid4
from django.db import models
from django.db.models import Q


class OwnedEvidence(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    user = models.ForeignKey("myapp.User", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class LongitudinalState(OwnedEvidence):
    user = models.OneToOneField("myapp.User", on_delete=models.CASCADE)
    enabled = models.BooleanField(default=False)
    provider_processing = models.BooleanField(default=False)
    provider_context = models.BooleanField(default=False)
    use_confirmed_context = models.BooleanField(default=False)
    source_revision = models.PositiveBigIntegerField(default=0)
    privacy_revision = models.PositiveBigIntegerField(default=0)
    review_revision = models.PositiveBigIntegerField(default=0)
    suppressed_public_keys = models.JSONField(default=list)
    source_manifest_digest = models.CharField(max_length=64, blank=True)


class EntryAuditPreference(OwnedEvidence):
    entry = models.OneToOneField("myapp.JournalEntry", on_delete=models.CASCADE)
    included = models.BooleanField(default=False)
    origin = models.CharField(max_length=16, default="unknown")
    language = models.CharField(max_length=16, default="unknown")
    independent_occurrence = models.BooleanField(default=False)


class DreamReflection(OwnedEvidence):
    entry = models.ForeignKey("myapp.JournalEntry", on_delete=models.CASCADE, null=True, blank=True)
    text = models.TextField(max_length=4000)
    revision = models.PositiveBigIntegerField(default=1)
    use_for_profile = models.BooleanField(default=False)
    use_for_chat = models.BooleanField(default=False)


class EntryAudit(OwnedEvidence):
    entry = models.ForeignKey("myapp.JournalEntry", on_delete=models.CASCADE)
    source_hash = models.CharField(max_length=64)
    source_updated_at = models.DateTimeField()
    extraction_fingerprint = models.CharField(max_length=64)
    schema_version = models.CharField(max_length=32)
    codebook_version = models.CharField(max_length=32)
    extractor_version = models.CharField(max_length=32)
    prompt_version = models.CharField(max_length=32, default="none")
    model = models.CharField(max_length=100, default="local_rules")
    consent_mode = models.CharField(max_length=16, default="local")
    language = models.CharField(max_length=16)
    status = models.CharField(max_length=20, default="queued")
    lease_token = models.UUIDField(null=True, blank=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    attempts = models.PositiveIntegerField(default=0)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "entry", "source_hash", "extraction_fingerprint"], name="unique_owned_audit_slot")]
        indexes = [models.Index(fields=["user", "status"])]


class DreamObservation(OwnedEvidence):
    audit = models.ForeignKey(EntryAudit, on_delete=models.CASCADE, related_name="observations")
    kind = models.CharField(max_length=16)
    key = models.CharField(max_length=240)
    start = models.PositiveIntegerField()
    end = models.PositiveIntegerField()
    origin = models.CharField(max_length=20, default="literal_candidate")
    polarity = models.CharField(max_length=16)
    experiencer = models.CharField(max_length=16)

    class Meta:
        constraints = [models.CheckConstraint(condition=Q(end__gt=models.F("start")), name="observation_nonempty_span")]


class ThemeRecord(OwnedEvidence):
    kind = models.CharField(max_length=16)
    key = models.CharField(max_length=240)
    namespace = models.CharField(max_length=64)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "kind", "key", "namespace"], name="unique_owned_theme")]


class ObservationSuppression(OwnedEvidence):
    """Source-bound public-codebook correction; stores no quote/private entity."""
    entry = models.ForeignKey("myapp.JournalEntry", on_delete=models.CASCADE)
    source_hash = models.CharField(max_length=64)
    key = models.CharField(max_length=64)
    start = models.PositiveIntegerField()
    end = models.PositiveIntegerField()

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "entry", "source_hash", "key", "start", "end"], name="unique_source_suppression")]


class ThemeReview(OwnedEvidence):
    theme = models.OneToOneField(ThemeRecord, on_delete=models.CASCADE, related_name="review")
    entry = models.ForeignKey("myapp.JournalEntry", on_delete=models.CASCADE, null=True, blank=True)
    reflection = models.ForeignKey(DreamReflection, on_delete=models.CASCADE, null=True, blank=True)
    status = models.CharField(max_length=16, default="unreviewed")
    label = models.CharField(max_length=200, blank=True)
    revision = models.PositiveBigIntegerField(default=1)


class ThemeMerge(OwnedEvidence):
    label = models.CharField(max_length=200)


class ThemeMergeMember(OwnedEvidence):
    merge = models.ForeignKey(ThemeMerge, on_delete=models.CASCADE, related_name="members")
    theme = models.ForeignKey(ThemeRecord, on_delete=models.CASCADE)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["merge", "theme"], name="unique_merge_member")]


class ProfileSnapshot(OwnedEvidence):
    as_of = models.DateTimeField()
    source_revision = models.PositiveBigIntegerField()
    privacy_revision = models.PositiveBigIntegerField()
    review_revision = models.PositiveBigIntegerField()
    schema_version = models.CharField(max_length=32)
    codebook_version = models.CharField(max_length=32)
    aggregator_version = models.CharField(max_length=32)
    manifest_digest = models.CharField(max_length=64)
    window = models.CharField(max_length=8, default="all")
    status = models.CharField(max_length=16)
    payload = models.JSONField(default=dict)


class SnapshotSource(OwnedEvidence):
    snapshot = models.ForeignKey(ProfileSnapshot, on_delete=models.CASCADE, related_name="sources")
    audit = models.ForeignKey(EntryAudit, on_delete=models.CASCADE, null=True, blank=True)
    reflection = models.ForeignKey(DreamReflection, on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        constraints = [models.CheckConstraint(condition=(Q(audit__isnull=False, reflection__isnull=True) |
            Q(audit__isnull=True, reflection__isnull=False)), name="snapshot_exactly_one_source")]


class AuditRun(OwnedEvidence):
    auth_user_id = models.PositiveBigIntegerField()
    login_deadline = models.DateTimeField()
    idempotency_key = models.CharField(max_length=128)
    request_fingerprint = models.CharField(max_length=64)
    manifest_digest = models.CharField(max_length=64)
    source_revision = models.PositiveBigIntegerField()
    privacy_revision = models.PositiveBigIntegerField()
    review_revision = models.PositiveBigIntegerField()
    as_of = models.DateTimeField()
    status = models.CharField(max_length=20, default="queued")
    mode = models.CharField(max_length=16, default="local")
    lease_token = models.UUIDField(null=True, blank=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    error_code = models.CharField(max_length=40, blank=True)
    max_new_sources = models.PositiveIntegerField(default=10)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "idempotency_key"], name="unique_owner_run_key")]
        indexes = [models.Index(fields=["user", "status", "created_at"])]


class AuditRunEntry(OwnedEvidence):
    run = models.ForeignKey(AuditRun, on_delete=models.CASCADE, related_name="entries")
    entry = models.ForeignKey("myapp.JournalEntry", on_delete=models.CASCADE)
    audit = models.ForeignKey(EntryAudit, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, default="queued")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["run", "entry"], name="unique_run_source")]
