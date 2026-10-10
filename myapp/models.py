# models.py
from uuid import uuid4
from django.db import models
from django.conf import settings as django_settings
import uuid

class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    email = models.EmailField(unique=True)
    auth_user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                     on_delete=models.PROTECT, related_name='dream_owner')
    active_settings = models.ForeignKey('Settings', null=True, blank=True,
                                        on_delete=models.SET_NULL, related_name='+')
    is_local_fixture = models.BooleanField(default=False, editable=False)
    import_ready = models.BooleanField(default=True, editable=False)

    def __str__(self):
        return self.email


class WorkflowExecution(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # What workflow is running
    workflow_type = models.CharField(max_length=50) 
    routine_name = models.CharField(max_length=200)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    analysis_id = models.CharField(max_length=255, null=True, blank=True)
    accepted_auth_user_id = models.PositiveBigIntegerField(null=True, blank=True, editable=False)
    is_historical = models.BooleanField(default=False, editable=False)

    # Results
    final_result = models.TextField(null=True, blank=True)
    overall_confidence = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    # Metadata
    total_citations = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-start_time']

class WorkflowStep(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    execution = models.ForeignKey(WorkflowExecution, on_delete=models.CASCADE, related_name='steps')
    
    # Step info
    step_number = models.IntegerField()
    name = models.CharField(max_length=200)
    step_type = models.CharField(max_length=50)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)
    
    # Data
    input_data = models.JSONField(null=True, blank=True)
    output_data = models.JSONField(null=True, blank=True)
    
    # Observability
    confidence = models.FloatField(null=True, blank=True)
    reasoning = models.TextField(null=True, blank=True)
    model_used = models.CharField(max_length=50, null=True, blank=True)
    tokens_used = models.IntegerField(null=True, blank=True)
    error = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['step_number']

class StepCitation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    step = models.ForeignKey(WorkflowStep, on_delete=models.CASCADE, related_name='citations')
    source = models.CharField(max_length=50) 
    content = models.TextField()
    confidence = models.FloatField()
    reference = models.CharField(max_length=500, null=True, blank=True)
    
    class Meta:
        ordering = ['-confidence']


class JournalEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    user = models.ForeignKey('User', related_name='entries', on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    content = models.TextField()
    analysis = models.OneToOneField('Analysis', related_name='journal_entry', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f'{self.user.email if self.user else "Anonymous"} • {self.created_at:%Y-%m-%d}'

class Analysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    entry = models.OneToOneField('JournalEntry', related_name='entry_analysis', on_delete=models.CASCADE) 
    user = models.ForeignKey('User', related_name='analyses', on_delete=models.CASCADE, null=True, blank=True)
    mood = models.CharField(max_length=255)
    summary = models.TextField()
    color = models.CharField(max_length=64)
    interpretation = models.TextField()
    negative = models.BooleanField(default=False)
    subject = models.CharField(max_length=255)
    sentiment_score = models.FloatField(default=0)
    doctor_personality = models.TextField(blank=True, default='')
    weights = models.JSONField(default=dict)
    symbols = models.JSONField(default=list)
    
    class Meta:
        indexes = [models.Index(fields=['user'])]

class CumulativeAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey('User', related_name='cumulative_analyses', on_delete=models.CASCADE, null=True, blank=True)
    analysis = models.TextField()
    doctor_personality = models.TextField(blank=True, default='')
    weights = models.JSONField(default=dict)
    workflow_execution = models.ForeignKey(
            WorkflowExecution, 
            on_delete=models.SET_NULL, 
            null=True, 
            blank=True,
            related_name='cumulative_analyses'
        )

    class Meta:
        indexes = [models.Index(fields=['user'])]

class CustomQuestion(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey('User', related_name='custom_question', on_delete=models.CASCADE, null=True, blank=True)
    question = models.TextField()
    weights = models.JSONField(default=dict)
    answer = models.TextField()
    doctor_personality = models.TextField(blank=True, default='')

    workflow_execution = models.ForeignKey(
        WorkflowExecution, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='custom_question'
    )

    class Meta:
        indexes = [models.Index(fields=['user'])]

def default_doctor_weights():
    return {
        "theory": 0.0,
        "astrology": 0.0,
        "personality": 0.0,
        "medicalHistory": 0.0
    }


def default_settings_weights():
    return {
        "astrology": 0.15,
        "personality": 0.15,
        "medicalHistory": 0.10,
        "theory": 0.70
    }


class DoctorProfile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    archetype = models.CharField(max_length=100, blank=True)
    tone = models.CharField(max_length=200, blank=True)
    background = models.TextField(blank=True)
    personality_style = models.TextField(blank=True)
    prompt_style = models.TextField(blank=True)
    full_prompt = models.TextField(blank=True)
    weights = models.JSONField(default=default_doctor_weights)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Settings(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    user = models.ForeignKey('User', related_name='settings', on_delete=models.CASCADE, null=True, blank=True)
    doctor = models.ForeignKey('DoctorProfile', related_name='settings', on_delete=models.SET_NULL, null=True, blank=True)
    doctor_personality = models.TextField(blank=True, default='')
    doctor_image = models.TextField(blank=True, default='')
    personality_type = models.TextField(blank=True, default='')
    occupation = models.TextField(blank=True, default='')
    weights = models.JSONField(default=default_settings_weights)
    doctor_influence = models.FloatField(default=0.7)
    astrology = models.JSONField(default=dict)
    medical_history = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=['user'])]

    def __str__(self):
        return f"Settings for {self.user}"

class AnalysisFeedback(models.Model):
    """Track user feedback on analysis quality"""
    FEEDBACK_CHOICES = [
        ('good', 'Good'),
        ('bad', 'Bad'),
        ('helpful', 'Helpful'),
        ('unhelpful', 'Unhelpful'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    analysis = models.ForeignKey('Analysis', on_delete=models.CASCADE, related_name='feedback', null=True, blank=True)
    user = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Basic feedback
    rating = models.CharField(max_length=10, choices=FEEDBACK_CHOICES)
    
    # Optional detailed feedback
    comment = models.TextField(blank=True, null=True)
    
    # What aspects were good/bad
    accuracy = models.BooleanField(null=True, blank=True)  # Was it accurate?
    relevance = models.BooleanField(null=True, blank=True)  # Was it relevant?
    helpful = models.BooleanField(null=True, blank=True)  # Was it helpful?
    
    # For custom questions
    custom_question = models.ForeignKey('CustomQuestion', on_delete=models.CASCADE, null=True, blank=True, related_name='feedback')
    
    # For cumulative analysis
    cumulative_analysis = models.ForeignKey('CumulativeAnalysis', on_delete=models.CASCADE, related_name='feedback', null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, blank=True)  # Track anonymous users
    
    # What the user was trying to improve
    improvement_areas = models.JSONField(default=list, blank=True)  # ["interpretation", "symbols", "tone", etc]
    
    class Meta:
        indexes = [
            models.Index(fields=['analysis']),
            models.Index(fields=['rating']),
            models.Index(fields=['created_at']),
        ]
        unique_together = [['analysis', 'user']]  # One feedback per user per analysis


class LoginAttemptBucket(models.Model):
    key = models.CharField(max_length=64, primary_key=True)
    started_at = models.DateTimeField()
    attempts = models.PositiveIntegerField(default=0)
    failures = models.PositiveIntegerField(default=0)
    blocked_until = models.DateTimeField(null=True)
    reservations = models.JSONField(default=dict)


class DataCopyRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    key = models.CharField(max_length=100, unique=True)
    owner = models.ForeignKey(User, on_delete=models.PROTECT)
    source_identity = models.JSONField(default=dict)
    manifest_digest = models.CharField(max_length=64)
    state = models.CharField(max_length=32, default='staged')
    settings_preimage = models.UUIDField(null=True)
    active_settings_copy = models.UUIDField(null=True)
    counts = models.JSONField(default=dict)
    files = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)


class DataCopyRecord(models.Model):
    run = models.ForeignKey(DataCopyRun, on_delete=models.CASCADE, related_name='records')
    model_label = models.CharField(max_length=100)
    source_id = models.UUIDField()
    destination_id = models.UUIDField()
    digest = models.CharField(max_length=64)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['run', 'model_label', 'source_id'],
                                              name='unique_copy_source_record')]


# Parent integrates this explicit bottom import after the auth-owned model changes.
from .longitudinal_models import (  # noqa: F401, E402
    ObservationSuppression,
    LongitudinalState, EntryAuditPreference, DreamReflection, EntryAudit,
    DreamObservation, ThemeRecord, ThemeReview, ThemeMerge, ThemeMergeMember,
    ProfileSnapshot, SnapshotSource, AuditRun, AuditRunEntry,
)
