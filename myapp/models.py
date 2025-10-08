# models.py
from uuid import uuid4
from django.db import models
from django.contrib.auth.models import User
import uuid

class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    email = models.EmailField(unique=True)

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
    workflow_type = models.CharField(max_length=50)  # 'cumulative_analysis' or 'custom_question'
    routine_name = models.CharField(max_length=200)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    
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
    
    source = models.CharField(max_length=50)  # 'jungian_symbols', 'natal_chart', etc.
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

    class Meta:
        indexes = [models.Index(fields=['user'])]

class CumulativeAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey('User', related_name='cumulative_analyses', on_delete=models.CASCADE, null=True, blank=True)
    analysis = models.TextField()
    doctor_personality = models.TextField(blank=True, default='')
   
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


