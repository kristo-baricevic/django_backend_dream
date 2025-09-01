# models.py
from uuid import uuid4
from django.db import models


class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.email


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
