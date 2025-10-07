from rest_framework import serializers
from .models import Analysis, JournalEntry, CumulativeAnalysis, CustomQuestion

class AnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = '__all__'


class JournalEntrySerializer(serializers.ModelSerializer):
    analysis = AnalysisSerializer(read_only=True)

    class Meta:
        model = JournalEntry
        fields = '__all__'

# First, create a serializer in serializers.py

class CumulativeAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = CumulativeAnalysis
        fields = ['id', 'created_at', 'updated_at', 'user', 'analysis', 'doctor_personality']
        read_only_fields = ['id', 'created_at', 'updated_at']


class CustomQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomQuestion
        fields = ['id', 'created_at', 'updated_at', 'user', 'question', 'answer', 'doctor_personality']
        read_only_fields = ['id', 'created_at', 'updated_at']

