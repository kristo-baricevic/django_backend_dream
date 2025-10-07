from rest_framework import serializers
from .models import Analysis, JournalEntry, CumulativeAnalysis

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
