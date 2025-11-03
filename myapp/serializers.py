from rest_framework import serializers
from .models import Analysis, JournalEntry, CumulativeAnalysis, CustomQuestion, Settings

class AnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = '__all__'


class JournalEntrySerializer(serializers.ModelSerializer):
    analysis = AnalysisSerializer(read_only=True)

    class Meta:
        model = JournalEntry
        fields = '__all__'

class CumulativeAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = CumulativeAnalysis
        fields = ['id', 'created_at', 'updated_at', 'user', 'analysis', 'doctor_personality', 'weights']
        read_only_fields = ['id', 'created_at', 'updated_at']


class CustomQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomQuestion
        fields = ['id', 'created_at', 'updated_at', 'user', 'question', 'answer', 'doctor_personality', 'weights']
        read_only_fields = ['id', 'created_at', 'updated_at']

class SettingsSerializer(serializers.ModelSerializer):
    doctorPersonality = serializers.CharField(source='doctor_personality')
    doctorImage = serializers.CharField(source='doctor_image')
    personality = serializers.CharField(
        source='personality_type',
        allow_blank=True,
        required=False
    )
    medicalHistory = serializers.JSONField(source='medical_history')
    influence = serializers.JSONField(source='weights')

    class Meta:
        model = Settings
        fields = [
            'id', 'doctorPersonality', 'doctorImage', 'personality',
            'occupation', 'medicalHistory', 'influence', 'astrology'
        ]
