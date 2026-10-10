from rest_framework import serializers
from .models import Analysis, JournalEntry, CumulativeAnalysis, CustomQuestion, Settings
from .security.settings_validation import validate_settings
from .security.freshness import result_current, source_preimage
from .security.owner import require_owner

class AnalysisSerializer(serializers.ModelSerializer):
    source_validation = serializers.SerializerMethodField()

    def get_source_validation(self, row):
        return 'historical_unvalidated' if row.is_historical else 'current'
    class Meta:
        model = Analysis
        fields = '__all__'
        read_only_fields = ['user', 'entry', 'id', 'created_at', 'updated_at']


class JournalEntrySerializer(serializers.ModelSerializer):
    analysis = serializers.SerializerMethodField()
    source_preimage = serializers.SerializerMethodField()
    analysis_status = serializers.SerializerMethodField()

    def get_source_preimage(self, row):
        return source_preimage(row)

    def get_analysis(self, row):
        if not row.analysis_id:
            return None
        request = self.context.get('request')
        if request is None:
            return None
        if not result_current(row.analysis, require_owner(request)):
            return None
        return AnalysisSerializer(row.analysis, context=self.context).data

    def get_analysis_status(self, row):
        return 'available' if self.get_analysis(row) is not None else 'unavailable'

    class Meta:
        model = JournalEntry
        fields = '__all__'
        read_only_fields = ['user', 'analysis', 'id', 'created_at', 'updated_at']

class CumulativeAnalysisSerializer(serializers.ModelSerializer):
    source_validation = serializers.SerializerMethodField()

    def get_source_validation(self, row):
        return 'historical_unvalidated' if row.is_historical else 'current'
    class Meta:
        model = CumulativeAnalysis
        fields = ['id', 'created_at', 'updated_at', 'user', 'analysis', 'doctor_personality', 'weights', 'source_validation']
        read_only_fields = ['id', 'created_at', 'updated_at', 'user']


class CustomQuestionSerializer(serializers.ModelSerializer):
    source_validation = serializers.SerializerMethodField()

    def get_source_validation(self, row):
        return 'historical_unvalidated' if row.is_historical else 'current'
    class Meta:
        model = CustomQuestion
        fields = ['id', 'created_at', 'updated_at', 'user', 'question', 'answer', 'doctor_personality', 'weights', 'source_validation']
        read_only_fields = ['id', 'created_at', 'updated_at', 'user']

class SettingsSerializer(serializers.ModelSerializer):
    def validate(self, attrs):
        validate_settings(attrs)
        return attrs

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
