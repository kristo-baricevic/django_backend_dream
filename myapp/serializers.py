from rest_framework import serializers
from .models import Analysis, JournalEntry

class AnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = '__all__'


class JournalEntrySerializer(serializers.ModelSerializer):
    analysis = AnalysisSerializer(read_only=True)

    class Meta:
        model = JournalEntry
        fields = '__all__'
