from django.contrib import admin
from .models import User, JournalEntry, Analysis, CumulativeAnalysis, CustomQuestion, Settings, AnalysisFeedback

class AnalysisInline(admin.StackedInline):
    model = Analysis
    extra = 0
    max_num = 1


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("id", "email", "created_at", "updated_at")
    search_fields = ("email",)


@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email",)
    list_filter = ("created_at",)
    inlines = [AnalysisInline]


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "entry", "user", "mood", "negative", "created_at")
    search_fields = ("user__email", "mood", "subject")
    list_filter = ("negative", "created_at")

@admin.register(CumulativeAnalysis)
class CumulativeAnalysisAdmin(admin.ModelAdmin):
    list_display = ['id', 'created_at', 'doctor_personality', 'user']
    list_editable = ['doctor_personality']

@admin.register(CustomQuestion)
class CustomQuestionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'created_at', 'question', 'answer', 'doctor_personality', ]
    list_editable = ['doctor_personality']

@admin.register(Settings)
class SettingsAdmin(admin.ModelAdmin):
    pass
    
@admin.register(AnalysisFeedback)
class AnalysisFeedbackAdmin(admin.ModelAdmin):
    list_display = ('id', 'rating', 'user', 'analysis', 'cumulative_analysis', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('comment',)
