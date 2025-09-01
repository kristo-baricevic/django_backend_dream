from django.contrib import admin
from .models import User, JournalEntry, Analysis

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
