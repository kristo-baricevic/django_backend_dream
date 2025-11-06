# myapp/urls.py
from django.urls import path
from .views import get_data, get_entries, submit_feedback, get_feedback_stats, get_user_preferences, get_user_recommendations, get_entry, create_entry, update_entry, delete_entry, get_workflow_execution
from . import views

urlpatterns = [
    path('data/', get_data, name='get_data'),
    # path('entries/', get_entries, name='get_entries'),
    path('entries/', views.JournalEntryListView.as_view(), name='get_entries'),
    path('symbols/', views.SymbolsListView.as_view(), name='get-symbols'),
    path('entries/<uuid:id>/', get_entry, name='get_entry'),
    path('entries/create/', create_entry, name='create_entry'),
    path('entries/<uuid:id>/update/', update_entry, name='update_entry'),
    path('entries/<uuid:id>/delete/', delete_entry, name='delete_entry'),
    path('entries/moods/', views.MoodListView.as_view(), name='get_moods'),
    path('cumulative-analyses/', views.CumulativeAnalysisListView.as_view(), name='cumulative-analysis-list'),
    path('custom-questions/', views.CustomQuestionListView.as_view(), name='custom-question-list'),
    path('workflows/<uuid:workflow_id>/', get_workflow_execution, name='get_workflow_execution'),
    path('settings/', views.SettingsListView.as_view(), name='settings-list'),
    path('settings/update/', views.SettingsUpdateView.as_view(), name='settings-update'), 
    path('feedback/submit/', submit_feedback, name='submit_feedback'),
    path('feedback/stats/', get_feedback_stats, name='feedback_stats'),
    path('feedback/preferences/', get_user_preferences),
    path('feedback/recommendations/', get_user_recommendations),
]