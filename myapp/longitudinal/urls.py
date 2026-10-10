from django.urls import path
from . import views

urlpatterns = [
    path("longitudinal/preferences/", views.Preferences.as_view()),
    path("longitudinal/profile/", views.Profile.as_view()),
    path("longitudinal/context/", views.Context.as_view()),
    path("longitudinal/source-reset-preview/", views.ResetPreview.as_view()),
    path("longitudinal/sources/", views.Sources.as_view()),
    path("longitudinal/audits/preview/", views.Preview.as_view()),
    path("longitudinal/audits/", views.Runs.as_view()),
    path("longitudinal/audits/<uuid:id>/", views.Run.as_view()),
    path("longitudinal/themes/<uuid:id>/", views.Review.as_view()),
    path("longitudinal/rejections/undo/", views.UndoPublicRejection.as_view()),
    path("longitudinal/decisions/", views.Decisions.as_view()),
    path("longitudinal/reflections/", views.Reflections.as_view()),
    path("longitudinal/reflections/<uuid:id>/", views.Reflection.as_view()),
    path("longitudinal/merges/", views.Merges.as_view()),
    path("longitudinal/merges/<uuid:id>/", views.Merge.as_view()),
    path("entries/<uuid:entry_id>/audit-preferences/", views.SourcePreferences.as_view()),
    path("entries/<uuid:entry_id>/audit/", views.EntryAudit.as_view()),
    path("entries/<uuid:entry_id>/observations/", views.Observations.as_view()),
    path("entries/<uuid:entry_id>/observations/<uuid:id>/", views.Observation.as_view()),
]
