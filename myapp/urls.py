# myapp/urls.py
from django.urls import path
from .views import get_data, get_entries, get_entry, create_entry, update_entry, delete_entry

urlpatterns = [
    path('data/', get_data, name='get_data'),
    path('entries/', get_entries, name='get_entries'),
    path('entries/<uuid:id>/', get_entry, name='get_entry'),
    path('entries/create/', create_entry, name='create_entry'),
    path('entries/<uuid:id>/update/', update_entry, name='update_entry'),
    path('entries/<uuid:id>/delete/', delete_entry, name='delete_entry'),
]