import requests
from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import Analysis, JournalEntry
from .serializers import AnalysisSerializer, JournalEntrySerializer
from rest_framework.permissions import AllowAny

@api_view(['GET'])
@permission_classes([AllowAny])
def get_data(request):
    if request.user.is_authenticated:
        analyses = Analysis.objects.filter(user=request.user).order_by('created_at')
    else:
        # For anonymous users, show all analyses or return empty
        analyses = Analysis.objects.all().order_by('created_at')  # or Analysis.objects.none()
    
    serializer = AnalysisSerializer(analyses, many=True)
    scores = [a.sentiment_score for a in analyses]
    avg = round(sum(scores) / len(scores)) if scores else 0
    return Response({"analyses": serializer.data, "avg": avg})

@api_view(['GET'])
@permission_classes([AllowAny])
def get_entries(request):
    if request.user.is_authenticated:
        entries = JournalEntry.objects.filter(user=request.user).order_by('-created_at')
    else:
        entries = JournalEntry.objects.all().order_by('-created_at')  # show all if anonymous
    serializer = JournalEntrySerializer(entries, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_entry(request, id):
    try:
        if request.user.is_authenticated:
            entry = JournalEntry.objects.get(id=id, user=request.user)
        else:
            # For anonymous users, get any entry by ID
            entry = JournalEntry.objects.get(id=id)
    except JournalEntry.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    
    serializer = JournalEntrySerializer(entry)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([AllowAny])
def create_entry(request):
    """Create a new journal entry."""
    data = request.data.copy()
    
    if request.user.is_authenticated:
        data['user'] = request.user.id
    # Don't set user for anonymous requests
    
    serializer = JournalEntrySerializer(data=data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['PUT', 'PATCH'])
@permission_classes([AllowAny])
def update_entry(request, id):
    """Update an existing journal entry."""
    try:
        if request.user.is_authenticated:
            entry = JournalEntry.objects.get(id=id, user=request.user)
        else:
            # Allow anonymous users to edit any entry (security risk!)
            entry = JournalEntry.objects.get(id=id)
    except JournalEntry.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    
    data = request.data.copy()
    if request.user.is_authenticated:
        data['user'] = request.user.id
    # Don't set user for anonymous requests
    
    partial = request.method == 'PATCH'
    serializer = JournalEntrySerializer(entry, data=data, partial=partial)
    
    if serializer.is_valid():
        updated_entry = serializer.save()
        
        # Check if analysis exists for this entry
        try:
            analysis = Analysis.objects.get(entry=updated_entry)
        except Analysis.DoesNotExist:
            # No analysis exists, create one by calling FastAPI
            try:
                personality_type = request.data.get('personality', 'empathetic')
                
                # Call FastAPI analyze endpoint
                fastapi_response = requests.post(
                    'http://localhost:8001/analyze',
                    json={
                        'content': updated_entry.content,
                        'personality_type': personality_type
                    },
                    timeout=30
                )
                
                if fastapi_response.status_code == 200:
                    analysis_data = fastapi_response.json()
                    
                    # Create Analysis object in Django
                    new_analysis = Analysis.objects.create(
                        entry=updated_entry,
                        user=updated_entry.user if updated_entry.user else None,
                        mood=analysis_data['mood'],
                        summary=analysis_data['summary'],
                        color=analysis_data['color'],
                        interpretation=analysis_data['interpretation'],
                        negative=analysis_data['negative'],
                        subject=analysis_data['subject'],
                        sentiment_score=analysis_data['sentiment_score']
                    )

                    # Link the entry to the analysis
                    updated_entry.analysis = new_analysis
                    updated_entry.save()
                    
                    # Re-serialize the entry to include the new analysis
                    updated_serializer = JournalEntrySerializer(updated_entry)
                    return Response(updated_serializer.data)
                    
                else:
                    print(f"FastAPI analysis failed: {fastapi_response.text}")
                    
            except requests.RequestException as e:
                print(f"Failed to connect to FastAPI service: {e}")
            except Exception as e:
                print(f"Error creating analysis: {e}")
        
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['DELETE'])
@permission_classes([AllowAny])
def delete_entry(request, id):
    """Delete a journal entry."""
    try:
        if request.user.is_authenticated:
            entry = JournalEntry.objects.get(id=id, user=request.user)
        else:
            # For anonymous users, get any entry by ID
            entry = JournalEntry.objects.get(id=id)
    except JournalEntry.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    
    entry.delete()
    return Response({"detail": "Entry deleted successfully."}, status=status.HTTP_204_NO_CONTENT)