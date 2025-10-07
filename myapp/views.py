import requests
from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.generics import ListAPIView
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
from .models import Analysis, JournalEntry, CumulativeAnalysis, CustomQuestion
from .serializers import AnalysisSerializer, JournalEntrySerializer, CumulativeAnalysisSerializer, CustomQuestionSerializer
from rest_framework.permissions import AllowAny
from django.db.models import Q
from django.utils.dateparse import parse_date


class JournalEntryPagination(PageNumberPagination):
    page_size = 3
    page_size_query_param = 'page_size'
    max_page_size = 100
    page_query_param = 'page'



class JournalEntryListView(ListAPIView):
    serializer_class = JournalEntrySerializer
    pagination_class = JournalEntryPagination
    permission_classes = [AllowAny]

    def get_queryset(self):
        qs = JournalEntry.objects.all().order_by('-created_at')
        # if self.request.user.is_authenticated:
        #     qs = qs.filter(user=self.request.user)

        entries = self.request.query_params.get('entries')
        title = self.request.query_params.get('title')
        moods = self.request.query_params.get('moods')
        analysis = self.request.query_params.get('analysis')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')

        if entries:
            qs = qs.filter(content__icontains=entries)
        if title:
            qs = qs.filter(analysis__subject__icontains=title)
        if moods:
            qs = qs.filter(analysis__mood__icontains=moods)
        if analysis:
            qs = qs.filter(analysis__summary__icontains=analysis)

        if start_date:
            start_date = parse_date(start_date)
            if start_date:
                qs = qs.filter(created_at__date__gte=start_date)
        if end_date:
            end_date = parse_date(end_date)
            if end_date:
                qs = qs.filter(created_at__date__lte=end_date)

        return qs

class MoodListView(ListAPIView):
    def get(self, request):
        if request.user.is_authenticated:
            qs = Analysis.objects.filter(user=request.user)
        else:
            qs = Analysis.objects.all()
        
        moods = list(
            qs.values('mood', 'color')
            .order_by('mood', 'color')
            .distinct('mood', 'color')
        )
        
        return Response({'moods': moods})

class CumulativeAnalysisPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100
    page_query_param = 'page'

class CumulativeAnalysisListView(ListAPIView):
    serializer_class = CumulativeAnalysisSerializer
    pagination_class = CumulativeAnalysisPagination
    permission_classes = [AllowAny]

    def get_queryset(self):
        qs = CumulativeAnalysis.objects.all().order_by('-created_at')
        
        # if self.request.user and self.request.user.is_authenticated:
        #     qs = qs.filter(user=self.request.user)
    

        doctor_personality = self.request.query_params.get('doctor_personality')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        search = self.request.query_params.get('search')

        if doctor_personality:
            qs = qs.filter(doctor_personality__icontains=doctor_personality)
        
        if search:
            qs = qs.filter(analysis__icontains=search)

        if start_date:
            start_date = parse_date(start_date)
            if start_date:
                qs = qs.filter(created_at__date__gte=start_date)
        
        if end_date:
            end_date = parse_date(end_date)
            if end_date:
                qs = qs.filter(created_at__date__lte=end_date)

        return qs

class CustomQuestionPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100
    page_query_param = 'page'


class CustomQuestionListView(ListAPIView):
    serializer_class = CustomQuestionSerializer
    pagination_class = CustomQuestionPagination
    permission_classes = [AllowAny]

    def get_queryset(self):
        qs = CustomQuestion.objects.all().order_by('-created_at')
        
        # if self.request.user and self.request.user.is_authenticated:
        #     qs = qs.filter(user=self.request.user)
    
        doctor_personality = self.request.query_params.get('doctor_personality')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        search = self.request.query_params.get('search')

        if doctor_personality:
            qs = qs.filter(doctor_personality__icontains=doctor_personality)
        
        if search:
            qs = qs.filter(analysis__icontains=search)

        if start_date:
            start_date = parse_date(start_date)
            if start_date:
                qs = qs.filter(created_at__date__gte=start_date)
        
        if end_date:
            end_date = parse_date(end_date)
            if end_date:
                qs = qs.filter(created_at__date__lte=end_date)

        return qs



@api_view(['GET'])
@permission_classes([AllowAny])
def get_data(request):
    if request.user.is_authenticated:
        analyses = Analysis.objects.filter(user=request.user).order_by('created_at')
    else:
        analyses = Analysis.objects.all().order_by('created_at')
    
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
        entries = JournalEntry.objects.all().order_by('-created_at')
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
                    'http://104.236.96.193:8001/analyze',
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