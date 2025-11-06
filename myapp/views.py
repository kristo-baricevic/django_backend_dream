import requests
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.generics import ListAPIView, UpdateAPIView
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
from .models import Analysis, JournalEntry, CumulativeAnalysis, CustomQuestion, WorkflowExecution, Settings, AnalysisFeedback
from .serializers import AnalysisSerializer, JournalEntrySerializer, CumulativeAnalysisSerializer, CustomQuestionSerializer, SettingsSerializer
from rest_framework.permissions import AllowAny
from django.db.models import Q
from django.utils.dateparse import parse_date
# views.py
from rest_framework.views import APIView
from django.db import connection
from collections import Counter


class SettingsListView(ListAPIView):
    queryset = Settings.objects.all()
    serializer_class = SettingsSerializer

    def get_queryset(self):
        # For demo, just return all settings or first one
        return Settings.objects.all()

class SettingsUpdateView(UpdateAPIView):
    serializer_class = SettingsSerializer

    def get_object(self):
        settings, _ = Settings.objects.get_or_create(pk=1)
        return settings

    def get_serializer(self, *args, **kwargs):
        kwargs['partial'] = True
        return super().get_serializer(*args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)


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

class SymbolsListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        table = Analysis._meta.db_table
        with connection.cursor() as cur:
            cur.execute(f"""
                WITH s_text AS (
                  SELECT jsonb_array_elements_text(symbols) AS sym
                  FROM {table}
                  WHERE jsonb_typeof(symbols)='array'
                ),
                s_obj AS (
                  SELECT elem->>'symbol' AS sym
                  FROM {table}
                  CROSS JOIN LATERAL jsonb_array_elements(symbols) AS elem
                  WHERE jsonb_typeof(symbols)='array' AND jsonb_typeof(elem)='object' AND elem ? 'symbol'
                ),
                s AS (
                  SELECT sym FROM s_text
                  UNION ALL
                  SELECT sym FROM s_obj
                )
                SELECT sym, COUNT(*) AS cnt
                FROM s
                WHERE sym IS NOT NULL AND sym <> ''
                GROUP BY sym
                ORDER BY cnt DESC, sym ASC;
            """)
            rows = cur.fetchall()
        return Response([{"symbol": r[0], "count": r[1]} for r in rows])


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
            entry = JournalEntry.objects.get(id=id)
    except JournalEntry.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    
    data = request.data.copy()
    if request.user.is_authenticated:
        data['user'] = request.user.id

    partial = request.method == 'PATCH'
    serializer = JournalEntrySerializer(entry, data=data, partial=partial)
    
    if serializer.is_valid():
        updated_entry = serializer.save()

        # If no analysis exists, call FastAPI to create one
        try:
            Analysis.objects.get(entry=updated_entry)
        except Analysis.DoesNotExist:
            try:
                settings = request.data.get('settings', {})
                doctor_personality = settings.get('doctorPersonality', 'Academic')
                influence = settings.get('influence', {
                    'astrology': 0.15,
                    'personality': 0.15,
                    'medicalHistory': 0.10
                })
                doctor_influence = influence.get('doctor', 0.5)

                fastapi_response = requests.post(
                    # 'http://localhost:8001/analyze',
                    'http://104.236.96.193:8001/analyze',
                    json={
                        'content': updated_entry.content,
                        'settings': {
                            'user_id': str(updated_entry.user.id) if updated_entry.user else None,
                            'doctorPersonality': doctor_personality,
                            'influence': influence,
                            'doctor_influence': doctor_influence
                        }
                    },
                    timeout=30
                )
                
                if fastapi_response.status_code == 200:
                    analysis_data = fastapi_response.json()
                    print(f"📦 FastAPI response: {analysis_data}")
                    
                    try:
                        new_analysis = Analysis.objects.create(
                            entry=updated_entry,
                            user=updated_entry.user if updated_entry.user else None,
                            mood=analysis_data['mood'],
                            summary=analysis_data['summary'],
                            color=analysis_data['color'],
                            interpretation=analysis_data['interpretation'],
                            negative=analysis_data['negative'],
                            subject=analysis_data['subject'],
                            sentiment_score=analysis_data['sentiment_score'],
                            doctor_personality=analysis_data['doctor_personality'],
                            weights=analysis_data['weights'],
                            symbols=analysis_data['symbols']
                        )
                        print(f"✅ Created analysis: {new_analysis.id}")
                        updated_entry.analysis = new_analysis
                        updated_entry.save()
                        updated_serializer = JournalEntrySerializer(updated_entry)
                        return Response(updated_serializer.data)
                    except KeyError as e:
                        print(f"❌ Missing key in response: {e}")
                        print(f"Available keys: {analysis_data.keys()}")
                    except Exception as e:
                        print(f"❌ Error creating Analysis: {e}")
                else:
                    print(f"FastAPI analysis failed: {fastapi_response.status_code} - {fastapi_response.text}")
                    
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

@require_http_methods(["GET"])
def get_workflow_execution(request, workflow_id):
    """Get workflow execution details"""
    try:
        execution = get_object_or_404(
            WorkflowExecution.objects.prefetch_related('steps__citations'),
            id=workflow_id
        )

        # if execution.workflow_type == "custom-question":
        #     analysis_id = str(execution.analysis_id) if execution.analysis_id else None
        # else:
        #     ca = execution.cumulative_analyses.order_by('-created_at').first()
        #     analysis_id = str(ca.id) if ca else None
        analysis_id = str(execution.analysis_id) if execution.analysis_id else None

        print(f"analysis id === from workflow execution -==- {analysis_id}")

        steps_data = []
        for step in execution.steps.all():
            citations_data = [
                {
                    'id': str(citation.id),
                    'source': citation.source,
                    'content': citation.content,
                    'confidence': citation.confidence,
                    'reference': citation.reference,
                }
                for citation in step.citations.all()
            ]
            
            steps_data.append({
                'id': str(step.id),
                'step_number': step.step_number,
                'name': step.name,
                'step_type': step.step_type,
                'status': step.status,
                'start_time': step.start_time.isoformat() if step.start_time else None,
                'end_time': step.end_time.isoformat() if step.end_time else None,
                'duration_ms': step.duration_ms,
                'confidence': step.confidence,
                'reasoning': step.reasoning,
                'model_used': step.model_used,
                'tokens_used': step.tokens_used,
                'error': step.error,
                'citations': citations_data,
            })
        
        data = {
            'id': str(execution.id),
            'workflow_type': execution.workflow_type,
            'routine_name': execution.routine_name,
            'status': execution.status,
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'final_result': execution.final_result,
            'overall_confidence': execution.overall_confidence,
            'total_citations': execution.total_citations,
            'error_message': execution.error_message,
            'analysis_id': analysis_id,
            'steps': steps_data,
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
        
@api_view(['POST'])
@permission_classes([AllowAny])
def submit_feedback(request):
    """Submit feedback for Analysis, CumulativeAnalysis, or CustomQuestion"""
    try:
        analysis_id = request.data.get('analysis_id')
        analysis_type = request.data.get('analysis_type', 'analysis')  # 'analysis', 'cumulative', 'custom_question'
        rating = request.data.get('rating')  # 'good' or 'bad'
        comment = request.data.get('comment', '')
        details = request.data.get('details', {})  # accuracy, relevance, helpful

        user = request.user if request.user.is_authenticated else None

        if analysis_type == 'cumulative':
            cumulative = CumulativeAnalysis.objects.get(id=analysis_id)
            feedback, created = AnalysisFeedback.objects.update_or_create(
                cumulative_analysis=cumulative,
                user=user,
                defaults={
                    'rating': rating,
                    'comment': comment,
                    'accuracy': details.get('accuracy'),
                    'relevance': details.get('relevance'),
                    'helpful': details.get('helpful'),
                    'session_id': request.session.session_key or '',
                }
            )
        elif analysis_type == 'custom_question':
            question = CustomQuestion.objects.get(id=analysis_id)
            feedback, created = AnalysisFeedback.objects.update_or_create(
                custom_question=question,
                user=user,
                defaults={
                    'rating': rating,
                    'comment': comment,
                    'accuracy': details.get('accuracy'),
                    'relevance': details.get('relevance'),
                    'helpful': details.get('helpful'),
                    'session_id': request.session.session_key or '',
                }
            )
        else:
            analysis = Analysis.objects.get(id=analysis_id)
            feedback, created = AnalysisFeedback.objects.update_or_create(
                analysis=analysis,
                user=user,
                defaults={
                    'rating': rating,
                    'comment': comment,
                    'accuracy': details.get('accuracy'),
                    'relevance': details.get('relevance'),
                    'helpful': details.get('helpful'),
                    'session_id': request.session.session_key or '',
                }
            )

        return Response({
            'success': True,
            'feedback_id': str(feedback.id),
            'created': created
        })

    except (Analysis.DoesNotExist, CumulativeAnalysis.DoesNotExist, CustomQuestion.DoesNotExist):
        return Response({'error': 'Analysis not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_feedback_stats(request):
    """Get feedback statistics for improvement"""
    from django.db.models import Count, Q
    
    stats = {
        'total_feedback': AnalysisFeedback.objects.count(),
        'good_ratings': AnalysisFeedback.objects.filter(rating='good').count(),
        'bad_ratings': AnalysisFeedback.objects.filter(rating='bad').count(),
        
        # By doctor personality
        'by_doctor': {},
        
        # Recent bad feedback for review
        'recent_bad': []
    }
    
    # Stats by doctor personality
    for analysis in Analysis.objects.filter(feedback__isnull=False).distinct():
        doctor = analysis.doctor_personality
        if doctor not in stats['by_doctor']:
            stats['by_doctor'][doctor] = {'good': 0, 'bad': 0}
        
        feedback = analysis.feedback.first()
        if feedback.rating == 'good':
            stats['by_doctor'][doctor]['good'] += 1
        else:
            stats['by_doctor'][doctor]['bad'] += 1
    
    # Get recent bad feedback with comments
    recent_bad = AnalysisFeedback.objects.filter(
        rating='bad',
        comment__isnull=False
    ).exclude(comment='').order_by('-created_at')[:10]
    
    for fb in recent_bad:
        stats['recent_bad'].append({
            'date': fb.created_at.isoformat(),
            'comment': fb.comment,
            'doctor': fb.analysis.doctor_personality,
            'mood': fb.analysis.mood,
        })
    
    return Response(stats)

@api_view(['GET'])
def get_user_preferences(request):
    analyzer = FeedbackAnalyzer()
    prefs = analyzer.get_user_preferences(request.user.id)
    return Response(prefs)

@api_view(['GET'])
def get_user_recommendations(request):
    recs = PersonalizationHelper.get_user_recommendations(request.user.id)
    return Response(recs)
