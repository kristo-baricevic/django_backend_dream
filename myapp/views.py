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
from .security.owner import require_owner, owned_queryset, require_owned_object, initialize_settings, owned_atomic_write
from django.http import Http404
from django.conf import settings as runtime_settings
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
        owner = require_owner(self.request)
        row = initialize_settings(owner)
        return Settings.objects.filter(pk=row.pk, user_id=owner.domain_user_id)

class SettingsUpdateView(UpdateAPIView):
    serializer_class = SettingsSerializer

    def get_object(self):
        return initialize_settings(require_owner(self.request))

    def perform_update(self, serializer):
        owner = require_owner(self.request)
        def update(profile):
            row = initialize_settings(owner)
            serializer.instance = Settings.objects.select_for_update().get(pk=row.pk, user=profile)
            serializer.save(user=profile)
        owned_atomic_write(owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
                           workflow_id=None, write=update)

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
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = owned_queryset(JournalEntry, require_owner(self.request)).order_by('-created_at')
        # if self.request.user.is_authenticated:
        #     qs = qs.filter(user=self.request.user)

        entries = self.request.query_params.get('entries')
        title = self.request.query_params.get('title')
        moods = self.request.query_params.get('moods')
        symbols = self.request.query_params.get('symbols')
        analysis = self.request.query_params.get('analysis')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')

        if entries:
            qs = qs.filter(content__icontains=entries)
        if title:
            qs = qs.filter(analysis__subject__icontains=title)
        if moods:
            qs = qs.filter(analysis__mood__icontains=moods)
        if symbols:
            qs = qs.filter(analysis__symbols__icontains=symbols)
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
        qs = owned_queryset(Analysis, require_owner(request))
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
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = owned_queryset(CumulativeAnalysis, require_owner(self.request)).order_by('-created_at')
        
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
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = owned_queryset(CustomQuestion, require_owner(self.request)).order_by('-created_at')
        
        # if self.request.user and self.request.user.is_authenticated:
        #     qs = qs.filter(user=self.request.user)
    
        doctor_personality = self.request.query_params.get('doctor_personality')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        search = self.request.query_params.get('search')

        if doctor_personality:
            qs = qs.filter(doctor_personality__icontains=doctor_personality)
        
        if search:
            qs = qs.filter(Q(answer__icontains=search) | Q(question__icontains=search))

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
    permission_classes = [IsAuthenticated]

    def get(self, request):
        table = Analysis._meta.db_table
        with connection.cursor() as cur:
            cur.execute(f"""
                WITH s_text AS (
                  SELECT jsonb_array_elements_text(symbols) AS sym
                  FROM {table}
                  WHERE user_id=%s AND jsonb_typeof(symbols)='array'
                ),
                s_obj AS (
                  SELECT elem->>'symbol' AS sym
                  FROM {table}
                  CROSS JOIN LATERAL jsonb_array_elements(symbols) AS elem
                  WHERE user_id=%s AND jsonb_typeof(symbols)='array' AND jsonb_typeof(elem)='object' AND elem ? 'symbol'
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
            """, [require_owner(request).domain_user_id] * 2)
            rows = cur.fetchall()
        return Response([{"symbol": r[0], "count": r[1]} for r in rows])


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_data(request):
    if request.user.is_authenticated:
        analyses = Analysis.objects.filter(user_id=require_owner(request).domain_user_id).order_by('created_at')
    else:
        analyses = owned_queryset(Analysis, require_owner(request)).order_by('created_at')
    
    serializer = AnalysisSerializer(analyses, many=True)
    scores = [a.sentiment_score for a in analyses]
    avg = round(sum(scores) / len(scores)) if scores else 0
    return Response({"analyses": serializer.data, "avg": avg})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_entries(request):
    if request.user.is_authenticated:
        entries = JournalEntry.objects.filter(user_id=require_owner(request).domain_user_id).order_by('-created_at')
    else:
        entries = owned_queryset(JournalEntry, require_owner(request)).order_by('-created_at')
    serializer = JournalEntrySerializer(entries, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_entry(request, id):
    try:
        if request.user.is_authenticated:
            entry = JournalEntry.objects.get(id=id, user_id=require_owner(request).domain_user_id)
        else:
            # For anonymous users, get any entry by ID
            entry = require_owned_object(JournalEntry, require_owner(request), id)
    except JournalEntry.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    
    serializer = JournalEntrySerializer(entry)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_entry(request):
    owner = require_owner(request)
    serializer = JournalEntrySerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    def create(profile):
        return serializer.save(user=profile)
    row = owned_atomic_write(owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
                             workflow_id=None, write=create)
    return Response(JournalEntrySerializer(row).data, status=201)


@api_view(['PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def update_entry(request, id):
    owner = require_owner(request)
    def update(profile):
        row = require_owned_object(JournalEntry, owner, id, for_update=True)
        serializer = JournalEntrySerializer(row, data=request.data, partial=request.method == 'PATCH')
        serializer.is_valid(raise_exception=True)
        return serializer.save()
    row = owned_atomic_write(owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
                             workflow_id=None, write=update)
    # The external work is outside the transaction; the result gets a fresh guarded commit.
    if row.analysis_id is None and getattr(runtime_settings, 'DREAM_ANALYZER_URL', ''):
        try:
            response = requests.post(runtime_settings.DREAM_ANALYZER_URL.rstrip('/') + '/analyze',
                json={'entry_id': str(row.pk), 'content': row.content},
                headers={key: value for key, value in {
                    'Cookie': request.META.get('HTTP_COOKIE', ''), 'X-CSRFToken': request.META.get('HTTP_X_CSRFTOKEN', ''),
                    'Origin': request.META.get('HTTP_ORIGIN', ''), 'Referer': request.META.get('HTTP_REFERER', ''),
                    'Host': request.get_host()}.items() if value}, allow_redirects=False, timeout=30)
            if response.status_code == 200:
                result = response.json()
                def save_analysis(profile):
                    entry = require_owned_object(JournalEntry, owner, id, for_update=True)
                    if entry.content != row.content or entry.analysis_id is not None:
                        return entry
                    fields = {key: result[key] for key in ('mood', 'summary', 'color', 'interpretation',
                        'negative', 'subject', 'sentiment_score', 'doctor_personality', 'weights', 'symbols')}
                    entry.analysis = Analysis.objects.create(entry=entry, user=profile, **fields)
                    entry.save(update_fields=['analysis', 'updated_at'])
                    return entry
                row = owned_atomic_write(owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
                                         workflow_id=None, write=save_analysis)
        except (requests.RequestException, ValueError, KeyError):
            pass  # Saved draft remains available for retry; private response bodies are never logged.
    return Response(JournalEntrySerializer(row).data)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_entry(request, id):
    owner = require_owner(request)
    def delete(profile):
        require_owned_object(JournalEntry, owner, id, for_update=True).delete()
    owned_atomic_write(owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
                       workflow_id=None, write=delete)
    return Response(status=204)

@require_http_methods(["GET"])
def get_workflow_execution(request, workflow_id):
    """Get workflow execution details"""
    try:
        execution = get_object_or_404(
            WorkflowExecution.objects.prefetch_related('steps__citations'),
            id=workflow_id, user_id=require_owner(request).domain_user_id
        )

        # if execution.workflow_type == "custom-question":
        #     analysis_id = str(execution.analysis_id) if execution.analysis_id else None
        # else:
        #     ca = execution.cumulative_analyses.order_by('-created_at').first()
        #     analysis_id = str(ca.id) if ca else None
        analysis_id = str(execution.analysis_id) if execution.analysis_id else None

        pass  # private payload logging removed

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
        
    except Http404:
        return JsonResponse({'detail': 'Not found.'}, status=404)
    except Exception:
        return JsonResponse({'detail': 'Workflow unavailable.'}, status=500)
        
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_feedback(request):
    owner = require_owner(request)
    kind = request.data.get('analysis_type', 'analysis')
    targets = {'analysis': (Analysis, 'analysis'), 'cumulative': (CumulativeAnalysis, 'cumulative_analysis'),
               'custom_question': (CustomQuestion, 'custom_question')}
    rating = request.data.get('rating')
    details = request.data.get('details', {})
    comment = request.data.get('comment', '')
    if kind not in targets or rating not in dict(AnalysisFeedback.FEEDBACK_CHOICES) or not isinstance(details, dict) or not isinstance(comment, str) or len(comment) > 10000:
        return Response({'detail': 'Invalid feedback.'}, status=400)
    if any(details.get(key) not in (True, False, None) for key in ('accuracy', 'relevance', 'helpful')):
        return Response({'detail': 'Invalid feedback.'}, status=400)
    model, relation = targets[kind]
    def write(profile):
        target = require_owned_object(model, owner, request.data.get('analysis_id'), for_update=True)
        return AnalysisFeedback.objects.update_or_create(**{relation: target, 'user': profile},
            defaults={'rating': rating, 'comment': comment, 'accuracy': details.get('accuracy'),
                      'relevance': details.get('relevance'), 'helpful': details.get('helpful'), 'session_id': ''})
    feedback, created = owned_atomic_write(owner_id=owner.domain_user_id, expected_auth_user_id=owner.auth_user_id,
                                           workflow_id=None, write=write)
    return Response({'success': True, 'feedback_id': str(feedback.pk), 'created': created})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_feedback_stats(request):
    """Get feedback statistics for improvement"""
    from django.db.models import Count, Q
    
    stats = {
        'total_feedback': owned_queryset(AnalysisFeedback, require_owner(request)).count(),
        'good_ratings': owned_queryset(AnalysisFeedback, require_owner(request)).filter(rating='good').count(),
        'bad_ratings': owned_queryset(AnalysisFeedback, require_owner(request)).filter(rating='bad').count(),
        
        # By doctor personality
        'by_doctor': {},
        
        # Recent bad feedback for review
        'recent_bad': []
    }
    
    # Stats by doctor personality
    for analysis in owned_queryset(Analysis, require_owner(request)).filter(feedback__user_id=require_owner(request).domain_user_id).distinct():
        doctor = analysis.doctor_personality
        if doctor not in stats['by_doctor']:
            stats['by_doctor'][doctor] = {'good': 0, 'bad': 0}
        
        feedback = analysis.feedback.filter(user_id=require_owner(request).domain_user_id).first()
        if feedback.rating == 'good':
            stats['by_doctor'][doctor]['good'] += 1
        else:
            stats['by_doctor'][doctor]['bad'] += 1
    
    # Get recent bad feedback with comments
    recent_bad = owned_queryset(AnalysisFeedback, require_owner(request)).filter(
        rating='bad',
        comment__isnull=False
    ).exclude(comment='').order_by('-created_at')[:10]
    
    for fb in recent_bad:
        target = fb.analysis or fb.cumulative_analysis or fb.custom_question
        if target is None or target.user_id != require_owner(request).domain_user_id:
            continue
        stats['recent_bad'].append({
            'date': fb.created_at.isoformat(),
            'comment': fb.comment,
            'doctor': target.doctor_personality,
            'mood': fb.analysis.mood if fb.analysis else '',
        })
    
    return Response(stats)

@api_view(['GET'])
def get_user_preferences(request):
    from .feedback_analyzer import FeedbackAnalyzer
    analyzer = FeedbackAnalyzer()
    prefs = analyzer.get_user_preferences(require_owner(request).domain_user_id)
    return Response(prefs)

@api_view(['GET'])
def get_user_recommendations(request):
    from .feedback_analyzer import PersonalizationHelper
    recs = PersonalizationHelper.get_user_recommendations(require_owner(request).domain_user_id)
    return Response(recs)
