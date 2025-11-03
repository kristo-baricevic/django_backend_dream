from django.db.models import Count, Avg, Q
from myapp.models import AnalysisFeedback, Analysis, User
import json

class FeedbackAnalyzer:
    """Analyze feedback to understand user preferences and identify technical issues"""
    
    def __init__(self):
        self.feedback_data = []
        
    def get_user_preferences(self, user_id):
        """Track individual user's preferences for personalization"""
        
        user_feedback = AnalysisFeedback.objects.filter(
            user_id=user_id
        ).select_related('analysis')
        
        preferences = {
            'preferred_doctors': {},
            'preferred_weights': {},
            'feedback_history': []
        }
        
        # Track which doctor styles this user likes
        for feedback in user_feedback:
            doctor = feedback.analysis.doctor_personality
            if doctor not in preferences['preferred_doctors']:
                preferences['preferred_doctors'][doctor] = {'liked': 0, 'disliked': 0}
            
            if feedback.rating == 'good':
                preferences['preferred_doctors'][doctor]['liked'] += 1
            else:
                preferences['preferred_doctors'][doctor]['disliked'] += 1
        
        # Track weight preferences
        good_analyses = user_feedback.filter(rating='good')
        for feedback in good_analyses:
            weights = feedback.analysis.weights
            # Store weights that this user responded well to
            preferences['preferred_weights'] = weights  # Use their most recent liked config
        
        return preferences
    
    def get_technical_issues(self):
        """Identify actual technical problems, not subjective preferences"""
        
        issues = {
            'technical_errors': [],
            'common_complaints': [],
            'usage_stats': {}
        }
        
        # Look for technical issue keywords
        technical_keywords = {
            'error': ['error', 'crash', 'failed', 'broken', 'doesn\'t work'],
            'wrong_dream': ['wrong dream', 'different dream', 'not my dream'],
            'missing': ['missing', 'no analysis', 'blank', 'empty'],
            'slow': ['slow', 'takes forever', 'timeout', 'stuck'],
            'formatting': ['formatting', 'can\'t read', 'garbled', 'messed up']
        }
        
        all_feedback = AnalysisFeedback.objects.filter(
            comment__isnull=False
        ).exclude(comment='')
        
        issue_counts = {issue: 0 for issue in technical_keywords}
        
        for feedback in all_feedback:
            comment_lower = feedback.comment.lower()
            for issue, keywords in technical_keywords.items():
                if any(keyword in comment_lower for keyword in keywords):
                    issue_counts[issue] += 1
                    issues['technical_errors'].append({
                        'type': issue,
                        'comment': feedback.comment[:200],
                        'date': feedback.created_at.isoformat(),
                        'analysis_id': str(feedback.analysis_id)
                    })
        
        # Usage statistics
        issues['usage_stats'] = {
            'total_analyses': Analysis.objects.count(),
            'total_feedback': AnalysisFeedback.objects.count(),
            'feedback_rate': (AnalysisFeedback.objects.count() / Analysis.objects.count() * 100) 
                           if Analysis.objects.count() > 0 else 0
        }
        
        return issues
    
    def get_usage_patterns(self):
        """Track how users are actually using the system"""
        
        patterns = {
            'doctor_usage': {},
            'feature_usage': {},
            'time_patterns': {}
        }
        
        # Track which doctors are being used (not which are "best")
        for analysis in Analysis.objects.all():
            doctor = analysis.doctor_personality
            if doctor not in patterns['doctor_usage']:
                patterns['doctor_usage'][doctor] = 0
            patterns['doctor_usage'][doctor] += 1
        
        # Track feature usage
        patterns['feature_usage'] = {
            'single_analysis': Analysis.objects.count(),
            'cumulative_analysis': CumulativeAnalysis.objects.count(),
            'custom_questions': CustomQuestion.objects.count(),
            'feedback_submitted': AnalysisFeedback.objects.count()
        }
        
        return patterns


# Django management command: analyze_feedback.py
from django.core.management.base import BaseCommand
from myapp.feedback_analyzer import FeedbackAnalyzer

class Command(BaseCommand):
    help = 'Analyze feedback for technical issues and usage patterns'
    
    def handle(self, *args, **options):
        analyzer = FeedbackAnalyzer()
        
        # Get technical issues
        issues = analyzer.get_technical_issues()
        
        self.stdout.write("\n=== TECHNICAL ISSUES REPORT ===\n")
        
        # Technical errors found
        if issues['technical_errors']:
            self.stdout.write("\nTechnical Issues Found:")
            for error in issues['technical_errors'][:10]:  # Show first 10
                self.stdout.write(f"  - {error['type']}: {error['comment'][:100]}...")
        else:
            self.stdout.write("\n✓ No technical issues reported")
        
        # Usage stats
        self.stdout.write(f"\nUsage Statistics:")
        self.stdout.write(f"  Total Analyses: {issues['usage_stats']['total_analyses']}")
        self.stdout.write(f"  Total Feedback: {issues['usage_stats']['total_feedback']}")
        self.stdout.write(f"  Feedback Rate: {issues['usage_stats']['feedback_rate']:.1f}%")
        
        # Usage patterns
        patterns = analyzer.get_usage_patterns()
        self.stdout.write(f"\nDoctor Personality Usage:")
        for doctor, count in patterns['doctor_usage'].items():
            self.stdout.write(f"  {doctor}: {count} times")


# Personalization helper (instead of "optimization")
class PersonalizationHelper:
    """Help users find their preferences, not "optimal" settings"""
    
    @classmethod
    def get_user_recommendations(cls, user_id):
        """Suggest settings based on individual user's past preferences"""
        
        analyzer = FeedbackAnalyzer()
        preferences = analyzer.get_user_preferences(user_id)
        
        recommendations = {
            'suggested_doctor': None,
            'suggested_weights': None,
            'reason': ''
        }
        
        # Find which doctor this specific user tends to like
        liked_most = None
        max_likes = 0
        for doctor, stats in preferences['preferred_doctors'].items():
            if stats['liked'] > max_likes:
                max_likes = stats['liked']
                liked_most = doctor
        
        if liked_most:
            recommendations['suggested_doctor'] = liked_most
            recommendations['reason'] = f"You've liked the {liked_most} style {max_likes} times"
        
        # Suggest weights if we have data
        if preferences['preferred_weights']:
            recommendations['suggested_weights'] = preferences['preferred_weights']
        
        return recommendations
