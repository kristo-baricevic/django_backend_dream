from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction, connection
from myapp.models import User

class Command(BaseCommand):
    help = 'Explicit local-only fixture; requires separate authorization before execution.'
    def add_arguments(self, parser):
        parser.add_argument('--confirm-local-only', action='store_true')
    def handle(self, **options):
        if not (options['confirm_local_only'] and settings.DREAM_ENV == 'local' and settings.DREAM_ENABLE_LOCAL_FIXTURE
                and str(connection.settings_dict['NAME']).startswith(('dream_auth_', 'test_dream_auth_'))):
            raise CommandError('Explicit isolated local fixture configuration required.')
        with transaction.atomic():
            if get_user_model().objects.filter(username='admin').exists():
                raise CommandError('Existing accounts are never changed by this fixture.')
            auth = get_user_model().objects.create_user(username='admin', password='admin', is_staff=True, is_superuser=False)
            User.objects.create(email='admin@local.invalid', auth_user=auth, is_local_fixture=True)
            models = ('user', 'journalentry', 'analysis', 'cumulativeanalysis', 'customquestion', 'settings', 'analysisfeedback')
            auth.user_permissions.set(Permission.objects.filter(content_type__app_label='myapp',
                codename__in=['view_' + model for model in models]))
        self.stdout.write('Created local fixture. Keep every service bound to loopback.')
