"""Bounded durable local worker. Schedule externally; no process/thread in HTTP."""
from django.core.exceptions import PermissionDenied
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import F
from rest_framework.exceptions import PermissionDenied as ApiPermissionDenied
from myapp.longitudinal_models import AuditRun
from myapp.longitudinal import services


class Command(BaseCommand):
    help = "Process up to 10 queued local dream-audit jobs; no paid inference."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=10)

    def handle(self, *args, **options):
        limit = options["limit"]
        if not 1 <= limit <= 10:
            raise CommandError("limit must be between 1 and 10")
        # Exclude revoked actor links before scheduling to avoid starvation.
        # This metadata filter is not authorization; process_run revalidates under locks.
        eligible = AuditRun.objects.filter(status="queued", mode="local", user__auth_user__is_active=True,
            user__auth_user_id=F("auth_user_id"), user__import_ready=True)
        if not (getattr(settings, "DREAM_ENV", "") == "local" and getattr(settings, "DREAM_ENABLE_LOCAL_FIXTURE", False)):
            eligible = eligible.filter(user__is_local_fixture=False)
        selected = list(eligible.order_by("created_at").values_list("pk", flat=True)[:limit])
        processed = rejected = 0
        for id in selected:
            try:
                services.process_run(id)
                processed += 1
            except (PermissionDenied, ApiPermissionDenied, AuditRun.DoesNotExist):
                # Auth/link changes are a fail-closed integration condition. Never
                # write private data outside the approved authenticated transaction.
                rejected += 1
        self.stdout.write(f"processed={processed} rejected={rejected}")
