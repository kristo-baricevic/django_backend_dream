from datetime import datetime, timezone
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from .owner import DEADLINE_KEY


@receiver(user_logged_in, dispatch_uid='dream_absolute_session_deadline')
def set_login_deadline(sender, request, user, **kwargs):
    request.session[DEADLINE_KEY] = datetime.now(timezone.utc).timestamp() + 8 * 60 * 60
    request.session.set_expiry(0)
