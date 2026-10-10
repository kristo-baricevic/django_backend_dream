"""Shared atomic login reservations with bounded interruption recovery."""
from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4
from django.db import transaction
from django.utils import timezone
from django.utils.crypto import salted_hmac
from myapp.models import LoginAttemptBucket

LEASE_SECONDS = 30

def _now():
    return timezone.now()

@dataclass
class LoginLimited(Exception):
    retry_after: int = 1


def reserve(username, address):
    now = _now()
    account = str(username).casefold()
    descriptors = [("account", account, 900, None)]
    if address is not None:
        descriptors += [("pair", account + chr(0) + address, 300, 5), ("ip", address, 300, 30)]
    descriptors.sort(key=lambda item: salted_hmac("dream-login-" + item[0], item[1]).hexdigest())
    token = str(uuid4())
    reservations = []
    with transaction.atomic():
        for kind, material, window, cap in descriptors:
            key = salted_hmac("dream-login-" + kind, material).hexdigest()
            LoginAttemptBucket.objects.get_or_create(key=key, defaults={"started_at": now})
            bucket = LoginAttemptBucket.objects.select_for_update().get(pk=key)
            if now >= bucket.started_at + timedelta(seconds=window):
                bucket.started_at, bucket.attempts, bucket.failures, bucket.blocked_until = now, 0, 0, None
                bucket.reservations = {}
            expired = [key for key, until in bucket.reservations.items() if until <= now.timestamp()]
            for stale in expired:
                del bucket.reservations[stale]
            if kind == "ip":
                bucket.failures += len(expired)
            else:
                bucket.attempts = max(0, bucket.attempts - len(expired))
            if bucket.blocked_until and bucket.blocked_until > now:
                raise LoginLimited(max(1, int((bucket.blocked_until - now).total_seconds()) + 1))
            if cap is not None and bucket.attempts >= cap:
                raise LoginLimited(max(1, int((bucket.started_at + timedelta(seconds=window) - now).total_seconds())))
            if kind == "account" and len(bucket.reservations) >= (1 if bucket.failures >= 10 else 10):
                raise LoginLimited(1)
            bucket.attempts += 1
            bucket.reservations[token] = (now + timedelta(seconds=LEASE_SECONDS)).timestamp()
            bucket.save()
            reservations.append((key, kind, bucket.started_at, token))
    return reservations


def finish(reservations, *, success):
    with transaction.atomic():
        buckets = [(LoginAttemptBucket.objects.select_for_update().get(pk=key), kind, started_at, token)
                   for key, kind, started_at, token in reservations]
        now = _now()
        valid = all(bucket.started_at == started and bucket.reservations.get(token, 0) > now.timestamp()
                    for bucket, kind, started, token in buckets)
        for bucket, kind, started, token in buckets:
            if bucket.started_at != started or token not in bucket.reservations:
                continue
            del bucket.reservations[token]
            if kind == "ip":
                bucket.failures += 1
            elif success is None or not valid:
                bucket.attempts = max(0, bucket.attempts - 1)
            elif success:
                bucket.attempts = len(bucket.reservations)
                bucket.failures, bucket.blocked_until = 0, None
            else:
                bucket.failures += 1
                if kind == "account" and bucket.failures >= 10:
                    bucket.blocked_until = now + timedelta(seconds=min(30, 2 ** min(5, bucket.failures - 10)))
            bucket.save()
        return valid
