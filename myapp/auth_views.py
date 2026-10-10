import json
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_GET, require_POST
from rest_framework.exceptions import APIException
from myapp.models import User
from .security.owner import require_owner, _valid_profile
from .security.throttle import reserve, finish, LoginLimited
from .security.client_address import login_address


def _reply(payload, status=200):
    response = JsonResponse(payload, status=status)
    response['Cache-Control'] = 'no-store, private'
    response['Vary'] = 'Cookie'
    return response


def _principal(request):
    owner = require_owner(request)
    return {'user': {'id': str(owner.domain_user_id), 'username': request.user.username,
                     'isAdmin': owner.is_staff}, 'csrfToken': get_token(request),
            'authUserId': owner.auth_user_id, 'loginDeadline': owner.login_deadline.isoformat()}


@require_GET
def csrf_token(request):
    return _reply({'csrfToken': get_token(request)})


@csrf_protect
@require_POST
def sign_in(request):
    if len(request.body) > 8192:
        return _reply({'detail': 'Invalid login.'}, 400)
    try:
        payload = json.loads(request.body)
        username, password = payload.get('username'), payload.get('password')
    except (ValueError, AttributeError):
        return _reply({'detail': 'Invalid login.'}, 400)
    if not isinstance(username, str) or not isinstance(password, str) or not username or len(username)>150 or len(password)>1024:
        return _reply({'detail': 'Invalid login.'}, 400)
    try:
        attempts = reserve(username, login_address(request))
    except LoginLimited as error:
        response = _reply({'detail': 'Try again later.'}, 429)
        response['Retry-After'] = str(error.retry_after)
        return response
    try:
        user = authenticate(request, username=username, password=password)
        profile = User.objects.filter(auth_user_id=user.pk).first() if user else None
        valid = user is not None and profile is not None and _valid_profile(profile, user)
    except Exception:
        finish(attempts, success=None)
        return _reply({'detail': 'Login temporarily unavailable.'}, 503)
    if not finish(attempts, success=valid):
        return _reply({'detail': 'Login temporarily unavailable.'}, 503)
    if not valid:
        return _reply({'detail': 'Invalid login.'}, 401)
    login(request, user)
    return _reply(_principal(request))


@require_GET
def session(request):
    try:
        return _reply(_principal(request))
    except APIException:
        return _reply({'detail': 'Authentication required.'}, 401)


@csrf_protect
@require_POST
def validate_session(request):
    try:
        return _reply(_principal(request))
    except APIException:
        return _reply({'detail': 'Authentication required.'}, 401)


@csrf_protect
@require_POST
def sign_out(request):
    logout(request)
    return _reply({'ok': True})
