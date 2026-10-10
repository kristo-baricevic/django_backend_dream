from django.http import JsonResponse
from django.contrib.auth import logout
from rest_framework.exceptions import APIException
from .owner import require_owner
from .throttle import reserve, finish, LoginLimited
from .client_address import login_address


class OwnerSessionMiddleware:
    PUBLIC_PATHS = {'/api/auth/csrf/', '/api/auth/login/', '/api/auth/logout/'}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        private = request.path.startswith('/api/') and request.path not in self.PUBLIC_PATHS
        private_admin = request.path.startswith('/admin/') and request.path not in {'/admin/login/', '/admin/logout/'}
        if request.path == '/admin/login/' and request.user.is_authenticated:
            try:
                require_owner(request)
            except APIException:
                pass  # require_owner clears an expired or invalid session.
        if private or private_admin:
            try:
                request.dream_owner = require_owner(request)
            except APIException as error:
                return JsonResponse({'detail': 'Authentication required.'}, status=403 if error.status_code == 403 else 401,
                                    headers={'Cache-Control': 'no-store, private'})
        response = self.get_response(request)
        attempts = getattr(request, '_dream_admin_attempts', None)
        if attempts is not None:
            settled = finish(attempts, success=(None if response.status_code >= 500 else
                             request.user.is_authenticated and response.status_code == 302))
            if not settled:
                logout(request)
                response = JsonResponse({'detail': 'Login temporarily unavailable.'}, status=503)
        if request.path.startswith(('/api/', '/admin/')):
            response['Cache-Control'] = 'no-store, private'
            response['Vary'] = 'Cookie'
        return response

    def process_view(self, request, callback, callback_args, callback_kwargs):
        if request.path == '/admin/login/' and request.method == 'POST':
            username = request.POST.get('username', '')
            password = request.POST.get('password', '')
            if len(username) > 150 or len(password) > 1024:
                return JsonResponse({'detail': 'Invalid login.'}, status=400)
            try:
                request._dream_admin_attempts = reserve(username, login_address(request))
            except LoginLimited as error:
                return JsonResponse({'detail': 'Try again later.'}, status=429,
                                    headers={'Retry-After': str(error.retry_after)})
