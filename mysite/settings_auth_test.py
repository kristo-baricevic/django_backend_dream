"""Synthetic-only settings; never import deployed database configuration."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'synthetic-task12-test-signing-key-not-for-serving'
DEBUG = False
DREAM_ENV = 'test'
DREAM_ENABLE_LOCAL_FIXTURE = False
ALLOWED_HOSTS = ['testserver', 'localhost', '127.0.0.1']
CSRF_TRUSTED_ORIGINS = ['http://testserver']
CORS_ALLOWED_ORIGINS = ['http://testserver', 'http://localhost:3100']
from corsheaders.defaults import default_headers
CORS_ALLOW_HEADERS = (*default_headers, 'x-dream-principal', 'x-dream-session')
CORS_EXPOSE_HEADERS = ['X-Dream-Principal']
INSTALLED_APPS = [
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
    'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
    'rest_framework', 'corsheaders', 'myapp',
]
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'myapp.security.middleware.OwnerSessionMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]
ROOT_URLCONF = 'mysite.urls'
DATABASES = {'default': {
    'ENGINE': 'django.db.backends.postgresql', 'NAME': 'dream_auth_test',
    'USER': 'dream_test', 'HOST': '127.0.0.1', 'PORT': 5432,
    'TEST': {'NAME': 'test_dream_auth_task12'},
}}
TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
        'django.template.context_processors.request',
    ]},
}]
USE_TZ = True
TIME_ZONE = 'UTC'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
STATIC_URL = '/static/'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = True
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = 8 * 60 * 60
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': ['rest_framework.authentication.SessionAuthentication'],
    'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.IsAuthenticated'],
}
