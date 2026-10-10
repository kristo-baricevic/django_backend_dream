"""Isolated synthetic-test settings. Never import deployed settings or connect to its DB."""
SECRET_KEY = "synthetic-tests-only"
INSTALLED_APPS = ["django.contrib.auth", "django.contrib.contenttypes", "myapp"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
MIGRATION_MODULES = {"myapp": None}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "UTC"
ROOT_URLCONF = "myapp.longitudinal_tests.urls"
LONGITUDINAL_NETWORK_ENABLED = False
LONGITUDINAL_CONTEXT_ENABLED = False
LONGITUDINAL_SECURITY_MODULE = "myapp.longitudinal_tests.security_double"
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
