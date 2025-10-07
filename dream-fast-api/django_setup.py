import os
import sys  
import django

# Detect environment
if os.path.exists('/django_backend_dream'):
    # Docker environment
    sys.path.insert(0, '/django_backend_dream')
else:
    # Local environment
    sys.path.insert(0, '/Users/kristo/django_backend_dream')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()
