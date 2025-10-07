import os
import sys  
import django

sys.path.insert(0, '/Users/kristo/django_backend_dream')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()
