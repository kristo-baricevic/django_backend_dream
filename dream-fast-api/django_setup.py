import os
import sys  
import django
from pathlib import Path

# Detect environment
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()
