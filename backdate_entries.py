#!/usr/bin/env python
"""
Django Shell Script to Backdate Test Entries
Run this after generate_test_data.py to properly backdate entries

Usage:
  python manage.py shell < backdate_entries.py
"""

from django.utils import timezone
from datetime import datetime, timedelta
import random
from myapp.models import JournalEntry, Analysis

# Get the most recent 40 entries (assuming these are your test entries)
recent_entries = JournalEntry.objects.all().order_by('-created_at')[:40]

if not recent_entries:
    print("No entries found to backdate!")
    exit()

print(f"Found {len(recent_entries)} recent entries to backdate")

# Date range
start_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 10, 28, tzinfo=timezone.utc)

# Generate random dates
delta = end_date - start_date
random_days = sorted([random.randint(0, delta.days) for _ in range(len(recent_entries))])

# Apply dates to entries (from newest to oldest, so oldest gets earliest date)
for entry, day_offset in zip(reversed(recent_entries), random_days):
    new_date = start_date + timedelta(days=day_offset)
    
    # Add random time of day (between 6 AM and 11 PM)
    hour = random.randint(6, 23)
    minute = random.randint(0, 59)
    new_date = new_date.replace(hour=hour, minute=minute)
    
    # Update the entry
    entry.created_at = new_date
    entry.save(update_fields=['created_at'])
    
    # Also update the related analysis if it exists
    if hasattr(entry, 'entry_analysis'):
        entry.entry_analysis.created_at = new_date
        entry.entry_analysis.save(update_fields=['created_at'])
    
    print(f"✓ Updated entry {entry.id} to {new_date.date()} {new_date.time()}")

print(f"\n✅ Successfully backdated {len(recent_entries)} entries!")
print(f"Date range: {start_date.date()} to {end_date.date()}")
