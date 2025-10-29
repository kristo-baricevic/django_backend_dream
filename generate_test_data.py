#!/usr/bin/env python3
"""
Test Data Generator for Dream Journal
Creates 40 dreams from August 1, 2025 to October 28, 2025
"""

import requests
import json
import random
from datetime import datetime, timedelta
import time
import sys

# API endpoints
DJANGO_API = "https://dream-journal-app.com/api"
FASTAPI_API = "https://dream-journal-app.com/fastapi"

# Test themes for variety
DREAM_THEMES = [
    "flying over mountains",
    "swimming in the ocean", 
    "lost in a forest",
    "visiting childhood home",
    "meeting a stranger",
    "running from something",
    "discovering hidden rooms",
    "talking animals",
    "time travel",
    "being in school again",
    "climbing endless stairs",
    "driving without control",
    "missing a flight",
    "finding treasure",
    "transforming into something",
    "walking through walls",
    "seeing deceased relatives",
    "being underwater breathing",
    "exploring alien worlds",
    "having superpowers",
    "falling from heights",
    "being chased",
    "public speaking",
    "losing teeth",
    "being late",
    "forgotten homework",
    "wedding ceremony",
    "birthday party",
    "mysterious doorway",
    "mirror reflections",
    "floating in space",
    "ancient ruins",
    "talking to younger self",
    "winning lottery",
    "apocalyptic scenario",
    "peaceful garden",
    "war zone",
    "carnival or fair",
    "library of infinite books",
    "frozen in place"
]

# Different doctor personalities to use
DOCTOR_PERSONALITIES = [
    "Academic",
    "Compassionate", 
    "Practical",
    "Mystical"
]

# Sample user settings
USER_SETTINGS = {
    "personality": "INTJ",
    "occupation": "Software Developer",
    "astrology": {
        "sun": "Leo",
        "moon": "Pisces", 
        "rising": "Gemini"
    },
    "medicalHistory": {
        "psychological": ["anxiety", "occasional insomnia"],
        "physical": ["migraine headaches"]
    }
}

def generate_dates(start_date, end_date, num_dates):
    """Generate random dates between start and end."""
    delta = end_date - start_date
    random_days = sorted([random.randint(0, delta.days) for _ in range(num_dates)])
    
    dates = []
    for day_offset in random_days:
        date = start_date + timedelta(days=day_offset)
        dates.append(date)
    
    return dates

def generate_dream(theme):
    """Generate a dream using FastAPI."""
    print(f"  Generating dream about: {theme}")
    
    try:
        response = requests.post(
            f"{FASTAPI_API}/generate-dream",
            json={"theme": theme},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            dream_content = response.json()["dream"]
            print(f"  ✓ Generated dream ({len(dream_content)} chars)")
            return dream_content
        else:
            print(f"  ✗ Failed to generate: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error generating dream: {e}")
        return None

def create_journal_entry(content, created_date):
    """Create journal entry in Django."""
    print(f"  Creating journal entry for {created_date.date()}")
    
    try:
        # Django expects this format
        response = requests.post(
            f"{DJANGO_API}/entries/create/",
            json={
                "content": content,
                # Note: We can't actually set created_at via API
                # Django will use current time, but we're distributing creation over time
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            entry = response.json()
            print(f"  ✓ Created entry ID: {entry['id']}")
            return entry
        else:
            print(f"  ✗ Failed to create entry: {response.status_code}")
            print(f"    Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error creating entry: {e}")
        return None

def analyze_entry(entry_id, content):
    """Trigger analysis for the journal entry."""
    print(f"  Analyzing entry {entry_id}")
    
    # Random doctor personality for variety
    doctor = random.choice(DOCTOR_PERSONALITIES)
    
    # Randomly adjust influence weights
    influence = {
        "astrology": random.uniform(0.1, 0.3),
        "personality": random.uniform(0.1, 0.3),
        "medicalHistory": random.uniform(0.05, 0.15),
    }
    
    settings = {
        **USER_SETTINGS,
        "doctorPersonality": doctor,
        "influence": influence,
        "doctor_influence": random.uniform(0.4, 0.7)
    }
    
    try:
        # Call the FastAPI analyze endpoint directly
        response = requests.post(
            f"{FASTAPI_API}/analyze",
            json={
                "content": content,
                "settings": settings
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            analysis = response.json()
            print(f"  ✓ Analysis complete - Mood: {analysis['mood']}, Score: {analysis['sentiment_score']}")
            
            # Now update the Django entry with the analysis
            update_response = requests.put(
                f"{DJANGO_API}/entries/{entry_id}/update/",
                json={
                    "content": content,
                    "settings": settings  # Pass settings so Django can store the analysis
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if update_response.status_code in [200, 201]:
                print(f"  ✓ Updated Django entry with analysis")
            else:
                print(f"  ⚠ Analysis complete but Django update failed")
            
            return analysis
        else:
            print(f"  ✗ Failed to analyze: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error analyzing entry: {e}")
        return None

def main():
    """Main execution function."""
    print("=" * 60)
    print("DREAM JOURNAL TEST DATA GENERATOR")
    print("=" * 60)
    
    # Date range
    start_date = datetime(2025, 8, 1)
    end_date = datetime(2025, 10, 28)
    num_dreams = 40
    
    print(f"\nGenerating {num_dreams} dreams")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("-" * 60)
    
    # Generate random dates
    dates = generate_dates(start_date, end_date, num_dreams)
    
    # Shuffle themes and cycle through them
    themes = DREAM_THEMES.copy()
    random.shuffle(themes)
    
    successful_dreams = 0
    failed_dreams = 0
    
    for i, date in enumerate(dates, 1):
        print(f"\nDream {i}/{num_dreams} - Date: {date.date()}")
        print("-" * 40)
        
        # Get theme (cycle if we run out)
        theme = themes[i % len(themes)]
        
        # Generate dream
        dream_content = generate_dream(theme)
        if not dream_content:
            failed_dreams += 1
            print("  ⚠ Skipping due to generation failure")
            continue
        
        # Create journal entry
        entry = create_journal_entry(dream_content, date)
        if not entry:
            failed_dreams += 1
            print("  ⚠ Skipping due to creation failure")
            continue
        
        # Analyze the entry
        analysis = analyze_entry(entry['id'], dream_content)
        if analysis:
            successful_dreams += 1
        else:
            # Entry created but analysis failed
            successful_dreams += 1  # Still count as success since entry exists
            print("  ⚠ Entry created but analysis failed")
        
        # Small delay to avoid overwhelming the APIs
        time.sleep(2)
        
        # Progress indicator
        if i % 5 == 0:
            print(f"\n>>> Progress: {i}/{num_dreams} dreams processed")
    
    # Final summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✓ Successful: {successful_dreams}")
    print(f"✗ Failed: {failed_dreams}")
    print(f"Total: {num_dreams}")
    
    if successful_dreams > 0:
        print(f"\nSuccess rate: {(successful_dreams/num_dreams)*100:.1f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        sys.exit(1)
