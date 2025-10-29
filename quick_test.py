#!/usr/bin/env python3
"""
Quick Test Data Generator - Creates just 5 dreams for testing
"""

import requests
import json
import random
import time
from datetime import datetime, timedelta

# Configuration
DJANGO_API = "https://dream-journal-app.com/api"
FASTAPI_API = "https://dream-journal-app.com/fastapi"

# For local testing, uncomment these:
# DJANGO_API = "http://localhost:8000/api"
# FASTAPI_API = "http://localhost:8001"

def quick_test():
    """Generate 5 test dreams quickly."""
    
    print("QUICK TEST: Generating 5 dreams\n")
    
    themes = ["flying", "ocean", "forest", "school", "mirror"]
    base_date = datetime(2025, 10, 20)  # Recent dates
    
    for i, theme in enumerate(themes):
        print(f"\n--- Dream {i+1}/5: {theme} ---")
        
        # Generate dream
        try:
            print("1. Generating dream...")
            resp = requests.post(
                f"{FASTAPI_API}/generate-dream",
                json={"theme": theme},
                timeout=30
            )
            if resp.status_code != 200:
                print(f"   Failed: {resp.status_code}")
                continue
            
            content = resp.json()["dream"]
            print(f"   ✓ Generated {len(content)} chars")
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
        
        # Create entry
        try:
            print("2. Creating journal entry...")
            resp = requests.post(
                f"{DJANGO_API}/entries/create/",
                json={"content": content},
                timeout=30
            )
            if resp.status_code not in [200, 201]:
                print(f"   Failed: {resp.status_code}")
                print(f"   Response: {resp.text}")
                continue
                
            entry_id = resp.json()["id"]
            print(f"   ✓ Created entry {entry_id}")
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
        
        # Analyze (using update endpoint)
        try:
            print("3. Analyzing dream...")
            
            settings = {
                "doctorPersonality": "Academic",
                "personality": "INTJ",
                "astrology": {
                    "sun": "Leo",
                    "moon": "Pisces",
                    "rising": "Gemini"
                },
                "influence": {
                    "astrology": 0.15,
                    "personality": 0.15,
                    "medicalHistory": 0.10
                },
                "doctor_influence": 0.5
            }
            
            resp = requests.put(
                f"{DJANGO_API}/entries/{entry_id}/update/",
                json={
                    "content": content,
                    "settings": settings
                },
                timeout=60
            )
            
            if resp.status_code in [200, 201]:
                print(f"   ✓ Analysis complete")
            else:
                print(f"   Failed: {resp.status_code}")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        # Small delay
        time.sleep(1)
    
    print("\n✅ Quick test complete!")

if __name__ == "__main__":
    quick_test()
