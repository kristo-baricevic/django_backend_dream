"""Closed value maps for every setting used to select reference files."""
from pathlib import Path
import math
from rest_framework.exceptions import ValidationError

SIGNS = frozenset('aries taurus gemini cancer leo virgo libra scorpio sagittarius capricorn aquarius pisces'.split())
PERSONALITIES = frozenset('INTJ ENTJ ESTJ ISTJ ESFJ ISFJ INFJ ENFJ ENFP INFP ISFP ESFP ISTP ESTP ENTP INTP'.split())
DOCTORS = frozenset('scientific mystical academic compassionate artistic'.split())


def validate_settings(values):
    if not isinstance(values, dict):
        raise ValidationError('Invalid settings.')
    values = dict(values)
    values.pop('user_id', None)
    values.pop('auth_user_id', None)
    doctor = values.get('doctorPersonality', values.get('doctor_personality', ''))
    if doctor and (not isinstance(doctor, str) or doctor.casefold() not in DOCTORS):
        raise ValidationError('Invalid doctor selection.')
    personality = values.get('personality', values.get('personality_type', ''))
    if personality and (not isinstance(personality, str) or personality.upper() not in PERSONALITIES):
        raise ValidationError('Invalid personality selection.')
    astrology = values.get('astrology', {})
    if not isinstance(astrology, dict) or set(astrology) - {'sun', 'moon', 'rising'}:
        raise ValidationError('Invalid astrology selection.')
    for sign in astrology.values():
        if sign and (not isinstance(sign, str) or sign.casefold() not in SIGNS):
            raise ValidationError('Invalid astrology selection.')
    weights = values.get('influence', values.get('weights', {}))
    if not isinstance(weights, dict) or set(weights) - {'theory', 'astrology', 'personality', 'medicalHistory', 'doctor'}:
        raise ValidationError('Invalid weights.')
    for weight in weights.values():
        if isinstance(weight, bool) or not isinstance(weight, (int,float)) or not math.isfinite(weight) or not 0<=weight<=1:
            raise ValidationError('Invalid weights.')
    return values


def reference_path(root, category, selection):
    maps = {
        'doctor_profiles': {name: name+'.txt' for name in DOCTORS},
        'personality': {name: name+'.txt' for name in PERSONALITIES},
        **{f'astrology/{kind}_signs': {name: name+'.txt' for name in SIGNS}
           for kind in ('sun','moon','rising')},
    }
    key = selection.upper() if category == 'personality' else selection.casefold()
    if category not in maps or key not in maps[category]:
        raise ValidationError('Invalid reference selection.')
    base = Path(root).resolve()
    candidate = base / category / maps[category][key]
    if candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents if p != base):
        raise ValidationError('Invalid reference path.')
    if not candidate.resolve().is_relative_to(base):
        raise ValidationError('Invalid reference path.')
    return str(candidate)
