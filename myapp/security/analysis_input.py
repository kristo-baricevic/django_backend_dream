from .owner import initialize_settings, load_owned_entries, owned_atomic_write
from .settings_validation import validate_settings


def analysis_input(owner, entry_ids):
    """Read authoritative saved settings and entries under the account locks."""
    initialize_settings(owner)
    def read(profile):
        entries = load_owned_entries(owner, entry_ids)
        row = profile.active_settings
        if row is None or row.user_id != profile.pk:
            from rest_framework.exceptions import ValidationError
            raise ValidationError('Settings selection is required.')
        values = validate_settings({
            'doctor_personality': row.doctor_personality,
            'personality_type': row.personality_type,
            'astrology': row.astrology, 'weights': row.weights,
        })
        config = {'doctorPersonality': row.doctor_personality or 'Academic',
                  'personality': row.personality_type,
                  'occupation': row.occupation, 'astrology': row.astrology,
                  'medicalHistory': row.medical_history,
                  'influence': row.weights, 'doctor_influence': row.doctor_influence,
                  'user_id': str(profile.pk)}
        return ([{'id': str(entry.pk), 'content': entry.content,
                  'created_at': entry.created_at} for entry in entries], config)
    return owned_atomic_write(owner_id=owner.domain_user_id,
                              expected_auth_user_id=owner.auth_user_id,
                              workflow_id=None, write=read)
