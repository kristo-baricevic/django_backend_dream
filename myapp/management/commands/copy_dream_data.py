import json
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from myapp.security.data_copy import copy_bundle, rollback, CopyError

class Command(BaseCommand):
    help = 'Inspect/copy a restricted partial API export into an isolated local owner; never provision login access.'
    def add_arguments(self, parser):
        parser.add_argument('--bundle')
        parser.add_argument('--owner')
        parser.add_argument('--key')
        parser.add_argument('--source-origin')
        parser.add_argument('--settings-id')
        parser.add_argument('--private-bundle')
        parser.add_argument('--acknowledge-partial', action='store_true')
        parser.add_argument('--apply', action='store_true')
        parser.add_argument('--rollback')
    def handle(self, **options):
        try:
            if options['rollback']:
                if not options['apply']: raise CopyError('Rollback requires --apply.')
                report = rollback(options['rollback'])
            else:
                if not all(options[key] for key in ('bundle', 'owner', 'key', 'source_origin')):
                    raise CopyError('Bundle, owner, key and source origin required.')
                path = Path(options['bundle'])
                if path.is_symlink() or path.stat().st_mode & 0o077:
                    raise CopyError('Input must be a restricted regular file (0600).')
                private = None
                if options.get('private_bundle'):
                    private_path = Path(options['private_bundle'])
                    if private_path.is_symlink() or not private_path.is_file() or private_path.stat().st_mode & 0o077:
                        raise CopyError('Private input must be a restricted regular file (0600).')
                    private = json.loads(private_path.read_text())
                report = copy_bundle(json.loads(path.read_text()), owner_id=options['owner'], key=options['key'],
                    source_identity={'origin': options['source_origin'], 'kind': 'partial-api-export'},
                    selected_settings_id=options['settings_id'], partial=options['acknowledge_partial'], dry_run=not options['apply'], private_bundle=private)
            self.stdout.write(json.dumps(report, sort_keys=True))  # counts/hashes only; no private fields
        except (CopyError, OSError, ValueError) as error:
            raise CommandError(str(error))
