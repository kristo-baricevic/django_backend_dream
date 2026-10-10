# Authentication and isolated migration

Private API data belongs to the existing UUID `myapp.User`. Migration 0019 adds
a unique nullable Django auth-user link, explicit active-settings selection,
local-fixture/import visibility flags, accepted-job metadata, shared login
reservations and a copy ledger. It does not replace the domain user model.

Username/password authentication uses Django hashing and server sessions. Login
requires CSRF; private API and admin requests require an active linked owner and
an immutable eight-hour session deadline. Registration is disabled. Admin
provides owner-scoped read-only domain views and no auth-user/group management.
Unsafe requests require CSRF. Browser cookies are HttpOnly/Lax and secure in
production; private responses are no-store. Throttle reservations are held in
PostgreSQL and settle atomically, with account/IP/pair bounds and short recovery
backoff. Missing edge attestation uses account throttling rather than one shared
proxy-IP budget.

Analyzer validates sessions and CSRF locally using Django in a threadpool. Chat
has no database/signing secret: each request uses a fixed-origin HTTP client and
revalidates with Django before tools and each returned stream chunk. Redirects
and production HTTP credential forwarding are rejected. No provider is called
by security tests or the isolated configuration.

## Configuration

Production requires `DJANGO_SECRET_KEY` (at least 50 characters), explicit
`DJANGO_ALLOWED_HOSTS`, `DREAM_BROWSER_ORIGINS`, `DREAM_LOGIN_PROXY_CIDRS`, database
variables, and HTTPS analyzer/chat service destinations. `DREAM_ENV` defaults
to production. Only configure TLS proxy trust when that proxy overwrites the
header and prevents bypass. Optional `DREAM_ACCEPT_VERIFIED_CLIENT_IP=1` accepts
`X-Dream-Client-IP` only from declared trusted peers; the matching frontend edge
must overwrite that header on every request and prevent direct BFF access.

`docker-compose.auth-local.yml` is a separate loopback-only project. It requires
fresh local signing/database secrets, uses its own volume/names, and does not
automatically migrate, create users or call providers. Do not reuse the original
root Compose to start an isolated environment. Provider startup is disabled in
the analyzer local fixture; chatbot provider clients are lazy.

The opt-in `create_local_admin_fixture --confirm-local-only` command requires
`DREAM_ENV=local`, `DREAM_ENABLE_LOCAL_FIXTURE=1`, an isolated `dream_auth_` database
and loopback hosts/origins. Existing accounts are never overwritten. It hashes
the local development password and grants only domain view permissions; it is
never enabled by default or allowed in production. No persistent fixture account
has been provisioned as part of these changes.

## Copy-first migration

`copy_dream_data` accepts a restricted 0600 JSON export, an explicit destination
owner, idempotency key, verified source origin and partial-coverage acknowledgement.
It defaults to dry inspection; `--apply` is explicit and permitted only in an
isolated local/test database. It never creates a login or modifies the source.

New UUIDs and ledger digests bind every copied record to its source. Relationships
and timestamps are remapped/preserved; ambiguous settings require selection.
Replays reconcile the original copy. `--rollback <run-id> --apply` restores the
prior settings pointer and deletes only unchanged copied rows; changed records,
new dependent references and changed settings block rollback. Input containing
session credentials or unknown resources is rejected.

The initial adapter is intentionally partial: serializers omit workflow links,
some settings/doctor metadata, feedback and private file/index inventory. A
complete source database/files/index export and explicit private-model adapters
are required before claiming a complete account migration. Never infer that
missing API relations are absent from the database. Do not publish exports,
dreams, medical data, backups or indexes in Git/PRs.

## Accepted jobs and integration

`OwnerContext` carries the UUID owner, accepted numeric auth ID and login deadline.
`owned_atomic_write` locks auth → owner → optional `lock_state(profile)` → workflow
through commit. Resolve all private relations under those locks and assign the
owner explicitly. Accepted jobs may finish after logout; account deactivation,
relinking, withdrawn import visibility, removed/foreign sources or historical
workflows block publication. Never persist session cookies in job metadata.

Longitudinal migration 0020, privacy/source revisions and private-model copy
adapters are a coordinated integration follow-up. Outbound longitudinal provider
context stays disabled until all owner/retention/revocation gates pass combined
tests. Private dream indexes must not be placed in the shared reference FAISS.

## Validation

Synthetic tests use Python 3.11/Django 5.2/PostgreSQL 15 and standalone
`mysite.settings_auth_test`; they never resolve deployed credentials. Run:

```bash
rtk proxy python manage.py test myapp.test_auth_security myapp.test_service_auth myapp.test_data_copy myapp.test_throttle myapp.test_private_routes myapp.test_fastapi_routes myapp.test_proxy_address myapp.test_fixture_and_jobs --settings=mysite.settings_auth_test --noinput
rtk proxy python manage.py makemigrations --check --dry-run --settings=mysite.settings_auth_test
```

47 tests pass, including separate-connection transaction lock checks. Chat's
`test_security.py` passes four tests with its unchanged requirements. Production
configuration and the isolated Compose config pass checks. Actual partial export
copy/replay/digest/rollback was rehearsed privately in a disposable DB, destroyed
afterward. Fresh image build, interactive combined browser QA, complete source
export and real account provisioning remain separate prerequisites.
