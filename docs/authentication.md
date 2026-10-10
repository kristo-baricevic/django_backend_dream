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

Migration 0020 adds the fourteen private longitudinal resources. The opt-in
profile worker stores accepted owner/auth IDs, not credentials, and extracts
literal English candidates locally. Auditing starts disabled and sources start
excluded. Generated and unknown provenance requires explicit reclassification.
Provider profile context stays disabled and empty. Private dream indexes must
never be placed in the shared reference FAISS.

## Source and session fences

`X-Dream-Principal` and `X-Dream-Session` are optional rejection-only browser
preconditions. They must match the authenticated owner and immutable deadline;
they never grant access. Before a request-bound read/write callback, the server
locks and rechecks the stored Session row, its current expiry, backend, auth hash
and deadline. Password changes, logout and expiry while waiting for a lock block
the callback. Accepted durable jobs omit session credentials and deliberately
outlive logout; deactivation, relinking and source/settings/privacy changes still
prevent provider dispatch or result publication.

Source edits/deletes carry `expected_revision` (source/privacy/review),
`expected_session`, `expected_source` (entry ID/full UTF-8 hash/update timestamp)
and `expected_operation`, plus the principal header. The frontend captures a
preview and reconfirms the session after the warning. Under auth → domain →
longitudinal state → workflow locks, the backend compares that complete ticket
and invalidates private derived state before touching the source. No-op writes
preserve review history; stale current analysis pointers detach for reanalysis.

Migration 0021 adds durable analysis/workflow grounding metadata and preserves
analysis versions. Server-owned snapshots seal the accepted account, every
selected source's whole-content hash, all three revision counters and canonical
Settings/doctor configuration. They are rechecked before provider work,
publication and immutable DTO serialization. Result reads, export-shaped list
responses, generated-field searches, statistics and feedback ignore stale or
ineligible derivations. Selected sources omitted by retrieval remain dependencies.
Deleting a workflow cannot turn a current result into historical data.

Existing analyses are explicitly marked historical rather than assigned invented
current consent/revisions. They remain labelled historical in the UI and cannot
contribute generated fields to current journal searches, statistics or chat.
Historical cumulative/question result lists remain owner-scoped and searchable.
Reverse
migration refuses to discard additional versions or grounding metadata.
Copy-ledger digest version 2 rebases old unchanged records only after matching
their old-schema digest and the exact permitted additive defaults. Changed or
unknown rows remain incompatible and block reconciliation/rollback.

The private-copy contract inventories all fourteen resources. Only explicitly
selected authored reflections may be copied, with remapped relations, preserved
timestamps and consent reset. Audits, themes, profiles, jobs and private indexes
are discarded and rebuilt privately. Unknown resources/index namespaces fail
closed. This does not fill the missing full source export described above.

## RAG and chat scope

Analysis accepts at most 500 entries, 200,000 content characters and a 4,000
character question. Evidence identifies source spans and distinguishes record
observations, public references and hypotheses. Calibration and semantic
entailment are unavailable; structural evidence checks are not clinical or
semantic validation. Fictional persona metadata is explicit, medical history is
excluded, and the existing medical-influence control has no numerical effect.

Chat discards caller-supplied prior conversation and compares Django's server
context fence before tools, model dispatch and each released chunk. Saving a
model-generated dream is disabled; the journal editor is the persistence path.
No retained private conversation or longitudinal provider context is enabled.

## Validation

Synthetic tests use Python 3.11/Django 5.2/PostgreSQL 15 and standalone
`mysite.settings_auth_test`; they never resolve deployed credentials. Run:

```bash
rtk proxy python manage.py test myapp.test_auth_security myapp.test_data_copy myapp.test_fastapi_routes myapp.test_private_routes myapp.test_service_auth myapp.test_throttle myapp.test_proxy_address myapp.test_fixture_and_jobs myapp.test_source_integration myapp.test_grounded_retention myapp.test_private_copy_integration myapp.test_grounded_migration myapp.test_request_fences myapp.test_rag_orm_integration myapp.longitudinal_tests.test_actual_auth myapp.longitudinal_tests.test_core myapp.longitudinal_tests.test_copy_contract --settings=mysite.settings_auth_test --noinput
rtk proxy python manage.py makemigrations --check --dry-run --settings=mysite.settings_auth_test
rtk proxy python manage.py test myapp.longitudinal_tests.test_api myapp.longitudinal_tests.test_services myapp.longitudinal_tests.test_core myapp.longitudinal_tests.test_copy_contract --settings=myapp.longitudinal_tests.settings --noinput
rtk proxy python -m unittest discover -s rag_tests
```

The actual-auth PostgreSQL aggregate has 112 distinct tests, including separate-connection
expiry/lock checks and production analyzer/tracker/serving with synthetic
provider boundaries. The independent longitudinal suite has 50 tests (21 overlap
the aggregate); pure RAG regressions have 35. Chat's ten security tests use its
unchanged requirements in a separate environment. No provider is called.

The actual partial export copy/replay/digest/rollback was rehearsed privately in
a disposable database. New tests rehearse pre-0021 ledgers, reverse-migration
refusal and changed-record rollback rejection. Full combined browser QA, complete
source database/files/index inventory, fresh coordinated images and action-time
approval for real account provisioning remain prerequisites for live use. No
live migration, persistent account provisioning or deployment occurred.
