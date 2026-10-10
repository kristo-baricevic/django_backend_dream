# Exact RAG dependency persistence and serving contract

This is the current implementation contract plus a concrete auth integration recipe. The auth agent owns persistence, serving routes and migrations. No auth/longitudinal checkout was written here. Do not replace secured views with the audit baseline.

## What is actually captured now

For both cumulative and custom analysis, the completed generation `WorkflowStep.output_data` contains:

```json
{
  "coverage": {
    "selected_entries": 24,
    "retrieved_entries": 6,
    "included_entry_spans": 4,
    "omitted_retrieved_entries": 2,
    "source_scope": "selected_subset",
    "retrieval": {"queries_executed": 3},
    "source_dependencies": [
      {"entry_id": "00000000-0000-4000-8000-000000000001", "content_hash": "<64 lowercase SHA-256 hex characters>"}
    ]
  },
  "claims": [],
  "evidence": [],
  "semantic_support": "not_evaluated",
  "span_status": "validated_at_pack_time",
  "calibration_status": "unavailable"
}
```

The example is structural; counts/list length are abbreviated, not a valid complete fixture. Dependencies contain **every selected saved record**, including records omitted from the final evidence pack. Hash is `sha256(entry.content.encode('utf-8')).hexdigest()` over exact whole content; no normalization, prefix hashing, or excerpt hash. Evidence spans are half-open Unicode code points against the same source content. Private copied quotes are omitted from saved workflow evidence. Public-reference evidence hashes identify the reference chunk, not a whole PDF.

Owner authority is durable in `WorkflowExecution.user` plus `accepted_auth_user_id`, checked through auth's locked owner/workflow writer. The response metadata does **not** duplicate an owner UUID or capture source/privacy/review counters. The first profile-resolution step stores profile provenance/persona version, but that is **not** a longitudinal revision or full Settings revision. Do not fabricate zero revision captures or call content hashes privacy-consent revisions.

For single saved-entry analysis, `JournalAnalysis` returns `coverage.source_dependencies`, `claims`, `evidence`, `provenance` and semantic/span/calibration statuses. However the inspected auth bridge copies only legacy `Analysis` fields; **this metadata is not currently persisted by that bridge**. Its locked owner/content equality check exists. `Analysis` has no grounding-metadata field or workflow relation in the inspected model. Add nullable, versioned `Analysis.grounding_metadata` and workflow `source_metadata` through the auth-owned additive migration, and persist the returned metadata under the bridge's same locked save; do not overload weights, symbols, or interpretation text. These field names are task12's requested integration contract, not fields implemented here. Historical rows with null metadata remain unvalidated. No migration/backfill was made here.

Unsaved drafts return `user_draft`, null saved entry ID and null recording date. Their `source_dependencies` is empty. Their request-local evidence retains the real draft content hash/span without a saved identifier. Do not persist a draft dependency until a saved entry is explicitly created and analysis reruns against that owner-scoped saved row. Draft output still revalidates accepted account authority before return.

## Existing pure check to reuse

`core.rag_contracts.validate_source_dependencies(expected, entries)` is implemented and tested:

```python
actual = {str(entry.id): sha256(entry.content.encode('utf-8')).hexdigest()
          for entry in entries}
wanted = {item['entry_id']: item['content_hash'] for item in expected}
if len(wanted) != len(expected) or actual != wanted:
    raise ContractError('analysis_sources_changed')
```

The caller must obtain `entries` from `load_owned_entries(accepted_owner, dependency_ids)` inside its accepted locked write/read boundary. The pure function never authorizes arbitrary entry objects or body owner IDs. It deliberately fails when selected IDs are absent or the exact whole-source digest changed. If Django needs this helper, move the pure implementation into an agreed shared importable module; do not modify `sys.path` in request views or import a provider/analyzer to read saved results.

## Concrete auth serving recipe

The following is integration pseudocode using actual auth interfaces. `verify_revision_and_eligibility` is an integration hook to be implemented by the longitudinal authority; it is **not** an existing function or an approval to enable provider context.

```python
def read_current_result(owner, workflow_id):
    def read(profile):
        # owned_atomic_write already validates/locks the accepted auth account,
        # domain owner and nonhistorical workflow in this exact lock order.
        execution = WorkflowExecution.objects.get(pk=workflow_id, user=profile)
        if execution.status != 'completed':
            raise ResultUnavailable()
        output = latest_completed_generation_step(execution).output_data
        deps = require_nonempty_bounded_dependencies(output['coverage'])
        # Validate each stored ID as UUID and each hash as 64 lowercase hex;
        # max 500 unique IDs, no drafts or missing metadata in durable results.
        sources = load_owned_entries(owner, tuple(UUID(d['entry_id']) for d in deps))
        validate_source_dependencies(deps, sources)
        verify_revision_and_eligibility(owner, output, sources)
        result = load_result_for_execution(execution)
        require_matching_owner_and_workflow(result, execution, profile)
        # Materialize all derived fields/relations now; return no lazy ORM rows.
        return serialize_immutable_payload(execution, result, output)
    return owned_atomic_write(
        owner_id=owner.domain_user_id,
        expected_auth_user_id=owner.auth_user_id,
        workflow_id=workflow_id,
        lock_state=longitudinal_lock_state_hook,
        write=read,
    )
```

`latest_completed_generation_step` must require a unique current complete claims/evidence/dependency envelope belonging to this execution, not choose an arbitrary user-context step. Missing/malformed/historical metadata fails closed for derived answers. Only literal journal content remains available through its separately authorized route. Hydrate the cumulative/custom result, verify its `user` and `workflow_execution_id`, and serialize every derived field and relation inside the locked callback. Returning ORM objects to serialize later would weaken the read boundary. All entry, Settings and exclusion mutations must acquire the same owner/state locks; `load_owned_entries` itself does not select source rows for update. Audit that writer discipline or add coordinated source-row locks before claiming atomic freshness.

Use the same check for workflow detail/polling, cumulative/custom list/detail, journal `analysis` nesting, all query/statistics paths that include or aggregate generated fields, feedback-derived text, exports and chat answer/history replay. Literal-source statistics retain their separately authorized contract. List responses omit invalid derived bodies/citations/claims rather than serialize then mark them stale. Return a generic unavailable/stale state to the authenticated owner; foreign/missing IDs retain the auth 404 behavior. Do not include a stale `final_result`, old quotes, profile snippets or citations in error responses. Invalidate materialized answer/history copies and retained traces; a guard on one endpoint does not purge other copies. Do not retain or replay chat history containing derived context until a server-stamped Settings/source fence exists and is checked on every use.

Single bridge: under `owned_atomic_write`, load and lock the current entry, compare the accepted saved content, validate the returned complete dependency envelope against that entry, capture/recheck the same privacy/review eligibility before committing both Analysis metadata and entry association. Single read: hydrate the owned associated entry plus durable metadata and run the same hash/revision/eligibility gate before exposing any generated field. A content comparison only at save does not cover later edits/deletions.

## Revision and consent integration: mandatory, still missing

Capture a server-owned immutable entries/Settings snapshot and fence under the agreed auth→domain→longitudinal-state→source locks before retrieval. Capture the exact authoritative `{source, privacy, review}` tuple there, recheck the accepted snapshot/fence at publication, persist it additively with the grounding envelope, and compare it again at every serving/replay/export read. A body-supplied snapshot or counter is not authority. Use the auth/longitudinal owners' agreed fence schema and live source eligibility/exclusion rules. Metadata-only changes, consent revocation or exclusion can invalidate answers while content hashes remain equal. A content digest cannot substitute for those revisions. Also capture and compare the authoritative Settings/profile version necessary for purpose-specific context inclusion; current `persona_version` alone does not cover user Settings or consent.

If no revision/eligibility contract was captured, do not silently substitute the current revision to bless an old result. Mark it unvalidated and require explicit regeneration in the approved environment. Keep longitudinal provider context OFF; the DTO's source/privacy/review and retained-history obligations must pass before activation. No endorsed assistant hypothesis may become a user interpretation or established psychological fact. Recording time remains recording time; occurrence time is unknown.

Account remapping/deactivation/expiry must still reject publication and result reads. The inspected locked writer validates active account/mapping/import/workflow authority, but does not compare `OwnerContext.login_deadline`; expiry enforcement is an auth integration gate, not an implemented guarantee. The narrow redacted failure helper retains the existing auth/owner/workflow checks; revoked accounts cannot obtain even a cleanup write. Auth lifecycle cleanup must expire/purge abandoned jobs without reinstating revoked authority. No release claim is made until the combined two-owner/edit/delete/exclusion/revocation/expiry/poll/export/chat-replay matrix passes.
