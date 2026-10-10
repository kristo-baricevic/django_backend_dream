"""Context follows async task execution; authority is revalidated at commit."""
from contextvars import ContextVar
from dataclasses import dataclass
from asgiref.sync import sync_to_async
from rest_framework.exceptions import PermissionDenied
from .owner import owned_atomic_write, load_owned_entries, validate_request_session
from .freshness import lock_state, require_snapshot


@dataclass(frozen=True)
class JobContext:
    owner: object
    entry_ids: tuple
    source_metadata: object = None
    request_bound: bool = False


current_job = ContextVar('dream_current_job', default=None)


def job_write(callback, workflow_id=None):
    job = current_job.get()
    if job is None:
        raise PermissionDenied('Job owner unavailable.')
    def write(profile):
        # Source deletion or ownership changes invalidate a late job result.
        load_owned_entries(job.owner, job.entry_ids)
        return callback(profile)
    def fence(profile):
        lock_state(profile)
        if job.request_bound:
            validate_request_session(job.owner)
        if job.source_metadata is not None:
            require_snapshot(job.source_metadata, job.owner)
    return owned_atomic_write(owner_id=job.owner.domain_user_id,
                              expected_auth_user_id=job.owner.auth_user_id,
                              workflow_id=workflow_id, lock_state=fence, write=write)


async def async_job_write(callback, workflow_id=None):
    return await sync_to_async(job_write, thread_sensitive=True)(callback, workflow_id)
