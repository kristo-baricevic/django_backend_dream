"""Context follows async task execution; authority is revalidated at commit."""
from contextvars import ContextVar
from dataclasses import dataclass
from asgiref.sync import sync_to_async
from rest_framework.exceptions import PermissionDenied
from .owner import owned_atomic_write, load_owned_entries


@dataclass(frozen=True)
class JobContext:
    owner: object
    entry_ids: tuple


current_job = ContextVar('dream_current_job', default=None)


def job_write(callback, workflow_id=None):
    job = current_job.get()
    if job is None:
        raise PermissionDenied('Job owner unavailable.')
    def write(profile):
        # Source deletion or ownership changes invalidate a late job result.
        load_owned_entries(job.owner, job.entry_ids)
        return callback(profile)
    return owned_atomic_write(owner_id=job.owner.domain_user_id,
                              expected_auth_user_id=job.owner.auth_user_id,
                              workflow_id=workflow_id, write=write)


async def async_job_write(callback, workflow_id=None):
    return await sync_to_async(job_write, thread_sensitive=True)(callback, workflow_id)
