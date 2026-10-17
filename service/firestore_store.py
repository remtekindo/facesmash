"""
Firestore persistence for the Face Cloak service.

Every completed /protect job is stored as one document in the
`protection_jobs` collection (configurable via FIRESTORE_COLLECTION).
Only parameters, metrics and the verdict are persisted — NEVER the
photo bytes or any user-identifying data.

Design rules (competition-critical):
- Auth: uses the Cloud Run default service account via Application
  Default Credentials. The service account needs the `datastore.user`
  IAM role on the project (see FIRESTORE_SETUP.md). No key files.
- Fail-safe: Firestore must NEVER break /protect. Client init and
  every write are wrapped; on any failure the job is skipped and a
  warning is logged. The prototype's "must stay functional during
  evaluation" rule takes precedence over persistence.
- Synchronous write (~50–200 ms) — negligible next to the 16–25 s
  PGD run, and guaranteed before the response returns (a Cloud Run
  instance with min-instances=0 must not be relied on to finish
  background work after responding).
- Controlled by env: FIRESTORE_ENABLED=0 disables persistence
  entirely (useful for local dev without GCP credentials).
"""
import logging
import os

log = logging.getLogger("firestore_store")

_ENABLED = os.environ.get("FIRESTORE_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "protection_jobs")

_client = None
_client_failed = False


def _get_client():
    """Lazily create the Firestore client; None if unavailable (logged once)."""
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None
    try:
        from google.cloud import firestore

        project = (
            os.environ.get("GCP_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        _client = firestore.Client(project=project) if project else firestore.Client()
        return _client
    except Exception as exc:  # missing creds / API not enabled / no project
        _client_failed = True
        log.warning("firestore disabled: client init failed (%s)", exc)
        return None


def save_protection_job(job: dict) -> str | None:
    """
    Persist one protection job. Returns the Firestore document id, or None
    if persistence is disabled or the write failed (never raises).
    """
    if not _ENABLED:
        return None
    try:
        from google.cloud import firestore

        client = _get_client()
        if client is None:
            return None
        doc = {
            "created_at": firestore.SERVER_TIMESTAMP,
            **job,
        }
        _, ref = client.collection(_COLLECTION).add(doc)
        return ref.id
    except Exception as exc:
        log.warning("firestore write skipped (%s)", exc)
        return None
