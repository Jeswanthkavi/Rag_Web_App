"""Storage Provider Factory.

Instantiates the configured storage provider (LocalStorageProvider by default, extensible
for Cloud Object Storage providers such as S3 / GCS / Azure / Cloudflare R2).
"""

from typing import Optional
from backend.app.core.config import get_settings
from backend.app.storage.base import BaseStorageProvider
from backend.app.storage.local import LocalStorageProvider


def get_storage_provider(provider_type: Optional[str] = None) -> BaseStorageProvider:
    """Returns an instance of the configured document storage provider."""
    # Currently defaults to LocalStorageProvider; cloud object storage providers
    # can be injected via provider_type or configuration settings in production.
    return LocalStorageProvider()
