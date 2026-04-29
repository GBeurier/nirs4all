"""Data contracts for synthetic datasets, latents, views, and tasks."""

from .latents import CanonicalLatentBatch, CanonicalLatentBatchError
from .multiview import (
    SpectralViewVariantConfig,
    build_same_latent_spectral_views,
)
from .task_sampling import (
    ContextQuerySplit,
    ContextQuerySplitConfig,
    ContextQuerySplitError,
    sample_context_query_split,
    sample_nirs_prior_task,
)
from .tasks import NIRSPriorTask, NIRSPriorTaskError
from .views import SpectralViewBatch, SpectralViewBatchError

__all__ = [
    "CanonicalLatentBatch",
    "CanonicalLatentBatchError",
    "ContextQuerySplit",
    "ContextQuerySplitConfig",
    "ContextQuerySplitError",
    "NIRSPriorTask",
    "NIRSPriorTaskError",
    "SpectralViewBatch",
    "SpectralViewBatchError",
    "SpectralViewVariantConfig",
    "build_same_latent_spectral_views",
    "sample_context_query_split",
    "sample_nirs_prior_task",
]
