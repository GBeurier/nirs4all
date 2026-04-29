"""Adapters from existing nirs4all synthesis APIs to bench contracts."""

from .prior_adapter import (
    DOMAIN_ALIASES,
    PriorCanonicalizationError,
    PriorConfigRecord,
    PriorValidationIssue,
    canonicalize_domain,
    canonicalize_prior_config,
    sample_canonical_prior,
    summarize_prior_coverage,
)

__all__ = [
    "DOMAIN_ALIASES",
    "PriorCanonicalizationError",
    "PriorConfigRecord",
    "PriorValidationIssue",
    "canonicalize_domain",
    "canonicalize_prior_config",
    "sample_canonical_prior",
    "summarize_prior_coverage",
]
