"""Cache configuration for nirs4all pipeline execution.

Provides a single, typed entry point for all cache-related settings.
Each feature is independently toggleable for rollback without code changes.

The config flows through PipelineRunner -> PipelineOrchestrator -> RuntimeContext.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuration for step-level caching and observability.

    Attributes:
        step_cache_enabled: Enable step-level caching for cross-variant reuse.
            OFF by default until correctness tests confirm no regressions.
        step_cache_max_mb: Maximum memory budget for step cache in megabytes.
        step_cache_max_entries: Maximum number of cached step entries.
        use_cow_snapshots: Use copy-on-write snapshots for branch features (Phase 3).
        log_cache_stats: Log end-of-run cache statistics at verbose >= 1.
        log_step_memory: Log per-step memory stats at verbose >= 2.
        memory_warning_threshold_mb: RSS threshold in MB for memory warnings.
    """

    # Step cache (Phase 2) -- OFF by default until stress-tested
    step_cache_enabled: bool = False
    step_cache_max_mb: int = 2048
    step_cache_max_entries: int = 200

    # Branch snapshots (Phase 3)
    use_cow_snapshots: bool = True

    # Observability (Phase 1.5)
    log_cache_stats: bool = True
    log_step_memory: bool = True
    memory_warning_threshold_mb: int = 3072
