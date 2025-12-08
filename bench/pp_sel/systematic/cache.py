"""Caching utilities for systematic preprocessing selection.

Saves and loads intermediate results to avoid recomputing pipelines
when the same dataset and preprocessing config are used.
"""

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_classes import DiversityAnalysis, PipelineResult


def compute_data_hash(X: np.ndarray, y: Optional[np.ndarray] = None) -> str:
    """Compute a hash representing the dataset.

    Args:
        X: Input data matrix.
        y: Target values (optional).

    Returns:
        SHA256 hash string (first 16 chars) representing the data.
    """
    hasher = hashlib.sha256()

    # Hash X shape and a sample of values
    hasher.update(str(X.shape).encode())
    hasher.update(X.tobytes()[:10000])  # First 10KB of data

    if y is not None:
        hasher.update(str(y.shape).encode())
        hasher.update(y.tobytes()[:1000])

    return hasher.hexdigest()[:16]


def compute_config_hash(
    preprocessings: Dict[str, Any],
    max_depth: int,
    **kwargs
) -> str:
    """Compute a hash representing the configuration.

    Args:
        preprocessings: Dictionary of available transforms.
        max_depth: Maximum pipeline depth.
        **kwargs: Additional config parameters.

    Returns:
        SHA256 hash string (first 16 chars) representing the config.
    """
    hasher = hashlib.sha256()

    # Hash preprocessing names (sorted for consistency)
    pp_names = sorted(preprocessings.keys()) if preprocessings else []
    hasher.update(json.dumps(pp_names).encode())

    # Hash max_depth
    hasher.update(str(max_depth).encode())

    # Hash additional kwargs
    for key in sorted(kwargs.keys()):
        hasher.update(f"{key}={kwargs[key]}".encode())

    return hasher.hexdigest()[:16]


class ResultCache:
    """Cache manager for systematic selection results.

    Stores intermediate results in a cache directory, organized by
    data hash and config hash. Each stage's results are stored separately.
    """

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached results.
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_stage_path(
        self,
        data_hash: str,
        config_hash: str,
        stage: str
    ) -> str:
        """Get the file path for a cached stage result.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.
            stage: Stage identifier (e.g., "stage1", "stage2").

        Returns:
            Path to the cache file.
        """
        cache_key = f"{data_hash}_{config_hash}"
        stage_dir = os.path.join(self.cache_dir, cache_key)
        os.makedirs(stage_dir, exist_ok=True)
        return os.path.join(stage_dir, f"{stage}.pkl")

    def has_cached(
        self,
        data_hash: str,
        config_hash: str,
        stage: str
    ) -> bool:
        """Check if a cached result exists for a stage.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.
            stage: Stage identifier.

        Returns:
            True if cached result exists, False otherwise.
        """
        path = self._get_stage_path(data_hash, config_hash, stage)
        return os.path.exists(path)

    def save_stage1(
        self,
        data_hash: str,
        config_hash: str,
        results: List[PipelineResult],
    ) -> None:
        """Save Stage 1 results to cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.
            results: List of PipelineResult objects from Stage 1.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage1")

        # Convert to serializable format
        cached_data = []
        for r in results:
            cached_data.append({
                "name": r.name,
                "depth": r.depth,
                "pipeline_type": r.pipeline_type,
                "components": r.components,
                "X_transformed": r.X_transformed,
                "metrics": r.metrics,
                "total_score": r.total_score,
            })

        with open(path, "wb") as f:
            pickle.dump(cached_data, f)

    def load_stage1(
        self,
        data_hash: str,
        config_hash: str,
    ) -> Optional[List[PipelineResult]]:
        """Load Stage 1 results from cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.

        Returns:
            List of PipelineResult objects if cached, None otherwise.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage1")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                cached_data = pickle.load(f)

            results = []
            for d in cached_data:
                r = PipelineResult(
                    name=d["name"],
                    depth=d["depth"],
                    pipeline_type=d["pipeline_type"],
                    components=d["components"],
                    X_transformed=d["X_transformed"],
                    metrics=d["metrics"],
                    total_score=d["total_score"],
                )
                results.append(r)
            return results
        except Exception:
            return None

    def save_stage2(
        self,
        data_hash: str,
        config_hash: str,
        diversity_analysis: DiversityAnalysis,
        filtered_results: List[PipelineResult],
    ) -> None:
        """Save Stage 2 results to cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.
            diversity_analysis: DiversityAnalysis object.
            filtered_results: Filtered PipelineResult list.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage2")

        # Serialize diversity analysis
        da_data = {
            "grassmann_matrix": diversity_analysis.grassmann_matrix,
            "cka_matrix": diversity_analysis.cka_matrix,
            "rv_matrix": diversity_analysis.rv_matrix,
            "procrustes_matrix": diversity_analysis.procrustes_matrix,
            "trustworthiness_matrix": diversity_analysis.trustworthiness_matrix,
            "covariance_matrix": diversity_analysis.covariance_matrix,
            "subspace_matrix": diversity_analysis.subspace_matrix,
            "geometry_matrix": diversity_analysis.geometry_matrix,
            "combined_matrix": diversity_analysis.combined_matrix,
            "pipeline_names": diversity_analysis.pipeline_names,
            "subspace_ranking": diversity_analysis.subspace_ranking,
            "geometry_ranking": diversity_analysis.geometry_ranking,
            "combined_ranking": diversity_analysis.combined_ranking,
        }

        # Serialize filtered results
        filtered_data = []
        for r in filtered_results:
            filtered_data.append({
                "name": r.name,
                "depth": r.depth,
                "pipeline_type": r.pipeline_type,
                "components": r.components,
                "X_transformed": r.X_transformed,
                "metrics": r.metrics,
                "total_score": r.total_score,
                "diversity_scores": r.diversity_scores,
            })

        with open(path, "wb") as f:
            pickle.dump({"diversity_analysis": da_data, "filtered_results": filtered_data}, f)

    def load_stage2(
        self,
        data_hash: str,
        config_hash: str,
    ) -> Optional[Tuple[DiversityAnalysis, List[PipelineResult]]]:
        """Load Stage 2 results from cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.

        Returns:
            Tuple of (DiversityAnalysis, filtered_results) if cached, None otherwise.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage2")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)

            da_data = cached["diversity_analysis"]
            diversity_analysis = DiversityAnalysis(
                grassmann_matrix=da_data["grassmann_matrix"],
                cka_matrix=da_data["cka_matrix"],
                rv_matrix=da_data["rv_matrix"],
                procrustes_matrix=da_data["procrustes_matrix"],
                trustworthiness_matrix=da_data["trustworthiness_matrix"],
                covariance_matrix=da_data["covariance_matrix"],
                subspace_matrix=da_data["subspace_matrix"],
                geometry_matrix=da_data["geometry_matrix"],
                combined_matrix=da_data["combined_matrix"],
                pipeline_names=da_data["pipeline_names"],
                subspace_ranking=da_data["subspace_ranking"],
                geometry_ranking=da_data["geometry_ranking"],
                combined_ranking=da_data["combined_ranking"],
            )

            filtered_results = []
            for d in cached["filtered_results"]:
                r = PipelineResult(
                    name=d["name"],
                    depth=d["depth"],
                    pipeline_type=d["pipeline_type"],
                    components=d["components"],
                    X_transformed=d["X_transformed"],
                    metrics=d["metrics"],
                    total_score=d["total_score"],
                    diversity_scores=d.get("diversity_scores", {}),
                )
                filtered_results.append(r)

            return diversity_analysis, filtered_results
        except Exception:
            return None

    def save_stage3(
        self,
        data_hash: str,
        config_hash: str,
        results: List[PipelineResult],
    ) -> None:
        """Save Stage 3 results to cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.
            results: List of evaluated PipelineResult objects.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage3")

        cached_data = []
        for r in results:
            cached_data.append({
                "name": r.name,
                "depth": r.depth,
                "pipeline_type": r.pipeline_type,
                "components": r.components,
                "X_transformed": r.X_transformed,
                "metrics": r.metrics,
                "total_score": r.total_score,
                "proxy_scores": r.proxy_scores,
                "final_score": r.final_score,
                "diversity_scores": r.diversity_scores,
            })

        with open(path, "wb") as f:
            pickle.dump(cached_data, f)

    def load_stage3(
        self,
        data_hash: str,
        config_hash: str,
    ) -> Optional[List[PipelineResult]]:
        """Load Stage 3 results from cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.

        Returns:
            List of PipelineResult objects if cached, None otherwise.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage3")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                cached_data = pickle.load(f)

            results = []
            for d in cached_data:
                r = PipelineResult(
                    name=d["name"],
                    depth=d["depth"],
                    pipeline_type=d["pipeline_type"],
                    components=d["components"],
                    X_transformed=d["X_transformed"],
                    metrics=d["metrics"],
                    total_score=d["total_score"],
                    proxy_scores=d.get("proxy_scores", {}),
                    final_score=d.get("final_score", 0.0),
                    diversity_scores=d.get("diversity_scores", {}),
                )
                results.append(r)
            return results
        except Exception:
            return None

    def save_stage4(
        self,
        data_hash: str,
        config_hash: str,
        results: List[PipelineResult],
    ) -> None:
        """Save Stage 4 results to cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.
            results: List of augmented PipelineResult objects.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage4")

        cached_data = []
        for r in results:
            cached_data.append({
                "name": r.name,
                "depth": r.depth,
                "pipeline_type": r.pipeline_type,
                "components": r.components,
                "X_transformed": r.X_transformed,
                "metrics": r.metrics,
                "total_score": r.total_score,
                "proxy_scores": r.proxy_scores,
                "final_score": r.final_score,
            })

        with open(path, "wb") as f:
            pickle.dump(cached_data, f)

    def load_stage4(
        self,
        data_hash: str,
        config_hash: str,
    ) -> Optional[List[PipelineResult]]:
        """Load Stage 4 results from cache.

        Args:
            data_hash: Hash of the dataset.
            config_hash: Hash of the configuration.

        Returns:
            List of PipelineResult objects if cached, None otherwise.
        """
        path = self._get_stage_path(data_hash, config_hash, "stage4")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                cached_data = pickle.load(f)

            results = []
            for d in cached_data:
                r = PipelineResult(
                    name=d["name"],
                    depth=d["depth"],
                    pipeline_type=d["pipeline_type"],
                    components=d["components"],
                    X_transformed=d["X_transformed"],
                    metrics=d["metrics"],
                    total_score=d["total_score"],
                    proxy_scores=d.get("proxy_scores", {}),
                    final_score=d.get("final_score", 0.0),
                )
                results.append(r)
            return results
        except Exception:
            return None

    def clear_cache(self, data_hash: str = None, config_hash: str = None) -> int:
        """Clear cached results.

        Args:
            data_hash: If provided, only clear caches matching this data hash.
            config_hash: If provided, only clear caches matching this config hash.

        Returns:
            Number of cache entries cleared.
        """
        import shutil

        cleared = 0

        if not os.path.exists(self.cache_dir):
            return 0

        for entry in os.listdir(self.cache_dir):
            entry_path = os.path.join(self.cache_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            # Entry format: {data_hash}_{config_hash}
            parts = entry.split("_")
            if len(parts) != 2:
                continue

            entry_data_hash, entry_config_hash = parts

            should_clear = True
            if data_hash and entry_data_hash != data_hash:
                should_clear = False
            if config_hash and entry_config_hash != config_hash:
                should_clear = False

            if should_clear:
                shutil.rmtree(entry_path)
                cleared += 1

        return cleared
