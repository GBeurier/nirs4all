"""Main systematic selector class with all stages."""

import os
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import ResultCache, compute_config_hash, compute_data_hash
from .data_classes import DiversityAnalysis, PipelineResult
from .distances import compute_all_distances
from .metrics import evaluate_unsupervised
from .pipelines import apply_pipeline, generate_stacked_pipelines, get_base_preprocessings
from .proxy import evaluate_with_proxies


def compute_combinatorics(
    n_preprocessings: int,
    max_depth: int,
    top_stage1: int,
    top_stage2: int,
    top_stage3: int,
    top_stage4: int,
    top_final: int,
    similarity_ratio: float,
    augmentation_order: int,
) -> Dict[str, Any]:
    """Compute the number of items that will be evaluated at each stage.

    Args:
        n_preprocessings: Number of base preprocessing transforms.
        max_depth: Maximum pipeline depth.
        top_stage1: Number of candidates from Stage 1 to Stage 2.
        top_stage2: Number of diverse candidates from Stage 2 to Stage 3.
        top_stage3: Number of top candidates from Stage 3.
        top_stage4: Number of top augmentations from Stage 4.
        top_final: Number of final configurations.
        similarity_ratio: Similarity threshold for diversity filtering.
        augmentation_order: Maximum augmentation order (2 or 3).

    Returns:
        Dictionary with counts for each stage.
    """
    from math import factorial, comb

    # Stage 1: All stacked pipeline combinations
    # For depth d: P(n, d) = n! / (n-d)!
    stage1_total = 0
    stage1_by_depth = {}
    for d in range(1, max_depth + 1):
        if d <= n_preprocessings:
            count = factorial(n_preprocessings) // factorial(n_preprocessings - d)
            stage1_by_depth[d] = count
            stage1_total += count

    # Stage 2: Pairwise distance computations
    stage2_input = min(top_stage1, stage1_total)
    stage2_pairs = stage2_input * (stage2_input - 1) // 2

    # Stage 3: Proxy model evaluations
    stage3_input = min(top_stage2, stage2_input)
    stage3_evaluations = stage3_input

    # Stage 4: Augmentation evaluations
    stage4_input = min(top_stage3, stage3_input)
    aug_2way = comb(stage4_input, 2) if stage4_input >= 2 else 0
    aug_3way = 0
    if augmentation_order >= 3:
        triplets = comb(stage4_input, 3) if stage4_input >= 3 else 0
        aug_3way = min(triplets, 50)  # Limited to 50 as in code
    stage4_total = aug_2way + aug_3way

    # Final
    final_count = min(top_final, top_stage3 + top_stage4)

    return {
        "n_preprocessings": n_preprocessings,
        "max_depth": max_depth,
        "stage1_by_depth": stage1_by_depth,
        "stage1_total": stage1_total,
        "stage2_input": stage2_input,
        "stage2_pairs": stage2_pairs,
        "stage3_input": stage3_input,
        "stage3_evaluations": stage3_evaluations,
        "stage4_input": stage4_input,
        "stage4_2way": aug_2way,
        "stage4_3way": aug_3way,
        "stage4_total": stage4_total,
        "final_count": final_count,
    }


def print_combinatorics_overview(
    n_preprocessings: int,
    max_depth: int,
    top_stage1: int,
    top_stage2: int,
    top_stage3: int,
    top_stage4: int,
    top_final: int,
    similarity_ratio: float,
    augmentation_order: int,
) -> Dict[str, Any]:
    """Print a summary of what will be computed at each stage.

    Args:
        n_preprocessings: Number of base preprocessing transforms.
        max_depth: Maximum pipeline depth.
        top_stage1: Number of candidates from Stage 1 to Stage 2.
        top_stage2: Number of diverse candidates from Stage 2 to Stage 3.
        top_stage3: Number of top candidates from Stage 3.
        top_stage4: Number of top augmentations from Stage 4.
        top_final: Number of final configurations.
        similarity_ratio: Similarity threshold for diversity filtering.
        augmentation_order: Maximum augmentation order (2 or 3).

    Returns:
        Dictionary with the computed combinatorics.
    """
    stats = compute_combinatorics(
        n_preprocessings, max_depth, top_stage1, top_stage2, top_stage3,
        top_stage4, top_final, similarity_ratio, augmentation_order
    )

    print("\n" + "=" * 70)
    print("📊 COMBINATORICS OVERVIEW")
    print("=" * 70)

    print(f"\n🔧 Base preprocessings: {stats['n_preprocessings']}")
    print(f"   Max depth: {stats['max_depth']}")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ STAGE 1: Unsupervised Evaluation                                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    for depth, count in stats["stage1_by_depth"].items():
        print(f"│   Depth {depth}: {count:>6} pipelines (permutations P({stats['n_preprocessings']},{depth}))          │")
    print("│   ─────────────────────────────────────────────────                │")
    print(f"│   Total: {stats['stage1_total']:>6} pipeline evaluations                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ STAGE 2: Diversity Analysis                                         │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│   Input: top {stats['stage2_input']:>4} from Stage 1                                     │")
    print(f"│   Pairs: {stats['stage2_pairs']:>6} distance computations (6 metrics each)           │")
    print(f"│   Filtering: similarity > {similarity_ratio:.2f} removed                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ STAGE 3: Proxy Model Evaluation                                     │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│   Input: top {stats['stage3_input']:>4} diverse candidates from Stage 2                 │")
    print(f"│   Evaluations: {stats['stage3_evaluations']:>4} × 3 models (Ridge, KNN, XGBoost)            │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ STAGE 4: Augmentation Evaluation                                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│   Input: top {stats['stage4_input']:>4} from Stage 3                                     │")
    print(f"│   2-way augmentations: C({stats['stage4_input']},2) = {stats['stage4_2way']:>4}                               │")
    if augmentation_order >= 3:
        print(f"│   3-way augmentations: min(C({stats['stage4_input']},3), 50) = {stats['stage4_3way']:>4}                       │")
    print("│   ─────────────────────────────────────────────────────────────    │")
    print(f"│   Total: {stats['stage4_total']:>4} augmentation evaluations                          │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ FINAL OUTPUT                                                        │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│   Top {stats['final_count']:>4} configurations (from Stage 3 + Stage 4)               │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    total_evaluations = stats["stage1_total"] + stats["stage4_total"]
    total_distances = stats["stage2_pairs"] * 6
    print(f"\n📈 Total: {total_evaluations} pipeline evaluations + {total_distances} distance computations")
    print("=" * 70 + "\n")

    return stats


class SystematicSelector:
    """Systematic preprocessing selection with exhaustive evaluation.

    Stages:
        1. Exhaustive unsupervised evaluation of all pipelines
        2. Diversity analysis with pairwise distances
        3. Proxy model evaluation of top candidates
        4. Augmentation evaluation (2nd and 3rd order concatenations)

    Supports caching to avoid recomputing results when the same dataset
    and preprocessing configuration are used.
    """

    def __init__(self, verbose: int = 1, cache_dir: str = ".cache", use_cache: bool = True):
        """Initialize the selector.

        Args:
            verbose: Verbosity level (0=silent, 1=normal, 2=debug).
            cache_dir: Directory to store cached results.
            use_cache: Whether to use caching (default: True).
        """
        self.verbose = verbose
        self.results: List[PipelineResult] = []
        self.distance_matrix: Optional[pd.DataFrame] = None
        self.use_cache = use_cache
        self.cache = ResultCache(cache_dir) if use_cache else None
        self._data_hash: Optional[str] = None
        self._config_hash: Optional[str] = None

    def _log(self, msg: str, level: int = 1):
        """Log a message if verbosity level allows."""
        if self.verbose >= level:
            print(msg)

    def run_stage1_unsupervised(
        self,
        X: np.ndarray,
        preprocessings: Dict[str, Any],
        max_depth: int = 3,
    ) -> pd.DataFrame:
        """Stage 1: Exhaustive unsupervised evaluation of all pipelines.

        Args:
            X: Input data matrix (n_samples, n_features).
            preprocessings: Dictionary of available transforms.
            max_depth: Maximum pipeline depth.

        Returns:
            DataFrame with all results sorted by total_score.
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 1: Exhaustive Unsupervised Evaluation")
        self._log("=" * 70)

        # Compute hashes for caching
        self._data_hash = compute_data_hash(X)
        self._config_hash = compute_config_hash(preprocessings, max_depth)

        # Check cache
        if self.use_cache and self.cache:
            cached = self.cache.load_stage1(self._data_hash, self._config_hash)
            if cached is not None:
                self._log(f"\n✓ Loaded {len(cached)} results from cache")
                self.results = cached
                return self._results_to_dataframe()

        # Generate all stacked pipelines
        pipelines = generate_stacked_pipelines(preprocessings, max_depth)
        self._log(f"\nGenerated {len(pipelines)} pipeline combinations (depth 1-{max_depth})")

        self.results = []

        for i, (name, components, transforms) in enumerate(pipelines):
            if self.verbose >= 1:
                print(f"\r  Evaluating [{i+1}/{len(pipelines)}] {name}...", end="", flush=True)

            try:
                # Apply pipeline
                X_t = apply_pipeline(X, transforms)

                # Check for valid output
                if np.any(np.isnan(X_t)) or np.any(np.isinf(X_t)):
                    self._log(f"\n  Warning: {name} produced NaN/Inf, skipping", 2)
                    continue

                # Compute metrics
                metrics = evaluate_unsupervised(X_t)

                result = PipelineResult(
                    name=name,
                    depth=len(components),
                    pipeline_type="stacked" if len(components) > 1 else "single",
                    components=components,
                    X_transformed=X_t,
                    metrics=metrics,
                    total_score=metrics["total_score"],
                )
                self.results.append(result)

            except Exception as e:
                self._log(f"\n  Error with {name}: {e}", 2)
                continue

        print()  # Newline after progress

        # Save to cache
        if self.use_cache and self.cache:
            self.cache.save_stage1(self._data_hash, self._config_hash, self.results)
            self._log("\n💾 Stage 1 results saved to cache")

        return self._results_to_dataframe()

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert self.results to a DataFrame.

        Returns:
            DataFrame with all results sorted by total_score.
        """
        df_data = []
        for r in self.results:
            row = {
                "name": r.name,
                "depth": r.depth,
                "type": r.pipeline_type,
                "components": "|".join(r.components),
                **r.metrics,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)

        self._log(f"\n✓ Evaluated {len(self.results)} valid pipelines")
        self._log("\nTop 10 by total score:")
        self._log(
            df[["name", "depth", "total_score", "variance_ratio", "effective_dim", "snr"]]
            .head(10)
            .to_string()
        )

        return df

    def run_stage2_diversity(
        self,
        top_k_in: int = 15,
        top_k_out: int = 15,
        similarity_ratio: float = 0.95,
        n_components: int = 5,
        k_neighbors: int = 10,
    ) -> Tuple[DiversityAnalysis, pd.DataFrame, List[PipelineResult]]:
        """Stage 2: Comprehensive diversity analysis with multiple metrics.

        Takes top_k_in candidates, computes pairwise distances using both
        subspace-based metrics (Grassmann, CKA, RV) and geometry-based metrics
        (Procrustes, trustworthiness, covariance). Produces two rankings:

        1. **Subspace diversity**: Emphasizes how different the transformed
           feature spaces are (principal component structure).
        2. **Geometry diversity**: Emphasizes how different the sample
           distributions are (shape, neighborhoods, alignment).

        Pipelines too similar (by combined distance) to a better-scored one
        are filtered out.

        Args:
            top_k_in: Number of top candidates from Stage 1 to analyze.
            top_k_out: Number of diverse candidates to output.
            similarity_ratio: Threshold for removing similar pipelines (0-1).
                Pipelines with similarity > ratio to a better-scored one are removed.
            n_components: Number of PCA components for subspace metrics.
            k_neighbors: Number of neighbors for trustworthiness metric.

        Returns:
            Tuple of (DiversityAnalysis object, metrics DataFrame, filtered results).
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 2: Diversity Analysis")
        self._log("=" * 70)

        # Compute config hash for stage 2 caching
        stage2_config_hash = compute_config_hash(
            {},
            0,
            stage="2",
            top_k_in=top_k_in,
            top_k_out=top_k_out,
            similarity_ratio=similarity_ratio,
            n_components=n_components,
            k_neighbors=k_neighbors,
            base_config=self._config_hash,
        )

        # Check cache
        if self.use_cache and self.cache and self._data_hash:
            cached = self.cache.load_stage2(self._data_hash, stage2_config_hash)
            if cached is not None:
                diversity_analysis, filtered_results = cached
                self._log(f"\n✓ Loaded diversity analysis from cache ({len(filtered_results)} candidates)")
                # Rebuild metrics DataFrame from diversity analysis
                metrics_df = self._build_stage2_metrics_df(diversity_analysis, filtered_results)
                self.distance_matrix = pd.DataFrame(
                    diversity_analysis.combined_matrix,
                    index=diversity_analysis.pipeline_names,
                    columns=diversity_analysis.pipeline_names,
                )
                return diversity_analysis, metrics_df, filtered_results

        # Get top results
        top_results = sorted(self.results, key=lambda x: x.total_score, reverse=True)[:top_k_in]
        names = [r.name for r in top_results]
        n = len(top_results)

        self._log(f"\nAnalyzing {n} top pipelines with 6 distance metrics:")
        self._log("  Subspace-based: Grassmann, CKA, RV coefficient")
        self._log("  Geometry-based: Procrustes, Trustworthiness, Covariance")

        # Initialize distance matrices
        grassmann_matrix = np.zeros((n, n))
        cka_matrix = np.zeros((n, n))
        rv_matrix = np.zeros((n, n))
        procrustes_matrix = np.zeros((n, n))
        trustworthiness_matrix = np.zeros((n, n))
        covariance_matrix = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        pair_count = 0

        self._log(f"\nComputing {total_pairs} pairwise distance comparisons...")

        for i in range(n):
            for j in range(i + 1, n):
                pair_count += 1
                if self.verbose >= 1:
                    print(
                        f"\r  Computing distances [{pair_count}/{total_pairs}]...",
                        end="",
                        flush=True,
                    )

                X1 = top_results[i].X_transformed
                X2 = top_results[j].X_transformed

                # Compute all distances
                distances = compute_all_distances(X1, X2, n_components, k_neighbors)

                # Fill matrices (symmetric)
                grassmann_matrix[i, j] = distances["grassmann"]
                grassmann_matrix[j, i] = distances["grassmann"]

                cka_matrix[i, j] = distances["cka"]
                cka_matrix[j, i] = distances["cka"]

                rv_matrix[i, j] = distances["rv"]
                rv_matrix[j, i] = distances["rv"]

                procrustes_matrix[i, j] = distances["procrustes"]
                procrustes_matrix[j, i] = distances["procrustes"]

                trustworthiness_matrix[i, j] = distances["trustworthiness"]
                trustworthiness_matrix[j, i] = distances["trustworthiness"]

                covariance_matrix[i, j] = distances["covariance"]
                covariance_matrix[j, i] = distances["covariance"]

                # Store per-pipeline diversity scores
                top_results[i].diversity_scores[f"vs_{names[j]}"] = distances
                top_results[j].diversity_scores[f"vs_{names[i]}"] = distances

        print()

        # Compute combined matrices
        subspace_matrix = 0.4 * grassmann_matrix + 0.4 * cka_matrix + 0.2 * rv_matrix
        geometry_matrix = 0.4 * procrustes_matrix + 0.3 * trustworthiness_matrix + 0.3 * covariance_matrix
        combined_matrix = 0.5 * subspace_matrix + 0.5 * geometry_matrix

        # Create distance DataFrames for output
        self.distance_matrix = pd.DataFrame(combined_matrix, index=names, columns=names)

        # Create DiversityAnalysis object
        diversity_analysis = DiversityAnalysis(
            grassmann_matrix=grassmann_matrix,
            cka_matrix=cka_matrix,
            rv_matrix=rv_matrix,
            procrustes_matrix=procrustes_matrix,
            trustworthiness_matrix=trustworthiness_matrix,
            covariance_matrix=covariance_matrix,
            subspace_matrix=subspace_matrix,
            geometry_matrix=geometry_matrix,
            combined_matrix=combined_matrix,
            pipeline_names=names,
        )

        # Compute diversity rankings
        self._log("\n📊 Computing diversity rankings...")

        # For each pipeline, compute mean distance to all others
        metrics_data = []
        for i, r in enumerate(top_results):
            mean_grassmann = np.mean([grassmann_matrix[i, j] for j in range(n) if j != i])
            mean_cka = np.mean([cka_matrix[i, j] for j in range(n) if j != i])
            mean_rv = np.mean([rv_matrix[i, j] for j in range(n) if j != i])
            mean_procrustes = np.mean([procrustes_matrix[i, j] for j in range(n) if j != i])
            mean_trust = np.mean([trustworthiness_matrix[i, j] for j in range(n) if j != i])
            mean_cov = np.mean([covariance_matrix[i, j] for j in range(n) if j != i])
            mean_subspace = np.mean([subspace_matrix[i, j] for j in range(n) if j != i])
            mean_geometry = np.mean([geometry_matrix[i, j] for j in range(n) if j != i])
            mean_combined = np.mean([combined_matrix[i, j] for j in range(n) if j != i])

            metrics_data.append({
                "name": names[i],
                "unsupervised_score": r.total_score,
                "depth": r.depth,
                # Individual distances
                "grassmann_dist": mean_grassmann,
                "cka_dist": mean_cka,
                "rv_dist": mean_rv,
                "procrustes_dist": mean_procrustes,
                "trust_dist": mean_trust,
                "cov_dist": mean_cov,
                # Combined distances
                "subspace_dist": mean_subspace,
                "geometry_dist": mean_geometry,
                "combined_dist": mean_combined,
            })

        metrics_df = pd.DataFrame(metrics_data)

        # Print metric statistics
        self._log("\nDistance metric summary (mean ± std across pipeline pairs):")
        for metric, matrix in [
            ("Grassmann", grassmann_matrix),
            ("CKA", cka_matrix),
            ("RV", rv_matrix),
            ("Procrustes", procrustes_matrix),
            ("Trustworthiness", trustworthiness_matrix),
            ("Covariance", covariance_matrix),
        ]:
            upper = matrix[np.triu_indices(n, k=1)]
            self._log(f"  {metric:15s}: {np.mean(upper):.4f} ± {np.std(upper):.4f}")

        # Create rankings
        # Subspace ranking: high subspace diversity, sorted by unsupervised score among diverse ones
        subspace_ranking = metrics_df.sort_values(
            ["subspace_dist", "unsupervised_score"], ascending=[False, False]
        )["name"].tolist()

        # Geometry ranking: high geometry diversity
        geometry_ranking = metrics_df.sort_values(
            ["geometry_dist", "unsupervised_score"], ascending=[False, False]
        )["name"].tolist()

        # Combined ranking
        combined_ranking = metrics_df.sort_values(
            ["combined_dist", "unsupervised_score"], ascending=[False, False]
        )["name"].tolist()

        diversity_analysis.subspace_ranking = subspace_ranking
        diversity_analysis.geometry_ranking = geometry_ranking
        diversity_analysis.combined_ranking = combined_ranking

        # Display rankings
        self._log("\n🏆 Diversity Rankings:")
        self._log("\n  Subspace Diversity (Grassmann + CKA + RV):")
        for i, name in enumerate(subspace_ranking[:5]):
            row = metrics_df[metrics_df["name"] == name].iloc[0]
            self._log(f"    {i+1}. {name} (dist={row['subspace_dist']:.4f})")

        self._log("\n  Geometry Diversity (Procrustes + Trust + Cov):")
        for i, name in enumerate(geometry_ranking[:5]):
            row = metrics_df[metrics_df["name"] == name].iloc[0]
            self._log(f"    {i+1}. {name} (dist={row['geometry_dist']:.4f})")

        # Filter out similar pipelines using combined distance
        distance_threshold = 1.0 - similarity_ratio
        kept_indices = []
        removed_info = []

        for i in range(n):
            is_similar_to_better = False
            for kept_i in kept_indices:
                if combined_matrix[kept_i, i] < distance_threshold:
                    is_similar_to_better = True
                    removed_info.append({
                        "removed": names[i],
                        "similar_to": names[kept_i],
                        "combined_dist": combined_matrix[kept_i, i],
                        "subspace_dist": subspace_matrix[kept_i, i],
                        "geometry_dist": geometry_matrix[kept_i, i],
                    })
                    break
            if not is_similar_to_better:
                kept_indices.append(i)

        filtered_results = [top_results[i] for i in kept_indices]

        self._log(f"\n✂️ Similarity Filtering (threshold: {similarity_ratio:.2f}):")
        self._log(f"   Removed {len(removed_info)} similar pipelines")
        if self.verbose >= 2 and removed_info:
            for info in removed_info[:5]:
                self._log(
                    f"   - {info['removed']} ≈ {info['similar_to']} "
                    f"(combined={info['combined_dist']:.4f})", 2
                )

        # Take top_k_out from filtered results
        filtered_results = filtered_results[:top_k_out]

        self._log(f"\n✓ Kept {len(filtered_results)} diverse pipelines")

        # Find most diverse pairs (best for augmentation)
        self._log("\n🔗 Most diverse pairs (best for augmentation):")
        diverse_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                diverse_pairs.append({
                    "pair": (names[i], names[j]),
                    "combined": combined_matrix[i, j],
                    "subspace": subspace_matrix[i, j],
                    "geometry": geometry_matrix[i, j],
                })

        # Sort by combined distance (descending = most diverse first)
        diverse_pairs.sort(key=lambda x: x["combined"], reverse=True)

        for pair_info in diverse_pairs[:5]:
            p1, p2 = pair_info["pair"]
            self._log(
                f"  {p1} + {p2}: "
                f"combined={pair_info['combined']:.4f}, "
                f"subspace={pair_info['subspace']:.4f}, "
                f"geometry={pair_info['geometry']:.4f}"
            )

        # Save to cache
        if self.use_cache and self.cache and self._data_hash:
            self.cache.save_stage2(self._data_hash, stage2_config_hash, diversity_analysis, filtered_results)
            self._log("\n💾 Stage 2 results saved to cache")

        return diversity_analysis, metrics_df, filtered_results

    def _build_stage2_metrics_df(
        self,
        diversity_analysis: DiversityAnalysis,
        filtered_results: List[PipelineResult],
    ) -> pd.DataFrame:
        """Build the metrics DataFrame for Stage 2 from cached data.

        Args:
            diversity_analysis: The DiversityAnalysis object.
            filtered_results: The filtered results list.

        Returns:
            DataFrame with diversity metrics.
        """
        n = len(diversity_analysis.pipeline_names)
        metrics_data = []

        # Find results matching pipeline names
        result_map = {r.name: r for r in filtered_results}

        for i, name in enumerate(diversity_analysis.pipeline_names):
            r = result_map.get(name)
            if r is None:
                continue

            mean_grassmann = np.mean([diversity_analysis.grassmann_matrix[i, j] for j in range(n) if j != i])
            mean_cka = np.mean([diversity_analysis.cka_matrix[i, j] for j in range(n) if j != i])
            mean_rv = np.mean([diversity_analysis.rv_matrix[i, j] for j in range(n) if j != i])
            mean_procrustes = np.mean([diversity_analysis.procrustes_matrix[i, j] for j in range(n) if j != i])
            mean_trust = np.mean([diversity_analysis.trustworthiness_matrix[i, j] for j in range(n) if j != i])
            mean_cov = np.mean([diversity_analysis.covariance_matrix[i, j] for j in range(n) if j != i])
            mean_subspace = np.mean([diversity_analysis.subspace_matrix[i, j] for j in range(n) if j != i])
            mean_geometry = np.mean([diversity_analysis.geometry_matrix[i, j] for j in range(n) if j != i])
            mean_combined = np.mean([diversity_analysis.combined_matrix[i, j] for j in range(n) if j != i])

            metrics_data.append({
                "name": name,
                "unsupervised_score": r.total_score,
                "depth": r.depth,
                "grassmann_dist": mean_grassmann,
                "cka_dist": mean_cka,
                "rv_dist": mean_rv,
                "procrustes_dist": mean_procrustes,
                "trust_dist": mean_trust,
                "cov_dist": mean_cov,
                "subspace_dist": mean_subspace,
                "geometry_dist": mean_geometry,
                "combined_dist": mean_combined,
            })

        return pd.DataFrame(metrics_data)

    def run_stage3_proxy_evaluation(
        self,
        y: np.ndarray,
        candidates: List[PipelineResult],
        top_k: int = 15,
        cv_folds: int = 3,
    ) -> Tuple[pd.DataFrame, List[PipelineResult]]:
        """Stage 3: Evaluate candidates with proxy models.

        Takes candidates from Stage 2, evaluates them using proxy models
        (Ridge, KNN, and XGBoost) and returns the top_k best performers.

        Args:
            y: Target values.
            candidates: List of candidates from Stage 2 to evaluate.
            top_k: Number of top candidates to return.
            cv_folds: Number of cross-validation folds.

        Returns:
            Tuple of (DataFrame with results, list of top_k PipelineResults).
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 3: Proxy Model Evaluation")
        self._log("=" * 70)

        # Compute hash for y to include in stage 3 cache key
        y_hash = compute_data_hash(y.reshape(-1, 1) if y.ndim == 1 else y)
        stage3_config_hash = compute_config_hash(
            {},
            0,
            stage="3",
            top_k=top_k,
            cv_folds=cv_folds,
            base_config=self._config_hash,
            y_hash=y_hash,
            candidate_names="|".join(sorted(c.name for c in candidates)),
        )

        # Check cache
        if self.use_cache and self.cache and self._data_hash:
            cached = self.cache.load_stage3(self._data_hash, stage3_config_hash)
            if cached is not None:
                self._log(f"\n✓ Loaded {len(cached)} proxy evaluation results from cache")
                # Return top_k
                cached.sort(key=lambda x: x.final_score, reverse=True)
                top_results = cached[:top_k]
                df = self._build_stage3_dataframe(cached)
                return df, top_results

        # Detect if classification
        unique_y = np.unique(y)
        is_classification = len(unique_y) < 20 and np.all(unique_y == unique_y.astype(int))
        task_type = "classification" if is_classification else "regression"
        self._log(f"\nDetected task: {task_type}")

        evaluated_results = []

        # Evaluate all candidates with proxy models
        self._log(f"\nEvaluating {len(candidates)} pipelines with proxy models (Ridge, KNN, XGBoost)...")

        for i, r in enumerate(candidates):
            if self.verbose >= 1:
                print(f"\r  [{i+1}/{len(candidates)}] {r.name}...", end="", flush=True)

            proxy_scores = evaluate_with_proxies(r.X_transformed, y, cv_folds, is_classification)
            r.proxy_scores = proxy_scores
            r.final_score = 0.4 * r.total_score + 0.6 * proxy_scores["proxy_score"]
            evaluated_results.append(r)

        print()

        # Sort by final_score and take top_k
        evaluated_results.sort(key=lambda x: x.final_score, reverse=True)
        top_results = evaluated_results[:top_k]

        # Save to cache
        if self.use_cache and self.cache and self._data_hash:
            self.cache.save_stage3(self._data_hash, stage3_config_hash, evaluated_results)
            self._log("\n💾 Stage 3 results saved to cache")

        df = self._build_stage3_dataframe(evaluated_results)

        self._log(f"\n✓ Evaluated {len(evaluated_results)} pipelines")
        self._log(f"\nTop {top_k} by proxy score:")
        self._log(
            df[["name", "type", "proxy_score", "final_score"]].head(top_k).to_string()
        )

        return df, top_results

    def _build_stage3_dataframe(self, results: List[PipelineResult]) -> pd.DataFrame:
        """Build DataFrame from Stage 3 results.

        Args:
            results: List of evaluated PipelineResult objects.

        Returns:
            DataFrame with proxy evaluation results.
        """
        df_data = []
        for r in results:
            row = {
                "name": r.name,
                "type": r.pipeline_type,
                "depth": r.depth,
                "unsupervised_score": r.total_score,
                "ridge_r2": r.proxy_scores.get("ridge_r2", 0),
                "knn_score": r.proxy_scores.get("knn_score", 0),
                "xgb_score": r.proxy_scores.get("xgb_score", 0),
                "proxy_score": r.proxy_scores.get("proxy_score", 0),
                "final_score": r.final_score,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        return df

    def run_stage4_augmentation(
        self,
        y: np.ndarray,
        top_results: List[PipelineResult],
        top_k: int = 15,
        augmentation_order: int = 3,
        cv_folds: int = 3,
    ) -> Tuple[pd.DataFrame, List[PipelineResult]]:
        """Stage 4: Generate and evaluate augmented concatenations.

        Takes the top_k results from Stage 3, generates augmentations up to
        the specified order, evaluates them with proxy models, and returns
        the top_k augmentations to extend the final list.

        Args:
            y: Target values.
            top_results: Top results from Stage 3.
            top_k: Number of top augmentations to return.
            augmentation_order: Maximum order of augmentations to generate (2 or 3).
                2 = only 2-way concatenations, 3 = 2-way and 3-way.
            cv_folds: Number of cross-validation folds.

        Returns:
            Tuple of (DataFrame with augmentation results, list of top_k augmented results).
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 4: Augmentation Evaluation")
        self._log("=" * 70)

        # Compute hash for stage 4 caching
        y_hash = compute_data_hash(y.reshape(-1, 1) if y.ndim == 1 else y)
        stage4_config_hash = compute_config_hash(
            {},
            0,
            stage="4",
            top_k=top_k,
            augmentation_order=augmentation_order,
            cv_folds=cv_folds,
            base_config=self._config_hash,
            y_hash=y_hash,
            input_names="|".join(sorted(r.name for r in top_results)),
        )

        # Check cache
        if self.use_cache and self.cache and self._data_hash:
            cached = self.cache.load_stage4(self._data_hash, stage4_config_hash)
            if cached is not None:
                self._log(f"\n✓ Loaded {len(cached)} augmentation results from cache")
                cached.sort(key=lambda x: x.final_score, reverse=True)
                top_augmented = cached[:top_k]
                df = self._build_stage4_dataframe(cached)
                return df, top_augmented

        # Detect if classification
        unique_y = np.unique(y)
        is_classification = len(unique_y) < 20 and np.all(unique_y == unique_y.astype(int))

        augmented_results = []

        # Generate 2-way augmentations (all pairs from top_results)
        pairs = list(combinations(range(len(top_results)), 2))
        self._log(f"\nEvaluating {len(pairs)} 2-way augmentations...")

        for idx, (i, j) in enumerate(pairs):
            r1, r2 = top_results[i], top_results[j]

            if self.verbose >= 1:
                print(
                    f"\r  [{idx+1}/{len(pairs)}] {r1.name} + {r2.name}...",
                    end="",
                    flush=True,
                )

            # Create augmented features
            X_aug = np.hstack([r1.X_transformed, r2.X_transformed])

            # Compute metrics
            metrics = evaluate_unsupervised(X_aug)
            proxy_scores = evaluate_with_proxies(X_aug, y, cv_folds, is_classification)

            aug_result = PipelineResult(
                name=f"[{r1.name}+{r2.name}]",
                depth=2,
                pipeline_type="augmented_2",
                components=[r1.name, r2.name],
                X_transformed=X_aug,
                metrics=metrics,
                total_score=metrics["total_score"],
                proxy_scores=proxy_scores,
                final_score=0.4 * metrics["total_score"] + 0.6 * proxy_scores["proxy_score"],
            )
            augmented_results.append(aug_result)

        print()

        # Generate 3-way augmentations if order >= 3
        if augmentation_order >= 3:
            triplets = list(combinations(range(len(top_results)), 3))
            max_triplets = min(len(triplets), 50)  # Limit to avoid explosion
            triplets = triplets[:max_triplets]

            self._log(f"\nEvaluating {len(triplets)} 3-way augmentations...")

            for idx, (i, j, k) in enumerate(triplets):
                r1, r2, r3 = top_results[i], top_results[j], top_results[k]

                if self.verbose >= 1:
                    print(
                        f"\r  [{idx+1}/{len(triplets)}] {r1.name} + {r2.name} + {r3.name}...",
                        end="",
                        flush=True,
                    )

                # Create augmented features
                X_aug = np.hstack([r1.X_transformed, r2.X_transformed, r3.X_transformed])

                # Compute metrics
                metrics = evaluate_unsupervised(X_aug)
                proxy_scores = evaluate_with_proxies(X_aug, y, cv_folds, is_classification)

                aug_result = PipelineResult(
                    name=f"[{r1.name}+{r2.name}+{r3.name}]",
                    depth=3,
                    pipeline_type="augmented_3",
                    components=[r1.name, r2.name, r3.name],
                    X_transformed=X_aug,
                    metrics=metrics,
                    total_score=metrics["total_score"],
                    proxy_scores=proxy_scores,
                    final_score=0.4 * metrics["total_score"] + 0.6 * proxy_scores["proxy_score"],
                )
                augmented_results.append(aug_result)

            print()

        # Sort by final_score and take top_k
        augmented_results.sort(key=lambda x: x.final_score, reverse=True)
        top_augmented = augmented_results[:top_k]

        # Save to cache
        if self.use_cache and self.cache and self._data_hash:
            self.cache.save_stage4(self._data_hash, stage4_config_hash, augmented_results)
            self._log("\n💾 Stage 4 results saved to cache")

        df = self._build_stage4_dataframe(augmented_results)

        self._log(f"\n✓ Evaluated {len(augmented_results)} augmentations")
        self._log(f"\nTop {top_k} augmentations:")
        self._log(
            df[["name", "type", "proxy_score", "final_score"]].head(top_k).to_string()
        )

        return df, top_augmented

    def _build_stage4_dataframe(self, results: List[PipelineResult]) -> pd.DataFrame:
        """Build DataFrame from Stage 4 results.

        Args:
            results: List of augmented PipelineResult objects.

        Returns:
            DataFrame with augmentation evaluation results.
        """
        df_data = []
        for r in results:
            row = {
                "name": r.name,
                "type": r.pipeline_type,
                "depth": r.depth,
                "unsupervised_score": r.total_score,
                "ridge_r2": r.proxy_scores.get("ridge_r2", 0),
                "knn_score": r.proxy_scores.get("knn_score", 0),
                "xgb_score": r.proxy_scores.get("xgb_score", 0),
                "proxy_score": r.proxy_scores.get("proxy_score", 0),
                "final_score": r.final_score,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        return df

    def run_full_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocessings: Dict[str, Any] = None,
        max_depth: int = 3,
        top_stage1: int = 15,
        top_stage2: int = 15,
        top_stage3: int = 15,
        top_stage4: int = 15,
        top_final: int = 15,
        similarity_ratio: float = 0.95,
        augmentation_order: int = 3,
        cv_folds: int = 3,
        output_dir: str = ".",
        cache_dir: str = None,
    ) -> Dict[str, Any]:
        """Run the complete systematic selection pipeline.

        This runs all 4 stages with configurable top_k values at each stage:
        - Stage 1: Unsupervised evaluation -> take top_stage1
        - Stage 2: Diversity filtering -> filter by similarity_ratio, take top_stage2
        - Stage 3: Proxy model evaluation -> take top_stage3
        - Stage 4: Augmentation evaluation (up to augmentation_order) -> take top_stage4
        - Final: top_stage3 + top_stage4 candidates, ranked, take top_final

        Args:
            X: Input data matrix (n_samples, n_features).
            y: Target values.
            preprocessings: Dictionary of transforms (default: base set).
            max_depth: Maximum pipeline depth for Stage 1.
            top_stage1: Number of top candidates to pass from Stage 1 to Stage 2.
            top_stage2: Number of diverse candidates to pass from Stage 2 to Stage 3.
            top_stage3: Number of top candidates from Stage 3 proxy evaluation.
            top_stage4: Number of top augmentations from Stage 4.
            top_final: Number of final candidates to return.
            similarity_ratio: Threshold for removing similar pipelines in Stage 2.
                Pipelines with similarity > ratio to a better-scored one are removed.
            augmentation_order: Maximum augmentation order in Stage 4 (2 or 3).
            cv_folds: Number of cross-validation folds.
            output_dir: Directory to save output files.
            cache_dir: Directory for caching results. If None, uses output_dir/.cache.

        Returns:
            Dictionary with all results and DataFrames.
        """
        if preprocessings is None:
            preprocessings = get_base_preprocessings()

        # Update cache directory if specified
        if cache_dir is not None and self.use_cache:
            self.cache = ResultCache(cache_dir)
        elif self.use_cache and self.cache is None:
            self.cache = ResultCache(os.path.join(output_dir, ".cache"))

        start_time = time.time()

        # Print combinatorics overview
        n_preprocessings = len(preprocessings)
        print_combinatorics_overview(
            n_preprocessings=n_preprocessings,
            max_depth=max_depth,
            top_stage1=top_stage1,
            top_stage2=top_stage2,
            top_stage3=top_stage3,
            top_stage4=top_stage4,
            top_final=top_final,
            similarity_ratio=similarity_ratio,
            augmentation_order=augmentation_order,
        )

        self._log("\n" + "=" * 70)
        self._log("SYSTEMATIC PREPROCESSING SELECTION")
        self._log("=" * 70)
        self._log("\nParameters:")
        self._log(f"  - max_depth: {max_depth}")
        self._log(f"  - top_stage1: {top_stage1}")
        self._log(f"  - top_stage2: {top_stage2}")
        self._log(f"  - top_stage3: {top_stage3}")
        self._log(f"  - top_stage4: {top_stage4}")
        self._log(f"  - top_final: {top_final}")
        self._log(f"  - similarity_ratio: {similarity_ratio}")
        self._log(f"  - augmentation_order: {augmentation_order}")

        # Stage 1: Unsupervised evaluation
        df_stage1 = self.run_stage1_unsupervised(X, preprocessings, max_depth)

        # Stage 2: Diversity analysis with filtering
        diversity_analysis, df_stage2, stage2_candidates = self.run_stage2_diversity(
            top_k_in=top_stage1,
            top_k_out=top_stage2,
            similarity_ratio=similarity_ratio,
        )

        # Stage 3: Proxy evaluation of Stage 2 candidates, get top_stage3
        df_stage3, top_stage3_results = self.run_stage3_proxy_evaluation(
            y, stage2_candidates, top_stage3, cv_folds
        )

        # Stage 4: Augmentation evaluation, get top_stage4 augmentations
        df_stage4, top_stage4_results = self.run_stage4_augmentation(
            y, top_stage3_results, top_stage4, augmentation_order, cv_folds
        )

        # Combine results: top_stage3 + top_stage4
        all_results = top_stage3_results + top_stage4_results

        # Sort and take top_final
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = all_results[:top_final]

        total_time = time.time() - start_time

        # Create final DataFrame
        df_data = []
        for r in final_results:
            row = {
                "name": r.name,
                "type": r.pipeline_type,
                "depth": r.depth,
                "unsupervised_score": r.total_score,
                "ridge_r2": r.proxy_scores.get("ridge_r2", 0),
                "knn_score": r.proxy_scores.get("knn_score", 0),
                "xgb_score": r.proxy_scores.get("xgb_score", 0),
                "proxy_score": r.proxy_scores.get("proxy_score", 0),
                "final_score": r.final_score,
            }
            df_data.append(row)

        df_final = pd.DataFrame(df_data)
        if not df_final.empty:
            df_final = df_final.sort_values("final_score", ascending=False).reset_index(drop=True)

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        stage1_path = os.path.join(output_dir, "stage1_unsupervised.csv")
        df_stage1.to_csv(stage1_path, index=False)
        self._log(f"\n📄 Stage 1 results saved to: {stage1_path}")

        stage2_path = os.path.join(output_dir, "stage2_diversity.csv")
        df_stage2.to_csv(stage2_path, index=False)
        self._log(f"📄 Stage 2 diversity metrics saved to: {stage2_path}")

        stage3_path = os.path.join(output_dir, "stage3_proxy.csv")
        df_stage3.to_csv(stage3_path, index=False)
        self._log(f"📄 Stage 3 results saved to: {stage3_path}")

        stage4_path = os.path.join(output_dir, "stage4_augmentation.csv")
        df_stage4.to_csv(stage4_path, index=False)
        self._log(f"📄 Stage 4 results saved to: {stage4_path}")

        final_path = os.path.join(output_dir, "final_ranking.csv")
        df_final.to_csv(final_path, index=False)
        self._log(f"📄 Final ranking saved to: {final_path}")

        # Save distance matrices
        if diversity_analysis.combined_matrix is not None:
            combined_dist_df = pd.DataFrame(
                diversity_analysis.combined_matrix,
                index=diversity_analysis.pipeline_names,
                columns=diversity_analysis.pipeline_names,
            )
            dist_path = os.path.join(output_dir, "distance_matrix_combined.csv")
            combined_dist_df.to_csv(dist_path)
            self._log(f"📄 Combined distance matrix saved to: {dist_path}")

            # Save subspace and geometry matrices
            subspace_dist_df = pd.DataFrame(
                diversity_analysis.subspace_matrix,
                index=diversity_analysis.pipeline_names,
                columns=diversity_analysis.pipeline_names,
            )
            subspace_path = os.path.join(output_dir, "distance_matrix_subspace.csv")
            subspace_dist_df.to_csv(subspace_path)
            self._log(f"📄 Subspace distance matrix saved to: {subspace_path}")

            geometry_dist_df = pd.DataFrame(
                diversity_analysis.geometry_matrix,
                index=diversity_analysis.pipeline_names,
                columns=diversity_analysis.pipeline_names,
            )
            geometry_path = os.path.join(output_dir, "distance_matrix_geometry.csv")
            geometry_dist_df.to_csv(geometry_path)
            self._log(f"📄 Geometry distance matrix saved to: {geometry_path}")

        # Print final summary
        self._log("\n" + "=" * 70)
        self._log("FINAL RANKING")
        self._log("=" * 70)
        self._log(f"\n⏱️ Total time: {total_time:.1f}s")
        self._log(f"\n🏆 Top {len(final_results)} Configurations:")
        if not df_final.empty:
            self._log(
                df_final[["name", "type", "unsupervised_score", "proxy_score", "final_score"]]
                .to_string()
            )

        return {
            "stage1_df": df_stage1,
            "stage2_df": df_stage2,
            "diversity_analysis": diversity_analysis,
            "distance_matrix": self.distance_matrix,  # Combined matrix as DataFrame
            "stage2_candidates": stage2_candidates,
            "stage3_df": df_stage3,
            "stage4_df": df_stage4,
            "final_df": df_final,
            "top_stage3": top_stage3_results,
            "top_stage4": top_stage4_results,
            "final_results": final_results,
            "total_time": total_time,
        }
