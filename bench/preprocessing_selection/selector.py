"""
Preprocessing Selector
======================

Main orchestrator class for the preprocessing selection framework.
Coordinates all stages (A, B, C, D) to filter and rank preprocessing techniques
before running full ML/DL pipelines.

Usage:
    selector = PreprocessingSelector(verbose=1)
    results = selector.select(X, y, preprocessings, stages=['A', 'B', 'C'], top_k=5)
"""

import time
import numpy as np
from typing import Dict, List, Union, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin

from .metrics import evaluate_unsupervised, evaluate_supervised
from .proxy_models import evaluate_proxies
from .combinations import analyze_combinations


class PreprocessingSelector:
    """
    Main class for preprocessing selection framework.

    Coordinates filtering and ranking of preprocessing techniques through
    multiple stages to reduce the search space before full model training.

    Stages:
        A: Unsupervised filtering (PCA, SNR, Roughness, Distance)
        B: Supervised ranking (RV, CKA, Correlation, PLS)
        C: Proxy models (Ridge, KNN)
        D: Combination analysis (Mutual Info, Grassmann)

    Args:
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        random_state: Random seed for reproducibility
    """

    def __init__(self, verbose: int = 1, random_state: int = 42):
        self.verbose = verbose
        self.random_state = random_state
        np.random.seed(random_state)

    def _log(self, message: str, level: int = 1):
        """Print message if verbosity level is sufficient."""
        if self.verbose >= level:
            print(message)

    def _apply_preprocessing(
        self,
        X: np.ndarray,
        preprocessing: Union[BaseEstimator, TransformerMixin]
    ) -> np.ndarray:
        """Apply a preprocessing transformer to data."""
        try:
            X_transformed = preprocessing.fit_transform(X)
            return X_transformed
        except Exception as e:
            self._log(f"  Warning: Preprocessing failed: {e}", level=2)
            return None

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocessings: Dict[str, Any],
        stages: List[str] = None,
        top_k: int = 5,
        # Stage A parameters
        min_variance_ratio: float = 0.90,
        max_first_component_ratio: float = 0.99,
        min_snr_ratio: float = 0.5,
        max_roughness_ratio: float = 20.0,
        min_separation_ratio: float = 0.8,
        # Stage B parameters
        pls_n_components: int = 2,
        correlation_aggregation: str = 'max',
        cka_kernel: str = 'linear',
        # Stage C parameters
        cv_folds: int = 3,
        knn_neighbors: int = 3,
        # Stage D parameters
        grassmann_n_components: int = 5,
        mi_top_k_features: int = 50
    ) -> dict:
        """
        Run the preprocessing selection pipeline.

        Args:
            X: Original spectra data (n_samples, n_features)
            y: Target values (n_samples,)
            preprocessings: Dict of {name: preprocessing_transformer}
            stages: List of stages to run ['A', 'B', 'C', 'D'] (default: all)
            top_k: Number of top preprocessings to return

            Stage A parameters:
                min_variance_ratio: Minimum PCA cumulative variance
                max_first_component_ratio: Maximum 1st component variance
                min_snr_ratio: Minimum SNR ratio (after/before)
                max_roughness_ratio: Maximum roughness ratio
                min_separation_ratio: Minimum distance separation ratio

            Stage B parameters:
                pls_n_components: Number of PLS components
                correlation_aggregation: How to aggregate correlations
                cka_kernel: Kernel for CKA ('linear' or 'rbf')

            Stage C parameters:
                cv_folds: Number of cross-validation folds
                knn_neighbors: Number of neighbors for KNN

            Stage D parameters:
                grassmann_n_components: PCA components for Grassmann distance
                mi_top_k_features: Top features for MI redundancy

        Returns:
            dict with:
                - 'ranking': List of (name, score) tuples
                - 'stage_results': Dict of results per stage
                - 'valid_preprocessings': List of valid preprocessing names
                - 'eliminated': List of eliminated preprocessing names with reasons
                - 'combinations_2d': Recommended 2D combinations (if Stage D run)
                - 'timing': Dict of timing per stage
        """
        if stages is None:
            stages = ['A', 'B', 'C', 'D']

        start_time = time.time()
        results = {
            'stage_results': {},
            'valid_preprocessings': list(preprocessings.keys()),
            'eliminated': [],
            'timing': {}
        }

        # Determine task type
        task = 'classification' if len(np.unique(y)) < 10 else 'regression'
        self._log(f"Task type: {task}")
        self._log(f"Data shape: {X.shape}, Target shape: {y.shape}")
        self._log(f"Evaluating {len(preprocessings)} preprocessings...")

        # Transform all preprocessings
        self._log("\nApplying preprocessings...")
        transformed = {}
        for name, pp in preprocessings.items():
            X_transformed = self._apply_preprocessing(X.copy(), pp)
            if X_transformed is not None:
                transformed[name] = X_transformed
            else:
                results['eliminated'].append({
                    'name': name,
                    'stage': 'transform',
                    'reason': 'Transformation failed'
                })

        results['valid_preprocessings'] = list(transformed.keys())
        self._log(f"Successfully transformed: {len(transformed)}/{len(preprocessings)}")

        # =====================================================================
        # Stage A: Unsupervised Filtering
        # =====================================================================
        if 'A' in stages:
            stage_start = time.time()
            self._log("\n" + "="*60)
            self._log("Stage A: Unsupervised Filtering")
            self._log("="*60)

            stage_a_results = {}
            valid_after_a = []

            for name, X_pp in transformed.items():
                result = evaluate_unsupervised(
                    X_original=X,
                    X_preprocessed=X_pp,
                    y=y,
                    min_variance_ratio=min_variance_ratio,
                    max_first_component_ratio=max_first_component_ratio,
                    min_snr_ratio=min_snr_ratio,
                    max_roughness_ratio=max_roughness_ratio,
                    min_separation_ratio=min_separation_ratio
                )
                stage_a_results[name] = result

                if result['is_valid']:
                    valid_after_a.append(name)
                    self._log(f"  ✓ {name}: PASSED", level=2)
                else:
                    reasons = '; '.join(result['reasons'])
                    results['eliminated'].append({
                        'name': name,
                        'stage': 'A',
                        'reason': reasons
                    })
                    self._log(f"  ✗ {name}: FAILED - {reasons}", level=2)

            results['stage_results']['A'] = stage_a_results
            results['valid_preprocessings'] = valid_after_a
            results['timing']['A'] = time.time() - stage_start

            eliminated_count = len(transformed) - len(valid_after_a)
            self._log(f"\nStage A complete: {len(valid_after_a)} passed, {eliminated_count} eliminated")
            self._log(f"Time: {results['timing']['A']:.2f}s")

            # Update transformed dict to only include valid ones
            transformed = {k: v for k, v in transformed.items() if k in valid_after_a}

        # =====================================================================
        # Stage B: Supervised Ranking
        # =====================================================================
        if 'B' in stages and len(transformed) > 0:
            stage_start = time.time()
            self._log("\n" + "="*60)
            self._log("Stage B: Supervised Ranking")
            self._log("="*60)

            stage_b_results = {}

            for name, X_pp in transformed.items():
                result = evaluate_supervised(
                    X_preprocessed=X_pp,
                    y=y,
                    pls_n_components=pls_n_components,
                    correlation_aggregation=correlation_aggregation,
                    cka_kernel=cka_kernel
                )
                stage_b_results[name] = result
                self._log(f"  {name}: score={result['composite_score']:.4f}", level=2)

            results['stage_results']['B'] = stage_b_results
            results['timing']['B'] = time.time() - stage_start

            # Rank by composite score
            b_ranking = sorted(
                [(name, r['composite_score']) for name, r in stage_b_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            self._log(f"\nStage B Rankings:")
            for i, (name, score) in enumerate(b_ranking[:10]):
                self._log(f"  {i+1}. {name}: {score:.4f}")

            self._log(f"Time: {results['timing']['B']:.2f}s")

        # =====================================================================
        # Stage C: Proxy Models
        # =====================================================================
        if 'C' in stages and len(transformed) > 0:
            stage_start = time.time()
            self._log("\n" + "="*60)
            self._log("Stage C: Proxy Models")
            self._log("="*60)

            stage_c_results = {}

            for name, X_pp in transformed.items():
                result = evaluate_proxies(
                    X_preprocessed=X_pp,
                    y=y,
                    cv_folds=cv_folds,
                    knn_neighbors=knn_neighbors,
                    task=task
                )
                stage_c_results[name] = result
                self._log(f"  {name}: Ridge R²={result['ridge']['ridge_r2']:.4f}, "
                         f"KNN={result['knn']['knn_score']:.4f}", level=2)

            results['stage_results']['C'] = stage_c_results
            results['timing']['C'] = time.time() - stage_start

            # Rank by composite score
            c_ranking = sorted(
                [(name, r['composite_score']) for name, r in stage_c_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            self._log(f"\nStage C Rankings:")
            for i, (name, score) in enumerate(c_ranking[:10]):
                self._log(f"  {i+1}. {name}: {score:.4f}")

            self._log(f"Time: {results['timing']['C']:.2f}s")

        # =====================================================================
        # Stage D: Combination Analysis
        # =====================================================================
        if 'D' in stages and len(transformed) > 1:
            stage_start = time.time()
            self._log("\n" + "="*60)
            self._log("Stage D: Combination Analysis")
            self._log("="*60)

            stage_d_results = analyze_combinations(
                preprocessed_variants=transformed,
                y=y,
                task=task,
                n_components=grassmann_n_components,
                top_k_features=mi_top_k_features
            )

            results['stage_results']['D'] = stage_d_results
            results['timing']['D'] = time.time() - stage_start

            # Extract recommended combinations
            results['combinations_2d'] = stage_d_results['recommended_combinations']

            self._log(f"\nRecommended 2D Combinations:")
            for combo in stage_d_results['recommended_combinations'][:5]:
                self._log(f"  {combo['combination']}: score={combo['combined_score']:.4f}")

            self._log(f"Time: {results['timing']['D']:.2f}s")

        # =====================================================================
        # Final Ranking
        # =====================================================================
        self._log("\n" + "="*60)
        self._log("Final Results")
        self._log("="*60)

        # Compute final ranking based on available stages
        final_scores = {}
        for name in transformed.keys():
            scores = []
            weights = []

            if 'B' in results['stage_results'] and name in results['stage_results']['B']:
                scores.append(results['stage_results']['B'][name]['composite_score'])
                weights.append(1.0)

            if 'C' in results['stage_results'] and name in results['stage_results']['C']:
                scores.append(results['stage_results']['C'][name]['composite_score'])
                weights.append(2.0)  # Give more weight to proxy models

            if scores:
                final_scores[name] = np.average(scores, weights=weights)
            else:
                final_scores[name] = 0.0

        # Sort and take top_k
        ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        results['ranking'] = ranking[:top_k]

        self._log(f"\nTop {top_k} Preprocessings:")
        for i, (name, score) in enumerate(results['ranking']):
            self._log(f"  {i+1}. {name}: {score:.4f}")

        results['timing']['total'] = time.time() - start_time
        self._log(f"\nTotal time: {results['timing']['total']:.2f}s")

        return results

    def select_from_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocessing_list: List[Any],
        **kwargs
    ) -> dict:
        """
        Run selection from a list of preprocessings (auto-generate names).

        Args:
            X: Original spectra data
            y: Target values
            preprocessing_list: List of preprocessing transformers
            **kwargs: Additional arguments passed to select()

        Returns:
            Same as select()
        """
        # Generate names from class names
        preprocessings = {}
        name_counts = {}

        for pp in preprocessing_list:
            base_name = pp.__class__.__name__
            if base_name in name_counts:
                name_counts[base_name] += 1
                name = f"{base_name}_{name_counts[base_name]}"
            else:
                name_counts[base_name] = 1
                name = base_name

            preprocessings[name] = pp

        return self.select(X, y, preprocessings, **kwargs)


def print_selection_report(results: dict) -> None:
    """
    Print a formatted report of selection results.

    Args:
        results: Results dict from PreprocessingSelector.select()
    """
    print("\n" + "="*70)
    print("PREPROCESSING SELECTION REPORT")
    print("="*70)

    print("\n📊 FINAL RANKING:")
    print("-"*40)
    for i, (name, score) in enumerate(results['ranking']):
        print(f"  {i+1}. {name}: {score:.4f}")

    if results.get('eliminated'):
        print("\n❌ ELIMINATED PREPROCESSINGS:")
        print("-"*40)
        for item in results['eliminated']:
            print(f"  • {item['name']} (Stage {item['stage']}): {item['reason']}")

    if results.get('combinations_2d'):
        print("\n🔗 RECOMMENDED 2D COMBINATIONS:")
        print("-"*40)
        for combo in results['combinations_2d'][:5]:
            print(f"  • {combo['combination']}: {combo['combined_score']:.4f}")

    print("\n⏱️ TIMING:")
    print("-"*40)
    for stage, duration in results['timing'].items():
        print(f"  Stage {stage}: {duration:.2f}s")

    print("\n" + "="*70)
