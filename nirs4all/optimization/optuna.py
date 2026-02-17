"""
Optuna Manager - External hyperparameter optimization logic

This module combines the best practices from the original optuna_manager for parameter handling
and sampling with fold-based optimization strategies. It provides a clean interface for
hyperparameter optimization across different strategies and frameworks.
"""

import os
os.environ['DISABLE_EMOJIS'] = '1'  # Set to '1' to disable emojis in print statements

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext

try:
    import optuna
    from optuna.samplers import TPESampler, GridSampler, RandomSampler, CmaEsSampler, BaseSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    BaseSampler = object  # Fallback for type checking

VALID_SAMPLERS = {"auto", "grid", "tpe", "random", "cmaes", "binary"}
VALID_APPROACHES = {"single", "grouped", "individual"}
VALID_EVAL_MODES = {"best", "mean", "robust_best"}
VALID_PRUNERS = {"none", "median", "successive_halving", "hyperband"}

# Metrics supported for the unified ``metric`` field in finetune_params.
# Direction is auto-inferred from the metric when not explicitly set.
METRIC_DIRECTION = {
    # Regression (minimize)
    "mse": "minimize",
    "rmse": "minimize",
    "mae": "minimize",
    # Regression (maximize)
    "r2": "maximize",
    # Classification (maximize)
    "accuracy": "maximize",
    "balanced_accuracy": "maximize",
    "f1": "maximize",
}

# Aliases normalized at entry time
_EVAL_MODE_ALIASES = {"avg": "mean"}
_SAMPLER_ALIASES = {"sample": "sampler"}


# ============================================================================
# Binary Search Sampler for Unimodal Integer Parameters
# ============================================================================


class BinarySearchSampler(BaseSampler):
    """Gradient-based binary search sampler for unimodal integer parameters.

    This sampler uses gradient information to climb toward the optimum, making it
    highly efficient for parameters with unimodal behavior (single peak), such as
    PLS n_components.

    Typically reduces optimization from ~30-50 trials (TPE) to ~10-15 trials.

    Strategy:
        1. Initial phase: Test boundaries (low, high) and midpoint
        2. Gradient detection: Test neighbors to determine direction of improvement
        3. Search narrowing: Move search bounds in direction of gradient
        4. Local refinement: Exhaustive search when range is small

    Key difference from naive binary search: Instead of just refining around the
    best value found, this sampler follows the gradient direction. If the optimum
    is at n_components=5 but the initial midpoint (13) is better than boundaries,
    the algorithm will detect the gradient pointing left and explore [1, 13],
    eventually converging to 5.

    Best for:
        - PLS/PCR n_components (most common use case)
        - KNN n_neighbors
        - Polynomial degree
        - Tree depth, number of leaves
        - Any integer parameter with clear unimodal behavior

    Not suitable for:
        - Multi-modal parameters (multiple peaks)
        - Continuous float parameters (use TPE instead)
        - Categorical parameters (use Grid instead)

    Args:
        seed: Random seed for reproducibility.

    Example:
        >>> finetune_params = {
        ...     "n_trials": 12,
        ...     "sampler": "binary",
        ...     "model_params": {
        ...         "n_components": ('int', 1, 30),
        ...     }
        ... }

    Note:
        Works best with single integer parameter. For multiple parameters,
        will apply gradient-based search to integer parameters and random
        sampling to others.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize binary search sampler.

        Args:
            seed: Random seed (for API compatibility, currently unused).
        """
        self._search_space: Dict[str, Any] = {}
        self._rng = np.random.RandomState(seed)

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, Any]:
        """Infer relative search space (not used by binary search).

        Args:
            study: Optuna study.
            trial: Current trial.

        Returns:
            Empty dict (binary search doesn't use relative search space).
        """
        return {}

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sample relative parameters (not used by binary search).

        Args:
            study: Optuna study.
            trial: Current trial.
            search_space: Search space dict.

        Returns:
            Empty dict (binary search doesn't use relative sampling).
        """
        return {}

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: "optuna.distributions.BaseDistribution",
    ) -> Any:
        """Sample using ternary search for unimodal integer parameters.

        Ternary search strategy:
            1. Test low, mid, high
            2. Divide range into thirds with points m1, m2
            3. Compare scores at m1 and m2 to eliminate one third
            4. Repeat until convergence

        For unimodal functions, this guarantees finding the optimum in O(log n) trials.

        Args:
            study: Optuna study.
            trial: Current trial.
            param_name: Parameter name.
            param_distribution: Parameter distribution.

        Returns:
            Sampled parameter value.
        """
        # Only handle integer distributions
        if not isinstance(param_distribution, optuna.distributions.IntDistribution):
            if isinstance(param_distribution, optuna.distributions.FloatDistribution):
                return self._rng.uniform(param_distribution.low, param_distribution.high)
            elif isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
                return self._rng.choice(param_distribution.choices)
            else:
                raise ValueError(f"Unsupported distribution type: {type(param_distribution)}")

        # Initialize search space
        if param_name not in self._search_space:
            self._search_space[param_name] = {
                "low": param_distribution.low,
                "high": param_distribution.high,
                "tested": set(),
                "search_low": param_distribution.low,
                "search_high": param_distribution.high,
            }

        space = self._search_space[param_name]
        low, high = space["low"], space["high"]
        search_low, search_high = space["search_low"], space["search_high"]

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and param_name in t.params
        ]

        # Phase 1: Initial triplet
        if len(completed_trials) < 3:
            if len(completed_trials) == 0:
                candidate = search_low
            elif len(completed_trials) == 1:
                candidate = search_high
            else:
                candidate = (search_low + search_high) // 2
            space["tested"].add(candidate)
            return candidate

        # Build value->score map
        value_to_score = {t.params[param_name]: t.value for t in completed_trials}
        is_better = (lambda a, b: a < b) if study.direction == optuna.study.StudyDirection.MINIMIZE else (lambda a, b: a > b)

        # Find current best
        best_score = min(value_to_score.values()) if study.direction == optuna.study.StudyDirection.MINIMIZE else max(value_to_score.values())
        best_values = [v for v, s in value_to_score.items() if s == best_score]
        best_value = int(np.median(best_values))

        search_range = search_high - search_low

        # Phase 2: Ternary search
        if search_range <= 3:
            # Small range - exhaust remaining
            untested = [v for v in range(search_low, search_high + 1) if v not in space["tested"]]
            candidate = min(untested, key=lambda x: abs(x - best_value)) if untested else best_value
        else:
            # Divide into thirds
            third = max(1, search_range // 3)
            m1 = search_low + third
            m2 = search_high - third

            # Test division points if not yet tested
            if m1 not in space["tested"]:
                candidate = m1
            elif m2 not in space["tested"]:
                candidate = m2
            else:
                # Both tested - compare and narrow
                m1_score = value_to_score.get(m1)
                m2_score = value_to_score.get(m2)

                if m1_score is not None and m2_score is not None:
                    if is_better(m1_score, m2_score):
                        # Left third is better - eliminate right third
                        space["search_high"] = m2
                        search_high = m2
                    else:
                        # Right third is better - eliminate left third
                        space["search_low"] = m1
                        search_low = m1

                # Sample midpoint of new range
                new_mid = (search_low + search_high) // 2
                if new_mid not in space["tested"]:
                    candidate = new_mid
                else:
                    # Find closest untested to best
                    untested = [v for v in range(search_low, search_high + 1) if v not in space["tested"]]
                    if untested:
                        candidate = min(untested, key=lambda x: abs(x - best_value))
                    else:
                        # Range exhausted - expand
                        untested_full = [v for v in range(low, high + 1) if v not in space["tested"]]
                        candidate = min(untested_full, key=lambda x: abs(x - best_value)) if untested_full else best_value

        # Ensure no retesting
        if candidate in space["tested"]:
            untested = [v for v in range(search_low, search_high + 1) if v not in space["tested"]]
            if untested:
                candidate = min(untested, key=lambda x: abs(x - best_value))
            else:
                untested_full = [v for v in range(low, high + 1) if v not in space["tested"]]
                candidate = min(untested_full, key=lambda x: abs(x - best_value)) if untested_full else best_value

        space["tested"].add(candidate)
        return candidate


@dataclass
class TrialSummary:
    """Summary of a single Optuna trial.

    Attributes:
        number: Trial number (0-indexed).
        params: Hyperparameters used in the trial.
        value: Objective value achieved.
        duration_seconds: Wall-clock duration.
        state: Trial state ('COMPLETE', 'PRUNED', 'FAIL').
    """

    number: int
    params: Dict[str, Any]
    value: Optional[float]
    duration_seconds: float
    state: str


@dataclass
class FinetuneResult:
    """Structured result of hyperparameter optimization.

    Returned by ``OptunaManager.finetune()`` to expose the full optimization
    history, not just ``best_params``.

    Attributes:
        best_params: Best hyperparameters found.
        best_value: Best objective value.
        n_trials: Total number of trials.
        n_pruned: Number of pruned trials.
        n_failed: Number of failed trials.
        trials: Per-trial summaries.
        study_name: Optuna study name (if storage is used).
        metric: Metric name used for optimization.
        direction: Optimization direction ('minimize' or 'maximize').
    """

    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    n_pruned: int = 0
    n_failed: int = 0
    trials: List[TrialSummary] = field(default_factory=list)
    study_name: Optional[str] = None
    metric: Optional[str] = None
    direction: str = "minimize"

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a lightweight dict suitable for prediction payload storage."""
        return {
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "best_value": self.best_value,
            "study_name": self.study_name,
            "metric": self.metric,
            "direction": self.direction,
        }


class OptunaManager:
    """
    External Optuna manager for hyperparameter optimization.

    Combines robust parameter handling with flexible fold-based optimization strategies:
    - Individual fold optimization
    - Grouped fold optimization
    - Single optimization (no folds)
    - Smart sampler selection (TPE, Grid, Random, CMA-ES)
    - Pruning support (Median, Successive Halving, Hyperband)
    - Multiple evaluation modes (best, mean, robust_best)
    - Storage/resume support for persistent studies
    - Seed support for reproducible optimization
    """

    def __init__(self):
        """Initialize the Optuna manager."""
        self.is_available = OPTUNA_AVAILABLE
        if not self.is_available:
            logger.warning("Optuna not available - finetuning will be skipped")

    def _validate_and_normalize_finetune_params(self, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize finetune_params, raising on unknown values.

        Normalizes legacy aliases:
        - ``"sample"`` key → ``"sampler"``
        - ``"avg"`` eval_mode → ``"mean"``

        Args:
            finetune_params: Raw finetune configuration from the user.

        Returns:
            Normalized copy of finetune_params.

        Raises:
            ValueError: If sampler, approach, eval_mode, or pruner is not recognized.
        """
        params = finetune_params.copy()

        # Normalize "sample" key → "sampler"
        if "sample" in params:
            if "sampler" not in params:
                params["sampler"] = params.pop("sample")
            else:
                params.pop("sample")

        # Normalize eval_mode aliases
        eval_mode = params.get("eval_mode", "best")
        if eval_mode in _EVAL_MODE_ALIASES:
            params["eval_mode"] = _EVAL_MODE_ALIASES[eval_mode]
            eval_mode = params["eval_mode"]

        # Validate sampler
        sampler = params.get("sampler", "auto")
        if sampler not in VALID_SAMPLERS:
            raise ValueError(
                f"Unknown sampler '{sampler}'. Valid: {sorted(VALID_SAMPLERS)}"
            )

        # Validate approach
        approach = params.get("approach", "grouped")
        if approach not in VALID_APPROACHES:
            raise ValueError(
                f"Unknown approach '{approach}'. Valid: {sorted(VALID_APPROACHES)}"
            )

        # Validate eval_mode
        if eval_mode not in VALID_EVAL_MODES:
            raise ValueError(
                f"Unknown eval_mode '{eval_mode}'. Valid: {sorted(VALID_EVAL_MODES)}"
            )

        # Validate pruner
        pruner = params.get("pruner", "none")
        if pruner not in VALID_PRUNERS:
            raise ValueError(
                f"Unknown pruner '{pruner}'. Valid: {sorted(VALID_PRUNERS)}"
            )

        # Validate phases (if present)
        phases = params.get("phases")
        if phases is not None:
            if not isinstance(phases, list) or len(phases) == 0:
                raise ValueError("'phases' must be a non-empty list of phase configurations")
            for i, phase in enumerate(phases):
                if not isinstance(phase, dict):
                    raise ValueError(f"Phase {i} must be a dict, got {type(phase).__name__}")
                if "n_trials" not in phase:
                    raise ValueError(f"Phase {i} must include 'n_trials'")
                phase_sampler = phase.get("sampler", "tpe")
                if phase_sampler not in VALID_SAMPLERS:
                    raise ValueError(
                        f"Unknown sampler '{phase_sampler}' in phase {i}. Valid: {sorted(VALID_SAMPLERS)}"
                    )

        return params

    def _build_finetune_result(self, study: Any, finetune_params: Dict[str, Any]) -> FinetuneResult:
        """Build a FinetuneResult from a completed Optuna study.

        Args:
            study: Completed Optuna study.
            finetune_params: Finetune configuration (for metric/direction).

        Returns:
            FinetuneResult with trial history.
        """
        trials = []
        n_pruned = 0
        n_failed = 0
        for trial in study.trials:
            duration = (
                (trial.datetime_complete - trial.datetime_start).total_seconds()
                if trial.datetime_complete and trial.datetime_start
                else 0.0
            )
            state_str = trial.state.name  # COMPLETE, PRUNED, FAIL, etc.
            if trial.state == optuna.trial.TrialState.PRUNED:
                n_pruned += 1
            elif trial.state == optuna.trial.TrialState.FAIL:
                n_failed += 1
            trials.append(TrialSummary(
                number=trial.number,
                params=dict(trial.params),
                value=trial.value,
                duration_seconds=duration,
                state=state_str,
            ))

        metric = finetune_params.get('metric')
        direction = finetune_params.get('direction', 'minimize')

        return FinetuneResult(
            best_params=dict(study.best_params),
            best_value=study.best_value,
            n_trials=len(study.trials),
            n_pruned=n_pruned,
            n_failed=n_failed,
            trials=trials,
            study_name=study.study_name,
            metric=metric,
            direction=direction,
        )

    def finetune(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: Optional[List],
        finetune_params: Dict[str, Any],
        context: 'ExecutionContext',
        controller: Any  # The model controller instance
    ) -> Union[FinetuneResult, List[FinetuneResult]]:
        """
        Main finetune entry point - delegates to appropriate optimization strategy.

        Args:
            dataset: SpectroDataset for optimization.
            model_config: Model configuration.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            folds: List of (train_indices, val_indices) tuples or None.
            finetune_params: Finetuning configuration.
            context: Pipeline context.
            controller: Model controller instance.

        Returns:
            FinetuneResult (single model) or list of FinetuneResult (per-fold).
        """
        if not self.is_available:
            logger.warning("Optuna not available, skipping finetuning")
            return FinetuneResult(best_params={}, best_value=float('inf'), n_trials=0)

        # Validate and normalize configuration
        finetune_params = self._validate_and_normalize_finetune_params(finetune_params)

        # Resolve metric and direction
        finetune_params = self._resolve_metric_direction(finetune_params, dataset)

        # Extract configuration
        strategy = finetune_params.get('approach', 'grouped')
        eval_mode = finetune_params.get('eval_mode', 'best')
        n_trials = finetune_params.get('n_trials', 50)
        verbose = finetune_params.get('verbose', 0)
        phases = finetune_params.get('phases')

        if verbose > 1:
            logger.info("Starting hyperparameter optimization:")
            logger.info(f"   Strategy: {strategy}")
            logger.info(f"   Eval mode: {eval_mode}")
            logger.info(f"   Trials: {n_trials}")
            logger.info(f"   Folds: {len(folds) if folds else 0}")
            if phases:
                logger.info(f"   Phases: {len(phases)}")
            metric = finetune_params.get('metric')
            if metric:
                logger.info(f"   Metric: {metric} ({finetune_params.get('direction', 'minimize')})")

        # Multi-phase search: when 'phases' is present, run phases sequentially on one study
        if phases:
            return self._optimize_multiphase(
                dataset, model_config, X_train, y_train, X_test, y_test,
                folds, finetune_params, context, controller, eval_mode, verbose
            )

        # Route to appropriate optimization strategy
        if folds and strategy == 'individual':
            # Individual fold optimization: best_params = [], foreach fold: best_params.append(optuna.loop(...))
            return self._optimize_individual_folds(
                dataset,
                model_config, X_train, y_train, folds, finetune_params,
                n_trials, context, controller, verbose
            )

        elif folds and strategy == 'grouped':
            # Grouped fold optimization: return best_param = optuna.loop(objective(folds, data, evalMode))
            return self._optimize_grouped_folds(
                dataset,
                model_config, X_train, y_train, folds, finetune_params,
                n_trials, context, controller, eval_mode, verbose
            )

        else:
            # Single optimization (no folds): use holdout split for validation
            from sklearn.model_selection import train_test_split
            X_opt_train, X_val, y_opt_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            return self._optimize_single(
                dataset,
                model_config, X_opt_train, y_opt_train, X_val, y_val,
                finetune_params, n_trials, context, controller, verbose
            )

    def _resolve_metric_direction(self, finetune_params: Dict[str, Any], dataset: 'SpectroDataset') -> Dict[str, Any]:
        """Resolve metric name and direction from finetune_params and dataset task_type.

        When ``metric`` is set, auto-infers ``direction`` from METRIC_DIRECTION
        unless explicitly overridden. When ``metric`` is not set, preserves the
        existing default behavior (MSE for regression, -balanced_accuracy for
        classification).

        Args:
            finetune_params: Normalized finetune configuration.
            dataset: SpectroDataset (for task_type).

        Returns:
            Updated finetune_params with resolved metric/direction.
        """
        params = finetune_params.copy()
        metric = params.get('metric')

        if metric is not None:
            # Validate the metric is supported
            if metric not in METRIC_DIRECTION:
                raise ValueError(
                    f"Unknown metric '{metric}' for finetuning. "
                    f"Supported: {sorted(METRIC_DIRECTION.keys())}"
                )
            # Auto-infer direction if not explicitly set
            if 'direction' not in finetune_params:
                params['direction'] = METRIC_DIRECTION[metric]
        else:
            # No explicit metric — use defaults based on task_type
            task_type = getattr(dataset, 'task_type', 'regression')
            if 'classification' in task_type:
                params.setdefault('direction', 'maximize')
            else:
                params.setdefault('direction', 'minimize')

        return params

    def _optimize_individual_folds(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        folds: List,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        verbose: int
    ) -> List[FinetuneResult]:
        """
        Optimize each fold individually.

        Returns list of FinetuneResult for each fold.
        """
        results = []

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            if verbose > 1:
                logger.info(f"Optimizing fold {fold_idx + 1}/{len(folds)}")

            # Extract fold data
            X_train_fold = X_train[train_indices]
            y_train_fold = y_train[train_indices]
            X_val_fold = X_train[val_indices]
            y_val_fold = y_train[val_indices]

            # Run optimization for this fold
            fold_result = self._run_single_optimization(
                dataset,
                model_config, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                finetune_params, n_trials, context, controller, verbose=0
            )

            results.append(fold_result)

            if verbose > 1:
                logger.info(f"   Fold {fold_idx + 1} best: {fold_result.best_params}")

        return results

    def _optimize_grouped_folds(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        folds: List,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        eval_mode: str,
        verbose: int
    ) -> FinetuneResult:
        """
        Optimize using grouped fold evaluation.

        Single optimization where objective function evaluates across all folds.
        Supports pruning: reports intermediate scores after each fold and prunes
        unpromising trials early.
        """
        pruner_type = finetune_params.get('pruner', 'none')

        # Extract metric/direction for _evaluate_model
        opt_metric = finetune_params.get('metric')
        opt_direction = finetune_params.get('direction', 'minimize')

        # Create objective function that evaluates across all folds
        def objective(trial):
            # Sample hyperparameters (returns model_params and train_params separately)
            model_params, sampled_train_params = self.sample_hyperparameters(trial, finetune_params)

            # Process model parameters if controller supports it
            if hasattr(controller, 'process_hyperparameters'):
                model_params = controller.process_hyperparameters(model_params)

            if verbose > 2:
                logger.debug(f"Trial model_params: {model_params}, train_params: {sampled_train_params}")

            # Train on all folds and collect scores
            scores = []
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                X_train_fold = X_train[train_indices]
                y_train_fold = y_train[train_indices]
                X_val_fold = X_train[val_indices]
                y_val_fold = y_train[val_indices]
                try:
                    model = controller._get_model_instance(dataset, model_config, force_params=model_params)  # noqa: SLF001

                    # Prepare data
                    X_train_prep, y_train_prep = controller._prepare_data(X_train_fold, y_train_fold, context)  # noqa: SLF001
                    X_val_prep, y_val_prep = controller._prepare_data(X_val_fold, y_val_fold, context)  # noqa: SLF001

                    # Build train_params: sampled train params + processed model params (for TF compile/fit dicts)
                    train_params_for_trial = sampled_train_params.copy()
                    train_params_for_trial.update(model_params)

                    # Ensure task_type is passed for models that need it (e.g., TensorFlow)
                    if 'task_type' not in train_params_for_trial:
                        train_params_for_trial['task_type'] = dataset.task_type
                    trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **train_params_for_trial)  # noqa: SLF001
                    score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep, metric=opt_metric, direction=opt_direction)  # noqa: SLF001
                    scores.append(score)

                    # Pruning: report intermediate score and check for pruning
                    if pruner_type != 'none':
                        intermediate_score = self._aggregate_scores(scores, eval_mode)
                        trial.report(intermediate_score, fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                except optuna.TrialPruned:
                    raise  # Re-raise pruning exception
                except Exception as e:
                    if verbose >= 2:
                        logger.debug(f"   Fold failed: {e}")
                    scores.append(float('inf'))

            # Return evaluation based on eval_mode
            return self._aggregate_scores(scores, eval_mode)

        # Run optimization with the multi-fold objective
        study = self._create_study(finetune_params)
        self._configure_logging(verbose)

        if verbose > 1:
            logger.starting(f"Running grouped optimization ({n_trials} trials)...")

        # Extract n_jobs for parallel trial evaluation (default=1 to avoid conflicts with branch parallelization)
        n_jobs = finetune_params.get('n_jobs', 1)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)

        if verbose > 1:
            logger.success(f"Best score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

        return self._build_finetune_result(study, finetune_params)

    def _optimize_single(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        verbose: int
    ) -> FinetuneResult:
        """Optimize without folds - single train/val split."""
        return self._run_single_optimization(
            dataset,
            model_config, X_train, y_train, X_val, y_val,
            finetune_params, n_trials, context, controller, verbose
        )

    def _optimize_multiphase(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: Optional[List],
        finetune_params: Dict[str, Any],
        context: 'ExecutionContext',
        controller: Any,
        eval_mode: str,
        verbose: int
    ) -> FinetuneResult:
        """Run multi-phase optimization with different samplers on the same study.

        Each phase runs sequentially on a shared study. After phase 1 completes,
        later phases benefit from the accumulated trial history (e.g., TPE in
        phase 2 learns from random exploration in phase 1).

        Args:
            dataset: SpectroDataset.
            model_config: Model configuration.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            folds: CV fold indices or None.
            finetune_params: Full finetune configuration (must contain 'phases').
            context: Pipeline context.
            controller: Model controller instance.
            eval_mode: Score aggregation mode.
            verbose: Verbosity level.

        Returns:
            FinetuneResult with trial history across all phases.
        """
        phases = finetune_params['phases']

        # Extract metric/direction for _evaluate_model
        opt_metric = finetune_params.get('metric')
        opt_direction = finetune_params.get('direction', 'minimize')

        # Build the objective function (same for all phases)
        if folds:
            pruner_type = finetune_params.get('pruner', 'none')

            def objective(trial):
                model_params, sampled_train_params = self.sample_hyperparameters(trial, finetune_params)
                if hasattr(controller, 'process_hyperparameters'):
                    model_params = controller.process_hyperparameters(model_params)
                scores = []
                for fold_idx, (train_indices, val_indices) in enumerate(folds):
                    X_train_fold = X_train[train_indices]
                    y_train_fold = y_train[train_indices]
                    X_val_fold = X_train[val_indices]
                    y_val_fold = y_train[val_indices]
                    try:
                        model = controller._get_model_instance(dataset, model_config, force_params=model_params)  # noqa: SLF001
                        X_train_prep, y_train_prep = controller._prepare_data(X_train_fold, y_train_fold, context)  # noqa: SLF001
                        X_val_prep, y_val_prep = controller._prepare_data(X_val_fold, y_val_fold, context)  # noqa: SLF001
                        train_params_for_trial = sampled_train_params.copy()
                        train_params_for_trial.update(model_params)
                        if 'task_type' not in train_params_for_trial:
                            train_params_for_trial['task_type'] = dataset.task_type
                        trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **train_params_for_trial)  # noqa: SLF001
                        score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep, metric=opt_metric, direction=opt_direction)  # noqa: SLF001
                        scores.append(score)
                        if pruner_type != 'none':
                            intermediate_score = self._aggregate_scores(scores, eval_mode)
                            trial.report(intermediate_score, fold_idx)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                    except optuna.TrialPruned:
                        raise
                    except Exception as e:
                        if verbose >= 2:
                            logger.debug(f"   Fold failed: {e}")
                        scores.append(float('inf'))
                return self._aggregate_scores(scores, eval_mode)
        else:
            # Single split for no-folds case
            from sklearn.model_selection import train_test_split
            X_opt_train, X_val, y_opt_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            def objective(trial):
                model_params, sampled_train_params = self.sample_hyperparameters(trial, finetune_params)
                if hasattr(controller, 'process_hyperparameters'):
                    model_params = controller.process_hyperparameters(model_params)
                try:
                    model = controller._get_model_instance(dataset, model_config, force_params=model_params)  # noqa: SLF001
                    X_train_prep, y_train_prep = controller._prepare_data(X_opt_train, y_opt_train, context)  # noqa: SLF001
                    X_val_prep, y_val_prep = controller._prepare_data(X_val, y_val, context)  # noqa: SLF001
                    train_params_for_trial = sampled_train_params.copy()
                    train_params_for_trial.update(model_params)
                    if 'task_type' not in train_params_for_trial:
                        train_params_for_trial['task_type'] = dataset.task_type
                    trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **train_params_for_trial)  # noqa: SLF001
                    score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep, metric=opt_metric, direction=opt_direction)  # noqa: SLF001
                    return score
                except Exception as e:
                    if verbose >= 2:
                        logger.warning(f"Trial failed: {e}")
                    return float('inf')

        # Create study with first phase's config (inherits direction, pruner, storage, seed from top-level)
        study = self._create_study(finetune_params)
        self._configure_logging(verbose)

        # Run phases sequentially, changing sampler between phases
        for phase_idx, phase_config in enumerate(phases):
            phase_n_trials = phase_config['n_trials']
            phase_sampler_type = phase_config.get('sampler', 'tpe')
            seed = finetune_params.get('seed', None)

            # Create new sampler for this phase
            sampler = self._create_sampler(phase_sampler_type, finetune_params, seed)
            study.sampler = sampler

            if verbose > 1:
                logger.info(f"Phase {phase_idx + 1}/{len(phases)}: {phase_sampler_type} sampler, {phase_n_trials} trials")

            # Extract n_jobs for parallel trial evaluation (default=1)
            n_jobs = finetune_params.get('n_jobs', 1)
            study.optimize(objective, n_trials=phase_n_trials, show_progress_bar=False, n_jobs=n_jobs)

        if verbose > 1:
            logger.success(f"Multi-phase optimization complete. Best score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

        return self._build_finetune_result(study, finetune_params)

    def _create_sampler(self, sampler_type: str, finetune_params: Dict[str, Any], seed: Optional[int] = None) -> Any:
        """Create an Optuna sampler instance.

        Args:
            sampler_type: Sampler type string ('tpe', 'random', 'cmaes', 'grid', 'binary', 'auto').
            finetune_params: Full finetune config (needed for grid search space).
            seed: Random seed for reproducibility.

        Returns:
            Optuna sampler instance.
        """
        if sampler_type == 'grid':
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            if is_grid_suitable:
                search_space = self._create_grid_search_space(finetune_params)
                return GridSampler(search_space, seed=seed)
            else:
                logger.warning("Grid sampler requested but parameters are not all categorical. Using TPE.")
                return TPESampler(seed=seed)
        elif sampler_type == 'random':
            return RandomSampler(seed=seed)
        elif sampler_type == 'cmaes':
            return CmaEsSampler(seed=seed)
        elif sampler_type == 'binary':
            return BinarySearchSampler(seed=seed)
        elif sampler_type == 'auto':
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            if is_grid_suitable:
                search_space = self._create_grid_search_space(finetune_params)
                return GridSampler(search_space, seed=seed)
            return TPESampler(seed=seed)
        else:  # tpe (default)
            return TPESampler(seed=seed)

    def _run_single_optimization(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        verbose: int = 1
    ) -> FinetuneResult:
        """
        Run single optimization study for a train/val split.

        Core optimization logic used by both individual fold and single optimization.
        """
        # Extract metric/direction for _evaluate_model
        opt_metric = finetune_params.get('metric')
        opt_direction = finetune_params.get('direction', 'minimize')

        def objective(trial):
            # Sample hyperparameters (returns model_params and train_params separately)
            model_params, sampled_train_params = self.sample_hyperparameters(trial, finetune_params)

            # Process model parameters if controller supports it
            if hasattr(controller, 'process_hyperparameters'):
                model_params = controller.process_hyperparameters(model_params)

            if verbose > 2:
                logger.debug(f"Trial model_params: {model_params}, train_params: {sampled_train_params}")

            try:
                model = controller._get_model_instance(dataset, model_config, force_params=model_params)  # noqa: SLF001

                # Prepare data
                X_train_prep, y_train_prep = controller._prepare_data(X_train, y_train, context)  # noqa: SLF001
                X_val_prep, y_val_prep = controller._prepare_data(X_val, y_val, context)  # noqa: SLF001

                # Build train_params: sampled train params + processed model params (for TF compile/fit dicts)
                train_params_for_trial = sampled_train_params.copy()
                train_params_for_trial.update(model_params)

                # Ensure task_type is passed for models that need it (e.g., TensorFlow)
                if 'task_type' not in train_params_for_trial:
                    train_params_for_trial['task_type'] = dataset.task_type
                trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **train_params_for_trial)  # noqa: SLF001
                score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep, metric=opt_metric, direction=opt_direction)  # noqa: SLF001

                return score

            except Exception as e:
                if verbose >= 2:
                    logger.warning(f"Trial failed: {e}")
                return float('inf')

        # Create and run optimization
        study = self._create_study(finetune_params)
        self._configure_logging(verbose)

        if verbose > 1:
            logger.starting(f"Running optimization ({n_trials} trials)...")

        # Extract n_jobs for parallel trial evaluation (default=1 to avoid conflicts with branch parallelization)
        n_jobs = finetune_params.get('n_jobs', 1)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)

        if verbose > 1:
            logger.success(f"Best score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

        return self._build_finetune_result(study, finetune_params)

    def _create_study(self, finetune_params: Dict[str, Any]) -> Any:
        """
        Create an Optuna study with appropriate sampler, pruner, storage, and seed.

        Supports:
        - Sampler: auto, grid, tpe, random, cmaes
        - Pruner: none, median, successive_halving, hyperband
        - Storage: in-memory or SQLite/database URL
        - Seed: for reproducible optimization
        - Direction: minimize (default) or maximize
        """
        if not OPTUNA_AVAILABLE or optuna is None:
            raise ImportError("Optuna is not available")

        # Extract seed for reproducible sampling
        seed = finetune_params.get('seed', None)

        # Determine optimal sampler strategy (already normalized by _validate_and_normalize_finetune_params)
        sampler_type = finetune_params.get('sampler', 'auto')

        if sampler_type == 'auto':
            # Auto-detect best sampler based on parameter types
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            sampler_type = 'grid' if is_grid_suitable else 'tpe'
        elif sampler_type == 'grid':
            # Verify grid is suitable even if explicitly requested
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            if not is_grid_suitable:
                logger.warning("Grid sampler requested but parameters are not all categorical. Using TPE sampler instead.")
                sampler_type = 'tpe'

        # Create sampler instance with seed support
        if sampler_type == 'grid':
            search_space = self._create_grid_search_space(finetune_params)
            sampler = GridSampler(search_space, seed=seed)
        elif sampler_type == 'random':
            sampler = RandomSampler(seed=seed)
        elif sampler_type == 'cmaes':
            sampler = CmaEsSampler(seed=seed)
        elif sampler_type == 'binary':
            sampler = BinarySearchSampler(seed=seed)
        else:  # tpe (default)
            sampler = TPESampler(seed=seed)

        # Create pruner instance
        pruner_type = finetune_params.get('pruner', 'none')
        pruner = self._create_pruner(pruner_type)

        # Direction: minimize by default (loss-based metrics)
        direction = finetune_params.get('direction', 'minimize')

        # Storage/resume support
        storage = finetune_params.get('storage', None)
        study_name = finetune_params.get('study_name', None)
        resume = finetune_params.get('resume', False)

        # Create study
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=resume,
        )

        # Enqueue force_params as trial 0 if provided
        force_params = finetune_params.get('force_params')
        if force_params:
            study.enqueue_trial(force_params)

        return study

    def _create_pruner(self, pruner_type: str) -> Any:
        """Create an Optuna pruner instance.

        Args:
            pruner_type: One of 'none', 'median', 'successive_halving', 'hyperband'.

        Returns:
            Pruner instance, or None for 'none'.
        """
        if pruner_type == 'none':
            return None
        elif pruner_type == 'median':
            return MedianPruner()
        elif pruner_type == 'successive_halving':
            return SuccessiveHalvingPruner()
        elif pruner_type == 'hyperband':
            return HyperbandPruner()
        else:
            return None

    def _configure_logging(self, verbose: int):
        """Configure Optuna logging based on verbosity level."""
        if verbose < 2 and optuna is not None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _aggregate_scores(self, scores: List[float], eval_mode: str) -> float:
        """
        Aggregate fold scores based on evaluation mode.

        Args:
            scores: List of scores from different folds
            eval_mode: How to aggregate ('best', 'mean', 'robust_best')

        Returns:
            Aggregated score
        """
        if eval_mode == 'best':
            return min(scores)
        elif eval_mode == 'mean':
            return float(np.mean(scores))
        elif eval_mode == 'robust_best':
            # Exclude infinite scores (failed trials) then take best
            valid_scores = [s for s in scores if s != float('inf')]
            return min(valid_scores) if valid_scores else float('inf')
        else:
            raise ValueError(f"Unknown eval_mode '{eval_mode}'. Valid: 'best', 'mean', 'robust_best'")

    def sample_hyperparameters(
        self,
        trial: Any,
        finetune_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Sample hyperparameters for an Optuna trial.

        Samples both model_params (for model constructor) and train_params
        (for training configuration). train_params with range/choice specs
        are sampled by Optuna; static values are passed through.

        Supports nested parameter dicts (e.g. TabPFN's inference_config)
        by flattening with ``__`` separator for Optuna and reconstructing
        the nested structure afterward.

        Args:
            trial: Optuna trial instance
            finetune_params: Finetuning configuration

        Returns:
            Tuple of (model_params, train_params) dictionaries.
            model_params: sampled model constructor parameters (nested structure preserved).
            train_params: sampled and static training parameters.
        """
        # Get model parameters - support both nested and flat structure
        model_params_spec = finetune_params.get('model_params', {})

        # Legacy support: look for parameters directly in finetune_params
        if not model_params_spec:
            model_params_spec = {k: v for k, v in finetune_params.items()
                          if k not in ['n_trials', 'approach', 'eval_mode', 'sampler', 'train_params',
                                       'verbose', 'pruner', 'seed', 'storage', 'study_name', 'resume',
                                       'direction', 'force_params', 'model_params', 'phases']}

        # Flatten nested parameter dicts (e.g. inference_config: {PARAM: [True, False]})
        flat_params_spec = self._flatten_nested_params(model_params_spec)

        # Sample each flat parameter
        flat_sampled = {}
        for param_name, param_config in flat_params_spec.items():
            flat_sampled[param_name] = self._sample_single_parameter(trial, param_name, param_config)

        # Unflatten back to nested structure
        model_params = self._unflatten_params(flat_sampled)

        # Sample train_params (new in Phase 3)
        sampled_train_params = {}
        train_params_spec = finetune_params.get('train_params', {})
        for param_name, param_config in train_params_spec.items():
            if self._is_sampable(param_config):
                sampled_train_params[param_name] = self._sample_single_parameter(
                    trial, f"train_{param_name}", param_config
                )
            else:
                sampled_train_params[param_name] = param_config  # Pass through static values

        return model_params, sampled_train_params

    def _is_param_spec(self, value: Any) -> bool:
        """Check if a dict value is a parameter spec (not a nested param group).

        A parameter spec is a dict with 'type', 'min'/'max', or 'low'/'high' keys
        that defines how Optuna should sample a single parameter. A nested param
        group is a dict whose values are themselves parameter specs or further
        nested groups.

        Args:
            value: Value to check.

        Returns:
            True if the value is a parameter spec dict.
        """
        if not isinstance(value, dict):
            return False
        return 'type' in value or 'min' in value or 'low' in value or 'choices' in value or 'values' in value

    def _flatten_nested_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested parameter dicts into flat keys with ``__`` separator.

        Nested dicts that are NOT parameter specs (no ``type``/``min``/``max`` keys)
        are recursively flattened. Parameter specs and non-dict values are kept as-is.

        Args:
            params: Parameter dict, potentially with nested groups.
            prefix: Current key prefix for recursion.

        Returns:
            Flat dict with ``__``-separated keys.

        Example:
            >>> _flatten_nested_params({"inference_config": {"PARAM_A": [True, False]}})
            {"inference_config__PARAM_A": [True, False]}
        """
        flat = {}
        for key, value in params.items():
            full_key = f"{prefix}__{key}" if prefix else key
            if isinstance(value, dict) and not self._is_param_spec(value):
                # Nested param group — recurse
                flat.update(self._flatten_nested_params(value, full_key))
            else:
                flat[full_key] = value
        return flat

    def _unflatten_params(self, flat_params: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct nested dict from flat ``__``-separated keys.

        Args:
            flat_params: Flat dict with ``__``-separated keys.

        Returns:
            Nested dict.

        Raises:
            ValueError: If key structures conflict (e.g., ``config`` as both
                a scalar value and a nested group).

        Example:
            >>> _unflatten_params({"inference_config__PARAM_A": True})
            {"inference_config": {"PARAM_A": True}}
        """
        nested: Dict[str, Any] = {}
        for key, value in flat_params.items():
            parts = key.split("__")
            current = nested
            for part in parts[:-1]:
                if part in current and not isinstance(current[part], dict):
                    raise ValueError(
                        f"Conflicting nested parameter structure: '{part}' is both "
                        f"a scalar value and a nested group in key '{key}'"
                    )
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        return nested

    def _is_sampable(self, config: Any) -> bool:
        """Check if a parameter config represents an Optuna-sampable spec.

        Returns True for range tuples, choice lists, and dict param specs.
        Returns False for scalar values (int, float, str, bool, None) that
        should be passed through unchanged.

        Args:
            config: Parameter configuration value.

        Returns:
            True if the value should be sampled by Optuna.
        """
        if isinstance(config, (list, tuple)):
            return True
        if isinstance(config, dict):
            # Dict with 'type' key or 'min'/'max' keys is a param spec
            return 'type' in config or 'min' in config or 'low' in config
        return False

    # Supported parameter type strings for tuple format ('type', min, max)
    _INT_TYPES = ('int', int, 'builtins.int')
    _INT_LOG_TYPES = ('int_log', 'log_int')
    _FLOAT_TYPES = ('float', float, 'builtins.float')
    _FLOAT_LOG_TYPES = ('float_log', 'log_float')
    _ALL_RANGE_TYPES = _INT_TYPES + _INT_LOG_TYPES + _FLOAT_TYPES + _FLOAT_LOG_TYPES

    def _sample_single_parameter(self, trial: Any, param_name: str, param_config: Any) -> Any:
        """
        Sample a single parameter based on its configuration.

        Supported formats:

        **Tuple format** (most common):
            - ('int', min, max)        → Integer uniform sampling
            - ('int_log', min, max)    → Integer log-uniform sampling
            - ('float', min, max)      → Float uniform sampling
            - ('float_log', min, max)  → Float log-uniform sampling (recommended for learning rates, regularization)
            - (min, max)               → Inferred type based on values (int if both int, else float)

        **List format**:
            - [val1, val2, val3]       → Categorical sampling

        **Dict format** (most flexible):
            - {'type': 'categorical', 'choices': [v1, v2, v3]}
            - {'type': 'int', 'min': 1, 'max': 10}
            - {'type': 'int', 'min': 1, 'max': 10, 'step': 2}
            - {'type': 'int', 'min': 1, 'max': 1000, 'log': True}
            - {'type': 'float', 'min': 0.0, 'max': 1.0}
            - {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1}
            - {'type': 'float', 'min': 1e-5, 'max': 1e-1, 'log': True}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': ('int', 3, 5), 'min': 0.0, 'max': 2.0, 'element_type': 'float'}

        **Single value**:
            - Any scalar value → Passed through unchanged

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_config: Parameter configuration

        Returns:
            Sampled parameter value
        """
        if isinstance(param_config, list):
            # Check if this is actually a type specification that was converted from tuple to list
            # Pattern: ['int'/'float'/etc., min, max] from tuple ('type', min, max)
            if (len(param_config) == 3 and
                param_config[0] in self._ALL_RANGE_TYPES and
                isinstance(param_config[1], (int, float)) and
                isinstance(param_config[2], (int, float))):
                # This is a range specification, not a categorical list
                param_type, min_val, max_val = param_config
                return self._suggest_from_type(trial, param_name, param_type, min_val, max_val)

            # Pattern: ['bool'/'categorical', [choices]] from tuple ('bool'/'categorical', [choices])
            if (len(param_config) == 2 and
                isinstance(param_config[0], str) and
                param_config[0] in ('bool', 'categorical') and
                isinstance(param_config[1], list)):
                return trial.suggest_categorical(param_name, param_config[1])

            # Regular categorical parameter: [val1, val2, val3]
            return trial.suggest_categorical(param_name, param_config)

        elif isinstance(param_config, tuple) and len(param_config) == 3:
            # Explicit type tuple: ('type', min, max)
            param_type, min_val, max_val = param_config
            return self._suggest_from_type(trial, param_name, param_type, min_val, max_val)

        elif isinstance(param_config, tuple) and len(param_config) == 2:
            type_hint, values = param_config
            if isinstance(type_hint, str) and type_hint in ('bool', 'categorical') and isinstance(values, list):
                return trial.suggest_categorical(param_name, values)
            # Range tuple: (min, max) - infer type from values
            min_val, max_val = type_hint, values
            if isinstance(min_val, int) and isinstance(max_val, int):
                return trial.suggest_int(param_name, min_val, max_val)
            else:
                return trial.suggest_float(param_name, float(min_val), float(max_val))

        elif isinstance(param_config, dict):
            # Dictionary configuration with full options
            return self._suggest_from_dict(trial, param_name, param_config)

        else:
            # Single value - pass through unchanged
            return param_config

    def _suggest_from_type(
        self,
        trial: Any,
        param_name: str,
        param_type: Any,
        min_val: float,
        max_val: float,
        step: Optional[float] = None,
        log: bool = False
    ) -> Any:
        """
        Suggest a parameter value based on type string.

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_type: Type indicator ('int', 'float', 'int_log', 'float_log', etc.)
            min_val: Minimum value
            max_val: Maximum value
            step: Step size (optional, for discrete sampling)
            log: Whether to use log-uniform sampling (overridden by type suffix)

        Returns:
            Sampled value
        """
        # Determine if log scale from type suffix
        use_log = log or param_type in self._INT_LOG_TYPES or param_type in self._FLOAT_LOG_TYPES

        if param_type in self._INT_TYPES or param_type in self._INT_LOG_TYPES:
            # Integer sampling
            int_step = int(step) if step is not None else 1
            return trial.suggest_int(param_name, int(min_val), int(max_val), step=int_step, log=use_log)

        elif param_type in self._FLOAT_TYPES or param_type in self._FLOAT_LOG_TYPES:
            # Float sampling
            return trial.suggest_float(param_name, float(min_val), float(max_val), step=step, log=use_log)

        else:
            raise ValueError(
                f"Unknown parameter type '{param_type}' for parameter '{param_name}'. "
                f"Supported types: 'int', 'int_log', 'float', 'float_log' (or Python int/float types)"
            )

    def _suggest_from_dict(self, trial: Any, param_name: str, param_config: Dict[str, Any]) -> Any:
        """
        Suggest a parameter value from a dictionary configuration.

        Supports:
            - {'type': 'categorical', 'choices': [v1, v2, v3]}
            - {'type': 'int', 'min': 1, 'max': 10, 'step': 2, 'log': False}
            - {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1, 'log': True}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': ('int', 3, 5), 'min': 0.0, 'max': 2.0, 'element_type': 'float'}

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_config: Dictionary with 'type' and range/choices

        Returns:
            Sampled value
        """
        param_type = param_config.get('type', 'categorical')

        if param_type == 'categorical':
            choices = param_config.get('choices', param_config.get('values', []))
            if not choices:
                raise ValueError(f"Categorical parameter '{param_name}' requires 'choices' or 'values' list")
            return trial.suggest_categorical(param_name, choices)

        elif param_type in ('int', 'int_log'):
            min_val = param_config.get('min', param_config.get('low'))
            max_val = param_config.get('max', param_config.get('high'))
            step = param_config.get('step', 1)
            log = param_config.get('log', param_type == 'int_log')

            if min_val is None or max_val is None:
                raise ValueError(f"Integer parameter '{param_name}' requires 'min'/'max' or 'low'/'high'")

            return trial.suggest_int(param_name, int(min_val), int(max_val), step=int(step), log=log)

        elif param_type in ('float', 'float_log'):
            min_val = param_config.get('min', param_config.get('low'))
            max_val = param_config.get('max', param_config.get('high'))
            step = param_config.get('step')  # None means continuous
            log = param_config.get('log', param_type == 'float_log')

            if min_val is None or max_val is None:
                raise ValueError(f"Float parameter '{param_name}' requires 'min'/'max' or 'low'/'high'")

            return trial.suggest_float(param_name, float(min_val), float(max_val), step=step, log=log)

        elif param_type == 'sorted_tuple':
            return self._suggest_sorted_tuple(trial, param_name, param_config)

        else:
            raise ValueError(
                f"Unknown parameter type '{param_type}' for parameter '{param_name}'. "
                f"Supported types: 'categorical', 'int', 'int_log', 'float', 'float_log', 'sorted_tuple'"
            )

    def _suggest_sorted_tuple(self, trial: Any, param_name: str, param_config: Dict[str, Any]) -> tuple:
        """
        Suggest a sorted tuple of values.

        Generates N values within a range and returns them as a sorted tuple.
        Useful for parameters like 'alphas' that need ordered sequences.

        Supports:
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': ('int', 3, 5), 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0, 'element_type': 'int'}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0, 'step': 0.5}

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_config: Dictionary with length, min, max, and optional element_type/step

        Returns:
            Sorted tuple of sampled values
        """
        # Get tuple length - can be fixed or a range
        length_config = param_config.get('length', 3)
        if isinstance(length_config, int):
            length = length_config
        elif isinstance(length_config, (tuple, list)) and len(length_config) == 3:
            # ('int', min, max) or ['int', min, max] format
            _, min_len, max_len = length_config
            length = trial.suggest_int(f"{param_name}_length", int(min_len), int(max_len))
        elif isinstance(length_config, (tuple, list)) and len(length_config) == 2:
            # (min, max) or [min, max] format
            min_len, max_len = length_config
            length = trial.suggest_int(f"{param_name}_length", int(min_len), int(max_len))
        elif isinstance(length_config, dict):
            min_len = length_config.get('min', length_config.get('low', 2))
            max_len = length_config.get('max', length_config.get('high', 5))
            length = trial.suggest_int(f"{param_name}_length", int(min_len), int(max_len))
        else:
            length = int(length_config)

        # Get value range
        min_val = param_config.get('min', param_config.get('low', 0.0))
        max_val = param_config.get('max', param_config.get('high', 1.0))
        step = param_config.get('step')
        element_type = param_config.get('element_type', 'float')

        # Sample individual elements
        values = []
        for i in range(length):
            elem_name = f"{param_name}_{i}"
            if element_type in ('int', 'int_log'):
                log = param_config.get('log', element_type == 'int_log')
                int_step = int(step) if step is not None else 1
                val = trial.suggest_int(elem_name, int(min_val), int(max_val), step=int_step, log=log)
            else:  # float or float_log
                log = param_config.get('log', element_type == 'float_log')
                val = trial.suggest_float(elem_name, float(min_val), float(max_val), step=step, log=log)
            values.append(val)

        # Sort and return as tuple
        return tuple(sorted(values))

    def _is_grid_search_suitable(self, finetune_params: Dict[str, Any]) -> bool:
        """
        Check if grid search is suitable (all parameters are categorical).

        Grid search only works well when all parameters are categorical (discrete choices).
        Continuous parameters need random/TPE sampling.
        """
        model_params = finetune_params.get('model_params', {})

        # Legacy support
        if not model_params:
            model_params = {
                k: v for k, v in finetune_params.items()
                if k not in ['n_trials', 'approach', 'eval_mode', 'sampler', 'train_params',
                             'verbose', 'pruner', 'seed', 'storage', 'study_name', 'resume',
                             'direction', 'force_params', 'model_params']
            }

        for _, param_config in model_params.items():
            # Check if this is a range specification disguised as a list (from tuple-to-list conversion)
            is_list = isinstance(param_config, list)
            has_len_3 = len(param_config) == 3 if is_list else False
            is_type_spec = param_config[0] in self._ALL_RANGE_TYPES if is_list and has_len_3 else False
            is_min_num = isinstance(param_config[1], (int, float)) if is_list and has_len_3 else False
            is_max_num = isinstance(param_config[2], (int, float)) if is_list and has_len_3 else False

            is_range_spec = is_list and has_len_3 and is_type_spec and is_min_num and is_max_num

            if is_range_spec:
                return False

            # Dict-categorical is also grid-compatible
            if isinstance(param_config, dict):
                if param_config.get("type") == "categorical":
                    continue
                else:
                    return False

            # Only categorical (list) parameters are suitable for grid search
            if not isinstance(param_config, list):
                return False

        return len(model_params) > 0

    def _create_grid_search_space(self, finetune_params: Dict[str, Any]) -> Dict[str, List]:
        """
        Create grid search space for categorical parameters only.

        Returns search space suitable for GridSampler.
        """
        model_params = finetune_params.get('model_params', {})

        # Legacy support
        if not model_params:
            model_params = {
                k: v for k, v in finetune_params.items()
                if k not in ['n_trials', 'approach', 'eval_mode', 'sampler', 'train_params',
                             'verbose', 'pruner', 'seed', 'storage', 'study_name', 'resume',
                             'direction', 'force_params', 'model_params']
            }

        search_space = {}
        for param_name, param_config in model_params.items():
            if isinstance(param_config, list):
                search_space[param_name] = param_config
            elif isinstance(param_config, dict) and param_config.get("type") == "categorical":
                search_space[param_name] = param_config.get("choices", param_config.get("values", []))

        return search_space


# ============================================================================
# Parameter helpers for complex sklearn models
# ============================================================================


def stack_params(
    final_estimator_params: Optional[Dict[str, Any]] = None,
    **other_params: Any
) -> Dict[str, Any]:
    """Create Optuna-compatible parameter structure for sklearn Stacking models.

    Helper to finetune the final_estimator (metamodel) of a StackingRegressor
    or StackingClassifier through Optuna. Automatically namespaces parameters
    with the ``final_estimator__`` prefix required by sklearn.

    The existing nested parameter flattening/unflattening system in
    OptunaManager will handle the ``__`` separator transparently.

    Args:
        final_estimator_params: Parameter specs for the metamodel.
            Each value can be an Optuna parameter spec (range tuple, list, dict).
        **other_params: Additional Stack parameters (e.g., cv, passthrough).

    Returns:
        Dict with properly namespaced parameters for Optuna sampling.

    Example:
        >>> from nirs4all.optimization.optuna import stack_params
        >>> finetune_params = {
        ...     "n_trials": 20,
        ...     "model_params": stack_params(
        ...         final_estimator_params={
        ...             "alpha": ("float", 1e-3, 1e0, "log"),
        ...             "fit_intercept": [True, False],
        ...         },
        ...         passthrough=True,  # Stack parameter
        ...     ),
        ... }
        >>> # Optuna will sample final_estimator__alpha and final_estimator__fit_intercept

    See Also:
        - sklearn.ensemble.StackingRegressor
        - sklearn.ensemble.StackingClassifier
    """
    params = {}

    # Namespace final_estimator params with final_estimator__ prefix
    if final_estimator_params:
        for key, value in final_estimator_params.items():
            params[f"final_estimator__{key}"] = value

    # Add other Stack-level params as-is (e.g., cv, passthrough)
    params.update(other_params)

    return params
