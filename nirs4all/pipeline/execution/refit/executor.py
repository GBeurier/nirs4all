"""Simple refit executor for the refit phase.

After cross-validation (Pass 1) selects the winning pipeline variant,
this module re-executes the exact same pipeline on the full training set
(no CV folds) to produce a single final model.

The refit pass:
1. Takes the winning variant's expanded steps.
2. Replaces the CV splitter with a single "all training data" fold.
3. Injects finetuned params and refit_params into the model step.
4. Executes the pipeline using the standard step execution flow.
5. Creates a prediction entry with ``fold_id="final"`` and
   ``refit_context="standalone"``.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import Any

from nirs4all.core.logging import get_logger
from nirs4all.pipeline.analysis.topology import _is_splitter_instance
from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.config.refit_params import resolve_refit_params
from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
from nirs4all.pipeline.storage.store_schema import REFIT_CONTEXT_STANDALONE

logger = get_logger(__name__)


@dataclass
class RefitResult:
    """Result of a refit execution.

    Attributes:
        success: Whether the refit completed successfully.
        model_artifact_id: Artifact ID of the final refit model.
        test_score: Score on the test set, or ``None`` if no test set.
        metric: Metric used for evaluation.
        fold_id: Always ``"final"`` for refit entries.
        refit_context: Always ``"standalone"`` for simple refit.
        predictions_count: Number of prediction entries created.
    """

    success: bool = False
    model_artifact_id: str | None = None
    test_score: float | None = None
    metric: str = ""
    fold_id: str = "final"
    refit_context: str = REFIT_CONTEXT_STANDALONE
    predictions_count: int = 0


def execute_simple_refit(
    refit_config: RefitConfig,
    dataset: Any,  # SpectroDataset
    context: Any,  # ExecutionContext
    runtime_context: RuntimeContext,
    artifact_registry: Any,  # ArtifactRegistry
    executor: Any,  # PipelineExecutor
    prediction_store: Any,  # Predictions
) -> RefitResult:
    """Re-execute the winning configuration on all training data.

    This function implements the "simple refit" strategy for pipelines
    without stacking.  It replays the exact winning pipeline variant
    from Pass 1, but replaces the CV splitter with a single fold that
    uses all training samples for fitting and no validation set.

    Args:
        refit_config: Winning variant configuration from
            :func:`extract_winning_config`.
        dataset: The original ``SpectroDataset`` (will be deep-copied
            internally so that the caller's dataset is not mutated).
        context: Execution context from the orchestrator.
        runtime_context: Runtime context (shared infrastructure).  The
            ``phase`` attribute will be set to ``ExecutionPhase.REFIT``.
        artifact_registry: Artifact registry from Pass 1 (shared --
            refit artifacts are appended to the same registry).
        executor: ``PipelineExecutor`` instance to drive step execution.
        prediction_store: ``Predictions`` instance for collecting refit
            prediction entries.

    Returns:
        A :class:`RefitResult` summarizing the refit outcome.
    """
    from nirs4all.data.predictions import Predictions

    result = RefitResult()

    # Deep-copy the dataset so refit transforms don't mutate the caller's data
    refit_dataset = copy.deepcopy(dataset)

    # Prepare the steps -- keep a reference to the originals for metadata
    original_steps = refit_config.expanded_steps
    steps = copy.deepcopy(original_steps)

    # Detect and handle the splitter step
    has_splitter = False
    for idx, step in enumerate(steps):
        if _step_is_splitter(step):
            has_splitter = True
            # Replace splitter with a single full-training-data fold
            steps[idx] = _make_full_train_fold_step(refit_dataset)
            break

    if not has_splitter:
        # No CV splitter in pipeline -- the model from Pass 1 IS the
        # final model.  Nothing to refit.
        logger.info(
            "No cross-validation detected. The trained model is already "
            "the final model. Skipping refit."
        )
        result.success = True
        return result

    # Inject best_params into the model step
    _inject_best_params(steps, refit_config.best_params)

    # Preserve runtime labels so this helper is non-destructive to callers.
    prev_refit_fold_id = runtime_context.refit_fold_id
    prev_refit_context_name = runtime_context.refit_context_name

    # Set execution phase to REFIT
    runtime_context.phase = ExecutionPhase.REFIT
    runtime_context.refit_fold_id = "final"
    runtime_context.refit_context_name = REFIT_CONTEXT_STANDALONE

    # Reset runtime context counters for a fresh execution pass
    runtime_context.step_number = 0
    runtime_context.operation_count = 0
    runtime_context.substep_number = -1

    # Create a fresh pipeline_uid for the refit pass so that the refit
    # predictions are distinguishable from Pass 1 predictions.
    refit_pipeline_name = f"{runtime_context.pipeline_name or 'pipeline'}_refit"

    # Create a local Predictions to capture refit entries
    refit_predictions = Predictions()

    # Re-initialise the execution context for a clean refit pass
    refit_context = executor.initialize_context(refit_dataset)

    logger.info("Starting refit pass: training winning model on full training set")

    try:
        executor.execute(
            steps=steps,
            config_name=refit_pipeline_name,
            dataset=refit_dataset,
            context=refit_context,
            runtime_context=runtime_context,
            prediction_store=refit_predictions,
            generator_choices=refit_config.generator_choices,
        )
    except Exception as e:
        logger.error(f"Refit execution failed: {e}")
        runtime_context.phase = ExecutionPhase.CV
        runtime_context.refit_fold_id = prev_refit_fold_id
        runtime_context.refit_context_name = prev_refit_context_name
        return result

    # Mark refit prediction entries with fold_id="final", refit_context, and metadata
    _relabel_refit_predictions(refit_predictions, refit_config, original_steps)

    # Merge refit predictions into the caller's prediction store
    if refit_predictions.num_predictions > 0:
        prediction_store.merge_predictions(refit_predictions)
        result.predictions_count = refit_predictions.num_predictions

    # Check for test score
    test_score = _extract_test_score(refit_predictions)
    if test_score is None:
        logger.warning(
            "No test set available for refit evaluation. "
            "final_score is set to None."
        )
    result.test_score = test_score
    result.metric = refit_config.metric
    result.success = True

    # Reset phase back to CV for any subsequent operations
    runtime_context.phase = ExecutionPhase.CV
    runtime_context.refit_fold_id = prev_refit_fold_id
    runtime_context.refit_context_name = prev_refit_context_name

    logger.success("Refit pass completed successfully")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _step_is_splitter(step: Any) -> bool:
    """Check if a step is a CV splitter.

    Handles both live operator instances and serialized dict steps.
    """
    # Live instance (e.g., KFold(), ShuffleSplit())
    if not isinstance(step, dict):
        return _is_splitter_instance(step)

    # Dict with 'split' keyword
    if "split" in step:
        return True

    # Serialized class from expanded_config
    class_path = step.get("class", "")
    if isinstance(class_path, str) and class_path:
        class_path_lower = class_path.lower()
        class_name = class_path.rsplit(".", 1)[-1]

        if "model_selection" in class_path_lower:
            return True

        splitter_fragments = {"Fold", "Split", "Splitter", "LeaveOne", "LeaveP"}
        if any(frag in class_name for frag in splitter_fragments):
            return True

    return False


class _FullTrainFoldSplitter:
    """Dummy splitter that yields a single fold with all samples in train."""

    def __init__(self, n_samples: int | None = None) -> None:
        # Optional fallback when X length cannot be inferred at runtime.
        self._n_samples = n_samples

    def split(self, X, y=None, groups=None):
        """Yield a single fold: all indices go to train, empty validation."""
        n_samples = self._n_samples
        if X is not None:
            with contextlib.suppress(Exception):
                n_samples = int(X.shape[0])
            if n_samples is None:
                with contextlib.suppress(Exception):
                    n_samples = int(len(X))

        if n_samples is None:
            n_samples = 0

        yield list(range(n_samples)), []

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return 1 (single fold)."""
        return 1


def _make_full_train_fold_step(dataset: Any) -> Any:
    """Create a splitter step that assigns all training samples to a single fold.

    Args:
        dataset: SpectroDataset to count training samples.

    Returns:
        A ``_FullTrainFoldSplitter`` instance that yields one fold.
    """
    try:
        X_train = dataset.x({"partition": "train"}, layout="2d")
        n_train = int(X_train.shape[0])
    except Exception:
        n_train = 100

    # The splitter re-measures from X at execution time so it remains valid
    # when upstream refit steps change active sample count.
    return _FullTrainFoldSplitter(n_train)


def _inject_best_params(steps: list[Any], best_params: dict[str, Any]) -> None:
    """Inject best hyperparameters and refit_params into model steps.

    Modifies the step list in place.  For each model step:
    1. Apply ``best_params`` (from Optuna/finetuning).
    2. Apply ``resolve_refit_params()`` to merge refit_params on top.

    Args:
        steps: Mutable list of pipeline steps.
        best_params: Best hyperparameters from CV.
    """
    for _idx, step in enumerate(steps):
        if not isinstance(step, dict):
            # Live model instance: set_params directly
            if hasattr(step, "fit") and hasattr(step, "predict") and not _step_is_splitter(step) and best_params:
                _apply_params_to_model(step, best_params)
            continue

        # Dict step with "model" keyword
        model_value = step.get("model")
        if model_value is None:
            continue

        # Apply best_params to the model instance or config
        if best_params:
            if hasattr(model_value, "set_params"):
                _apply_params_to_model(model_value, best_params)
            elif isinstance(model_value, dict) and "params" in model_value:
                model_value["params"].update(best_params)

        # Remove finetune_params to prevent re-triggering during refit
        step.pop("finetune_params", None)

        # Resolve refit_params (merge refit_params on top of train_params)
        resolved = resolve_refit_params(step)
        if resolved:
            # Apply resolved params to the model (sklearn-compatible models)
            if hasattr(model_value, "set_params"):
                _apply_params_to_model(model_value, resolved)
            elif isinstance(model_value, dict) and "params" in model_value:
                model_value["params"].update(resolved)

            # Write resolved params back to train_params so that
            # launch_training() picks them up for all frameworks
            # (PyTorch, TensorFlow, etc. read train_params, not set_params).
            step["train_params"] = resolved


def _apply_params_to_model(model: Any, params: dict[str, Any]) -> None:
    """Safely apply parameters to a model, skipping refit-only keys.

    Args:
        model: sklearn-compatible model with ``set_params()``.
        params: Parameters to apply.
    """
    # Filter out refit-specific keys that are not model params
    refit_only_keys = {"warm_start_fold"}
    applicable = {k: v for k, v in params.items() if k not in refit_only_keys}

    if not applicable or not hasattr(model, "set_params"):
        return

    try:
        model.set_params(**applicable)
    except (TypeError, ValueError) as e:
        # Some params may not apply to this model (e.g., warm_start on PLS).
        # Try one-by-one and skip failures.
        for key, value in applicable.items():
            try:
                model.set_params(**{key: value})
            except (TypeError, ValueError):
                logger.debug(f"Skipping unsupported param {key}={value}: {e}")


def _relabel_refit_predictions(
    predictions: Any,
    refit_config: RefitConfig | None = None,
    original_steps: list[Any] | None = None,
) -> None:
    """Relabel prediction entries as refit entries with metadata.

    Sets ``fold_id="final"`` and ``refit_context="standalone"`` on all
    buffered prediction entries.  When *refit_config* is provided, also
    injects refit metadata (CV strategy, generator choices, best params)
    into the entry's ``metadata`` dict.

    Args:
        predictions: ``Predictions`` instance with buffered entries.
        refit_config: Optional refit configuration for metadata enrichment.
        original_steps: Original expanded steps (before splitter replacement),
            used to extract CV splitter description.
    """
    # Build metadata from refit_config
    refit_metadata: dict[str, Any] = {}
    if refit_config is not None:
        if refit_config.generator_choices:
            refit_metadata["generator_choices"] = refit_config.generator_choices
        if refit_config.best_params:
            refit_metadata["best_params"] = refit_config.best_params

    # Extract CV strategy info from original steps
    if original_steps is not None:
        cv_strategy, cv_n_folds = _extract_cv_strategy(original_steps)
        if cv_strategy:
            refit_metadata["cv_strategy"] = cv_strategy
        if cv_n_folds is not None:
            refit_metadata["cv_n_folds"] = cv_n_folds

    for entry in predictions._buffer:
        entry["fold_id"] = "final"
        entry["refit_context"] = REFIT_CONTEXT_STANDALONE
        # Inject the CV selection score so final entries can be ranked
        # in mix mode by their originating chain's avg folds val_score.
        if refit_config is not None and refit_config.best_score:
            entry["cv_rank_score"] = refit_config.best_score
        if refit_metadata:
            existing = entry.get("metadata") or {}
            existing.update(refit_metadata)
            entry["metadata"] = existing


def _extract_cv_strategy(steps: list[Any]) -> tuple[str, int | None]:
    """Extract CV strategy description from pipeline steps.

    Scans the step list for the CV splitter and returns a human-readable
    description and the number of folds.

    Args:
        steps: List of pipeline steps (original, before splitter replacement).

    Returns:
        Tuple of (strategy_description, n_folds).  Returns ``("", None)``
        if no splitter is found.
    """
    for step in steps:
        if not _step_is_splitter(step):
            continue

        # Live instance
        if not isinstance(step, dict):
            class_name = type(step).__name__
            n_folds = None
            if hasattr(step, "get_n_splits"):
                with contextlib.suppress(Exception):
                    n_folds = step.get_n_splits()
            if hasattr(step, "n_splits"):
                n_folds = step.n_splits
            params_str = ""
            if n_folds is not None:
                params_str = f"({n_folds})"
            return f"{class_name}{params_str}", n_folds

        # Serialized dict step
        class_path = step.get("class", "")
        class_name = class_path.rsplit(".", 1)[-1] if class_path else "Unknown"
        params = step.get("params", {})
        n_folds = params.get("n_splits")
        params_str = ""
        if n_folds is not None:
            params_str = f"({n_folds})"
        return f"{class_name}{params_str}", n_folds

    return "", None


def _extract_test_score(predictions: Any) -> float | None:
    """Extract the test score from refit predictions.

    Looks for test-partition entries or entries with a non-None
    ``test_score`` field.

    Args:
        predictions: ``Predictions`` instance.

    Returns:
        Test score as float, or ``None`` if no test score available.
    """
    for entry in predictions._buffer:
        test_score = entry.get("test_score")
        if test_score is not None:
            return float(test_score)

    # Also check test partition entries
    test_entries = predictions.filter_predictions(partition="test")
    if test_entries:
        score = test_entries[0].get("test_score")
        if score is not None:
            return float(score)

    return None
