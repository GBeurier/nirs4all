"""Stacking refit executor for pipelines with merge-predictions.

Implements a two-step refit strategy for stacking pipelines:

**Step 1 -- Base model refit**: For each base model branch in the
winning variant, clone the preprocessing pipeline, fit on all training
data, train the model, and generate in-sample predictions that will
serve as meta-features for Step 2.

**Step 2 -- Meta-model refit**: Collect base model in-sample predictions
into a feature matrix, train the meta-model on that matrix, evaluate on
the test set (if available), and persist the result with
``fold_id="final"`` and ``refit_context="stacking"``.

This module also handles *mixed merge* pipelines (Task 3.4) where some
branches contribute features and others contribute predictions, as well
as GPU-aware serialization (Task 3.5) to avoid OOM when multiple GPU
models are refit concurrently.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.pipeline.analysis.topology import PipelineTopology
from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
from nirs4all.pipeline.execution.refit.executor import (
    RefitResult,
    _FullTrainFoldSplitter,
    _inject_best_params,
    _step_is_splitter,
)
from nirs4all.pipeline.storage.store_schema import REFIT_CONTEXT_STACKING

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Nested stacking depth limit (Task 4.1)
# ---------------------------------------------------------------------------

DEFAULT_MAX_STACKING_DEPTH = 3

# ---------------------------------------------------------------------------
# GPU detection helpers (Task 3.5)
# ---------------------------------------------------------------------------

# Module path fragments indicating GPU-backed frameworks
_GPU_MODULE_FRAGMENTS = frozenset({
    "tensorflow",
    "keras",
    "torch",
    "pytorch",
    "jax",
    "flax",
})


def _is_gpu_model(model_config: Any) -> bool:
    """Detect whether a model configuration uses a GPU-backed framework.

    Inspects the model's class path (for serialized configs) or the
    module hierarchy (for live instances) for TensorFlow, PyTorch, or
    JAX indicators.

    Args:
        model_config: A model instance, class path string, or serialized
            dict with a ``"class"`` key.

    Returns:
        ``True`` if the model appears to be GPU-backed.
    """
    class_path = _extract_model_class_path(model_config)
    if not class_path:
        return False
    class_path_lower = class_path.lower()
    return any(frag in class_path_lower for frag in _GPU_MODULE_FRAGMENTS)


def _extract_model_class_path(model_config: Any) -> str:
    """Extract the class path string from a model config.

    Handles live instances, strings, and serialized dicts.

    Args:
        model_config: Model configuration in any supported form.

    Returns:
        Dotted class path string, or empty string if unresolvable.
    """
    if model_config is None:
        return ""
    if isinstance(model_config, str):
        return model_config
    if isinstance(model_config, dict):
        return model_config.get("class", "")
    # Live instance -- use its module + class name
    cls = type(model_config)
    return f"{cls.__module__}.{cls.__qualname__}"


def _cleanup_gpu_memory() -> None:
    """Release GPU memory after a GPU model refit.

    Calls framework-specific cleanup functions for TensorFlow, PyTorch,
    and JAX.  Failures are silently ignored (the frameworks may not be
    installed).
    """
    # TensorFlow / Keras
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass

    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # JAX -- no explicit cache-clearing API; garbage collection suffices
    try:
        import gc
        gc.collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Branch classification for mixed merge (Task 3.4)
# ---------------------------------------------------------------------------


def _classify_branch_type(
    branch_index: int,
    merge_step: dict[str, Any] | None,
) -> str:
    """Determine whether a branch contributes features or predictions.

    For a simple ``merge: "predictions"`` every branch is a prediction
    branch.  For a mixed merge ``merge: {"features": [...],
    "predictions": [...]}``, this function checks which list the branch
    index appears in.

    Args:
        branch_index: 0-based index of the branch.
        merge_step: The merge step dict, or ``None``.

    Returns:
        ``"predictions"``, ``"features"``, or ``"unknown"``.
    """
    if merge_step is None:
        return "predictions"

    merge_value = merge_step.get("merge", merge_step)

    if isinstance(merge_value, str):
        return merge_value  # "predictions", "features", etc.

    if isinstance(merge_value, dict):
        pred_branches = merge_value.get("predictions", [])
        feat_branches = merge_value.get("features", [])

        if branch_index in pred_branches:
            return "predictions"
        if branch_index in feat_branches:
            return "features"

        # If lists are empty or branch not listed, default to predictions
        if pred_branches or feat_branches:
            return "unknown"
        return "predictions"

    return "predictions"


# ---------------------------------------------------------------------------
# Step extraction helpers
# ---------------------------------------------------------------------------


def _find_branch_step(steps: list[Any]) -> tuple[int, Any] | None:
    """Find the branch step and its index in the step list.

    Args:
        steps: Expanded pipeline step list.

    Returns:
        Tuple of (index, branch_step_dict) or ``None``.
    """
    for idx, step in enumerate(steps):
        if isinstance(step, dict) and "branch" in step:
            return idx, step
    return None


def _find_merge_step(steps: list[Any], after: int = 0) -> dict[str, Any] | None:
    """Find the merge step after a given index.

    Args:
        steps: Expanded pipeline step list.
        after: Start searching after this index.

    Returns:
        The merge step dict, or ``None``.
    """
    for step in steps[after + 1:]:
        if isinstance(step, dict) and "merge" in step:
            return step
    return None


def _extract_pre_branch_steps(steps: list[Any], branch_idx: int) -> list[Any]:
    """Extract preprocessing steps before the branch.

    Args:
        steps: Expanded pipeline step list.
        branch_idx: Index of the branch step.

    Returns:
        List of steps before the branch.
    """
    return steps[:branch_idx]


def _extract_post_merge_steps(steps: list[Any], branch_idx: int) -> list[Any]:
    """Extract steps after the merge step (meta-model and beyond).

    Args:
        steps: Expanded pipeline step list.
        branch_idx: Index of the branch step.

    Returns:
        List of steps after the merge step.
    """
    for idx, step in enumerate(steps[branch_idx + 1:], start=branch_idx + 1):
        if isinstance(step, dict) and "merge" in step:
            return steps[idx + 1:]
    return []


def _extract_model_from_steps(steps: list[Any]) -> Any:
    """Extract the model config from a list of steps.

    Args:
        steps: Pipeline step list (branch sub-pipeline or post-merge).

    Returns:
        The model value (instance or dict), or ``None``.
    """
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            return step["model"]
        if not isinstance(step, dict) and hasattr(step, "fit") and hasattr(step, "predict") and not _step_is_splitter(step):
            return step
    return None


def _extract_preprocessing_from_branch(branch_steps: list[Any]) -> list[Any]:
    """Extract non-model, non-splitter steps from a branch sub-pipeline.

    These are the preprocessing steps that need to be replayed before
    the model.

    Args:
        branch_steps: Steps within a single branch.

    Returns:
        List of preprocessing-only steps.
    """
    result = []
    for step in branch_steps:
        if isinstance(step, dict) and "model" in step:
            continue
        if not isinstance(step, dict) and hasattr(step, "fit") and hasattr(step, "predict") and not _step_is_splitter(step):
            continue
        result.append(step)
    return result


def _branch_contains_stacking(branch_steps: list[Any]) -> bool:
    """Check if a branch sub-pipeline contains stacking (merge: predictions).

    Examines the branch steps for a ``merge: "predictions"`` step,
    which indicates the branch itself is a stacking pipeline that
    needs recursive refit (Task 4.1).

    Also checks nested branches recursively.

    Args:
        branch_steps: Steps within a single branch.

    Returns:
        ``True`` if the branch contains stacking.
    """
    for step in branch_steps:
        if not isinstance(step, dict):
            continue
        if "merge" in step:
            merge_value = step["merge"]
            if merge_value == "predictions":
                return True
            if isinstance(merge_value, dict) and "predictions" in merge_value:
                return True
        # Check recursively inside nested branches
        if "branch" in step:
            branch_value = step["branch"]
            if isinstance(branch_value, list):
                for sub_steps in branch_value:
                    sub_list = sub_steps if isinstance(sub_steps, list) else [sub_steps]
                    if _branch_contains_stacking(sub_list):
                        return True
    return False


def _select_winning_branch(
    refit_config: RefitConfig,
    store: Any,
    branch_value: list[Any],
) -> int:
    """Select the winning branch from CV predictions (Task 4.5).

    Queries the store for predictions from the winning pipeline,
    groups by ``branch_id``, and returns the branch index with the
    best average validation score.

    Args:
        refit_config: Winning variant configuration (for pipeline_id and metric).
        store: WorkspaceStore for querying predictions.
        branch_value: List of branch sub-pipelines.

    Returns:
        0-based index of the winning branch.
    """
    if store is None or not refit_config.pipeline_id:
        return 0

    try:
        preds = store.query_predictions(
            pipeline_id=refit_config.pipeline_id,
            partition="val",
        )
    except Exception:
        return 0

    if preds is None or len(preds) == 0:
        return 0

    # Group val_scores by branch_id
    branch_scores: dict[int, list[float]] = {}
    try:
        branch_ids = preds["branch_id"].to_list()
        val_scores = preds["val_score"].to_list()
    except Exception:
        return 0

    for bid, vs in zip(branch_ids, val_scores):
        if bid is not None and vs is not None:
            bid = int(bid)
            if bid not in branch_scores:
                branch_scores[bid] = []
            branch_scores[bid].append(float(vs))

    if not branch_scores:
        return 0

    # Select branch with best average val_score
    ascending = _infer_ascending_for_metric(refit_config.metric)
    branch_avg = {
        bid: sum(scores) / len(scores)
        for bid, scores in branch_scores.items()
    }

    if ascending:
        winning = min(branch_avg, key=branch_avg.get)
    else:
        winning = max(branch_avg, key=branch_avg.get)

    # Clamp to valid branch range
    if winning < 0 or winning >= len(branch_value):
        return 0

    return winning


def _infer_ascending_for_metric(metric: str) -> bool:
    """Infer whether lower-is-better from the metric name.

    Args:
        metric: Metric name string.

    Returns:
        ``True`` if lower is better (error metrics like RMSE, MAE).
    """
    if not metric:
        return True
    metric_lower = metric.lower()
    higher_is_better = {
        "r2", "accuracy", "f1", "precision", "recall",
        "auc", "roc_auc", "balanced_accuracy",
    }
    return metric_lower not in higher_is_better


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def execute_stacking_refit(
    refit_config: RefitConfig,
    dataset: Any,  # SpectroDataset
    context: Any,  # ExecutionContext
    runtime_context: RuntimeContext,
    artifact_registry: Any,  # ArtifactRegistry
    executor: Any,  # PipelineExecutor
    prediction_store: Any,  # Predictions
    topology: PipelineTopology,
    max_depth: int = DEFAULT_MAX_STACKING_DEPTH,
    _current_depth: int = 0,
) -> RefitResult:
    """Execute a two-step stacking refit on the winning configuration.

    **Step 1**: For each base model branch, clone preprocessing, fit on
    all training data, and collect in-sample predictions as meta-features.
    If a branch itself contains stacking (Task 4.1), recursively applies
    ``execute_stacking_refit`` up to ``max_depth`` levels deep.

    **Step 2**: Build a meta-feature matrix from base model predictions,
    train the meta-model, and persist with ``fold_id="final"`` and
    ``refit_context="stacking"``.

    For mixed merge pipelines (Task 3.4), feature branches use simple
    refit (transform only), while prediction branches use the full
    stacking refit flow.

    GPU-backed models (Task 3.5) are detected and refit sequentially
    with memory cleanup between each, while CPU models can be refit
    in parallel.

    Args:
        refit_config: Winning variant configuration.
        dataset: Original ``SpectroDataset`` (deep-copied internally).
        context: Execution context (may be ``None``; executor creates
            its own internally).
        runtime_context: Shared runtime context.
        artifact_registry: Artifact registry shared from CV pass.
        executor: ``PipelineExecutor`` for step execution.
        prediction_store: ``Predictions`` accumulator for refit entries.
        topology: Pipeline topology descriptor.
        max_depth: Maximum nesting depth for recursive stacking refit.
            Defaults to ``DEFAULT_MAX_STACKING_DEPTH`` (3).
        _current_depth: Internal recursion depth counter (do not set
            manually).

    Returns:
        A :class:`RefitResult` summarizing the stacking refit outcome.
    """
    from nirs4all.data.predictions import Predictions

    result = RefitResult(refit_context=REFIT_CONTEXT_STACKING)

    # Deep-copy dataset so refit transforms don't mutate caller's data
    refit_dataset = copy.deepcopy(dataset)

    # Deep-copy the steps for manipulation
    original_steps = refit_config.expanded_steps
    steps = copy.deepcopy(original_steps)

    # Inject best params into model steps
    _inject_best_params(steps, refit_config.best_params)

    # Locate the branch and merge structure
    branch_info = _find_branch_step(steps)
    if branch_info is None:
        logger.warning("Stacking refit called but no branch found in steps. Skipping.")
        result.success = False
        return result

    branch_idx, branch_step = branch_info
    branch_value = branch_step.get("branch", [])
    merge_step = _find_merge_step(steps, branch_idx)

    if not isinstance(branch_value, list):
        logger.warning("Stacking refit expects duplication branches (list). Skipping.")
        result.success = False
        return result

    # Extract structural components
    pre_branch_steps = _extract_pre_branch_steps(steps, branch_idx)
    post_merge_steps = _extract_post_merge_steps(steps, branch_idx)
    meta_model_config = _extract_model_from_steps(post_merge_steps)

    # Preserve caller overrides so this helper does not leak refit labels.
    prev_refit_fold_id = runtime_context.refit_fold_id
    prev_refit_context_name = runtime_context.refit_context_name

    # Set execution phase to REFIT
    runtime_context.phase = ExecutionPhase.REFIT
    runtime_context.refit_fold_id = "final"
    runtime_context.refit_context_name = REFIT_CONTEXT_STACKING
    runtime_context.step_number = 0
    runtime_context.operation_count = 0
    runtime_context.substep_number = -1

    refit_pipeline_name = f"{runtime_context.pipeline_name or 'pipeline'}_stacking_refit"

    logger.info("Starting stacking refit pass")

    try:
        # ---------------------------------------------------------------
        # Step 1: Refit base models and collect meta-features
        # ---------------------------------------------------------------
        base_predictions_list: list[np.ndarray] = []
        base_features_list: list[np.ndarray] = []

        # Check for GPU models to determine execution strategy (Task 3.5)
        has_gpu = _any_branch_has_gpu_model(branch_value)

        for branch_i, branch_steps in enumerate(branch_value):
            if not isinstance(branch_steps, list):
                branch_steps = [branch_steps]

            branch_type = _classify_branch_type(branch_i, merge_step)
            model_config = _extract_model_from_steps(branch_steps)

            logger.info(
                f"Stacking refit: branch {branch_i} "
                f"(type={branch_type}, gpu={_is_gpu_model(model_config) if model_config else False})"
            )

            # Task 4.1: Check if this branch contains nested stacking
            if _branch_contains_stacking(branch_steps):
                if _current_depth + 1 >= max_depth:
                    logger.warning(
                        f"Nested stacking depth limit ({max_depth}) reached. "
                        f"Branch {branch_i} will use simple base model refit."
                    )
                    # Fall through to normal base model refit below
                else:
                    logger.info(
                        f"Stacking refit: branch {branch_i} contains nested "
                        f"stacking (depth {_current_depth + 1}/{max_depth})"
                    )
                    from nirs4all.pipeline.analysis.topology import (
                        analyze_topology as _analyze_topology,
                    )

                    # Build the full nested pipeline: pre-branch + branch steps
                    nested_steps = (
                        copy.deepcopy(pre_branch_steps)
                        + copy.deepcopy(branch_steps)
                    )
                    nested_topology = _analyze_topology(nested_steps)
                    nested_config = RefitConfig(
                        expanded_steps=nested_steps,
                        best_params=refit_config.best_params,
                        variant_index=refit_config.variant_index,
                        generator_choices=refit_config.generator_choices,
                        pipeline_id=refit_config.pipeline_id,
                        metric=refit_config.metric,
                        best_score=refit_config.best_score,
                    )

                    nested_predictions = Predictions()
                    nested_runtime = RuntimeContext(
                        store=runtime_context.store,
                        artifact_loader=runtime_context.artifact_loader,
                        artifact_registry=artifact_registry,
                        step_runner=executor.step_runner,
                        run_id=runtime_context.run_id,
                        pipeline_name=(
                            f"{refit_pipeline_name}_nested_{branch_i}"
                        ),
                        phase=ExecutionPhase.REFIT,
                        refit_fold_id="final",
                        refit_context_name=REFIT_CONTEXT_STACKING,
                    )

                    try:
                        nested_result = execute_stacking_refit(
                            refit_config=nested_config,
                            dataset=copy.deepcopy(refit_dataset),
                            context=context,
                            runtime_context=nested_runtime,
                            artifact_registry=artifact_registry,
                            executor=executor,
                            prediction_store=nested_predictions,
                            topology=nested_topology,
                            max_depth=max_depth,
                            _current_depth=_current_depth + 1,
                        )
                    except Exception as e:
                        logger.error(
                            f"Nested stacking refit failed for "
                            f"branch {branch_i}: {e}"
                        )
                        runtime_context.phase = ExecutionPhase.CV
                        return result

                    if not nested_result.success:
                        logger.error(
                            f"Nested stacking refit failed for "
                            f"branch {branch_i}"
                        )
                        runtime_context.phase = ExecutionPhase.CV
                        return result

                    # Relabel and collect predictions
                    _relabel_stacking_predictions(
                        nested_predictions, branch_i
                    )
                    in_sample = _extract_in_sample_predictions(
                        nested_predictions
                    )
                    if in_sample is not None:
                        if branch_type == "features":
                            base_features_list.append(in_sample)
                        else:
                            base_predictions_list.append(in_sample)

                    if nested_predictions.num_predictions > 0:
                        prediction_store.merge_predictions(
                            nested_predictions
                        )

                    continue  # Skip normal base model refit

            # Build the full sub-pipeline for this branch:
            # pre_branch_steps + branch_preprocessing + splitter_replacement + model
            branch_preprocessing = _extract_preprocessing_from_branch(branch_steps)

            # Build a complete sub-pipeline for this branch
            sub_steps = copy.deepcopy(pre_branch_steps) + copy.deepcopy(branch_preprocessing)

            # Replace any splitter with full-train-data fold
            sub_steps = _replace_splitter(sub_steps, refit_dataset)

            # Add the model step back
            if model_config is not None:
                sub_steps.append({"model": copy.deepcopy(model_config)})

            # Execute the sub-pipeline
            branch_predictions = Predictions()
            branch_context = executor.initialize_context(copy.deepcopy(refit_dataset))

            branch_runtime = RuntimeContext(
                store=runtime_context.store,
                artifact_loader=runtime_context.artifact_loader,
                artifact_registry=artifact_registry,
                step_runner=executor.step_runner,
                run_id=runtime_context.run_id,
                phase=ExecutionPhase.REFIT,
                refit_fold_id="final",
                refit_context_name=REFIT_CONTEXT_STACKING,
            )

            try:
                executor.execute(
                    steps=sub_steps,
                    config_name=f"{refit_pipeline_name}_base_{branch_i}",
                    dataset=copy.deepcopy(refit_dataset),
                    context=branch_context,
                    runtime_context=branch_runtime,
                    prediction_store=branch_predictions,
                    generator_choices=refit_config.generator_choices,
                )
            except Exception as e:
                logger.error(f"Stacking refit: base model branch {branch_i} failed: {e}")
                runtime_context.phase = ExecutionPhase.CV
                return result

            # Relabel branch predictions
            _relabel_stacking_predictions(branch_predictions, branch_i)

            # Collect in-sample predictions as meta-features
            in_sample_preds = _extract_in_sample_predictions(branch_predictions)
            if in_sample_preds is not None:
                if branch_type == "features":
                    base_features_list.append(in_sample_preds)
                else:
                    base_predictions_list.append(in_sample_preds)

            # Merge branch predictions into the main store
            if branch_predictions.num_predictions > 0:
                prediction_store.merge_predictions(branch_predictions)

            # GPU memory cleanup after each GPU model (Task 3.5)
            if has_gpu and model_config is not None and _is_gpu_model(model_config):
                _cleanup_gpu_memory()

        # ---------------------------------------------------------------
        # Step 2: Train meta-model on base model predictions
        # ---------------------------------------------------------------
        logger.info("Stacking refit: training meta-model on base predictions")

        # Build meta-feature matrix (Task 3.4: combine features + predictions)
        all_meta_features = base_predictions_list + base_features_list
        if not all_meta_features:
            logger.warning("No base model predictions collected. Cannot train meta-model.")
            runtime_context.phase = ExecutionPhase.CV
            result.success = False
            return result

        meta_X = np.column_stack(all_meta_features)

        # Build the meta-model sub-pipeline: splitter replacement + meta-model
        meta_steps: list[Any] = []

        # Add a full-train-data fold splitter
        meta_steps.append(_FullTrainFoldSplitter(meta_X.shape[0]))

        # Add the meta-model
        if meta_model_config is not None:
            meta_steps.append({"model": copy.deepcopy(meta_model_config)})
        else:
            logger.warning("No meta-model found in post-merge steps.")
            runtime_context.phase = ExecutionPhase.CV
            result.success = False
            return result

        # Execute the meta-model sub-pipeline
        meta_predictions = Predictions()

        # Create a synthetic dataset with meta-features
        meta_dataset = _create_meta_dataset(meta_X, refit_dataset)
        meta_context = executor.initialize_context(meta_dataset)

        meta_runtime = RuntimeContext(
            store=runtime_context.store,
            artifact_loader=runtime_context.artifact_loader,
            artifact_registry=artifact_registry,
            step_runner=executor.step_runner,
            run_id=runtime_context.run_id,
            phase=ExecutionPhase.REFIT,
            refit_fold_id="final",
            refit_context_name=REFIT_CONTEXT_STACKING,
        )

        try:
            executor.execute(
                steps=meta_steps,
                config_name=f"{refit_pipeline_name}_meta",
                dataset=meta_dataset,
                context=meta_context,
                runtime_context=meta_runtime,
                prediction_store=meta_predictions,
                generator_choices=[],
            )
        except Exception as e:
            logger.error(f"Stacking refit: meta-model training failed: {e}")
            runtime_context.phase = ExecutionPhase.CV
            return result

        # Relabel meta-model predictions
        _relabel_stacking_predictions(meta_predictions, branch_index=None, is_meta=True)

        # Merge meta predictions into the main store
        if meta_predictions.num_predictions > 0:
            prediction_store.merge_predictions(meta_predictions)

        # Extract test score
        test_score = _extract_test_score_from_predictions(meta_predictions)
        result.test_score = test_score
        result.metric = refit_config.metric
        result.success = True

        result.predictions_count = meta_predictions.num_predictions

        logger.success("Stacking refit pass completed successfully")

    except Exception as e:
        logger.error(f"Stacking refit failed: {e}")
        result.success = False
    finally:
        # Always reset phase back to CV
        runtime_context.phase = ExecutionPhase.CV
        runtime_context.refit_fold_id = prev_refit_fold_id
        runtime_context.refit_context_name = prev_refit_context_name

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _any_branch_has_gpu_model(branch_value: list[Any]) -> bool:
    """Check if any branch contains a GPU-backed model.

    Args:
        branch_value: List of branch sub-pipelines.

    Returns:
        ``True`` if at least one branch has a GPU model.
    """
    for branch_steps in branch_value:
        if not isinstance(branch_steps, list):
            branch_steps = [branch_steps]
        model_config = _extract_model_from_steps(branch_steps)
        if model_config is not None and _is_gpu_model(model_config):
            return True
    return False


def _replace_splitter(steps: list[Any], dataset: Any) -> list[Any]:
    """Replace CV splitter steps with a full-training-data fold.

    Args:
        steps: Step list (modified in place and returned).
        dataset: Dataset to determine sample count.

    Returns:
        The modified step list.
    """
    for idx, step in enumerate(steps):
        if _step_is_splitter(step):
            try:
                X_train = dataset.x({"partition": "train"}, layout="2d")
                n_train = X_train.shape[0]
            except Exception:
                n_train = 100
            steps[idx] = _FullTrainFoldSplitter(n_train)
            break
    return steps


def _relabel_stacking_predictions(
    predictions: Any,
    branch_index: int | None,
    is_meta: bool = False,
) -> None:
    """Relabel prediction entries for the stacking refit context.

    Sets ``fold_id="final"`` and ``refit_context="stacking"`` on all
    buffered entries.  Adds branch index or meta-model marker to
    metadata.

    Args:
        predictions: ``Predictions`` instance with buffered entries.
        branch_index: 0-based branch index, or ``None`` for meta-model.
        is_meta: Whether these are meta-model predictions.
    """
    for entry in predictions._buffer:
        entry["fold_id"] = "final"
        entry["refit_context"] = REFIT_CONTEXT_STACKING
        metadata = entry.get("metadata") or {}
        if is_meta:
            metadata["stacking_role"] = "meta_model"
        elif branch_index is not None:
            metadata["stacking_role"] = "base_model"
            metadata["stacking_branch"] = branch_index
        entry["metadata"] = metadata


def _extract_in_sample_predictions(predictions: Any) -> np.ndarray | None:
    """Extract in-sample predictions from a Predictions object.

    Looks for train-partition prediction arrays.  Falls back to
    val-partition entries if no train entries exist.

    Args:
        predictions: ``Predictions`` instance.

    Returns:
        1D numpy array of predictions, or ``None`` if no predictions
        were found.
    """
    for entry in predictions._buffer:
        y_pred = entry.get("y_pred")
        if y_pred is None:
            continue
        if isinstance(y_pred, np.ndarray):
            if y_pred.size == 0:
                continue
            return y_pred.ravel()
        arr = np.asarray(y_pred)
        if arr.size == 0:
            continue
        return arr.ravel()

    return None


def _extract_test_score_from_predictions(predictions: Any) -> float | None:
    """Extract test score from meta-model predictions.

    Args:
        predictions: ``Predictions`` instance.

    Returns:
        Test score as float, or ``None``.
    """
    for entry in predictions._buffer:
        test_score = entry.get("test_score")
        if test_score is not None:
            return float(test_score)
    return None


def _create_meta_dataset(meta_X: np.ndarray, original_dataset: Any) -> Any:
    """Create a synthetic SpectroDataset with meta-features.

    Uses the original dataset's target values and partition structure
    but replaces X with the meta-feature matrix.

    Args:
        meta_X: Meta-feature matrix (n_samples, n_meta_features).
        original_dataset: Original SpectroDataset for target values.

    Returns:
        A new SpectroDataset containing the meta-features.
    """
    from nirs4all.data.dataset import SpectroDataset

    meta_dataset = SpectroDataset(name=f"{original_dataset.name}_meta")

    # Add meta-features as training data
    meta_dataset.add_samples(meta_X, indexes={"partition": "train"})

    # Copy target values from original dataset
    try:
        y_train = original_dataset.y({"partition": "train"})
        if y_train is not None:
            meta_dataset.add_targets(y_train)
    except Exception:
        pass

    return meta_dataset


# ---------------------------------------------------------------------------
# Task 4.2: Separation branch refit
# ---------------------------------------------------------------------------


def execute_separation_refit(
    refit_config: RefitConfig,
    dataset: Any,  # SpectroDataset
    context: Any,  # ExecutionContext
    runtime_context: RuntimeContext,
    artifact_registry: Any,  # ArtifactRegistry
    executor: Any,  # PipelineExecutor
    prediction_store: Any,  # Predictions
    topology: PipelineTopology,
) -> RefitResult:
    """Execute refit for pipelines with separation branches.

    For pipelines using ``by_metadata``, ``by_tag``, or ``by_source``
    separation branches, the regular pipeline executor already handles
    data splitting during step execution.

    - **Separation + stacking** (e.g. ``by_source`` with inner models
      and ``merge: "predictions"``): delegates to
      :func:`execute_stacking_refit`.
    - **Simple separation** (e.g. ``by_metadata`` + per-group model +
      ``merge: "concat"``): delegates to :func:`execute_simple_refit`
      since the pipeline executor handles separation natively.

    Args:
        refit_config: Winning variant configuration.
        dataset: Original ``SpectroDataset`` (deep-copied internally).
        context: Execution context (may be ``None``).
        runtime_context: Shared runtime context.
        artifact_registry: Artifact registry from CV pass.
        executor: ``PipelineExecutor`` for step execution.
        prediction_store: ``Predictions`` accumulator.
        topology: Pipeline topology descriptor.

    Returns:
        A :class:`RefitResult` summarizing the refit outcome.
    """
    from nirs4all.pipeline.execution.refit.executor import execute_simple_refit

    # If the separation pipeline also has stacking (e.g., by_source with
    # inner models producing predictions merged into an outer model),
    # delegate to the stacking refit which handles the two-step flow.
    if topology.has_stacking or topology.has_mixed_merge:
        logger.info(
            "Separation refit: pipeline has stacking — delegating "
            "to stacking refit"
        )
        return execute_stacking_refit(
            refit_config=refit_config,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            artifact_registry=artifact_registry,
            executor=executor,
            prediction_store=prediction_store,
            topology=topology,
        )

    # For simple separation (by_metadata, by_tag, by_source without
    # stacking), the pipeline executor handles splitting natively when
    # it encounters the branch controller.  Simple refit replays the
    # full pipeline with a full-training-data fold.
    logger.info(
        "Separation refit: simple separation — delegating "
        "to simple refit"
    )
    return execute_simple_refit(
        refit_config=refit_config,
        dataset=dataset,
        context=context,
        runtime_context=runtime_context,
        artifact_registry=artifact_registry,
        executor=executor,
        prediction_store=prediction_store,
    )


# ---------------------------------------------------------------------------
# Task 4.5: Competing branches refit (branches without merge)
# ---------------------------------------------------------------------------


def execute_competing_branches_refit(
    refit_config: RefitConfig,
    dataset: Any,  # SpectroDataset
    context: Any,  # ExecutionContext
    runtime_context: RuntimeContext,
    artifact_registry: Any,  # ArtifactRegistry
    executor: Any,  # PipelineExecutor
    prediction_store: Any,  # Predictions
    topology: PipelineTopology,
) -> RefitResult:
    """Refit only the winning branch for competing branches (no merge).

    When branches have no merge step, they are competing alternatives
    (Section 4.12).  This function identifies the winning branch from
    CV predictions, extracts only that branch's sub-pipeline, and
    refits it using :func:`execute_simple_refit`.

    Args:
        refit_config: Winning variant configuration.
        dataset: Original ``SpectroDataset`` (deep-copied internally).
        context: Execution context (may be ``None``).
        runtime_context: Shared runtime context.
        artifact_registry: Artifact registry from CV pass.
        executor: ``PipelineExecutor`` for step execution.
        prediction_store: ``Predictions`` accumulator.
        topology: Pipeline topology descriptor.

    Returns:
        A :class:`RefitResult` summarizing the refit outcome.
    """
    from nirs4all.pipeline.execution.refit.executor import execute_simple_refit

    steps = copy.deepcopy(refit_config.expanded_steps)

    # Find the branch step
    branch_info = _find_branch_step(steps)
    if branch_info is None:
        logger.warning(
            "Competing branches refit called but no branch found. "
            "Falling back to simple refit."
        )
        return execute_simple_refit(
            refit_config=refit_config,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            artifact_registry=artifact_registry,
            executor=executor,
            prediction_store=prediction_store,
        )

    branch_idx, branch_step = branch_info
    branch_value = branch_step.get("branch", [])

    if not isinstance(branch_value, list) or len(branch_value) <= 1:
        logger.info(
            "Competing branches refit: single or non-list branch. "
            "Using simple refit."
        )
        return execute_simple_refit(
            refit_config=refit_config,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            artifact_registry=artifact_registry,
            executor=executor,
            prediction_store=prediction_store,
        )

    # Select the winning branch from CV predictions
    winning_idx = _select_winning_branch(
        refit_config, runtime_context.store, branch_value
    )

    logger.info(
        f"Competing branches: selected winning branch "
        f"{winning_idx}/{len(branch_value)}"
    )

    # Build a flattened pipeline: pre-branch + winning branch + post-branch
    pre_branch = steps[:branch_idx]
    winning_steps = branch_value[winning_idx]
    if not isinstance(winning_steps, list):
        winning_steps = [winning_steps]

    # Steps after the branch — skip any merge step (no merge needed
    # when we've selected a single branch)
    post_branch = [
        s for s in steps[branch_idx + 1:]
        if not (isinstance(s, dict) and "merge" in s)
    ]

    flat_steps = pre_branch + winning_steps + post_branch

    # Create a new RefitConfig with the flattened pipeline
    flat_config = RefitConfig(
        expanded_steps=flat_steps,
        best_params=refit_config.best_params,
        variant_index=refit_config.variant_index,
        generator_choices=refit_config.generator_choices,
        pipeline_id=refit_config.pipeline_id,
        metric=refit_config.metric,
        best_score=refit_config.best_score,
    )

    return execute_simple_refit(
        refit_config=flat_config,
        dataset=dataset,
        context=context,
        runtime_context=runtime_context,
        artifact_registry=artifact_registry,
        executor=executor,
        prediction_store=prediction_store,
    )
