"""Pipeline executor for executing a single pipeline on a single dataset."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.storage.manifest_manager import ManifestManager
from nirs4all.pipeline.steps.step_runner import StepRunner
from nirs4all.pipeline.trace import TraceRecorder

logger = get_logger(__name__)


class PipelineExecutor:
    """Executes a single pipeline configuration on a single dataset.

    Handles:
    - Step-by-step execution
    - Context propagation
    - Artifact management for one pipeline run
    - Predictions accumulation for this pipeline

    Attributes:
        step_runner: Executes individual steps
        manifest_manager: Manages pipeline manifests
        verbose: Verbosity level
        mode: Execution mode (train/predict/explain)
        continue_on_error: Whether to continue on step failures
        artifact_registry: Registry for v2 artifact management
    """

    def __init__(
        self,
        step_runner: StepRunner,
        manifest_manager: Optional[ManifestManager] = None,
        verbose: int = 0,
        mode: str = "train",
        continue_on_error: bool = False,
        saver: Any = None,
        artifact_loader: Any = None,
        artifact_registry: Any = None
    ):
        """Initialize pipeline executor.

        Args:
            step_runner: Step runner for executing individual steps
            manifest_manager: Optional manifest manager
            verbose: Verbosity level
            mode: Execution mode (train/predict/explain)
            continue_on_error: Whether to continue on step failures
            saver: Simulation saver for file operations
            artifact_loader: Artifact loader for predict/explain modes
            artifact_registry: Artifact registry for v2 artifact management
        """
        self.step_runner = step_runner
        self.manifest_manager = manifest_manager
        self.verbose = verbose
        self.mode = mode
        self.continue_on_error = continue_on_error
        self.saver = saver
        self.artifact_loader = artifact_loader
        self.artifact_registry = artifact_registry

        # Execution state
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0

    def initialize_context(self, dataset: SpectroDataset) -> ExecutionContext:
        """Initialize ExecutionContext for pipeline execution.

        Args:
            dataset: Dataset to create context for

        Returns:
            Initialized ExecutionContext
        """
        from nirs4all.pipeline.config.context import DataSelector, PipelineState, StepMetadata

        selector = DataSelector(
            partition=None,
            processing=[["raw"]] * dataset.features_sources(),
            layout="2d",
            concat_source=True
        )

        state = PipelineState(
            y_processing="numeric",
            step_number=0,
            mode=self.mode
        )

        metadata = StepMetadata()

        # Get aggregate setting from dataset for propagation through pipeline
        aggregate_column = dataset.aggregate

        return ExecutionContext(
            selector=selector,
            state=state,
            metadata=metadata,
            aggregate_column=aggregate_column
        )

    def execute(
        self,
        steps: List[Any],
        config_name: str,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,  # RuntimeContext
        prediction_store: Optional[Predictions] = None,
        generator_choices: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Execute pipeline steps sequentially on dataset.

        Args:
            steps: List of pipeline steps to execute
            config_name: Pipeline configuration name
            dataset: Dataset to process
            context: Initial execution context
            runtime_context: Runtime infrastructure context
            prediction_store: Prediction store for accumulating results
            generator_choices: List of generator choices that produced this pipeline

        Raises:
            RuntimeError: If pipeline execution fails
        """
        # Reset state for this execution
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0

        logger.starting(f"Starting pipeline {config_name} on dataset {dataset.name}")

        # Compute pipeline hash for identification
        pipeline_hash = self._compute_pipeline_hash(steps)

        # Create pipeline in manifest system (if in train mode)
        pipeline_uid = None
        if self.mode == "train" and self.manifest_manager:
            pipeline_config = {"steps": steps}
            pipeline_uid, pipeline_dir = self.manifest_manager.create_pipeline(
                name=config_name,
                dataset=dataset.name,
                pipeline_config=pipeline_config,
                pipeline_hash=pipeline_hash,
                generator_choices=generator_choices
            )

            # Register with saver
            if self.saver:
                self.saver.register(pipeline_uid)

            # Set pipeline_uid on runtime_context
            if runtime_context:
                runtime_context.pipeline_uid = pipeline_uid
        else:
            # For predict/explain modes, use temporary UID
            pipeline_uid = f"temp_{pipeline_hash}"

        # Save pipeline configuration
        if self.mode != "predict" and self.mode != "explain" and self.saver:
            self.saver.save_json("pipeline.json", steps)

        # Initialize prediction store if not provided
        if prediction_store is None:
            prediction_store = Predictions()

        # Initialize trace recorder for execution trace recording (Phase 2)
        trace_recorder = None
        if self.mode == "train" and runtime_context:
            trace_recorder = TraceRecorder(
                pipeline_uid=pipeline_uid or "",
                metadata={"dataset": dataset.name, "config_name": config_name}
            )
            runtime_context.trace_recorder = trace_recorder

        # Execute all steps
        all_artifacts = []
        try:
            context = self._execute_steps(
                steps,
                dataset,
                context,
                runtime_context,
                prediction_store,
                all_artifacts
            )

            # Save final pipeline configuration
            if self.mode != "predict" and self.mode != "explain" and self.saver:
                self.saver.save_json("pipeline.json", steps)

            # Finalize and save execution trace
            if trace_recorder is not None and self.manifest_manager and pipeline_uid:
                trace = trace_recorder.finalize(
                    preprocessing_chain=dataset.short_preprocessings_str(),
                    metadata={"n_steps": len(steps), "n_artifacts": len(all_artifacts)}
                )
                self.manifest_manager.save_execution_trace(pipeline_uid, trace)

            # Print best result if predictions were generated
            if prediction_store.num_predictions > 0:
                # Use None for ascending to let ranker infer from metric
                pipeline_best = prediction_store.get_best(
                    ascending=None
                )
                if pipeline_best:
                    logger.success(f"Pipeline Best: {Predictions.pred_short_string(pipeline_best)}")

                logger.debug(
                    f"Pipeline {config_name} completed successfully "
                    f"on dataset {dataset.name}"
                )

        except Exception as e:
            logger.error(
                f"Pipeline {config_name} on dataset {dataset.name} "
                f"failed: {str(e)}"
            )
            import traceback
            traceback.print_exc()
            raise

    def _execute_steps(
        self,
        steps: List[Any],
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,
        prediction_store: Predictions,
        all_artifacts: List[Any]
    ) -> ExecutionContext:
        """Execute all steps in sequence.

        Handles pipeline branching: when a branch step is encountered, subsequent
        steps are executed on each branch context independently.

        Args:
            steps: List of steps to execute
            dataset: Dataset to process
            context: Current execution context
            runtime_context: Runtime infrastructure context
            prediction_store: Prediction store
            all_artifacts: List to accumulate artifacts

        Returns:
            Updated execution context
        """
        from nirs4all.pipeline.execution.result import ArtifactMeta

        for step in steps:
            self.step_number += 1
            self.substep_number = 0
            self.operation_count = 0

            # Sync step number to runtime_context
            if runtime_context:
                runtime_context.step_number = self.step_number
                runtime_context.substep_number = self.substep_number
                runtime_context.operation_count = self.operation_count
                runtime_context.reset_processing_counter()  # Reset for unique artifact IDs within step

            # Update context with current step number
            if isinstance(context, ExecutionContext):
                context = context.with_step_number(self.step_number)

            # Load binaries if in prediction/explain mode
            loaded_binaries = None
            if self.mode in ("predict", "explain") and self.artifact_loader:
                loaded_binaries = self.artifact_loader.get_step_binaries(self.step_number)
                if self.verbose > 1 and loaded_binaries:
                    print(f"🔍 Loaded {', '.join(b[0] for b in loaded_binaries)} binaries for step {self.step_number}")

            # Check if we're in branch mode and this is NOT a branch step
            branch_contexts = context.custom.get("branch_contexts", [])
            is_branch_step = isinstance(step, dict) and "branch" in step
            # Merge steps need access to all branch contexts, so they execute globally
            is_merge_step = isinstance(step, dict) and "merge" in step

            if branch_contexts and not is_branch_step and not is_merge_step:
                # Execute step on each branch context
                context = self._execute_step_on_branches(
                    step=step,
                    dataset=dataset,
                    context=context,
                    runtime_context=runtime_context,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store,
                    all_artifacts=all_artifacts
                )
            else:
                # Normal execution (single context)
                context = self._execute_single_step(
                    step=step,
                    dataset=dataset,
                    context=context,
                    runtime_context=runtime_context,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store,
                    all_artifacts=all_artifacts
                )

        return context

    def _execute_step_on_branches(
        self,
        step: Any,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,
        loaded_binaries: Optional[List] = None,
        prediction_store: Optional[Predictions] = None,
        all_artifacts: Optional[List] = None
    ) -> ExecutionContext:
        """Execute a step on all branch contexts.

        Args:
            step: Step to execute
            dataset: Dataset to process
            context: Context containing branch_contexts in custom dict
            runtime_context: Runtime infrastructure context
            loaded_binaries: Pre-loaded binaries for predict mode
            prediction_store: Prediction store
            all_artifacts: List to accumulate artifacts

        Returns:
            Updated context with updated branch contexts
        """
        from nirs4all.pipeline.execution.result import ArtifactMeta

        branch_contexts = context.custom.get("branch_contexts", [])

        if not branch_contexts:
            # No branches, execute normally
            return self._execute_single_step(
                step, dataset, context, runtime_context,
                loaded_binaries, prediction_store, all_artifacts
            )

        logger.debug(f"Executing step on {len(branch_contexts)} branch(es)")

        updated_branch_contexts = []

        for branch_info in branch_contexts:
            branch_id = branch_info["branch_id"]
            branch_name = branch_info["name"]
            branch_context = branch_info["context"]

            # Restore dataset features from branch snapshot if available
            # This ensures each branch's post-branch steps (like model) use the correct
            # feature data that was produced by that branch's preprocessing steps
            features_snapshot = branch_info.get("features_snapshot")
            if features_snapshot is not None:
                import copy
                dataset._features.sources = copy.deepcopy(features_snapshot)

            # V3: Restore chain state from branch snapshot if available
            # This ensures each branch's post-branch steps use the correct operator chain
            # for artifact ID generation (fixes MetaModel chain_path issues across branches)
            chain_snapshot = branch_info.get("chain_snapshot")
            if chain_snapshot is not None and runtime_context and runtime_context.trace_recorder:
                runtime_context.trace_recorder.reset_chain_to(chain_snapshot)

            logger.debug(f"Branch {branch_id} ({branch_name})")

            # Update step number on branch context
            branch_context = branch_context.with_step_number(self.step_number)

            # For predict mode, load binaries specifically for this branch
            branch_binaries = None
            if self.mode in ("predict", "explain"):
                # Get the full branch_path from context (handles nested branches)
                branch_path = getattr(branch_context.selector, 'branch_path', None)

                # First try artifact_provider (for minimal pipeline)
                if runtime_context and hasattr(runtime_context, 'artifact_provider') and runtime_context.artifact_provider:
                    # Use artifact_provider for minimal pipeline prediction
                    # Try to get branch-specific artifacts
                    branch_binaries = runtime_context.artifact_provider.get_artifacts_for_step(
                        self.step_number, branch_path=branch_path, branch_id=branch_id
                    )
                    if not branch_binaries:
                        # Try without branch qualifier (may be a non-branch step artifact)
                        branch_binaries = runtime_context.artifact_provider.get_artifacts_for_step(self.step_number)
                elif self.artifact_loader:
                    # Fallback to artifact_loader for traditional prediction
                    if branch_path:
                        # Use full branch_path for proper nested branch matching
                        branch_binaries = self.artifact_loader.get_step_binaries(
                            self.step_number, branch_path=branch_path
                        )
                    else:
                        # Fallback to simple branch_id
                        branch_binaries = self.artifact_loader.get_step_binaries(
                            self.step_number, branch_id=branch_id
                        )

                if not branch_binaries:
                    # Fallback to non-branch binaries if no branch-specific ones exist
                    branch_binaries = loaded_binaries

            # Extract operator info for trace recording
            operator_type, operator_class, operator_config = self._extract_step_info(step)

            # Get branch_path from context
            branch_path = getattr(branch_context.selector, 'branch_path', [])
            branch_name_ctx = getattr(branch_context.selector, 'branch_name', '') or ''

            # Record step start in execution trace for this branch
            if runtime_context:
                runtime_context.record_step_start(
                    step_index=self.step_number,
                    operator_type=operator_type,
                    operator_class=operator_class,
                    operator_config=operator_config,
                    branch_path=branch_path,
                    branch_name=branch_name_ctx or branch_name,
                    mode=self.mode
                )
                # Record input shapes for branch step (parity with non-branch execution)
                self._record_dataset_shapes(
                    dataset,
                    branch_context,
                    runtime_context,
                    is_input=True,
                )

            # Execute step on this branch
            try:
                step_result = self.step_runner.execute(
                    step=step,
                    dataset=dataset,
                    context=branch_context,
                    runtime_context=runtime_context,
                    loaded_binaries=branch_binaries,
                    prediction_store=prediction_store
                )

                # Record output shapes after execution for branch steps
                if runtime_context:
                    self._record_dataset_shapes(
                        dataset,
                        step_result.updated_context,
                        runtime_context,
                        is_input=False,
                    )

                # Record step end in execution trace
                if runtime_context:
                    is_model = operator_type in ("model", "meta_model")
                    runtime_context.record_step_end(is_model=is_model)

                # Process artifacts
                processed_artifacts = self._process_step_artifacts(
                    step_result.artifacts,
                    branch_id=branch_id,
                    branch_name=branch_name
                )
                if all_artifacts is not None:
                    all_artifacts.extend(processed_artifacts)

                # Append artifacts to manifest
                if (self.mode == "train" and
                    self.manifest_manager and
                    runtime_context.pipeline_uid and
                    processed_artifacts):
                    self.manifest_manager.append_artifacts(
                        runtime_context.pipeline_uid,
                        processed_artifacts
                    )

                # Update branch context
                updated_branch_contexts.append({
                    "branch_id": branch_id,
                    "name": branch_name,
                    "context": step_result.updated_context,
                    # Preserve any additional metadata
                    **{k: v for k, v in branch_info.items()
                       if k not in ("branch_id", "name", "context")}
                })

            except Exception as e:
                # Record step end even on failure
                if runtime_context:
                    runtime_context.record_step_end(skip_trace=True)
                if self.continue_on_error:
                    logger.warning(f"Branch {branch_id} step {self.step_number} failed: {str(e)}")
                    # Keep original context on failure
                    updated_branch_contexts.append(branch_info)
                else:
                    raise RuntimeError(
                        f"Pipeline step {self.step_number} failed on branch {branch_id}: {str(e)}"
                    ) from e

        # Update context with new branch contexts
        result_context = context.copy()
        result_context.custom["branch_contexts"] = updated_branch_contexts

        # Sync operation_count back from runtime_context
        if runtime_context:
            self.operation_count = runtime_context.operation_count

        return result_context

    def _execute_single_step(
        self,
        step: Any,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,
        loaded_binaries: Optional[List] = None,
        prediction_store: Optional[Predictions] = None,
        all_artifacts: Optional[List] = None
    ) -> ExecutionContext:
        """Execute a single step (non-branched).

        Args:
            step: Step to execute
            dataset: Dataset to process
            context: Current execution context
            runtime_context: Runtime infrastructure context
            loaded_binaries: Pre-loaded binaries for predict mode
            prediction_store: Prediction store
            all_artifacts: List to accumulate artifacts

        Returns:
            Updated context
        """
        from nirs4all.pipeline.execution.result import ArtifactMeta

        # Extract operator info for trace recording
        operator_type, operator_class, operator_config = self._extract_step_info(step)

        # Record step start in execution trace
        if runtime_context:
            branch_path = getattr(context.selector, 'branch_path', [])
            branch_name = getattr(context.selector, 'branch_name', '') or ''
            runtime_context.record_step_start(
                step_index=self.step_number,
                operator_type=operator_type,
                operator_class=operator_class,
                operator_config=operator_config,
                branch_path=branch_path,
                branch_name=branch_name,
                mode=self.mode
            )
            # Record input shapes before execution
            self._record_dataset_shapes(dataset, context, runtime_context, is_input=True)

        try:
            # Execute step via step runner
            step_result = self.step_runner.execute(
                step=step,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store
            )

            logger.debug(f"Step {self.step_number} completed with {len(step_result.artifacts)} artifacts")
            if self.verbose > 1:
                logger.debug(str(dataset))

            # Record output shapes after execution
            if runtime_context:
                self._record_dataset_shapes(dataset, step_result.updated_context, runtime_context, is_input=False)

            # Process artifacts (persist if needed)
            processed_artifacts = self._process_step_artifacts(step_result.artifacts)
            if all_artifacts is not None:
                all_artifacts.extend(processed_artifacts)

            # Process outputs (save files)
            for output in step_result.outputs:
                if isinstance(output, dict):
                    # Legacy: already saved
                    pass
                elif isinstance(output, tuple) and len(output) >= 3:
                    # New: (data, name, type)
                    data, name, type_hint = output
                    if self.saver:
                        self.saver.save_output(
                            step_number=self.step_number,
                            name=name,
                            data=data,
                            extension=f".{type_hint}" if not type_hint.startswith('.') else type_hint
                        )

            # Update context
            context = step_result.updated_context

            # Sync operation_count back from runtime_context
            if runtime_context:
                self.operation_count = runtime_context.operation_count

            # Append artifacts to manifest if in train mode
            if (self.mode == "train" and
                self.manifest_manager and
                runtime_context.pipeline_uid and
                processed_artifacts):
                self.manifest_manager.append_artifacts(
                    runtime_context.pipeline_uid,
                    processed_artifacts
                )
                logger.debug(f"Appended {len(processed_artifacts)} artifacts to manifest")

            # Record step end in execution trace
            if runtime_context:
                # Determine if this is a model step (check for model operator type)
                is_model = operator_type in ("model", "meta_model")
                runtime_context.record_step_end(is_model=is_model)

        except Exception as e:
            # Record step end even on failure
            if runtime_context:
                runtime_context.record_step_end(skip_trace=True)
            if self.continue_on_error:
                logger.warning(f"Step {self.step_number} failed but continuing: {str(e)}")
            else:
                raise RuntimeError(f"Pipeline step {self.step_number} failed: {str(e)}") from e

        return context

    def _process_step_artifacts(
        self,
        artifacts: List[Any],
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None
    ) -> List[Any]:
        """Process and persist step artifacts.

        Args:
            artifacts: Raw artifacts from step execution
            branch_id: Optional branch ID for artifact naming
            branch_name: Optional branch name for metadata

        Returns:
            List of processed artifact metadata
        """
        from nirs4all.pipeline.execution.result import ArtifactMeta
        from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord

        processed_artifacts = []
        for artifact in artifacts:
            if isinstance(artifact, ArtifactRecord):
                # v2 system: ArtifactRecord from registry.register()
                # Convert to dict for manifest storage
                processed_artifacts.append(artifact.to_dict())
            elif isinstance(artifact, (ArtifactMeta, dict)):
                # Legacy: already persisted
                meta = artifact
                # Add branch metadata if applicable
                if branch_id is not None and isinstance(meta, dict):
                    meta = dict(meta)  # Copy to avoid mutation
                    meta["branch_id"] = branch_id
                    meta["branch_name"] = branch_name
                processed_artifacts.append(meta)
            elif isinstance(artifact, tuple) and len(artifact) >= 2:
                # New: (obj, name, format_hint)
                obj, name = artifact[0], artifact[1]
                format_hint = artifact[2] if len(artifact) > 2 else None

                # Add branch prefix to name if branching
                if branch_id is not None:
                    name = f"{name}_b{branch_id}"

                if self.saver:
                    meta = self.saver.persist_artifact(
                        step_number=self.step_number,
                        name=name,
                        obj=obj,
                        format_hint=format_hint,
                        branch_id=branch_id,
                        branch_name=branch_name
                    )
                    processed_artifacts.append(meta)

        return processed_artifacts

    def _filter_binaries_for_branch(
        self,
        loaded_binaries: List,
        branch_id: int
    ) -> List:
        """Filter loaded binaries for a specific branch.

        Filters artifacts by branch_id metadata. Artifacts without branch_id
        (pre-branch/shared) are included for all branches.

        Args:
            loaded_binaries: All loaded binaries for this step as (name, obj) tuples
            branch_id: Target branch ID

        Returns:
            Filtered list of binaries for this branch (including shared artifacts)
        """
        if not loaded_binaries:
            return loaded_binaries

        # Note: loaded_binaries are (name, obj) tuples from ArtifactLoader
        # The ArtifactLoader now handles branch filtering internally via get_step_binaries(step, branch_id)
        # This method is kept for backward compatibility but primarily relies on ArtifactLoader

        return loaded_binaries

    def _extract_step_info(self, step: Any) -> tuple:
        """Extract operator information from a step for trace recording.

        Args:
            step: Pipeline step configuration

        Returns:
            Tuple of (operator_type, operator_class, operator_config)
        """
        operator_type = ""
        operator_class = ""
        operator_config = {}

        if isinstance(step, dict):
            # Common step keys for operator type detection
            type_keywords = {
                "model": "model",
                "meta_model": "meta_model",
                "transform": "transform",
                "y_processing": "y_processing",
                "feature_augmentation": "feature_augmentation",
                "concat_transform": "concat_transform",
                "sample_partitioner": "sample_partitioner",
                "sample_augmentation": "sample_augmentation",
                "resampler": "resampler",
                "feature_selection": "feature_selection",
                "outlier_excluder": "outlier_excluder",
                "sample_filter": "sample_filter",
                "splitter": "splitter",
                "branch": "branch",
                "merge": "merge",
                "source_branch": "source_branch",
                "merge_sources": "merge_sources",
                "preprocessing": "preprocessing",
            }

            for key, op_type in type_keywords.items():
                if key in step:
                    operator_type = op_type
                    operator_value = step[key]

                    # Extract meaningful class name(s) from the operator value
                    operator_class = self._extract_operator_class_name(operator_value, key)

                    # Store sanitized config (avoid storing large objects)
                    try:
                        import json
                        json.dumps(step, default=str)  # Test if serializable
                        operator_config = step
                    except (TypeError, ValueError):
                        operator_config = {"_type": str(type(step))}
                    break
            else:
                # Check for serialized class format: {'class': 'module.ClassName', 'params': {...}}
                if 'class' in step:
                    class_path = step['class']
                    if isinstance(class_path, str) and '.' in class_path:
                        operator_class = class_path.split('.')[-1]
                        # Infer type from module path
                        class_path_lower = class_path.lower()
                        if 'model_selection' in class_path_lower or 'fold' in operator_class.lower() or 'split' in operator_class.lower():
                            operator_type = "splitter"
                        elif 'cross_decomposition' in class_path_lower or 'linear_model' in class_path_lower or 'ensemble' in class_path_lower:
                            operator_type = "model"
                        elif 'preprocessing' in class_path_lower or 'scaler' in operator_class.lower():
                            operator_type = "transform"
                        elif 'feature_selection' in class_path_lower:
                            operator_type = "feature_selection"
                        else:
                            operator_type = "operator"
                        operator_config = step
                    else:
                        operator_type = "config"
                        operator_class = str(class_path) if class_path else "config"
                else:
                    # Dict without recognized keyword - just record type as dict
                    operator_type = "config"
                    operator_class = str(type(step).__name__)
        elif hasattr(step, '__class__') and not isinstance(step, (str, int, float, bool, type(None))):
            # Raw class instance (e.g., sklearn transformer, cross-validator)
            class_name = step.__class__.__name__
            operator_class = class_name

            # Infer operator type from module or class patterns
            module = step.__class__.__module__
            if 'cross_decomposition' in module or 'linear_model' in module or 'ensemble' in module:
                operator_type = "model"
            elif 'model_selection' in module or 'Fold' in class_name or 'Split' in class_name:
                operator_type = "splitter"
            elif 'preprocessing' in module or 'Scaler' in class_name:
                operator_type = "transform"
            elif 'feature_selection' in module:
                operator_type = "feature_selection"
            else:
                operator_type = "operator"
        elif isinstance(step, str):
            # String step - could be a serialized class path like 'sklearn.preprocessing._data.MinMaxScaler'
            # or a command like 'chart_2d'
            if '.' in step:
                # Likely a fully qualified class path
                operator_class = step.split('.')[-1]  # Extract just the class name

                # Infer operator type from the module path
                step_lower = step.lower()
                if 'cross_decomposition' in step_lower or 'linear_model' in step_lower or 'ensemble' in step_lower:
                    operator_type = "model"
                elif 'model_selection' in step_lower or 'fold' in operator_class.lower() or 'split' in operator_class.lower():
                    operator_type = "splitter"
                elif 'preprocessing' in step_lower or 'scaler' in operator_class.lower():
                    operator_type = "transform"
                elif 'feature_selection' in step_lower:
                    operator_type = "feature_selection"
                elif 'nirs4all.operators.transforms' in step:
                    operator_type = "transform"
                else:
                    operator_type = "operator"
            else:
                # Simple command name (e.g., 'chart_2d')
                operator_type = "command"
                operator_class = step

        return operator_type, operator_class, operator_config

    def _extract_operator_class_name(self, value: Any, keyword: str = "") -> str:
        """Extract a meaningful class name from an operator value.

        Handles various cases:
        - Direct class instances (e.g., MinMaxScaler())
        - Class references (e.g., MinMaxScaler)
        - Lists of operators (e.g., [SNV(), FirstDerivative()])
        - Dicts with operator configuration
        - Strings (e.g., class paths)

        Args:
            value: The operator value to extract name from
            keyword: The keyword context (e.g., 'model', 'feature_augmentation')

        Returns:
            Human-readable operator class name
        """
        # Handle None
        if value is None:
            return "None"

        # Handle string values
        if isinstance(value, str):
            # For strings like 'sklearn.preprocessing.MinMaxScaler', extract last part
            return value.split('.')[-1] if '.' in value else value

        # Handle class references (not instances)
        if isinstance(value, type):
            return value.__name__

        # Handle lists of operators
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]"
            # Extract names from first few items
            names = []
            for item in value[:3]:  # Limit to first 3 for readability
                name = self._extract_operator_class_name(item, keyword)
                if name and name not in ('dict', 'list', 'tuple'):
                    names.append(name)
            if names:
                result = ", ".join(names)
                if len(value) > 3:
                    result += f" (+{len(value) - 3})"
                return result
            return f"[{len(value)} items]"

        # Handle dicts with class specification
        if isinstance(value, dict):
            # Check for common class keys
            for class_key in ('class', 'function', 'type', 'model', 'operator'):
                if class_key in value:
                    class_val = value[class_key]
                    if isinstance(class_val, str):
                        return class_val.split('.')[-1] if '.' in class_val else class_val
                    elif hasattr(class_val, '__name__'):
                        return class_val.__name__
                    elif hasattr(class_val, '__class__'):
                        return class_val.__class__.__name__

            # For sample_augmentation dicts, show transformer count
            if 'transformers' in value:
                transformers = value['transformers']
                count = value.get('count', 1)
                if isinstance(transformers, (list, tuple)):
                    names = [self._extract_operator_class_name(t) for t in transformers[:2]]
                    names = [n for n in names if n not in ('dict', 'list')]
                    if names:
                        return f"{', '.join(names)} ×{count}"
                return f"[{len(transformers)} aug] ×{count}"

            # For concat_transform with operations
            if 'operations' in value:
                ops = value['operations']
                return self._extract_operator_class_name(ops, keyword)

            return "config"

        # Handle class instances with __class__ attribute
        if hasattr(value, '__class__'):
            class_name = value.__class__.__name__
            # Skip generic Python types
            if class_name not in ('dict', 'list', 'tuple', 'set', 'str', 'int', 'float', 'bool', 'NoneType'):
                return class_name

        return str(type(value).__name__)

    def _record_dataset_shapes(
        self,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,
        is_input: bool = True
    ) -> None:
        """Record dataset shapes to the execution trace.

        Captures both 2D layout shape and 3D per-source feature shapes.

        Args:
            dataset: The dataset to measure
            context: Execution context with selector
            runtime_context: Runtime context with trace recorder
            is_input: True to record input shapes, False for output shapes
        """
        try:
            # Get 2D layout shape (samples × features)
            X_2d = dataset.x(context.selector, layout="2d", include_excluded=False)
            if isinstance(X_2d, list):
                # Multi-source with concat
                layout_shape = (X_2d[0].shape[0], sum(x.shape[1] for x in X_2d))
            else:
                layout_shape = X_2d.shape

            # Get 3D per-source shapes (samples × processings × features)
            X_3d = dataset.x(context.selector, layout="3d", concat_source=False, include_excluded=False)
            if not isinstance(X_3d, list):
                X_3d = [X_3d]

            features_shapes = [x.shape for x in X_3d]

            # Record to trace
            if is_input:
                runtime_context.record_input_shapes(
                    input_shape=layout_shape,
                    features_shape=features_shapes
                )
            else:
                runtime_context.record_output_shapes(
                    output_shape=layout_shape,
                    features_shape=features_shapes
                )

        except Exception:
            # Shape recording is non-critical, don't fail the step
            pass

    def _compute_pipeline_hash(self, steps: List[Any]) -> str:
        """Compute MD5 hash of pipeline configuration.

        Args:
            steps: Pipeline steps

        Returns:
            6-character hash string
        """
        pipeline_json = json.dumps(steps, sort_keys=True, default=str).encode('utf-8')
        return hashlib.md5(pipeline_json).hexdigest()[:6]

    def next_op(self) -> int:
        """Get the next operation ID (for compatibility)."""
        self.operation_count += 1
        return self.operation_count

    def execute_minimal(
        self,
        steps: List[Any],
        minimal_pipeline: Any,  # MinimalPipeline
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,  # RuntimeContext
        prediction_store: Optional[Predictions] = None
    ) -> None:
        """Execute minimal pipeline for prediction.

        This method executes only the steps from a MinimalPipeline, which
        represents the subset of the full pipeline needed to replay a prediction.
        It's the key optimization of Phase 5: instead of replaying the entire
        original pipeline, we only run the required steps.

        The method:
        1. Uses the minimal pipeline's step list (not full pipeline)
        2. Injects artifacts via the artifact_provider in runtime_context
        3. Runs controllers in predict mode
        4. Skips steps not in the minimal pipeline

        Args:
            steps: List of step configs (from minimal_pipeline.steps[i].step_config)
            minimal_pipeline: MinimalPipeline with artifact mappings
            dataset: Dataset to process
            context: Execution context
            runtime_context: Runtime context with artifact_provider
            prediction_store: Optional prediction store

        Note:
            The artifact_provider in runtime_context should be a MinimalArtifactProvider
            that provides artifacts by step index from the MinimalPipeline.
        """
        from nirs4all.pipeline.execution.result import ArtifactMeta

        logger.info(f"Executing minimal pipeline: {len(steps)} steps")

        # Reset state
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0

        if prediction_store is None:
            prediction_store = Predictions()

        # Get step indices from minimal pipeline for validation
        minimal_step_indices = set()
        if hasattr(minimal_pipeline, 'get_step_indices'):
            minimal_step_indices = set(minimal_pipeline.get_step_indices())

        # Get target branch_path from the model step
        # All steps need to use this branch for filtering artifacts
        target_branch_path = None
        if minimal_pipeline and hasattr(minimal_pipeline, 'model_step_index'):
            model_idx = minimal_pipeline.model_step_index
            if model_idx and hasattr(minimal_pipeline, 'get_step'):
                model_step = minimal_pipeline.get_step(model_idx)
                if model_step and model_step.branch_path:
                    target_branch_path = model_step.branch_path

        # Track previous step_index to avoid resetting counters for substeps with same step_index
        prev_step_idx = None

        # Execute each step using original step indices from MinimalPipeline
        # This ensures step_number matches training-time indices for artifact lookups
        for list_idx, step in enumerate(steps):
            # Get original step index from minimal pipeline
            if hasattr(minimal_pipeline, 'steps') and list_idx < len(minimal_pipeline.steps):
                step_idx = minimal_pipeline.steps[list_idx].step_index
            else:
                step_idx = list_idx + 1  # Fallback to 1-based enumeration

            if step is None:
                logger.debug(f"Step {step_idx}: skipped (no config)")
                continue

            self.step_number = step_idx
            self.substep_number = 0
            self.operation_count = 0

            # Sync to runtime_context
            if runtime_context:
                runtime_context.step_number = self.step_number
                runtime_context.substep_number = self.substep_number
                runtime_context.operation_count = self.operation_count
                # Only reset counters when step_index changes (not for substeps with same step_index)
                if step_idx != prev_step_idx:
                    runtime_context.reset_processing_counter()

            prev_step_idx = step_idx

            # Update context with step number
            context = context.with_step_number(self.step_number)

            # For branch steps, don't pre-filter artifacts by branch_path
            # The branch controller needs all artifacts, and internal transformers
            # will filter by their branch context when looking up artifacts
            is_branch_step = isinstance(step, dict) and "branch" in step
            is_merge_step = isinstance(step, dict) and "merge" in step
            branch_path = None if is_branch_step else target_branch_path

            # Get substep_index from minimal pipeline step for artifact filtering
            substep_index = None
            if hasattr(minimal_pipeline, 'steps') and list_idx < len(minimal_pipeline.steps):
                substep_index = minimal_pipeline.steps[list_idx].substep_index

            # Get binaries from artifact_provider instead of artifact_loader
            loaded_binaries = None
            if runtime_context and runtime_context.artifact_provider:
                # Use artifact_provider for minimal pipeline prediction
                # Pass branch_path to filter artifacts for multi-branch pipelines
                # (except for branch steps which need all artifacts)
                # Pass substep_index to filter artifacts for branch substeps
                artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                    step_idx, branch_path=branch_path, substep_index=substep_index
                )
                if artifacts:
                    loaded_binaries = artifacts  # Already in (name, obj) format
                    logger.debug(f"Loaded {len(artifacts)} artifact(s) for step {step_idx} (substep={substep_index})")
            elif self.mode in ("predict", "explain") and self.artifact_loader:
                # Fallback to artifact_loader
                loaded_binaries = self.artifact_loader.get_step_binaries(self.step_number)

            # Check for branch contexts (already computed is_branch_step above)
            branch_contexts = context.custom.get("branch_contexts", [])

            # Execute on branches for post-branch steps, but not for:
            # - branch steps (they create branches)
            # - merge steps (they consume branches and exit branch mode)
            if branch_contexts and not is_branch_step and not is_merge_step:
                # Execute on each branch
                context = self._execute_step_on_branches(
                    step=step,
                    dataset=dataset,
                    context=context,
                    runtime_context=runtime_context,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store,
                    all_artifacts=[]
                )
            else:
                # Single context execution
                context = self._execute_single_step(
                    step=step,
                    dataset=dataset,
                    context=context,
                    runtime_context=runtime_context,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store,
                    all_artifacts=[]
                )

        logger.success(f"Minimal pipeline completed: {prediction_store.num_predictions} predictions")

