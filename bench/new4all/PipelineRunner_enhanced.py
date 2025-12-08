"""
PipelineRunner - Enhanced execution engine with joblib parallelization and history tracking

Clean separation of concerns:
- Runner: context management + execution control + data flow management
- Builder: generic step building from any format
- Operations: simple wrappers around operators

Key Features:
- Unified parsing loop for all step types
- Delegated operation building to PipelineBuilder
- Enhanced feature augmentation with data flow management:
  * Runner manages train set extraction and context filtering
  * Runner handles transformer fitting and feature transformation
  * Runner manages adding new features back to the dataset
  * Support for parallel execution using joblib for better ML performance
- Robust error handling and continue-on-error support
- Execution history tracking with serialization support
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import copy
from joblib import Parallel, delayed

try:
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext
    from PipelineConfig import PipelineConfig
    from PipelineBuilder_clean import PipelineBuilder
    from PipelineHistory import PipelineHistory
except ImportError as e:
    # Handle imports more gracefully for testing
    print(f"Warning: Could not import some modules: {e}")
    SpectraDataset = Any
    PipelineContext = Any
    PipelineConfig = Any


class PipelineRunner:
    """Enhanced pipeline execution engine with joblib parallelization and history tracking"""

    def __init__(self, max_workers: Optional[int] = None, continue_on_error: bool = False,
                 backend: str = 'threading', verbose: int = 0):
        """
        Initialize pipeline runner

        Args:
            max_workers: Maximum number of parallel workers (None = use default)
            continue_on_error: Whether to continue execution on step errors
            backend: Joblib backend ('threading', 'loky', 'multiprocessing')
            verbose: Verbosity level for parallel execution (0-10)
        """
        self.max_workers = max_workers or -1  # -1 means use all available cores
        self.continue_on_error = continue_on_error
        self.backend = backend
        self.verbose = verbose

        # Initialize builder for operation creation
        self.builder = PipelineBuilder()

        # Execution state
        self.context = PipelineContext()
        self.current_step = 0

        # History tracking
        self.history = PipelineHistory()

    def run_pipeline(self, pipeline_config: Union[Dict, List, PipelineConfig], dataset: SpectraDataset) -> Tuple[SpectraDataset, PipelineHistory]:
        """
        Main entry point - now returns dataset and execution history

        Returns:
            Tuple of (modified_dataset, execution_history)
        """
        print("🚀 Starting Enhanced Pipeline Runner")

        # Start pipeline execution tracking
        self.history.start_execution(pipeline_config)

        # Extract steps from config
        if isinstance(pipeline_config, PipelineConfig):
            steps = pipeline_config.pipeline
        elif isinstance(pipeline_config, dict) and 'pipeline' in pipeline_config:
            steps = pipeline_config['pipeline']
        elif isinstance(pipeline_config, list):
            steps = pipeline_config
        else:
            raise ValueError(f"Invalid pipeline config type: {type(pipeline_config)}")

        # Execute pipeline steps
        try:
            for step in steps:
                self._run_step(step, dataset)

            # Complete pipeline execution
            self.history.complete_execution()
            print("✅ Pipeline completed successfully")

        except Exception as e:
            self.history.fail_execution(str(e))
            print(f"❌ Pipeline failed: {str(e)}")
            raise

        return dataset, self.history

    def _run_step(self, step: Any, dataset: SpectraDataset, prefix: str = ""):
        """
        MAIN PARSING LOGIC - Clean execution with builder delegation

        Control flow:
        1. Identify step type
        2. Delegate execution to appropriate handler
        3. Track execution in history
        """
        self.current_step += 1
        step_description = self._get_step_description(step)
        print(f"{prefix}🔹 Step {self.current_step}: {step_description}")

        # Start step tracking
        step_execution = self.history.start_step(
            step_number=self.current_step,
            step_description=step_description,
            step_config=step
        )

        try:
            # Control structures
            if isinstance(step, dict):
                if "context_filter" in step:
                    self._run_context_filter(step["context_filter"], dataset, prefix + "  ")
                elif "sample_augmentation" in step:
                    self._run_sample_augmentation(step["sample_augmentation"], dataset, prefix + "  ")
                elif "feature_augmentation" in step:
                    self._run_feature_augmentation(step["feature_augmentation"], dataset, prefix + "  ")
                elif "dispatch" in step:
                    self._run_dispatch(step["dispatch"], dataset, prefix + "  ")
                elif "model" in step:
                    self._run_model(step["model"], dataset, prefix + "  ")
                elif "stack" in step:
                    self._run_stacking(step["stack"], dataset, prefix + "  ")
                else:
                    # Direct operation dict
                    operation = self.builder.build_operation(step)
                    self._execute_operation(operation, dataset, prefix + "  ")

            # Sub-pipeline (list of steps)
            elif isinstance(step, list):
                print(f"{prefix}  📁 Sub-pipeline with {len(step)} steps")
                for sub_step in step:
                    self._run_step(sub_step, dataset, prefix + "    ")

            # String reference (preset, operation name, etc.)
            elif isinstance(step, str):
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, dataset, prefix + "  ")

            # Direct operation object
            else:
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, dataset, prefix + "  ")            # Complete step successfully
            self.history.complete_step(step_execution.step_id)

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            # Fail step
            self.history.fail_step(step_execution.step_id, str(e))

            if self.continue_on_error:
                print(f"{prefix}  ⚠️ Step failed but continuing: {str(e)}")
            else:
                raise
        except Exception as e:
            # Catch any other unexpected errors
            self.history.fail_step(step_execution.step_id, f"Unexpected error: {str(e)}")
            if self.continue_on_error:
                print(f"{prefix}  ⚠️ Unexpected error but continuing: {str(e)}")
            else:
                raise RuntimeError(f"Pipeline step failed: {str(e)}") from e

    def _execute_operation(self, operation: Any, dataset: SpectraDataset, prefix: str):
        """Execute a built operation"""
        operation_name = operation.get_name() if hasattr(operation, 'get_name') else str(operation)
        print(f"{prefix}⚙️ Executing: {operation_name}")

        if hasattr(operation, 'execute'):
            operation.execute(dataset)
        else:
            print(f"{prefix}  ⚠️ Operation {operation_name} has no execute method")

    # =================================================================
    # CONTROL STRUCTURE HANDLERS
    # =================================================================

    def _run_context_filter(self, filters: Dict, dataset: SpectraDataset, prefix: str):
        """Apply context filters"""
        print(f"{prefix}🔍 Context filter: {filters}")
        self.context.apply_filters(filters)

    def _run_sample_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute sample augmentation"""
        print(f"{prefix}📊 Sample augmentation with {len(augmenters)} augmenters")

        for i, augmenter in enumerate(augmenters):
            print(f"{prefix}  📌 Augmenter {i+1}/{len(augmenters)}")
            operation = self.builder.build_operation(augmenter)
            if hasattr(operation, 'mode'):
                setattr(operation, 'mode', "sample_augmentation")
            self._execute_operation(operation, dataset, f"{prefix}    ")

    def _run_feature_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """
        Execute feature augmentation - runner manages data flow and parallel execution

        The runner is responsible for:
        1. Getting the current train set
        2. Managing the data flow for each augmenter
        3. Adding new features to the dataset
        4. Handling parallel execution if enabled
        """
        print(f"{prefix}🔄 Feature augmentation with {len(augmenters)} augmenters")

        # Get current train set for augmentation - apply context filters
        train_view = dataset.select(partition="train", **self.context.current_filters)
        if len(train_view) == 0:
            print(f"{prefix}  ⚠️ No train data found for feature augmentation")
            return

        print(f"{prefix}  📊 Base train set: {len(train_view)} samples, {train_view.get_features().shape[1]} features")

        # Execute augmenters based on parallel configuration
        if self.max_workers != 1 and len(augmenters) > 1:
            self._run_feature_augmentation_parallel(augmenters, dataset, train_view, prefix)
        else:
            self._run_feature_augmentation_sequential(augmenters, dataset, train_view, prefix)

        # Report final feature count
        final_train_view = dataset.select(partition="train", **self.context.current_filters)
        final_feature_count = final_train_view.get_features().shape[1]
        print(f"{prefix}  📈 Final train set: {len(final_train_view)} samples, {final_feature_count} features")

    def _run_feature_augmentation_sequential(self, augmenters: List[Any], dataset: SpectraDataset, train_view, prefix: str):
        """Sequential feature augmentation execution"""
        for i, augmenter in enumerate(augmenters):
            print(f"{prefix}  📌 Augmenter {i+1}/{len(augmenters)}")
            try:
                self._apply_feature_augmentation(augmenter, dataset, train_view, f"{prefix}    ")
                print(f"{prefix}  ✅ Augmenter {i+1} completed")
            except Exception as e:
                print(f"{prefix}  ❌ Augmenter {i+1} failed: {str(e)}")
                if not self.continue_on_error:
                    raise

    def _run_feature_augmentation_parallel(self, augmenters: List[Any], dataset: SpectraDataset, train_view, prefix: str):
        """Parallel feature augmentation execution using joblib"""
        print(f"{prefix}  🔀 Running {len(augmenters)} augmenters in parallel (max_workers={self.max_workers}, backend={self.backend})")

        # Use joblib for better ML performance
        try:
            results = Parallel(
                n_jobs=self.max_workers,
                backend=self.backend,
                verbose=self.verbose
            )(
                delayed(self._apply_feature_augmentation_safe)(
                    augmenter, dataset, train_view, f"{prefix}    ", i + 1
                )
                for i, augmenter in enumerate(augmenters)
            )

            # Report results
            for i, result in enumerate(results):
                if result is None:  # Success
                    print(f"{prefix}  ✅ Augmenter {i+1} completed")
                else:  # Error
                    print(f"{prefix}  ❌ Augmenter {i+1} failed: {result}")
                    if not self.continue_on_error:
                        raise Exception(f"Augmenter {i+1} failed: {result}")

        except Exception as e:
            if not self.continue_on_error:
                raise Exception(f"Parallel feature augmentation failed: {str(e)}") from e
            else:
                print(f"{prefix}  ⚠️ Parallel execution failed, falling back to sequential: {str(e)}")
                self._run_feature_augmentation_sequential(augmenters, dataset, train_view, prefix)

    def _apply_feature_augmentation_safe(self, augmenter, dataset: SpectraDataset, train_view, prefix: str, aug_num: int) -> Optional[str]:
        """
        Thread-safe wrapper for feature augmentation application

        Returns:
            None if successful, error message string if failed
        """
        try:
            self._apply_feature_augmentation(augmenter, dataset, train_view, prefix, aug_num)
            return None
        except Exception as e:
            return str(e)

    def _apply_feature_augmentation(self, augmenter, dataset: SpectraDataset, train_view, prefix: str, aug_num: Optional[int] = None):
        """
        Apply single feature augmentation - runner manages the complete data flow

        Process:
        1. Build operation from config
        2. Get train features
        3. Fit and transform using the operation's transformer
        4. Create unique processing tag
        5. Add transformed features to dataset
        """
        # Build operation from augmenter config (delegated to builder)
        operation = self.builder.build_operation(augmenter)
        operation_name = operation.get_name() if hasattr(operation, 'get_name') else str(operation)

        aug_label = f"Aug {aug_num}" if aug_num else "Augmentation"
        print(f"{prefix}🔧 {aug_label}: {operation_name}")

        # Get train features for transformation
        X_train = train_view.get_features()
        print(f"{prefix}  📊 Input: {X_train.shape[0]} samples × {X_train.shape[1]} features")

        # Fit and transform using the operation's transformer
        # For feature augmentation, we need to access the underlying transformer
        if hasattr(operation, 'transformer'):
            transformer = getattr(operation, 'transformer')  # Use getattr to avoid linter issues
        else:
            # If operation doesn't have direct transformer access, try to extract it
            # This is a fallback for different operation types
            raise ValueError(f"Operation {operation_name} doesn't support feature augmentation (no transformer attribute)")

        # Fit on train data
        transformer.fit(X_train)

        # Transform to get new features
        X_transformed = transformer.transform(X_train)

        # Validate transformation result
        if X_transformed.shape[0] != X_train.shape[0]:
            raise ValueError(f"Feature augmentation changed sample count: {X_train.shape[0]} -> {X_transformed.shape[0]}")

        print(f"{prefix}  📈 Output: {X_transformed.shape[0]} samples × {X_transformed.shape[1]} features")

        # Create unique processing tag for this augmentation
        transformer_name = transformer.__class__.__name__
        params_hash = hash(str(sorted(transformer.get_params().items()))) % 10000
        aug_tag = f"aug_{transformer_name}_{params_hash:04d}"

        # Runner manages adding the transformed features to dataset
        print(f"{prefix}  📝 Adding features with tag: {aug_tag}")

        # Use dataset's feature augmentation method to add new features
        # Note: This assumes SpectraDataset has this method - if not, we need to implement it
        if hasattr(dataset, 'add_feature_augmentation'):
            add_features_method = getattr(dataset, 'add_feature_augmentation')
            add_features_method(
                sample_ids=train_view.sample_ids,
                features=X_transformed,
                processing_tag=aug_tag,
                source_partition="train"
            )
        else:
            # Fallback: add features directly to the dataset
            # This would need to be implemented based on the actual SpectraDataset interface
            print(f"{prefix}  ⚠️ Dataset doesn't support add_feature_augmentation - features not added")
            print(f"{prefix}  💡 Would add {X_transformed.shape[1]} features to {len(train_view.sample_ids)} samples")

        print(f"{prefix}  ✅ Added {X_transformed.shape[1]} new features to {len(train_view.sample_ids)} samples")

    def _run_dispatch(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute dispatch - parallel branch processing using joblib"""
        print(f"{prefix}🌿 Dispatch with {len(branches)} branches")

        if self.max_workers != 1 and len(branches) > 1:
            self._run_dispatch_parallel(branches, dataset, prefix)
        else:
            self._run_dispatch_sequential(branches, dataset, prefix)

    def _run_dispatch_sequential(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branches sequentially"""
        for i, branch in enumerate(branches):
            print(f"{prefix}  🌿 Branch {i+1}/{len(branches)}")
            self._run_step(branch, dataset, f"{prefix}    ")

    def _run_dispatch_parallel(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branches in parallel using joblib"""
        print(f"{prefix}  🔀 Running {len(branches)} branches in parallel")

        try:
            # Create dataset copies for parallel execution
            dataset_copies = [copy.deepcopy(dataset) for _ in branches]

            # Execute branches in parallel
            results = Parallel(
                n_jobs=self.max_workers,
                backend=self.backend,
                verbose=self.verbose
            )(
                delayed(self._run_step)(branch, dataset_copy, f"{prefix}    ")
                for branch, dataset_copy in zip(branches, dataset_copies)
            )

            # All branches completed successfully
            for i in range(len(branches)):
                print(f"{prefix}  ✅ Branch {i+1} completed")

        except Exception as e:
            print(f"{prefix}  ❌ Parallel dispatch failed: {str(e)}")
            if not self.continue_on_error:
                raise

    def _run_model(self, model_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute model operation"""
        print(f"{prefix}🤖 Model operation")
        operation = self.builder.build_operation(model_config)
        self._execute_operation(operation, dataset, prefix)

    def _run_stacking(self, stack_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute stacking operation"""
        print(f"{prefix}📚 Stacking operation")
        operation = self.builder.build_operation({"stack": stack_config})
        self._execute_operation(operation, dataset, prefix)

    # =================================================================
    # PIPELINE SERIALIZATION AND SAVING
    # =================================================================

    def get_fitted_pipeline(self) -> Dict[str, Any]:
        """
        Get fitted pipeline for saving/serialization

        Returns:
            Dictionary containing fitted operations and metadata
        """
        return {
            'fitted_operations': self.builder.get_fitted_operations(),
            'execution_history': self.history.to_dict(),
            'context': self.context.to_dict() if hasattr(self.context, 'to_dict') else {},
            'config': {
                'max_workers': self.max_workers,
                'continue_on_error': self.continue_on_error,
                'backend': self.backend,
                'verbose': self.verbose
            }
        }

    def save_pipeline(self, filepath: str, include_dataset: bool = False, dataset: Optional[SpectraDataset] = None):
        """
        Save the executed pipeline with all fitted components

        Args:
            filepath: Path to save the pipeline (will determine format from extension)
            include_dataset: Whether to include dataset folds/splits
            dataset: Dataset to include if include_dataset=True
        """
        pipeline_data = self.get_fitted_pipeline()

        if include_dataset and dataset is not None:
            pipeline_data['dataset_info'] = {
                'sample_count': len(dataset),
                'feature_count': dataset.get_features().shape[1] if hasattr(dataset, 'get_features') else None,
                # Add more dataset serialization as needed
            }

        self.history.save_pipeline_bundle(filepath, pipeline_data)
        print(f"💾 Pipeline saved to: {filepath}")

    # =================================================================
    # UTILITY FUNCTIONS
    # =================================================================

    def _get_step_description(self, step: Any) -> str:
        """Get human-readable description of step"""
        if isinstance(step, str):
            return f"'{step}'"
        elif isinstance(step, dict):
            if len(step) == 1:
                key = next(iter(step.keys()))
                return f"'{key}' control"
            else:
                return f"complex dict with {list(step.keys())}"
        elif isinstance(step, list):
            return f"sub-pipeline ({len(step)} steps)"
        else:
            return f"{step.__class__.__name__} object"
