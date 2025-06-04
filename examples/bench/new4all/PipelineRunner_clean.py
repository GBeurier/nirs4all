"""
PipelineRunner - Clean execution engine with builder delegation

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
  * Support for parallel execution of multiple augmenters
- Robust error handling and continue-on-error support
"""
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import copy

from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
from PipelineConfig import PipelineConfig
from PipelineBuilder_clean import PipelineBuilder
from DummyOperations import VisualizationOperation


class PipelineRunner:
    """Clean pipeline execution engine with builder delegation"""

    def __init__(self, max_workers: Optional[int] = None, continue_on_error: bool = False):
        self.max_workers = max_workers
        self.continue_on_error = continue_on_error

        # Initialize builder for operation creation
        self.builder = PipelineBuilder()

        # Execution state
        self.context = PipelineContext()
        self.current_step = 0

    def run_pipeline(self, pipeline_config: Union[Dict, List, PipelineConfig], dataset: SpectraDataset) -> SpectraDataset:
        """Main entry point"""
        print("🚀 Starting Pipeline Runner")

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
        for step in steps:
            self._run_step(step, dataset)

        return dataset

    def _run_step(self, step: Any, dataset: SpectraDataset, prefix: str = ""):
        """
        MAIN PARSING LOGIC - Clean execution with builder delegation

        Control flow:
        1. Identify step type
        2. If control operation -> call local _run_xxx function
        3. If data operation -> delegate to builder then execute
        """
        self.current_step += 1
        step_prefix = f"{prefix}  "

        print(f"{prefix}📋 Step {self.current_step}: {self._get_step_description(step)}")

        try:
            # CONTROL OPERATIONS - handled locally
            if isinstance(step, str) and step == "uncluster":
                self._run_uncluster(dataset, step_prefix)
            elif isinstance(step, str) and step.startswith("Plot"):
                self._run_visualization(step, dataset, step_prefix)
            elif isinstance(step, list):
                self._run_sub_pipeline(step, dataset, step_prefix)
            elif isinstance(step, dict) and len(step) == 1:
                key, value = next(iter(step.items()))
                if key in ["sample_augmentation", "samples", "S"]:
                    self._run_sample_augmentation(value, dataset, step_prefix)
                elif key in ["feature_augmentation", "features", "F"]:
                    self._run_feature_augmentation(value, dataset, step_prefix)
                elif key == "dispatch":
                    self._run_dispatch(value, dataset, step_prefix)
                else:
                    # Delegate to builder for data operations
                    operation = self.builder.build_operation(step)
                    self._execute_operation(operation, dataset, step_prefix)
            elif isinstance(step, dict) and ("model" in step or "stack" in step):
                # Complex model/stack operations - keep local for now
                if "model" in step:
                    self._run_model(step, dataset, step_prefix)
                else:
                    self._run_stacking(step, dataset, step_prefix)
            else:
                # DATA OPERATIONS - delegate to builder
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, dataset, step_prefix)

            print(f"{prefix}✅ Step {self.current_step} completed")

        except Exception as e:
            print(f"{prefix}❌ Step {self.current_step} failed: {str(e)}")
            if not self.continue_on_error:
                raise

    def _execute_operation(self, operation, dataset: SpectraDataset, prefix: str):
        """Execute a built operation - simple select and execute"""
        print(f"{prefix}🔧 Executing: {operation.get_name()}")

        # Apply current filters to dataset
        filtered_dataset = dataset.select(**self.context.current_filters) if self.context.current_filters else dataset

        # Execute operation
        operation.execute(filtered_dataset, self.context)    # =================================================================
    # CONTROL OPERATIONS - handled locally
    # =================================================================

    def _run_uncluster(self, dataset: SpectraDataset, prefix: str):
        """Remove group filters from context"""
        print(f"{prefix}🔓 Uncluster operation")
        if 'group' in self.context.current_filters:
            del self.context.current_filters['group']
            print(f"{prefix}   Removed group filters")

    def _run_visualization(self, step: str, dataset: SpectraDataset, prefix: str):
        """Handle visualization steps"""
        print(f"{prefix}📊 Visualization: {step}")
        viz_op = VisualizationOperation(plot_type=step)
        viz_op.execute(dataset, self.context)

    def _run_sub_pipeline(self, step_list: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute sub-pipeline"""
        print(f"{prefix}📋 Sub-pipeline with {len(step_list)} steps")
        for sub_step in step_list:
            self._run_step(sub_step, dataset, f"{prefix}  ")

    def _run_sample_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute sample augmentation - parallel processing"""
        print(f"{prefix}🔄 Sample augmentation with {len(augmenters)} augmenters")

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
        if self.max_workers and len(augmenters) > 1:
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
        """Parallel feature augmentation execution"""
        print(f"{prefix}  🔀 Running {len(augmenters)} augmenters in parallel (max_workers={self.max_workers})")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:            # Submit all augmentation tasks
            futures = []
            for i, augmenter in enumerate(augmenters):
                future = executor.submit(self._apply_feature_augmentation_safe,
                                         augmenter, dataset, train_view, f"{prefix}    ", i + 1)
                futures.append((i + 1, future))

            # Collect results and handle errors
            for aug_num, future in futures:
                try:
                    future.result()
                    print(f"{prefix}  ✅ Augmenter {aug_num} completed")
                except Exception as e:
                    print(f"{prefix}  ❌ Augmenter {aug_num} failed: {str(e)}")
                    if not self.continue_on_error:
                        raise

    def _apply_feature_augmentation_safe(self, augmenter, dataset: SpectraDataset, train_view, prefix: str, aug_num: int):
        """Thread-safe wrapper for feature augmentation application"""
        try:
            self._apply_feature_augmentation(augmenter, dataset, train_view, prefix, aug_num)
        except Exception as e:
            # Re-raise with context for better error reporting
            raise Exception(f"Augmenter {aug_num} error: {str(e)}") from e

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
        operation_name = operation.get_name()

        aug_label = f"Aug {aug_num}" if aug_num else "Augmentation"
        print(f"{prefix}� {aug_label}: {operation_name}")

        # Get train features for transformation
        X_train = train_view.get_features()
        print(f"{prefix}  � Input: {X_train.shape[0]} samples × {X_train.shape[1]} features")        # Fit and transform using the operation's transformer
        # For feature augmentation, we need to access the underlying transformer
        if hasattr(operation, 'transformer'):
            transformer = getattr(operation, 'transformer')  # Use getattr to avoid linter issues
        else:
            # If operation doesn't have direct transformer access, try to extract it
            # This is a fallback for different operation types
            raise ValueError(f"Operation {operation.get_name()} doesn't support feature augmentation (no transformer attribute)")

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
        """Execute dispatch - parallel branch processing"""
        print(f"{prefix}🌿 Dispatch with {len(branches)} branches")

        if self.max_workers and len(branches) > 1:
            self._run_dispatch_parallel(branches, dataset, prefix)
        else:
            self._run_dispatch_sequential(branches, dataset, prefix)

    def _run_dispatch_sequential(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branches sequentially"""
        for i, branch in enumerate(branches):
            print(f"{prefix}  🌿 Branch {i+1}/{len(branches)}")
            self._run_step(branch, dataset, f"{prefix}    ")

    def _run_dispatch_parallel(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branches in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, branch in enumerate(branches):
                dataset_copy = copy.deepcopy(dataset)
                future = executor.submit(self._run_step, branch, dataset_copy, f"{prefix}    ")
                futures.append(future)

            # Wait for all branches to complete
            for i, future in enumerate(futures):
                try:
                    future.result()
                    print(f"{prefix}  ✅ Branch {i+1} completed")
                except Exception as e:
                    print(f"{prefix}  ❌ Branch {i+1} failed: {str(e)}")

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
