"""
PipelineRunner - Simplified execution engine with branch-based context

Features:
- Simple branch-based context
- Direct operation execution
- Basic branching support
- Simplified data selection
"""
from typing import Any, Dict, List, Optional, Tuple

from nirs4all.core.SpectraDataset import SpectraDataset
from nirs4all.core.PipelineBuilder import PipelineBuilder
from nirs4all.core.PipelineHistory import PipelineHistory
from nirs4all.core.PipelineConfig import PipelineConfig
from nirs4all.core.PipelineOperation import PipelineOperation
from nirs4all.core.Pipeline import Pipeline


class PipelineRunner:
    """PipelineRunner - Executes a pipeline with enhanced context management and DatasetView support."""
    def __init__(self, max_workers: Optional[int] = None, continue_on_error: bool = False, backend: str = 'threading', verbose: int = 0):
        self.max_workers = max_workers or -1  # -1 means use all available cores
        self.continue_on_error = continue_on_error
        self.backend = backend
        self.verbose = verbose

        self.builder = PipelineBuilder()
        self.history = PipelineHistory()
        self.current_step = 0
        self.context = {"branch": 0}

        # Serialization support
        self.pipeline = Pipeline()

    def run(self, config: PipelineConfig, dataset: SpectraDataset) -> Tuple[SpectraDataset, Any, PipelineHistory, Any]:
        """Run the pipeline with the given configuration and dataset."""

        print("🚀 Starting Pipeline Runner")

        # Reset pipeline state
        if self.current_step > 0:
            print("  ⚠️ Warning: Previous run detected, resetting step count and context")
            self.current_step = 0
            self.context = {"branch": 0}

        # Start pipeline execution tracking
        # self.history.start_execution(config.steps)

        try:
            for step in config.steps:
                self._run_step(step, dataset)

            # Complete pipeline execution
            # self.history.complete_execution()
            print("✅ Pipeline completed successfully")

        except Exception as e:
            # self.history.fail_execution(str(e))
            # print(f"❌ Pipeline failed: {str(e)}")
            raise

        return dataset, self.history, self.pipeline

    command_operators = ["sample_augmentation", "feature_augmentation", "branch", "dispatch", "model", "stack", "scope", "cluster", "merge"]

    def _run_step(self, step: Any, dataset: SpectraDataset, prefix: str = ""):
        """
        Run a single pipeline step with enhanced context management and DatasetView support.
        """
        self.current_step += 1
        step_description = self._get_step_description(step)
        print(f"{prefix}🔹 Step {self.current_step}: {step_description}")
        print(f"{prefix}🔹 Current context: {self.context}")
        print(f"{prefix}🔹 Step config: {step}")

        # Start step tracking
        # step_execution = self.history.start_step(
        #     step_number=self.current_step,
        #     step_description=step_description,
        #     step_config=step
        # )

        try:
            # Control structures
            if isinstance(step, dict):

                if "sample_augmentation" in step:
                    self._run_sample_augmentation(step["sample_augmentation"], dataset, prefix + "  ")

                elif "feature_augmentation" in step:
                    self._run_feature_augmentation(step["feature_augmentation"], dataset, prefix + "  ")

                elif "branch" in step:
                    self._run_branch(step["branch"], dataset, prefix + "  ")

                elif "dispatch" in step:
                    self._run_dispatch(step["dispatch"], dataset, prefix + "  ")

                elif "model" in step:
                    self._run_model(step, dataset, prefix + "  ")

                elif "stack" in step:
                    self._run_stack(step, dataset, prefix + "  ")

                elif "scope" in step:
                    self._run_scope(step["scope"], dataset, prefix + "  ")

                elif "cluster" in step:
                    self._run_cluster(step["cluster"], dataset, prefix + "  ")

                elif "merge" in step:
                    self._run_merge(step["merge"], dataset, prefix + "  ")

                elif "context_filter" in step:
                    self._run_context_filter(step["context_filter"], dataset, prefix + "  ")

                else:
                    operation = self.builder.build_operation(step)
                    self._execute_operation(operation, dataset, prefix + "  ")

            # Sequential sub-pipeline (list of steps)
            elif isinstance(step, list):
                print(f"{prefix}  📁 Sub-pipeline with {len(step)} steps")
                for sub_step in step:
                    self._run_step(sub_step, dataset, prefix + "    ")

            # String reference (preset, operation name, etc.)
            elif isinstance(step, str):
                if step == "uncluster":
                    self._run_uncluster(dataset, prefix + "  ")
                elif step == "unscope":
                    self._run_unscope(dataset, prefix + "  ")
                else:
                    operation = self.builder.build_operation(step)
                    self._execute_operation(operation, dataset, prefix + "  ")

            # Direct operation object
            else:
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, dataset, prefix + "  ")

            # Complete step successfully
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
        finally:
            print("-" * 200)
            print(f"Step {self.current_step} completed: {step_description}")
            print(f"Dataset state after step {self.current_step}:")
            print(dataset)
            print("-" * 200)






    def _execute_operation(self, operation: Any, dataset: SpectraDataset, prefix: str):
        """Execute a built operation with proper context management and DatasetView"""
        operation_name = operation.get_name() if hasattr(operation, 'get_name') else str(operation)
        print(f"{prefix}⚙️ Executing: {operation_name}")

        if not self.data_selector:
            # Fallback to old behavior if DataSelector not available
            if hasattr(operation, 'execute'):
                operation.execute(dataset)
            else:
                print(f"{prefix}  ⚠️ Operation {operation_name} has no execute method")
            return        # Enhanced execution with DatasetView and proper scoping
        try:
            # Determine operation phase (fit, transform, predict)
            if hasattr(operation, 'fit') and not hasattr(operation, '_fitted'):
                self._execute_fit_phase(operation, dataset, prefix)
                operation._fitted = True  # Mark as fitted

            if hasattr(operation, 'transform'):
                self._execute_transform_phase(operation, dataset, prefix)
            elif hasattr(operation, 'predict'):
                self._execute_predict_phase(operation, dataset, prefix)

        except Exception as e:
            print(f"{prefix}  ❌ Operation {operation_name} failed: {str(e)}")
            if not self.continue_on_error:
                raise

    # def _execute_fit_phase(self, operation: Any, dataset: SpectraDataset, prefix: str):
    #     """Execute the fit phase with proper data scoping"""
    #     fit_filters = self.data_selector.get_enhanced_scope(operation, self.context, phase='fit')
    #     fit_view = DatasetView(dataset, filters=fit_filters)

    #     print(f"{prefix}  📊 Fitting on: {len(fit_view)} samples with filters {fit_filters}")

    #     X = fit_view.get_features()

    #     if hasattr(operation, 'fit'):
    #         # Check if operation needs targets
    #         try:
    #             y = fit_view.get_targets()
    #             operation.fit(X, y)
    #             print(f"{prefix}  ✅ Fitted with targets: X={X.shape}, y={y.shape}")
    #         except:
    #             # Unsupervised learning or transformer
    #             operation.fit(X)
    #             print(f"{prefix}  ✅ Fitted without targets: X={X.shape}")

    # def _execute_transform_phase(self, operation: Any, dataset: SpectraDataset, prefix: str):
    #     """Execute the transform phase with proper data scoping"""
    #     transform_filters = self.data_selector.get_enhanced_scope(operation, self.context, phase='transform')
    #     transform_view = DatasetView(dataset, filters=transform_filters)

    #     print(f"{prefix}  🔄 Transforming: {len(transform_view)} samples with filters {transform_filters}")

    #     X = transform_view.get_features()
    #     X_transformed = operation.transform(X)

    #     # Update dataset with transformed features
    #     # This is a simplified version - in practice we'd need to update the dataset properly
    #     print(f"{prefix}  ✅ Transformed: {X.shape} -> {X_transformed.shape}")

    # def _execute_predict_phase(self, operation: Any, dataset: SpectraDataset, prefix: str):
    #     """Execute the predict phase with proper data scoping"""
    #     predict_filters = self.data_selector.get_enhanced_scope(operation, self.context, phase='predict')
    #     predict_view = DatasetView(dataset, filters=predict_filters)

    #     print(f"{prefix}  🎯 Predicting: {len(predict_view)} samples with filters {predict_filters}")

    #     X = predict_view.get_features()

    #     if hasattr(operation, 'predict_proba'):
    #         predictions = operation.predict_proba(X)
    #         print(f"{prefix}  ✅ Predicted probabilities: {X.shape} -> {predictions.shape}")
    #     elif hasattr(operation, 'predict'):
    #         predictions = operation.predict(X)
    #         print(f"{prefix}  ✅ Predicted: {X.shape} -> {predictions.shape}")
    #     else:
    #         print(f"{prefix}  ⚠️ No prediction method available")

    # =================================================================
    # CONTROL STRUCTURE HANDLERS - Enhanced with DatasetView
    # =================================================================

    def _run_context_filter(self, filters: Dict, dataset: SpectraDataset, prefix: str):
        """Apply context filters"""
        print(f"{prefix}🔍 Context filter: {filters}")
        self.context.apply_filters(filters)

    def _run_sample_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute sample augmentation with proper scoping"""
        print(f"{prefix}📊 Sample augmentation with {len(augmenters)} augmenters")

        for i, augmenter in enumerate(augmenters):
            print(f"{prefix}  📌 Augmenter {i+1}/{len(augmenters)}")
            operation = self.builder.build_operation(augmenter)
            if hasattr(operation, 'mode'):
                setattr(operation, 'mode', "sample_augmentation")
            self._execute_operation(operation, dataset, f"{prefix}    ")

    def _run_feature_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute feature augmentation with enhanced context management"""
        print(f"{prefix}🔄 Feature augmentation with {len(augmenters)} augmenters")

        # Get current train set for augmentation using DatasetView
        if self.data_selector:
            train_filters = {**self.context.current_filters, "partition": "train"}
            train_view = DatasetView(dataset, filters=train_filters)
        else:
            # Fallback to old method
            train_view = dataset.select(partition="train", **self.context.current_filters)

        if len(train_view) == 0:
            print(f"{prefix}  ⚠️ No train data found for feature augmentation")
            return

        print(f"{prefix}  📊 Base train set: {len(train_view)} samples")

        # Execute augmenters based on parallel configuration
        if self.max_workers != 1 and len(augmenters) > 1:
            self._run_feature_augmentation_parallel(augmenters, dataset, train_view, prefix)
        else:
            self._run_feature_augmentation_sequential(augmenters, dataset, train_view, prefix)

    def _run_feature_augmentation_sequential(self, augmenters: List[Any], dataset: SpectraDataset, train_view, prefix: str):
        """Sequential feature augmentation execution"""
        for i, augmenter in enumerate(augmenters):
            print(f"{prefix}  📌 Augmenter {i+1}/{len(augmenters)}")
            try:
                operation = self.builder.build_operation(augmenter)
                self._execute_operation(operation, dataset, f"{prefix}    ")
                print(f"{prefix}  ✅ Augmenter {i+1} completed")
            except Exception as e:
                print(f"{prefix}  ❌ Augmenter {i+1} failed: {str(e)}")
                if not self.continue_on_error:
                    raise

    def _run_feature_augmentation_parallel(self, augmenters: List[Any], dataset: SpectraDataset, train_view, prefix: str):
        """Parallel feature augmentation execution using joblib"""
        print(f"{prefix}  🔀 Running {len(augmenters)} augmenters in parallel (max_workers={self.max_workers})")
        # Implementation would go here - simplified for now
        self._run_feature_augmentation_sequential(augmenters, dataset, train_view, prefix)

    # =================================================================
    # PLACEHOLDER METHODS FOR COMPLEX OPERATIONS
    # =================================================================

    def _run_branch(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branch operations"""
        print(f"{prefix}🌿 Branch with {len(branches)} paths")
        for i, branch in enumerate(branches):
            print(f"{prefix}  🔀 Branch {i+1}")
            self._run_step(branch, dataset, prefix + "    ")

    def _run_dispatch(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute dispatch operations"""
        print(f"{prefix}📤 Dispatch with {len(branches)} targets")
        for i, branch in enumerate(branches):
            print(f"{prefix}  📬 Dispatch {i+1}")
            self._run_step(branch, dataset, prefix + "    ")

    def _run_model(self, model_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute model operations"""
        print(f"{prefix}🤖 Model: {model_config}")
        operation = self.builder.build_operation(model_config)
        self._execute_operation(operation, dataset, prefix + "  ")

    def _run_stack(self, stack_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute stacking operations"""
        print(f"{prefix}📚 Stack: {stack_config}")
        operation = self.builder.build_operation(stack_config)
        self._execute_operation(operation, dataset, prefix + "  ")

    def _run_scope(self, scope_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute scope operations"""
        print(f"{prefix}🎯 Scope: {scope_config}")
        self.context.push_scope(**scope_config)

    def _run_cluster(self, cluster_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute cluster operations"""
        print(f"{prefix}🔘 Cluster: {cluster_config}")
        operation = self.builder.build_operation(cluster_config)
        self._execute_operation(operation, dataset, prefix + "  ")

    def _run_merge(self, merge_config: Dict, dataset: SpectraDataset, prefix: str):
        """Execute merge operations"""
        print(f"{prefix}🔗 Merge: {merge_config}")

    def _run_uncluster(self, dataset: SpectraDataset, prefix: str):
        """Execute uncluster operations"""
        print(f"{prefix}🔓 Uncluster")
        self.context.pop_cluster()

    def _run_unscope(self, dataset: SpectraDataset, prefix: str):
        """Execute unscope operations"""
        print(f"{prefix}🔄 Unscope")
        self.context.pop_scope()

    # =================================================================
    # UTILITY METHODS
    # =================================================================

    def _get_step_description(self, step: Any) -> str:
        """Get a human-readable description of a step"""
        if isinstance(step, dict):
            if len(step) == 1:
                key = next(iter(step.keys()))
                return f"{key}"
            else:
                return f"Dict with {len(step)} keys"
        elif isinstance(step, list):
            return f"Sub-pipeline ({len(step)} steps)"
        elif isinstance(step, str):
            return step
        else:
            return str(type(step).__name__)

    def _build_fitted_tree(self, steps: List[Any], dataset: SpectraDataset) -> Any:
        """Build fitted tree structure"""
        # Placeholder implementation
        from PipelineTree import PipelineTree
        return PipelineTree(steps)
