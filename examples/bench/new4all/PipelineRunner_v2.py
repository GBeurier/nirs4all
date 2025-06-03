"""
PipelineRunner - Direct execution engine for complex pipeline configurations

This runner takes the philosophy from the old pipeline.py but applies it to the current API.
It manages branches, context updates, and parallelization directly without building
PipelineOperation objects first, making the pipeline structure more visible and controllable.
"""
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy
from pathlib import Path

from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
from PipelineConfig import PipelineConfig
from TransformationOperation import TransformationOperation
from ClusteringOperation import ClusteringOperation, UnclusterOperation
from ModelOperation import ModelOperation
from StackOperation import StackOperation
from OptimizationOperation import OptimizationOperation
from DummyOperations import VisualizationOperation

# Import sklearn components for direct handling
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.model_selection import BaseCrossValidator


class PipelineRunner:
    """
    Direct pipeline execution engine that manages branches and context explicitly.
    
    Philosophy:
    - Direct interpretation of config structures
    - Explicit context and filter management
    - Visible branch parallelization
    - No intermediate operation objects unless needed
    """

    def __init__(self, max_workers: Optional[int] = None, continue_on_error: bool = False):
        """
        Initialize pipeline runner
        
        Parameters:
        -----------
        max_workers : Optional[int]
            Maximum number of parallel workers for dispatch operations
        continue_on_error : bool
            Whether to continue execution on operation errors
        """
        self.max_workers = max_workers
        self.continue_on_error = continue_on_error
        
        # Execution state
        self.context = PipelineContext()
        self.execution_history = []
        self.current_step = 0
        self.current_depth = 0
        
        # Branch management
        self.active_branches = [0]  # Start with branch 0
        self.branch_contexts = {0: {}}  # Store context per branch
        
    def run_pipeline(self, pipeline_config: Union[Dict[str, Any], List[Any], PipelineConfig],
                     dataset: SpectraDataset, prefix: str = "") -> SpectraDataset:
        """
        Main entry point for pipeline execution
        
        Parameters:
        -----------
        pipeline_config : Union[Dict, List, PipelineConfig]
            Pipeline configuration to execute
        dataset : SpectraDataset
            Dataset to process
        prefix : str
            Logging prefix for nested calls
            
        Returns:
        --------
        SpectraDataset
            Processed dataset
        """
        print(f"{prefix}🚀 Starting Pipeline Runner")
        
        # Extract pipeline steps from config
        if isinstance(pipeline_config, PipelineConfig):
            steps = pipeline_config.pipeline
            experiment_config = pipeline_config.experiment or {}
        elif isinstance(pipeline_config, dict) and 'pipeline' in pipeline_config:
            steps = pipeline_config['pipeline']
            experiment_config = pipeline_config.get('experiment', {})
        elif isinstance(pipeline_config, list):
            steps = pipeline_config
            experiment_config = {}
        else:
            raise ValueError(f"Invalid pipeline config type: {type(pipeline_config)}")
            
        # Initialize context with experiment settings
        self.context.current_filters = {}
        if experiment_config:
            self._apply_experiment_config(experiment_config)
            
        # Execute pipeline steps
        try:
            for step in steps:
                self._run_step(step, dataset, prefix)
                
            print(f"{prefix}✅ Pipeline completed successfully!")
            
        except Exception as e:
            print(f"{prefix}❌ Pipeline failed: {str(e)}")
            if not self.continue_on_error:
                raise
                
        return dataset
        
    def _apply_experiment_config(self, experiment_config: Dict[str, Any]):
        """Apply experiment configuration to context"""
        if 'action' in experiment_config:
            self.context.current_filters['task_type'] = experiment_config['action']
            
    def _run_step(self, step: Any, dataset: SpectraDataset, prefix: str = ""):
        """
        Execute a single pipeline step with unified parsing logic
        
        This is the MAIN PARSING LOGIC - all step types are identified here,
        then appropriate _run_XXXX functions are called.
        """
        self.current_step += 1
        step_prefix = f"{prefix}  "
        
        # Log step execution
        self._log_step_start(step, prefix)
        
        try:
            # UNIFIED PARSING LOGIC - Clear step type identification
            if isinstance(step, str):
                # String steps: control commands, plot operations
                if step == "uncluster":
                    self._run_uncluster(dataset, step_prefix)
                elif step.startswith("Plot"):
                    self._run_visualization(step, dataset, step_prefix)
                else:
                    print(f"{step_prefix}⚠️  Unknown string step: {step}")
                    
            elif isinstance(step, dict):
                # Dictionary steps: control structures or complex operations
                if len(step) == 1:
                    # Single-key control structures
                    key, value = next(iter(step.items()))
                    
                    if key == "cluster":
                        self._run_clustering(value, dataset, step_prefix)
                    elif key in ["sample_augmentation", "samples", "S"]:
                        self._run_sample_augmentation(value, dataset, step_prefix)
                    elif key in ["feature_augmentation", "features", "F"]:
                        self._run_feature_augmentation(value, dataset, step_prefix)
                    elif key == "dispatch":
                        self._run_dispatch(value, dataset, step_prefix)
                    elif key == "stack":
                        self._run_stacking(value, dataset, step_prefix)
                    elif key == "model":
                        self._run_model({"model": value}, dataset, step_prefix)
                    else:
                        print(f"{step_prefix}❓ Unknown control key: {key}")
                        
                else:
                    # Multi-key structures (model configurations, etc.)
                    if "model" in step:
                        self._run_model(step, dataset, step_prefix)
                    elif "stack" in step:
                        self._run_stacking(step["stack"], dataset, step_prefix)
                    else:
                        print(f"{step_prefix}❓ Unknown complex step: {list(step.keys())}")
                        
            elif isinstance(step, list):
                # List steps: sub-pipelines
                self._run_sub_pipeline(step, dataset, step_prefix)
                
            else:
                # Object instances: transformers, splitters, models
                self._run_object(step, dataset, step_prefix)
                
            # Log successful execution
            self._log_step_success(step, prefix)
            
        except Exception as e:
            self._log_step_error(step, e, prefix)
            if not self.continue_on_error:
                raise

    # =================================================================
    # EXECUTION FUNCTIONS - Called by the unified parsing logic above
    # =================================================================
    
    def _run_uncluster(self, dataset: SpectraDataset, prefix: str):
        """Remove group filters from context"""
        print(f"{prefix}🔓 Uncluster operation")
        if 'group' in self.context.current_filters:
            del self.context.current_filters['group']
        print(f"{prefix}   Removed group filters from context")
        
    def _run_visualization(self, step: str, dataset: SpectraDataset, prefix: str):
        """Handle visualization steps"""
        print(f"{prefix}📊 Visualization: {step}")
        viz_op = VisualizationOperation(plot_type=step)
        viz_op.execute(dataset, self.context)
        
    def _run_sub_pipeline(self, step_list: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute sub-pipeline (list of steps)"""
        print(f"{prefix}📋 Sub-pipeline with {len(step_list)} steps")
        for sub_step in step_list:
            self._run_step(sub_step, dataset, prefix)
            
    def _run_object(self, step: Any, dataset: SpectraDataset, prefix: str):
        """Handle object instances (transformers, splitters, models, etc.)"""
        step_type = step.__class__.__name__
        print(f"{prefix}🔧 Object: {step_type}")
        
        # Handle sklearn transformers
        if hasattr(step, "transform") and isinstance(step, TransformerMixin):
            self._run_transformation(step, dataset, prefix)
            
        # Handle splitters
        elif hasattr(step, "split"):
            self._run_splitting(step, dataset, prefix)
            
        # Handle clusterers
        elif hasattr(step, "fit") and isinstance(step, ClusterMixin):
            self._run_clustering(step, dataset, prefix)
            
        # Handle models
        elif isinstance(step, BaseEstimator):
            self._run_model({"model": step}, dataset, prefix)
            
        else:
            print(f"{prefix}❓ Unknown object type: {step_type}")
            
    def _run_transformation(self, transformer: Any, dataset: SpectraDataset, prefix: str):
        """Execute transformation operation"""
        print(f"{prefix}🔄 Transformation: {transformer.__class__.__name__}")
        
        # Create and execute transformation operation
        transform_op = TransformationOperation(
            transformer=transformer,
            fit_partition="train",
            mode="transformation"
        )
        transform_op.execute(dataset, self.context)
        
    def _run_splitting(self, splitter: Any, dataset: SpectraDataset, prefix: str):
        """Execute data splitting operation"""
        print(f"{prefix}✂️  Splitting: {splitter.__class__.__name__}")
        
        # Get current data view
        current_view = dataset.select(**self.context.current_filters)
        
        if isinstance(splitter, BaseCrossValidator):
            print(f"{prefix}   📊 Cross-validation setup")
            # Store CV splitter in context for model training
            self.context.cv_splitter = splitter
        else:
            print(f"{prefix}   🔀 Train/test split")
            # Perform train/test split
            X = current_view.get_features(concatenate=True)
            y = current_view.get_targets()
            
            if len(X) == 0:
                print(f"{prefix}   ⚠️  No data found for splitting")
                return
                
            # Get first split
            splits = list(splitter.split(X, y))
            if splits:
                train_idx, test_idx = splits[0]
                
                # Update partition labels in dataset
                sample_ids = current_view.sample_ids
                train_sample_ids = [sample_ids[i] for i in train_idx]
                test_sample_ids = [sample_ids[i] for i in test_idx]
                
                # Update dataset partitions
                dataset.update_processing(train_sample_ids, "train")
                dataset.update_processing(test_sample_ids, "test")
                
                print(f"{prefix}   📈 Split: {len(train_idx)} train, {len(test_idx)} test")
                
    def _run_clustering(self, clusterer: Any, dataset: SpectraDataset, prefix: str):
        """Execute clustering operation"""
        print(f"{prefix}🎯 Clustering: {clusterer.__class__.__name__ if hasattr(clusterer, '__class__') else str(clusterer)}")
        
        # Create and execute clustering operation
        cluster_op = ClusteringOperation(clusterer=clusterer)
        cluster_op.execute(dataset, self.context)
        
        # Update context with group filtering
        self.context.current_filters['group'] = True
        print(f"{prefix}   🏷️  Updated context with group filtering")
        
    def _run_sample_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute sample augmentation operations"""
        print(f"{prefix}🔄 Sample augmentation with {len(augmenters)} augmenter(s)")
        
        for i, augmenter in enumerate(augmenters):
            if augmenter is None:
                print(f"{prefix}   {i+1}. Identity (no augmentation)")
                continue
                
            print(f"{prefix}   {i+1}. {augmenter.__class__.__name__ if hasattr(augmenter, '__class__') else str(augmenter)}")
            
            # Create and execute transformation operation in sample augmentation mode
            if hasattr(augmenter, 'transform'):
                transform_op = TransformationOperation(
                    transformer=augmenter,
                    fit_partition="train",
                    mode="sample_augmentation"
                )
                transform_op.execute(dataset, self.context)
            else:
                print(f"{prefix}     ⚠️  Augmenter does not support transform")
                
    def _run_feature_augmentation(self, augmenters: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute feature augmentation operations"""
        print(f"{prefix}🔧 Feature augmentation with {len(augmenters)} augmenter(s)")
        
        for i, augmenter in enumerate(augmenters):
            if augmenter is None:
                print(f"{prefix}   {i+1}. Identity (no augmentation)")
                continue
                
            # Handle nested lists of transformers
            if isinstance(augmenter, list):
                print(f"{prefix}   {i+1}. Pipeline of {len(augmenter)} transformers")
                # For now, treat as sequential application - could be improved
                for j, sub_aug in enumerate(augmenter):
                    if hasattr(sub_aug, 'transform'):
                        transform_op = TransformationOperation(
                            transformer=sub_aug,
                            fit_partition="train",
                            mode="feature_augmentation"
                        )
                        transform_op.execute(dataset, self.context)
            else:
                print(f"{prefix}   {i+1}. {augmenter.__class__.__name__ if hasattr(augmenter, '__class__') else str(augmenter)}")
                
                if hasattr(augmenter, 'transform'):
                    transform_op = TransformationOperation(
                        transformer=augmenter,
                        fit_partition="train",
                        mode="feature_augmentation"
                    )
                    transform_op.execute(dataset, self.context)
                else:
                    print(f"{prefix}     ⚠️  Augmenter does not support transform")
                    
    def _run_dispatch(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute parallel dispatch of pipeline branches"""
        print(f"{prefix}🌳 Dispatch: {len(branches)} parallel branches")
        
        # Store original context state
        original_branch = self.context.current_branch
        original_filters = self.context.current_filters.copy()
        
        # Execute branches in parallel or sequential based on configuration
        if self.max_workers and self.max_workers > 1:
            self._run_dispatch_parallel(branches, dataset, prefix)
        else:
            self._run_dispatch_sequential(branches, dataset, prefix)
            
        # Restore original context state
        self.context.branch_stack = [original_branch]
        self.context.current_filters = original_filters
        
    def _run_dispatch_sequential(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branches sequentially"""
        for i, branch in enumerate(branches):
            branch_prefix = f"{prefix}  Branch {i+1}: "
            print(f"{branch_prefix}🌿 Starting")
            
            # Set up branch context
            self.context.push_branch(i)
            
            try:
                # Execute branch pipeline
                if isinstance(branch, list):
                    self._run_sub_pipeline(branch, dataset, branch_prefix)
                else:
                    self._run_step(branch, dataset, branch_prefix)
                    
                print(f"{branch_prefix}✅ Completed")
                
            except Exception as e:
                print(f"{branch_prefix}❌ Failed: {str(e)}")
                if not self.continue_on_error:
                    raise
            finally:
                # Clean up branch context
                self.context.pop_branch()
                
    def _run_dispatch_parallel(self, branches: List[Any], dataset: SpectraDataset, prefix: str):
        """Execute branches in parallel using ThreadPoolExecutor"""
        print(f"{prefix}   🔄 Parallel execution with {self.max_workers} workers")
        
        def execute_branch(branch_data):
            branch_idx, branch, dataset_copy = branch_data
            branch_prefix = f"{prefix}  Branch {branch_idx+1}: "
            
            # Create isolated context for this branch
            branch_context = PipelineContext()
            branch_context.current_filters = self.context.current_filters.copy()
            branch_context.push_branch(branch_idx)
            
            try:
                print(f"{branch_prefix}🌿 Starting (parallel)")
                
                # Create branch runner with isolated context
                branch_runner = PipelineRunner(max_workers=1, continue_on_error=self.continue_on_error)
                branch_runner.context = branch_context
                
                # Execute branch
                if isinstance(branch, list):
                    branch_runner._run_sub_pipeline(branch, dataset_copy, branch_prefix)
                else:
                    branch_runner._run_step(branch, dataset_copy, branch_prefix)
                    
                print(f"{branch_prefix}✅ Completed (parallel)")
                return branch_idx, dataset_copy, branch_context, None
                
            except Exception as e:
                print(f"{branch_prefix}❌ Failed (parallel): {str(e)}")
                return branch_idx, None, None, e
                
        # Prepare branch execution data
        branch_data = []
        for i, branch in enumerate(branches):
            # Create dataset copy for each branch (for parallel safety)
            dataset_copy = self._copy_dataset_for_branch(dataset)
            branch_data.append((i, branch, dataset_copy))
            
        # Execute branches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_branch = {executor.submit(execute_branch, bd): bd[0] for bd in branch_data}
            
            results = {}
            for future in as_completed(future_to_branch):
                branch_idx = future_to_branch[future]
                try:
                    result = future.result()
                    results[branch_idx] = result
                except Exception as e:
                    print(f"{prefix}  Branch {branch_idx+1} failed: {str(e)}")
                    if not self.continue_on_error:
                        raise
                        
        # Merge results back to main dataset (implementation depends on requirements)
        self._merge_branch_results(results, dataset, prefix)
        
    def _run_model(self, model_config: Dict[str, Any], dataset: SpectraDataset, prefix: str):
        """Execute model training and prediction"""
        model = model_config.get("model")
        model_name = model.__class__.__name__ if hasattr(model, '__class__') else str(model)
        print(f"{prefix}🤖 Model: {model_name}")
        
        # Check for hyperparameter optimization
        if "finetune_params" in model_config:
            print(f"{prefix}   🎯 Hyperparameter optimization")
            opt_op = OptimizationOperation(
                model=model,
                param_space=model_config["finetune_params"],
                y_pipeline=model_config.get("y_pipeline")
            )
            opt_op.execute(dataset, self.context)
        else:
            # Regular model training
            model_op = ModelOperation(
                model=model,
                y_pipeline=model_config.get("y_pipeline")
            )
            model_op.execute(dataset, self.context)
            
    def _run_stacking(self, stack_config: Dict[str, Any], dataset: SpectraDataset, prefix: str):
        """Execute stacking ensemble"""
        print(f"{prefix}🏗️  Stacking ensemble")
        
        stack_op = StackOperation(
            main_model=stack_config.get("model"),
            base_learners=stack_config.get("base_learners", []),
            y_pipeline=stack_config.get("y_pipeline")
        )
        stack_op.execute(dataset, self.context)
        
    # =================================================================
    # UTILITY FUNCTIONS
    # =================================================================
    
    def _copy_dataset_for_branch(self, dataset: SpectraDataset) -> SpectraDataset:
        """Create a copy of dataset for branch execution"""
        # For now, return the same dataset
        # In a full implementation, you might want to create a proper copy
        # or use copy-on-write semantics
        return dataset
        
    def _merge_branch_results(self, results: Dict[int, Tuple], dataset: SpectraDataset, prefix: str):
        """Merge results from parallel branches back to main dataset"""
        print(f"{prefix}   🔄 Merging {len(results)} branch results")
        
        # Implementation depends on what needs to be merged
        # For now, just collect predictions from contexts
        merged_predictions = {}
        
        for branch_idx, (_, dataset_copy, context, error) in results.items():
            if error is None and context:
                branch_predictions = context.get_predictions()
                for model_name, preds in branch_predictions.items():
                    key = f"branch_{branch_idx}_{model_name}"
                    merged_predictions[key] = preds
                    
        # Store merged predictions in main context
        for key, preds in merged_predictions.items():
            self.context.add_predictions(key, preds)
            
        print(f"{prefix}   ✅ Merged predictions from {len(merged_predictions)} models")
        
    def _log_step_start(self, step: Any, prefix: str):
        """Log step execution start"""
        step_desc = self._get_step_description(step)
        print(f"{prefix}🔄 Step {self.current_step}: {step_desc}")
        
        # Store in execution history
        self.execution_history.append({
            'step': self.current_step,
            'description': step_desc,
            'status': 'running',
            'context_filters': self.context.current_filters.copy(),
            'current_branch': self.context.current_branch
        })
        
    def _log_step_success(self, step: Any, prefix: str):
        """Log step execution success"""
        if self.execution_history:
            self.execution_history[-1]['status'] = 'success'
            
    def _log_step_error(self, step: Any, error: Exception, prefix: str):
        """Log step execution error"""
        step_desc = self._get_step_description(step)
        print(f"{prefix}❌ Step {self.current_step} failed: {step_desc} - {str(error)}")
        
        if self.execution_history:
            self.execution_history[-1]['status'] = 'failed'
            self.execution_history[-1]['error'] = str(error)
            
    def _get_step_description(self, step: Any) -> str:
        """Get human-readable description of step"""
        if isinstance(step, str):
            return f"String '{step}'"
        elif isinstance(step, dict):
            keys = list(step.keys())
            return f"Dict {keys}"
        elif isinstance(step, list):
            return f"List[{len(step)}]"
        elif hasattr(step, '__class__'):
            return f"{step.__class__.__name__}"
        else:
            return str(step)
            
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        return {
            'total_steps': self.current_step,
            'successful_steps': len([h for h in self.execution_history if h['status'] == 'success']),
            'failed_steps': len([h for h in self.execution_history if h['status'] == 'failed']),
            'execution_history': self.execution_history,
            'final_context_filters': self.context.current_filters,
            'predictions': self.context.get_predictions()
        }
