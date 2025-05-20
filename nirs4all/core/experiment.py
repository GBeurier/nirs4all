import copy
import numpy as np
import logging
from typing import Any, Optional, Dict

from .config import Config
from ..data.dataset_loader import get_dataset
from ..data.dataset import Dataset # For type hint
from .processor import get_transformer # Corrected: Removed duplicate import
from .model.model_builder_factory import ModelBuilderFactory

# Default logger for the module if no external logger is provided
module_logger = logging.getLogger(__name__)

class Experiment:
    def __init__(self, config: Config, external_logger: Optional[logging.Logger] = None):
        self.config: Config = config
        self.logger = external_logger if external_logger else module_logger

        # Initialized components
        self.dataset_obj: Optional[Dataset] = None
        self.x_pipeline_obj: Optional[Any] = None # Type could be TransformerMixin
        self.y_pipeline_obj: Optional[Any] = None # Type could be TransformerMixin
        self.model_obj: Optional[Any] = None
        self.model_framework: Optional[str] = None # e.g., 'sklearn', 'tensorflow', 'pytorch'
        
        # Parameters derived from config validation
        self.task: Optional[str] = None
        self.action: Optional[str] = None
        self.metrics: Optional[list] = None
        self.training_params: Optional[dict] = None
        self.finetune_params: Optional[dict] = None

        self._initialize_components()

    def _initialize_components(self):
        # 1. Initialize Dataset
        if self.config.dataset:
            try:
                self.dataset_obj = get_dataset(self.config.dataset)
                if self.dataset_obj:
                    self.logger.info(f"Dataset loaded: {getattr(self.dataset_obj, 'name', 'Unnamed Dataset')}")
                else:
                     self.logger.warning("Dataset configuration present but get_dataset returned None.")
            except Exception as e:
                self.logger.error(f"Error loading dataset: {e}")
                self.dataset_obj = None
        else:
            self.logger.info("No dataset configuration provided.")
            self.dataset_obj = None

        if not self.dataset_obj:
            self.logger.warning("Cannot proceed with full component initialization without a dataset.")
            # Try to initialize components that don't strictly depend on the dataset
            if self.config.x_pipeline:
                try:
                    self.x_pipeline_obj = get_transformer(self.config.x_pipeline)
                except Exception as e:
                    self.logger.error(f"Error initializing X pipeline without dataset context: {e}")
            if self.config.y_pipeline:
                try:
                    self.y_pipeline_obj = get_transformer(self.config.y_pipeline)
                except Exception as e:
                    self.logger.error(f"Error initializing Y pipeline without dataset context: {e}")
            return # Stop further initialization

        # Determine num_classes for the dataset_obj (using initial/raw labels)
        try:
            y_train_init_arr, y_test_init_arr = None, None
            if hasattr(self.dataset_obj, 'y_train_init') and self.dataset_obj.y_train_init is not None:
                y_train_init_arr = np.asarray(self.dataset_obj.y_train_init).ravel()
            if hasattr(self.dataset_obj, 'y_test_init') and self.dataset_obj.y_test_init is not None:
                y_test_init_arr = np.asarray(self.dataset_obj.y_test_init).ravel()

            if y_train_init_arr is not None and y_test_init_arr is not None:
                all_initial_labels = np.concatenate([y_train_init_arr, y_test_init_arr])
                self.dataset_obj.num_classes = len(np.unique(all_initial_labels))
                self.logger.info(f"Dataset num_classes (from y_train_init/y_test_init) set to: {self.dataset_obj.num_classes}")
            elif hasattr(self.dataset_obj, 'y') and self.dataset_obj.y is not None:
                y_arr = np.asarray(self.dataset_obj.y).ravel()
                self.dataset_obj.num_classes = len(np.unique(y_arr))
                self.logger.info(f"Dataset num_classes (from dataset.y) set to: {self.dataset_obj.num_classes}")
            elif not hasattr(self.dataset_obj, 'num_classes') or self.dataset_obj.num_classes is None:
                self.dataset_obj.num_classes = None
                self.logger.warning("Could not determine num_classes for the dataset. This may impact classification tasks.")
        except Exception as e:
            self.logger.error(f"Error determining num_classes: {e}")
            if not hasattr(self.dataset_obj, 'num_classes'): # Ensure attribute exists
                self.dataset_obj.num_classes = None

        # 2. Validate config and determine task, action, metrics, etc.
        if self.config.experiment:
            try:
                # Pass the loaded dataset_obj (with num_classes potentially set) to validate
                action, metrics, training_params, finetune_params, task = \
                    self.config.validate(self.dataset_obj)
                
                self.action = action
                self.metrics = metrics
                self.training_params = training_params
                self.finetune_params = finetune_params
                self.task = task
                self.logger.info(f"Config validated. Task: {self.task}, Action: {self.action}")
            except Exception as e:
                self.logger.error(f"Error during config.validate: {e}. Task and other params may not be reliably set.")
                # Fallback to direct extraction if validate fails
                self.task = self.config.experiment.get('task')
                self.action = self.config.experiment.get('action')
                self.metrics = self.config.experiment.get('metrics')
                self.training_params = self.config.experiment.get('training_params')
                self.finetune_params = self.config.experiment.get('finetune_params')
        else:
            self.logger.warning("No 'experiment' section in config. Task and other parameters cannot be determined via config.validate.")
            # Attempt to get task from dataset if possible (e.g., dataset.task_type)
            self.task = getattr(self.dataset_obj, 'task_type', None)

        # 3. Initialize X Pipeline
        if self.config.x_pipeline:
            try:
                self.x_pipeline_obj = get_transformer(self.config.x_pipeline)
                self.logger.info("X pipeline initialized.")
            except Exception as e:
                self.logger.error(f"Error initializing X pipeline: {e}")
        
        # 4. Initialize Y Pipeline
        if self.config.y_pipeline:
            try:
                self.y_pipeline_obj = get_transformer(self.config.y_pipeline)
                self.logger.info("Y pipeline initialized.")
            except Exception as e:
                self.logger.error(f"Error initializing Y pipeline: {e}")

        # 5. Initialize Model
        if self.config.model:
            if self.task is None:
                self.logger.warning("Task is not determined. Model building might be ambiguous or fail.")
            
            try:
                model_instance, model_framework = ModelBuilderFactory.build_single_model(
                    model_config=self.config.model,
                    dataset=self.dataset_obj, # Pass the loaded dataset
                    task=self.task
                )
                self.model_obj = model_instance
                self.model_framework = model_framework
                self.logger.info(f"Model initialized. Framework: {self.model_framework}")
            except Exception as e:
                self.logger.error(f"Error initializing model: {e}")
        else:
            self.logger.info("No model configuration provided.")

    # Getter methods
    def get_dataset(self) -> Optional[Dataset]:
        return self.dataset_obj

    def get_x_pipeline(self) -> Optional[Any]:
        return self.x_pipeline_obj

    def get_y_pipeline(self) -> Optional[Any]:
        return self.y_pipeline_obj

    def get_model(self) -> Optional[Any]:
        return self.model_obj

    def get_model_framework(self) -> Optional[str]:
        return self.model_framework

    def get_task(self) -> Optional[str]:
        return self.task
        
    def get_action(self) -> Optional[str]:
        return self.action

    def get_metrics(self) -> Optional[list]:
        return self.metrics

    def get_training_params(self) -> Optional[dict]:
        return self.training_params

    def get_finetune_params(self) -> Optional[dict]:
        return self.finetune_params

