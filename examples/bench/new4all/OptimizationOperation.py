"""
OptimizationOperation - Hyperparameter optimization for ML models
"""
from typing import Dict, List, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class OptimizationOperation(PipelineOperation):
    """Operation for hyperparameter optimization of models"""

    def __init__(self,
                 optimizer_type: str = "optuna",
                 model_params: Optional[Dict[str, Any]] = None,
                 training_params: Optional[Dict[str, Any]] = None,
                 n_trials: int = 100,
                 metrics: Optional[List[str]] = None,
                 task: str = "auto",
                 **optimizer_kwargs):
        """
        Initialize optimization operation

        Parameters:
        -----------
        optimizer_type : str
            Type of optimizer: "optuna", "sklearn", "torch", "tf"
        model_params : dict
            Model parameters to optimize
        training_params : dict
            Training parameters to optimize
        n_trials : int
            Number of optimization trials
        metrics : list
            Metrics to optimize for
        task : str
            Task type: "classification", "regression", or "auto"
        **optimizer_kwargs : dict
            Additional optimizer-specific parameters
        """
        super().__init__()
        self.optimizer_type = optimizer_type
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.n_trials = n_trials
        self.metrics = metrics or ["mean_squared_error"]
        self.task = task
        self.optimizer_kwargs = optimizer_kwargs

        # Optimization results
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        self.optimizer = None

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute hyperparameter optimization"""
        if not self.can_execute(dataset, context):
            print("Cannot execute optimization: insufficient data or configuration")
            return

        # Determine task type if auto
        if self.task == "auto":
            self.task = self._infer_task_type(dataset)

        print(f"Starting {self.optimizer_type} optimization for {self.task} task")
        print(f"Optimizing {len(self.model_params)} model parameters and {len(self.training_params)} training parameters")
        print(f"Running {self.n_trials} trials")

        # Create appropriate optimizer
        self.optimizer = self._create_optimizer()

        # Run optimization
        try:
            self.best_params, self.best_score = self.optimizer.optimize(
                dataset=dataset,
                context=context,
                model_params=self.model_params,
                training_params=self.training_params,
                n_trials=self.n_trials,
                metrics=self.metrics,
                task=self.task,
                **self.optimizer_kwargs
            )

            # Store results in context
            context.optimization_results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_history': self.optimization_history,
                'optimizer_type': self.optimizer_type,
                'task': self.task
            }

            print(f"Optimization completed. Best score: {self.best_score:.6f}")
            print(f"Best parameters: {self.best_params}")

        except Exception as e:
            print(f"Optimization failed: {e}")
            raise

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if optimization can be executed"""
        # Need training data
        train_view = dataset.select(partition="train", **context.current_filters)
        if len(train_view) == 0:
            return False

        # Need parameters to optimize
        if not self.model_params and not self.training_params:
            return False

        return True

    def get_name(self) -> str:
        """Get operation name"""
        return f"OptimizationOperation({self.optimizer_type})"

    def _infer_task_type(self, dataset: SpectraDataset) -> str:
        """Infer task type from dataset targets"""
        try:
            train_view = dataset.select(partition="train")
            if len(train_view) == 0:
                return "regression"  # Default

            sample_ids = train_view.sample_ids[:10]  # Sample first 10 for efficiency
            targets = dataset.get_targets(sample_ids)

            # Check if targets are discrete (classification) or continuous (regression)
            unique_values = np.unique(targets)
            if len(unique_values) <= 20 and np.all(targets == targets.astype(int)):
                return "classification"
            else:
                return "regression"

        except Exception:
            return "regression"  # Default fallback

    def _create_optimizer(self) -> 'BaseOptimizer':
        """Create appropriate optimizer based on type"""
        if self.optimizer_type == "optuna":
            return OptunaOptimizer()
        elif self.optimizer_type == "sklearn":
            return SklearnOptimizer()
        elif self.optimizer_type == "torch":
            return TorchOptimizer()
        elif self.optimizer_type == "tf":
            return TensorFlowOptimizer()
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if self.best_params is None:
            return {"status": "not_run"}

        return {
            "status": "completed",
            "optimizer_type": self.optimizer_type,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "n_trials": self.n_trials,
            "task": self.task,
            "metrics": self.metrics
        }


class BaseOptimizer(ABC):
    """Abstract base class for optimizers"""

    @abstractmethod
    def optimize(self, dataset: SpectraDataset, context: PipelineContext,
                 model_params: Dict[str, Any], training_params: Dict[str, Any],
                 n_trials: int, metrics: List[str], task: str,
                 **kwargs) -> tuple:
        """Run optimization and return best parameters and score"""
        pass


class OptunaOptimizer(BaseOptimizer):
    """Optuna-based optimizer"""

    def optimize(self, dataset: SpectraDataset, context: PipelineContext,
                 model_params: Dict[str, Any], training_params: Dict[str, Any],
                 n_trials: int, metrics: List[str], task: str,
                 **kwargs) -> tuple:
        """Run Optuna optimization"""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        def objective(trial):
            # Sample parameters
            trial_params = {}

            # Sample model parameters
            for param_name, param_config in model_params.items():
                trial_params[param_name] = self._sample_parameter(trial, param_name, param_config)

            # Sample training parameters
            for param_name, param_config in training_params.items():
                trial_params[param_name] = self._sample_parameter(trial, f"train_{param_name}", param_config)

            # Evaluate model with these parameters
            score = self._evaluate_parameters(
                dataset, context, trial_params, metrics, task
            )
            return score

        # Create study
        direction = "minimize" if "error" in metrics[0].lower() or "loss" in metrics[0].lower() else "maximize"
        study = optuna.create_study(direction=direction)

        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=kwargs.get('n_jobs', 1))

        return study.best_params, study.best_value

    def _sample_parameter(self, trial, param_name: str, param_config: Any):
        """Sample a parameter using Optuna trial"""
        if isinstance(param_config, list):
            return trial.suggest_categorical(param_name, param_config)
        elif isinstance(param_config, tuple) and len(param_config) >= 3:
            param_type, low, high = param_config[:3]
            if param_type == 'int':
                return trial.suggest_int(param_name, low, high)
            elif param_type == 'float':
                return trial.suggest_float(param_name, low, high)
            elif param_type == 'log':
                return trial.suggest_float(param_name, low, high, log=True)
        else:
            return param_config  # Fixed parameter

    def _evaluate_parameters(self, dataset: SpectraDataset, context: PipelineContext,
                           params: Dict[str, Any], metrics: List[str], task: str) -> float:
        """Evaluate model with given parameters"""
        # This is a simplified evaluation - in practice, you'd want to:
        # 1. Create model with sampled parameters
        # 2. Train on training data
        # 3. Evaluate on validation data
        # 4. Return appropriate metric

        # For now, return a dummy score (in real implementation, integrate with ModelOperation)
        return np.random.random()


class SklearnOptimizer(BaseOptimizer):
    """Scikit-learn GridSearchCV/RandomizedSearchCV optimizer"""

    def optimize(self, dataset: SpectraDataset, context: PipelineContext,
                 model_params: Dict[str, Any], training_params: Dict[str, Any],
                 n_trials: int, metrics: List[str], task: str,
                 **kwargs) -> tuple:
        """Run sklearn optimization"""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        # Convert parameter specifications to sklearn format
        param_grid = self._convert_params_to_sklearn(model_params)

        # Get training data
        train_view = dataset.select(partition="train", **context.current_filters)
        X_train = train_view.get_features(concatenate=True)
        y_train = train_view.get_targets("auto")

        # Create base estimator (placeholder - needs integration with actual models)
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        if task == "classification":
            estimator = RandomForestClassifier()
        else:
            estimator = RandomForestRegressor()

        # Choose search strategy
        approach = kwargs.get('approach', 'random')
        if approach == 'grid':
            search = GridSearchCV(
                estimator, param_grid,
                cv=kwargs.get('cv', 5),
                n_jobs=kwargs.get('n_jobs', -1),
                scoring=kwargs.get('scoring', None)
            )
        else:
            search = RandomizedSearchCV(
                estimator, param_grid,
                n_iter=n_trials,
                cv=kwargs.get('cv', 5),
                n_jobs=kwargs.get('n_jobs', -1),
                scoring=kwargs.get('scoring', None)
            )

        # Fit search
        search.fit(X_train, y_train)

        return search.best_params_, search.best_score_

    def _convert_params_to_sklearn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter specifications to sklearn format"""
        sklearn_params = {}

        for param_name, param_config in params.items():
            if isinstance(param_config, list):
                sklearn_params[param_name] = param_config
            elif isinstance(param_config, tuple) and len(param_config) >= 3:
                param_type, low, high = param_config[:3]
                if param_type == 'int':
                    sklearn_params[param_name] = list(range(low, high + 1))
                elif param_type == 'float':
                    sklearn_params[param_name] = np.linspace(low, high, num=10)
            else:
                sklearn_params[param_name] = [param_config]

        return sklearn_params


class TorchOptimizer(BaseOptimizer):
    """PyTorch-based optimizer"""

    def optimize(self, dataset: SpectraDataset, context: PipelineContext,
                 model_params: Dict[str, Any], training_params: Dict[str, Any],
                 n_trials: int, metrics: List[str], task: str,
                 **kwargs) -> tuple:
        """Run PyTorch optimization"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        # Placeholder implementation
        # In practice, this would integrate with PyTorch models and training loops
        print("PyTorch optimization not fully implemented yet")
        return {}, 0.0


class TensorFlowOptimizer(BaseOptimizer):
    """TensorFlow/Keras-based optimizer"""

    def optimize(self, dataset: SpectraDataset, context: PipelineContext,
                 model_params: Dict[str, Any], training_params: Dict[str, Any],
                 n_trials: int, metrics: List[str], task: str,
                 **kwargs) -> tuple:
        """Run TensorFlow optimization"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")

        # Placeholder implementation
        # In practice, this would integrate with TensorFlow models and Keras Tuner
        print("TensorFlow optimization not fully implemented yet")
        return {}, 0.0


class OptimizationStrategy:
    """Strategy patterns for common optimization scenarios"""

    @classmethod
    def sklearn_grid_search(cls, param_grid: Dict[str, Any], cv: int = 5) -> OptimizationOperation:
        """Create sklearn grid search optimization"""
        return OptimizationOperation(
            optimizer_type="sklearn",
            model_params=param_grid,
            approach="grid",
            cv=cv
        )

    @classmethod
    def sklearn_random_search(cls, param_distributions: Dict[str, Any],
                            n_trials: int = 100, cv: int = 5) -> OptimizationOperation:
        """Create sklearn random search optimization"""
        return OptimizationOperation(
            optimizer_type="sklearn",
            model_params=param_distributions,
            n_trials=n_trials,
            approach="random",
            cv=cv
        )

    @classmethod
    def optuna_optimization(cls, param_space: Dict[str, Any],
                          n_trials: int = 100, direction: str = "minimize") -> OptimizationOperation:
        """Create Optuna optimization"""
        return OptimizationOperation(
            optimizer_type="optuna",
            model_params=param_space,
            n_trials=n_trials,
            direction=direction
        )

    @classmethod
    def neural_network_optimization(cls, architecture_params: Dict[str, Any],
                                  training_params: Dict[str, Any],
                                  framework: str = "torch",
                                  n_trials: int = 50) -> OptimizationOperation:
        """Create neural network optimization"""
        return OptimizationOperation(
            optimizer_type=framework,
            model_params=architecture_params,
            training_params=training_params,
            n_trials=n_trials
        )
