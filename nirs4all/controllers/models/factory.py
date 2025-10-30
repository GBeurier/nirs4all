"""
Model Factory - Framework-specific model instantiation for controllers

This module provides a factory for building machine learning models from various configurations.
Relocated from utils/model_builder.py to be co-located with model controllers.

Supports multiple input formats:
- String: file path or class path
- Dict: configuration with class, import, or function keys
- Instance: pre-built model object
- Callable: function or class to instantiate
"""

import os
import importlib
import inspect

from nirs4all.utils.backend import TF_AVAILABLE, TORCH_AVAILABLE


class ModelFactory:
    """Factory class for building machine learning models from various configurations.

    Supports multiple input formats:
    - String: file path or class path
    - Dict: configuration with class, import, or function keys
    - Instance: pre-built model object
    - Callable: function or class to instantiate

    Also provides helper utilities for controllers:
    - filter_params: Filter parameters to match callable signature
    - compute_input_shape: Compute input shape from dataset
    - detect_framework: Detect framework from model instance
    - get_num_classes: Extract number of classes for classification
    """

    @staticmethod
    def filter_params(callable_obj, params: dict) -> dict:
        """Filter parameters to only those accepted by callable's signature.

        Args:
            callable_obj: Class or function to check signature
            params: Dictionary of parameters

        Returns:
            Filtered parameters matching signature
        """
        if inspect.isclass(callable_obj):
            sig = inspect.signature(callable_obj.__init__)
        else:
            sig = inspect.signature(callable_obj)

        # Check for **kwargs
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values())

        if has_kwargs:
            return params  # Accept all if **kwargs present

        valid_params = {name for name in sig.parameters if name != 'self'}
        return {k: v for k, v in params.items() if k in valid_params}

    @staticmethod
    def compute_input_shape(dataset, framework: str):
        """Compute input shape from dataset based on framework.

        Args:
            dataset: Dataset with X array
            framework: Framework name ('sklearn', 'tensorflow', 'pytorch', etc.)

        Returns:
            Input shape appropriate for framework (int or tuple)
        """
        # Get X data with correct layout
        from nirs4all.data.feature_components import FeatureLayout

        context = {'partition': 'train'}
        # Use VOLUME_3D_TRANSPOSE for TensorFlow to match Conv1D layout
        layout = FeatureLayout.VOLUME_3D_TRANSPOSE.value if framework == 'tensorflow' else FeatureLayout.FLAT_2D.value

        try:
            X = dataset.x(context, layout=layout, concat_source=True)
            if framework in ['sklearn', 'xgboost', 'lightgbm']:
                return X.shape[1]  # Flat features
            elif framework in ['tensorflow', 'pytorch']:
                return X.shape[1:]  # Tuple shape for neural networks
            else:
                return X.shape[1]
        except Exception as e:
            raise ValueError(f"Could not determine input shape from dataset: {str(e)}")

    @staticmethod
    def get_num_classes(dataset) -> int:
        """Extract number of classes for classification tasks.

        Args:
            dataset: Dataset with y labels and task type

        Returns:
            Number of unique classes, or None if not classification
        """
        if hasattr(dataset, 'is_classification') and dataset.is_classification:
            if hasattr(dataset, 'num_classes'):
                return dataset.num_classes
            elif hasattr(dataset, 'n_classes'):
                return dataset.n_classes
            else:
                # Fallback: compute from y
                import numpy as np
                context = {'partition': 'train'}
                y = dataset.y(context)
                return len(np.unique(y))
        return None

    @staticmethod
    def build_single_model(model_config, dataset, force_params={}):
        """Build a single model from the given configuration.

        Args:
            model_config: Configuration for the model. Can be a string (file path or class path),
                dict (with 'class', 'import', or 'function' keys), instance, or callable.
            dataset: Dataset object used to determine input dimensions and classification settings.
            force_params: Dictionary of parameters to force override on the model.

        Returns:
            The built model instance.

        Raises:
            ValueError: If the model_config format is invalid.
        """
        if hasattr(dataset, 'is_classification') and dataset.is_classification:
            if hasattr(dataset, 'num_classes'):
                force_params['num_classes'] = dataset.num_classes

        if isinstance(model_config, str):
            return ModelFactory._from_string(model_config, force_params)

        elif isinstance(model_config, dict):
            return ModelFactory._from_dict(model_config, dataset, force_params)

        elif hasattr(model_config, '__class__') and not inspect.isclass(model_config) and not inspect.isfunction(model_config):
            return ModelFactory._from_instance(model_config)

        elif callable(model_config):
            return ModelFactory._from_callable(model_config, dataset, force_params)

        else:
            raise ValueError("Invalid model_config format.")

    @staticmethod
    def _from_string(model_str, force_params=None):
        """Build a model from a string configuration.

        Args:
            model_str: String representing either a file path or a class path.
            force_params: Dictionary of parameters to force override.

        Returns:
            The built model instance.

        Raises:
            ValueError: If the string format is invalid.
        """
        if os.path.exists(model_str):
            model = ModelFactory._load_model_from_file(model_str)
            if force_params is not None:
                model = ModelFactory.reconstruct_object(model, force_params)
            return model
        else:
            try:
                cls = ModelFactory.import_class(model_str)
                model = ModelFactory.prepare_and_call(cls, force_params)
                return model
            except Exception as e:
                raise ValueError(f"Invalid model string format: {str(e)}") from e

    @staticmethod
    def _from_instance(model_instance, force_params=None):
        """Build a model from an existing instance.

        Args:
            model_instance: Pre-built model instance.
            force_params: Dictionary of parameters to force override.

        Returns:
            The model instance, possibly reconstructed with force_params.
        """
        if force_params is not None:
            model_instance = ModelFactory.reconstruct_object(model_instance, force_params)
        return model_instance

    @staticmethod
    def _from_dict(model_dict, dataset, force_params=None):
        """Build a model from a dictionary configuration.

        Args:
            model_dict: Dictionary containing model configuration.
                Can have 'class', 'import', 'function', 'model_instance', or new 'type'/'func' keys.
            dataset: Dataset object for input dimensions.
            force_params: Dictionary of parameters to force override.

        Returns:
            The built model instance.

        Raises:
            ValueError: If the dict format is invalid.
        """
        # Handle new dict format from deserialize_component (no _runtime_instance)
        if 'type' in model_dict and model_dict['type'] == 'function':
            # New format: {"type": "function", "func": <func>, "framework": "tensorflow", "params": {...}}
            callable_model = model_dict['func']
            params = model_dict.get('params', {}).copy()
            framework = model_dict.get('framework', None)

            if framework is None:
                raise ValueError("Framework must be specified in function config")

            input_dim = ModelFactory.compute_input_shape(dataset, framework)
            params['input_dim'] = input_dim
            params['input_shape'] = input_dim

            # Set num_classes for tensorflow classification models
            if framework == 'tensorflow' and hasattr(dataset, 'is_classification') and dataset.is_classification:
                if hasattr(dataset, 'num_classes'):
                    params['num_classes'] = dataset.num_classes

            model = ModelFactory.prepare_and_call(callable_model, params, force_params)
            return model

        # Handle model_instance key (used by base_model._extract_model_config)
        if 'model_instance' in model_dict:
            model_obj = model_dict['model_instance']
            # Recursively build the model instance through build_single_model
            # which will detect if it's an instance, callable, etc.
            return ModelFactory.build_single_model(model_obj, dataset, force_params)

        if 'model' in model_dict:
            model_dict = model_dict['model']

        if 'class' in model_dict:
            class_path = model_dict['class']
            params = model_dict.get('params', {})
            cls = ModelFactory.import_class(class_path)
            # Filter params for sklearn models
            framework = None
            try:
                framework = ModelFactory.detect_framework(cls)
            except Exception:
                pass
            if framework == 'sklearn':
                all_params = {**params, **(force_params or {})}
                model = ModelFactory.prepare_and_call(cls, all_params)
            else:
                model = ModelFactory.prepare_and_call(cls, params, force_params)
            return model

        elif 'import' in model_dict:
            object_path = model_dict['import']
            params = model_dict.get('params', {})
            obj = ModelFactory.import_object(object_path)

            if callable(obj):  # function or class
                model = ModelFactory.prepare_and_call(obj, params, force_params)
            else:  # instance
                model = obj
                if force_params is not None:
                    model = ModelFactory.reconstruct_object(model, params, force_params)

            return model

        elif 'function' in model_dict:
            callable_model = model_dict['function']

            # If function is a string path, import it first
            if isinstance(callable_model, str):
                try:
                    mod_name, _, func_name = callable_model.rpartition(".")
                    mod = importlib.import_module(mod_name)
                    callable_model = getattr(mod, func_name)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not import function {callable_model}: {e}")

            params = model_dict.get('params', {}).copy()  # copy to avoid mutating input
            framework = model_dict.get('framework', None)
            if framework is None:
                framework = getattr(callable_model, 'framework', None)
            if framework is None:
                raise ValueError("Cannot determine framework from callable model_config. Please set 'experiments.utils.framework' decorator on the function or add 'framework' key to the config.")
            input_dim = ModelFactory.compute_input_shape(dataset, framework)
            params['input_dim'] = input_dim
            params['input_shape'] = input_dim
            # Set num_classes for tensorflow classification models
            if framework == 'tensorflow' and hasattr(dataset, 'is_classification') and dataset.is_classification:
                if hasattr(dataset, 'num_classes'):
                    num_classes = dataset.num_classes
                    params['num_classes'] = num_classes
            model = ModelFactory.prepare_and_call(callable_model, params, force_params)
            return model
        else:
            raise ValueError("Dict model_config must contain 'class', 'path', or 'callable' with 'framework' key.")

    @staticmethod
    def _from_callable(model_callable, dataset, force_params=None):
        """Build a model from a callable (function or class).

        Args:
            model_callable: Callable to instantiate the model.
            dataset: Dataset object for input dimensions.
            force_params: Dictionary of parameters to force override.

        Returns:
            The built model instance.

        Raises:
            ValueError: If framework cannot be determined.
        """
        framework = None
        if inspect.isclass(model_callable):
            framework = ModelFactory.detect_framework(model_callable)
        elif inspect.isfunction(model_callable):
            framework = getattr(model_callable, 'framework', None)
        if framework is None:
            raise ValueError("Cannot determine framework from callable model_config. Please set 'experiments.utils.framework' decorator on the callable.")

        # Use framework-specific model creation for TensorFlow
        if framework == 'tensorflow':
            return ModelFactory._from_tensorflow_callable(model_callable, dataset, force_params)

        input_dim = ModelFactory.compute_input_shape(dataset, framework)
        sig = inspect.signature(model_callable)
        params = {}
        if 'input_shape' in sig.parameters:
            params['input_shape'] = input_dim
        if 'input_dim' in sig.parameters:
            params['input_dim'] = input_dim
        # Only set num_classes for tensorflow classification models
        if framework == 'tensorflow' and hasattr(dataset, 'is_classification') and dataset.is_classification:
            if hasattr(dataset, 'num_classes'):
                num_classes = dataset.num_classes
                # Only set num_classes if the function signature has it (for classification models)
                if 'num_classes' in sig.parameters:
                    params['num_classes'] = num_classes
        model = ModelFactory.prepare_and_call(model_callable, params, force_params)
        return model

    @staticmethod
    def _from_tensorflow_callable(model_callable, dataset, force_params=None):
        """Build a TensorFlow model from a callable (function or class).

        This method handles TensorFlow-specific model creation with proper
        input shape formatting and parameter passing.

        Args:
            model_callable: TensorFlow model function or class.
            dataset: Dataset object for input dimensions.
            force_params: Dictionary of parameters to force override.

        Returns:
            Built TensorFlow model instance.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")

        input_dim = ModelFactory.compute_input_shape(dataset, 'tensorflow')
        sig = inspect.signature(model_callable)
        params = {}

        # Add input shape parameter
        if 'input_shape' in sig.parameters:
            params['input_shape'] = input_dim
        if 'input_dim' in sig.parameters:
            params['input_dim'] = input_dim

        # Add num_classes for classification
        if hasattr(dataset, 'is_classification') and dataset.is_classification:
            if 'num_classes' in sig.parameters:
                if hasattr(dataset, 'num_classes'):
                    params['num_classes'] = dataset.num_classes
                elif hasattr(dataset, 'n_classes'):
                    params['num_classes'] = dataset.n_classes

        # Call the function with prepared parameters
        model = ModelFactory.prepare_and_call(model_callable, params, force_params)

        # If it's a class, we've instantiated it
        # If it's a function, it should return a model
        if not ModelFactory._is_tensorflow_model(model):
            raise ValueError(
                f"TensorFlow model function {model_callable.__name__} did not return a valid model. "
                f"Got {type(model)} instead."
            )

        return model

    @staticmethod
    def _is_tensorflow_model(obj):
        """Check if an object is a TensorFlow/Keras model.

        Args:
            obj: Object to check.

        Returns:
            True if object is a TensorFlow model, False otherwise.
        """
        if not TF_AVAILABLE:
            return False

        try:
            from tensorflow import keras
            return isinstance(obj, (keras.Model, keras.Sequential))
        except ImportError:
            return False

    @staticmethod
    def _clone_model(model, framework):
        """Clone the model using framework-specific cloning methods.

        Args:
            model: The model instance to clone.
            framework: The framework string ('sklearn', 'tensorflow', etc.).

        Returns:
            A cloned model instance.
        """
        if framework == 'sklearn':
            from sklearn.base import clone
            return clone(model)

        elif framework == 'tensorflow':
            if TF_AVAILABLE:
                from tensorflow.keras.models import clone_model
                cloned_model = clone_model(model)
                return cloned_model

        else:
            # Fallback to deepcopy
            from copy import deepcopy
            return deepcopy(model)

    @staticmethod
    def import_class(class_path):
        """Import a class from a module path.

        Args:
            class_path: String path like 'module.submodule.ClassName'.

        Returns:
            The imported class.

        Raises:
            ImportError: If TensorFlow or PyTorch is required but not available.
        """
        module_name, class_name = class_path.rsplit('.', 1)
        if module_name.startswith('tensorflow'):
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available but required to load this model.")
        elif module_name.startswith('torch'):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available but required to load this model.")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    @staticmethod
    def import_object(object_path):
        """Import an object from a module path.

        Args:
            object_path: String path like 'module.submodule.object_name'.

        Returns:
            The imported object.

        Raises:
            ImportError: If TensorFlow or PyTorch is required but not available.
        """
        module_name, object_name = object_path.rsplit('.', 1)
        if module_name.startswith('tensorflow'):
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available but required to load this model.")
        elif module_name.startswith('torch'):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available but required to load this model.")
        module = importlib.import_module(module_name)
        obj = getattr(module, object_name)
        return obj

    @staticmethod
    def detect_framework(model) -> str:
        """Detect the framework from the model instance.

        This is a helper for controllers to determine framework type.
        Returns simplified framework names for routing logic.

        Args:
            model: Model instance or class.

        Returns:
            String representing the framework ('sklearn', 'tensorflow', 'pytorch',
            'xgboost', 'lightgbm', 'catboost', or 'unknown').
        """
        # Special case for mocked objects in tests
        if hasattr(model, '_mock_name') or str(type(model)).startswith("<class 'unittest.mock."):
            return 'sklearn'  # By default, consider mocks as sklearn objects

        # Check for explicit framework attribute (e.g., from @framework decorator)
        if hasattr(model, 'framework'):
            return model.framework

        # Inspect module path
        if inspect.isclass(model):
            module = model.__module__
        else:
            module = model.__class__.__module__

        if 'sklearn' in module:
            return 'sklearn'
        elif 'tensorflow' in module or 'keras' in module:
            return 'tensorflow'
        elif 'torch' in module:
            return 'pytorch'
        elif 'xgboost' in module:
            return 'xgboost'
        elif 'lightgbm' in module:
            return 'lightgbm'
        elif 'catboost' in module:
            return 'catboost'
        else:
            return 'unknown'

    @staticmethod
    def _force_param_on_instance(model, force_params):
        """Force parameters on an existing model instance by reconstructing it.

        Args:
            model: Model instance to modify.
            force_params: Dictionary of parameters to force.

        Returns:
            New model instance with forced parameters.
        """
        try:
            filtered_params = ModelFactory.filter_params(model, force_params)
            new_model = model.__class__(**filtered_params)
            return new_model
        except Exception as e:
            print(f"Warning: Cannot force parameters on the model instance. Reason: {e}")
            return model

    @staticmethod
    def prepare_and_call(callable_obj, params_from_caller=None, force_params_from_caller=None):
        """Prepare arguments and call a callable with proper parameter handling.

        Args:
            callable_obj: The callable to invoke.
            params_from_caller: Parameters from the caller.
            force_params_from_caller: Force parameters that take precedence.

        Returns:
            The result of calling the callable.

        Raises:
            TypeError: If arguments cannot be bound to the callable.
        """
        if params_from_caller is None:
            params_from_caller = {}
        if force_params_from_caller is None:
            force_params_from_caller = {}

        all_available_args_from_caller = {**params_from_caller, **force_params_from_caller}

        signature = inspect.signature(callable_obj)
        sig_params_spec = signature.parameters

        final_named_args = {}
        remaining_args_for_bundle_or_kwargs = {}

        has_params_bundle_arg = 'params' in sig_params_spec and \
                                sig_params_spec['params'].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params_spec.values())

        # Only keep arguments that are in the callable's signature, unless **kwargs is present
        for name, value in all_available_args_from_caller.items():
            if name in sig_params_spec and (name != 'params' or not has_params_bundle_arg):
                final_named_args[name] = value
            else:
                remaining_args_for_bundle_or_kwargs[name] = value

        # If both params bundle and **kwargs are present, prioritize bundling all extras into params
        if has_params_bundle_arg:
            params_bundle_dict = {}
            if 'params' in remaining_args_for_bundle_or_kwargs and \
               isinstance(remaining_args_for_bundle_or_kwargs['params'], dict):
                params_bundle_dict = remaining_args_for_bundle_or_kwargs.pop('params')
            params_bundle_dict.update(remaining_args_for_bundle_or_kwargs)
            final_named_args['params'] = params_bundle_dict
            # Do not pass any extras to **kwargs if params is present
            if has_kwargs:
                # Remove any keys that would have gone to **kwargs
                for k in list(final_named_args.keys()):
                    if k not in sig_params_spec and k != 'params':
                        del final_named_args[k]
        elif has_kwargs:
            # If only **kwargs, add all remaining
            final_named_args.update(remaining_args_for_bundle_or_kwargs)
        # else: already filtered

        # Filter out any keys not in signature if no **kwargs
        if not has_kwargs:
            final_named_args = {k: v for k, v in final_named_args.items() if k in sig_params_spec or (has_params_bundle_arg and k == 'params')}

        try:
            bound_args = signature.bind(**final_named_args)
        except TypeError as e:
            detailed_error_message = (
                f"Error binding arguments for callable '{getattr(callable_obj, '__name__', str(callable_obj))}': {e}.\n"
                f"  Attempted to call with (processed arguments): {final_named_args}\n"
                f"  Original available arguments from caller: {all_available_args_from_caller}\n"
                f"  Callable signature: {signature}"
            )
            raise TypeError(detailed_error_message) from e

        return callable_obj(*bound_args.args, **bound_args.kwargs)

    @staticmethod
    def reconstruct_object(obj, params=None, force_params=None):
        """Reconstruct an object using its current attributes as default values.

        Args:
            obj: The object to be reconstructed.
            params: Dictionary of parameters to overwrite the object's current parameters.
            force_params: Dictionary of parameters that take precedence over both the object's
                current parameters and params.

        Returns:
            A new instance of the object with the updated parameters.

        Raises:
            TypeError: If a required parameter is missing and no default value is provided.
        """
        if params is None:
            params = {}
        if force_params is None:
            force_params = {}

        merged_params = {**params, **force_params}

        cls = obj.__class__
        signature = inspect.signature(cls)
        current_params = obj.__dict__.copy()  # This assumes the object stores its state in __dict__

        final_args = {}

        for name, param in signature.parameters.items():
            if name == 'self':  # Skip 'self'
                continue

            if name in force_params:
                final_args[name] = force_params[name]
            elif name in params:
                final_args[name] = params[name]
            elif name in current_params:
                final_args[name] = current_params[name]
            elif param.default is not inspect.Parameter.empty:
                final_args[name] = param.default
            elif name == "params" or name == "force_params":
                final_args[name] = merged_params
            else:
                raise TypeError(f"Missing required parameter: '{name}'")

        return cls(**final_args)

    @staticmethod
    def _load_model_from_file(model_path):
        """Load a model from a file path.

        Args:
            model_path: Path to the model file.

        Returns:
            The loaded model instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file extension is unsupported.
            ImportError: If required libraries are not available.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
        _, ext = os.path.splitext(model_path)

        # TensorFlow model
        if ext in ['.h5', '.hdf5', '.keras']:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available but required to load this model.")
            from tensorflow import keras

            # Pass custom objects if needed
            custom_objects = {}

            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            return model

        # PyTorch model
        elif ext == '.pt':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available but required to load this model.")
            import torch
            model = torch.load(model_path)
            return model

        # Sklearn model or pickled model
        elif ext == '.pkl':
            from nirs4all.pipeline.artifact_serialization import from_bytes
            with open(model_path, 'rb') as f:
                data = f.read()
            # Use cloudpickle format for compatibility
            model = from_bytes(data, 'cloudpickle')
            return model

        else:
            raise ValueError(f"Unsupported file extension '{ext}' for model file.")
