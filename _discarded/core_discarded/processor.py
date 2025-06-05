# processor.py

import numpy as np
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, clone
import importlib
import inspect
import hashlib
from typing import Any, List, Tuple, Optional, Dict
import logging
import zlib

from ..data.dataset import Dataset
from ..data_splitters import run_splitter
from .model.model_builder_factory import ModelBuilderFactory
from .model.model_manager import ModelManagerFactory

def instantiate_class(class_name: str, params: dict) -> Any:
    """
    Instantiate a class given its fully qualified name and parameters.

    Args:
        class_name (str): The fully qualified class name (e.g., 'module.submodule.ClassName').
        params (dict): A dictionary of parameters to pass to the class constructor.

    Returns:
        Any: An instance of the specified class.

    Raises:
        ValueError: If the class cannot be found or instantiated.
    """
    try:
        components = class_name.split('.')
        module_name = '.'.join(components[:-1])
        class_name_only = components[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name_only)
        return class_(**params)
    except Exception as e:
        raise ValueError(f"Class {class_name} not found: {str(e)}") from e


def get_transformer(config: Any) -> TransformerMixin:
    """
    Get a transformer instance from the config.

    Args:
        config (Any): Configuration for the transformer, can be an instance, a class, a string,
                      a tuple, or a dict.

    Returns:
        TransformerMixin: An instance of a transformer.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if isinstance(config, TransformerMixin):
        return config
    elif isinstance(config, tuple):
        class_def, params = config
        if isinstance(class_def, str):
            return instantiate_class(class_def, params)
        elif inspect.isclass(class_def):
            return class_def(**params)
        else:
            raise ValueError("Invalid transformer configuration in tuple.")
    elif isinstance(config, str):
        return instantiate_class(config, {})
    elif isinstance(config, dict):
        return instantiate_class(config['class'], config.get('params', {}))
    elif inspect.isclass(config):
        return config()
    else:
        raise ValueError(f"Invalid transformer configuration: {config}")


def run_pipeline(
    dataset: Dataset,
    x_pipeline: Any,
    y_pipeline: Any,
    logger: Any,
    cache: Optional[Dict[str, np.ndarray]] = None
) -> Dataset:
    """
    Run the pipeline on the dataset.

    Args:
        dataset (Dataset): The dataset to process.
        x_pipeline (Any): The pipeline configuration for x data.
        y_pipeline (Any): The pipeline configuration for y data.
        logger (Any): Logger object for logging.

    Returns:
        Dataset: The processed dataset.
    """
    dataset = exec_step_x(dataset, x_pipeline, logger, "", fit_test=False, cache=cache)
    dataset = exec_step_y(dataset, y_pipeline, fit_test=False)
    return dataset


def exec_step_y(
    dataset: Dataset,
    step: Any,
    fit_test: bool = False
) -> Dataset:
    """
    Execute the pipeline step for y data.

    Args:
        dataset (Dataset): The dataset containing y data.
        step (Any): The pipeline step to execute.
        fit_test (bool, optional): Whether to fit on test data. Defaults to False.

    Returns:
        Dataset: The dataset with transformed y data.

    Raises:
        NotImplementedError: If step is a dict or list, as augmentation is not authorized for y data.
        ValueError: If step format is invalid.
    """
    train_data, test_data = dataset.raw_y_train, dataset.raw_y_test
    if isinstance(step, dict):
        raise NotImplementedError("Sample and feature augmentation not authorized for y data.")
    elif isinstance(step, list):
        raise NotImplementedError("Sequential pipeline not authorized for y data.")
    elif isinstance(step, TransformerMixin) or isinstance(step, (str, tuple, dict)):
        transformer = get_transformer(step)
        if not fit_test:
            train_data = train_data if transformer is None else transformer.fit_transform(train_data)
            test_data = test_data if transformer is None else transformer.transform(test_data)
        else:
            combined_data = np.concatenate((train_data, test_data), axis=0)
            transformer.fit(combined_data)
            train_data = transformer.transform(train_data)
            test_data = transformer.transform(test_data)

        dataset.y_train, dataset.y_test, dataset.y_transformer = train_data, test_data, transformer
        return dataset
    elif step is None:
        return dataset
    else:
        raise ValueError(f"Invalid step format for y data: {step}")


def exec_step_x(
    dataset: Dataset,
    step: Any,
    logger: Any,
    indent: str,
    fit_test: bool = False,
    skip_test: bool = False,
    cache: Optional[Dict[str, np.ndarray]] = None
) -> Dataset:
    """
    Execute the pipeline step for x data.

    Args:
        dataset (Dataset): The dataset containing x data.
        step (Any): The pipeline step to execute.
        logger (Any): Logger object for logging.
        indent (str): Indentation string for logging.
        fit_test (bool, optional): Whether to fit on test data. Defaults to False.
        skip_test (bool, optional): Whether to skip processing test data. Defaults to False.
        cache (Optional[Dict[str, np.ndarray]], optional): Cache for storing transformed data. Defaults to None.

    Returns:
        Dataset: The dataset with transformed x data.

    Raises:
        ValueError: If the step format is invalid.
    """
    if cache is None:
        cache = {}

    train_data, test_data = dataset.raw_x_train, dataset.raw_x_test
    n_augmentations, n_samples, n_transformations, n_features = train_data.shape

    if isinstance(step, dict):
        sample_key = next((k for k in step if k in ["s", "samples", "sample_augmentation", "S"]), None)
        feature_key = next((k for k in step if k in ["f", "features", "feature_augmentation", "F", "parallel"]), None)
        split_key = next((k for k in step if k in ["split", "spl", "splitter", "SPL"]), None)

        if sample_key is not None:
            balanced = step.get("balance", False)
            sample_transformers = step[sample_key]

            if balanced:
                # Balanced augmentation
                y_train = dataset.raw_y_train
                classes, class_counts = np.unique(y_train, return_counts=True)
                max_count = np.max(class_counts)
                augmented_data_list = []
                augmented_labels_list = []

                for cls, count in zip(classes, class_counts):
                    num_needed = max_count - count
                    if num_needed > 0:
                        # Get indices of samples belonging to this class
                        class_indices = np.where(y_train.flatten() == cls)[0]
                        num_class_samples = len(class_indices)
                        repeats_per_sample = int(np.ceil(num_needed / num_class_samples))

                        for idx in class_indices:
                            sample_data = dataset.raw_x_train[:, idx:idx+1, :, :]
                            sample_label = y_train[idx:idx+1]

                            for _ in range(repeats_per_sample):
                                for st in sample_transformers:
                                    augmented_dataset = Dataset(_x_train=sample_data)
                                    augmented_dataset = exec_step_x(
                                        augmented_dataset, st, logger, indent + " ", fit_test, True, cache
                                    )
                                    augmented_data_list.append(augmented_dataset.raw_x_train)
                                    augmented_labels_list.append(sample_label)
                                    num_needed -= 1
                                    if num_needed <= 0:
                                        break
                                if num_needed <= 0:
                                    break
                            if num_needed <= 0:
                                break

                #print shape and number of element per class ({class: number of elements})
                print(dataset.raw_x_train.shape, dataset.raw_y_train.shape)
                print({cls: len(np.where(y_train.flatten() == cls)[0]) for cls in classes})
                print({cls: len(np.where(np.concatenate(augmented_labels_list).flatten() == cls)[0]) for cls in classes})
                if augmented_data_list:
                    # Concatenate augmented data and labels
                    augmented_data = np.concatenate(augmented_data_list, axis=1)
                    augmented_labels = np.concatenate(augmented_labels_list, axis=0)
                    # Update dataset with augmented data
                    dataset.raw_x_train = np.concatenate([dataset.raw_x_train, augmented_data], axis=1)
                    dataset._y_train = np.concatenate([dataset.raw_y_train, augmented_labels], axis=0)
                print(dataset.raw_x_train.shape, dataset.raw_y_train.shape)
                print({cls: len(np.where(dataset.y_train_().flatten() == cls)[0]) for cls in classes})
                # print({cls: len(np.where(np.concatenate(augmented_labels_list).flatten() == cls)[0]) for cls in classes})    
                
                return dataset
            else:
                # Original behavior (uniform augmentation)
                num_transformers = len(sample_transformers)
                train_data = np.repeat(train_data, num_transformers, axis=0)

                def process_sample_transformer(i_st):
                    i, st = i_st
                    aug_range = range(i * n_augmentations, (i + 1) * n_augmentations)
                    aug_data = train_data[aug_range, :, :, :]
                    augmented_dataset = Dataset(_x_train=aug_data)
                    augmented_dataset = exec_step_x(
                        augmented_dataset, st, logger, indent + " ", fit_test, True, cache
                    )
                    return (aug_range, augmented_dataset.raw_x_train)

                results = [process_sample_transformer((i, st)) for i, st in enumerate(sample_transformers)]

                for aug_range, aug_train_data in results:
                    train_data[aug_range, :, :, :] = aug_train_data

                dataset.raw_x_train = train_data
                return dataset

        elif feature_key is not None:
            # Feature augmentation
            # logger.info(indent + "Feature augmentation")
            feature_transformers = step[feature_key]
            num_transformers = len(feature_transformers)
            train_data = np.repeat(train_data, num_transformers, axis=2)
            if not skip_test:
                test_data = np.repeat(test_data, num_transformers, axis=2)

            def process_feature_transformer(i_ft):
                # logger.info(indent + f"Processing feature transformer {i_ft}")
                i, ft = i_ft
                tr_range = range(i * n_transformations, (i + 1) * n_transformations)
                tr_train_data = train_data[:, :, tr_range, :]
                tr_test_data = test_data[:, :, tr_range, :] if not skip_test else None

                transformed_dataset = Dataset(_x_train=tr_train_data, _x_test=tr_test_data)
                transformed_dataset = exec_step_x(
                    transformed_dataset, ft, logger, indent + " ", fit_test, skip_test, cache
                )
                return (tr_range, transformed_dataset.raw_x_train, transformed_dataset.raw_x_test)

            # Parallel execution
            # results = Parallel(n_jobs=-1, backend='loky')(
                # delayed(process_feature_transformer)((i, ft)) for i, ft in enumerate(feature_transformers)
            # )
            # Sequential execution
            results = [process_feature_transformer((i, ft)) for i, ft in enumerate(feature_transformers)]

            for tr_range, tr_train_data, tr_test_data in results:
                train_data[:, :, tr_range, :] = tr_train_data
                if not skip_test and tr_test_data is not None:
                    test_data[:, :, tr_range, :] = tr_test_data

            dataset.raw_x_train = train_data
            if not skip_test:
                dataset.raw_x_test = test_data
            return dataset

        elif split_key is not None:
            # Splitting dataset
            # logger.info(indent + "Splitting dataset")
            splitter_config = step[split_key]
            dataset.folds = run_splitter(splitter_config, dataset)
            return dataset

        elif "class" in step:
            transformer = get_transformer(step)
            return apply_step_x(dataset, transformer, logger, indent, fit_test, skip_test, cache)

        else:
            raise ValueError(f"Invalid step format in dict: {step}")

    elif isinstance(step, list):
        # Sequential pipeline
        # logger.info(indent + "Sequential pipeline")
        for s in step:
            dataset = exec_step_x(dataset, s, logger, indent + " ", fit_test, skip_test, cache)
        return dataset

    elif isinstance(step, TransformerMixin) or isinstance(step, (str, tuple, dict)):
        transformer = get_transformer(step)
        return apply_step_x(dataset, transformer, logger, indent, fit_test, skip_test, cache)

    elif inspect.isclass(step):
        transformer = get_transformer(step)
        return apply_step_x(dataset, transformer, logger, indent, fit_test, skip_test, cache)

    elif step is None:
        # Identity transformation
        # logger.info(indent + "Identity transformation")
        return dataset

    else:
        raise ValueError(f"Invalid step format: {step}")


def apply_step_x(
    dataset: Dataset,
    transformer: TransformerMixin,
    logger: Any,
    indent: str,
    fit_test: bool = False,
    skip_test: bool = False,
    cache: Optional[Dict[str, np.ndarray]] = None
) -> Dataset:
    """
    Apply a single transformer to the dataset.

    Args:
        dataset (Dataset): The dataset containing x data.
        transformer (TransformerMixin): The transformer to apply.
        logger (Any): Logger object for logging.
        indent (str): Indentation string for logging.
        fit_test (bool, optional): Whether to fit on test data. Defaults to False.
        skip_test (bool, optional): Whether to skip processing test data. Defaults to False.
        cache (Optional[Dict[str, np.ndarray]], optional): Cache for storing transformed data. Defaults to None.

    Returns:
        Dataset: The dataset with transformed x data.
    """
    logger = logging.getLogger(__name__)  # Ensure logging inside parallel workers
    logger.info(f"{indent}Processing ____")

    if cache is None:
        cache = {}
    
    train_data, test_data = dataset.raw_x_train, dataset.raw_x_test
    n_augmentations, n_samples, n_transformations, n_features = train_data.shape

    logger.info(indent + f"Applying transformer: {transformer.__class__.__name__}")

    # Generate transformer ID
    transformer_id = hashlib.md5(str(transformer).encode()).hexdigest()

    # Apply the transformer in parallel over transformations
    def process_transformation(transformation_index):
        logger = logging.getLogger('ExperimentRunner')
    
        train_slice = train_data[:, :, transformation_index, :]
        test_slice = test_data[:, :, transformation_index, :] if not skip_test else None
        
        # Clone the transformer to avoid parallel fitting conflicts
        # local_transformer = clone(transformer)
        local_transformer = transformer
        
        # Generate data IDs
        # train_data_id = hashlib.md5(train_slice.tobytes()).hexdigest()
        train_data_id = hash_data_crc32(train_slice.tobytes())
        # test_data_id = hashlib.md5(test_slice.tobytes()).hexdigest() if test_slice is not None else None
        test_data_id = hash_data_crc32(test_slice.tobytes()) if test_slice is not None else None
        
        # logger.info(indent + f"Processing transformation {transformation_index}: {transformer.__class__.__name__}")
        # logger.info(indent + f"Train data ID: {train_data_id}, Test data ID: {test_data_id}, cache keys: {cache.keys()}")

        # Check cache for transformed data
        cache_key_train = f"{train_data_id}_{transformer_id}"
        cache_key_test = f"{test_data_id}_{transformer_id}" if test_data_id else None

        if cache_key_train in cache:
            transformed_train = cache[cache_key_train]
            # logger.info(indent + f"Cache hit for train data at transformation {transformation_index}: {transformer.__class__.__name__}")
        else:
            # Reshape data for transformer
            train_data_view = train_slice.reshape(-1, n_features)
            if not fit_test:
                local_transformer.fit(train_data_view)
                transformed_train = local_transformer.transform(train_data_view)
            else:
                combined_data = train_data_view
                if test_slice is not None:
                    test_data_view = test_slice.reshape(-1, n_features)
                    combined_data = np.concatenate((train_data_view, test_data_view), axis=0)
                local_transformer.fit(combined_data)
                transformed_train = local_transformer.transform(train_data_view)
            transformed_train = transformed_train.reshape(n_augmentations, n_samples, n_features)
            cache[cache_key_train] = transformed_train

        if not skip_test and test_slice is not None:
            if cache_key_test in cache:
                transformed_test = cache[cache_key_test]
                # logger.info(indent + f"Cache hit for test data at transformation {transformation_index}")
            else:
                test_data_view = test_slice.reshape(-1, n_features)
                transformed_test = local_transformer.transform(test_data_view)
                transformed_test = transformed_test.reshape(test_slice.shape)
                cache[cache_key_test] = transformed_test
        else:
            transformed_test = None

        return (transformation_index, transformed_train, transformed_test)

    # Parallel execution
    # results = Parallel(n_jobs=4)(
        # delayed(process_transformation)(transformation_index) for transformation_index in range(n_transformations)
    # )
    # Sequential execution
    results = [process_transformation(transformation_index) for transformation_index in range(n_transformations)]

    for transformation_index, transformed_train, transformed_test in results:
        train_data[:, :, transformation_index, :] = transformed_train
        if not skip_test and transformed_test is not None:
            test_data[:, :, transformation_index, :] = transformed_test

    dataset.raw_x_train = train_data
    if not skip_test:
        dataset.raw_x_test = test_data
    return dataset


def hash_data_crc32(data):
    return zlib.crc32(data) & 0xffffffff  # Mask to ensure 32-bit output


class Processor:
    """
    Data processor class that handles the entire data pipeline.
    """
    
    def __init__(self, config=None, logger=None):
        """
        Initialize the processor with configuration.
        
        Args:
            config (dict): Configuration for the processor.
            logger: Logger instance.
        """
        self.config = config or {}
        self.logger = logger
        self.cache = {}
    
    def process_dataset(self, dataset):
        """
        Process the dataset through the pipeline.
        
        Args:
            dataset (Dataset): The dataset to process.
            
        Returns:
            Dataset: The processed dataset.
        """
        x_pipeline = self.config.get('x_pipeline')
        y_pipeline = self.config.get('y_pipeline')
        
        # Save categorical information before processing
        y_train_categorical_info = dataset.y_train_categorical_info
        y_test_categorical_info = dataset.y_test_categorical_info
        
        # Process data through pipeline
        processed_dataset = run_pipeline(dataset, x_pipeline, y_pipeline, self.logger, self.cache)
        
        # Restore categorical information after processing
        if y_train_categorical_info:
            processed_dataset.y_train_categorical_info = y_train_categorical_info
            
        if y_test_categorical_info:
            processed_dataset.y_test_categorical_info = y_test_categorical_info
            
        return processed_dataset
    
    def inverse_transform_predictions(self, y_pred, dataset):
        """
        Convert predictions back to original format, including categorical.
        
        Args:
            y_pred (numpy.ndarray): Predictions to transform
            dataset (Dataset): Dataset with transformation information
            
        Returns:
            numpy.ndarray: Transformed predictions
        """
        return dataset.inverse_transform(y_pred)

