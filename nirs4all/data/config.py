"""
DatasetConfigs - Configuration and caching for dataset loading.

This module provides DatasetConfigs class that handles dataset configuration,
name resolution, loader calls, and caching to avoid reloading the same dataset.
"""

import copy
import json
import hashlib
from pathlib import Path
from tabnanny import verbose
from typing import List, Union, Dict, Any

from nirs4all.utils.emoji import CROSS
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.loaders.loader import handle_data
from nirs4all.data.config_parser import parse_config


class DatasetConfigs:

    def __init__(
        self,
        configurations: Union[Dict[str, Any], List[Dict[str, Any]], str, List[str]],
        task_type: Union[str, List[str]] = "auto"
    ):
        '''Initialize dataset configurations.

        Args:
            configurations: Dataset configuration(s) as path(s), dict(s), or list of either
            task_type: Force task type. Can be:
                      - A single string applied to all datasets
                      - A list of strings (one per dataset)
                      Valid values per dataset:
                      - 'auto': Automatic detection based on target values (default)
                      - 'regression': Force regression task
                      - 'binary_classification': Force binary classification task
                      - 'multiclass_classification': Force multiclass classification task
        '''
        user_configs = configurations if isinstance(configurations, list) else [configurations]
        self.configs = []
        for config in user_configs:
            parsed_config, dataset_name = parse_config(config)
            if parsed_config is not None:
                self.configs.append((parsed_config, dataset_name))
            else:
                print(f"{CROSS} Skipping invalid dataset config: {config}")

        self.cache: Dict[str, Any] = {}

        # Normalize task_type to a list matching configs length
        if isinstance(task_type, str):
            self._task_types = [task_type] * len(self.configs)
        else:
            if len(task_type) != len(self.configs):
                raise ValueError(
                    f"task_type list length ({len(task_type)}) must match "
                    f"number of datasets ({len(self.configs)})"
                )
            self._task_types = list(task_type)
        # print(f"✅ {len(self.configs)} dataset configuration(s).")

    def iter_datasets(self):
        for idx, (config, name) in enumerate(self.configs):
            dataset = self._get_dataset_with_task_type(config, name, self._task_types[idx])
            yield dataset

    def _get_dataset_with_task_type(self, config, name, task_type: str) -> SpectroDataset:
        """Internal method to get dataset with specific task type."""
        dataset = self._load_dataset(config, name)

        # Apply forced task type if specified (not 'auto')
        if task_type != "auto":
            dataset.set_task_type(task_type, forced=True)

        return dataset

    def _load_dataset(self, config, name) -> SpectroDataset:
        # Handle preloaded datasets
        if isinstance(config, dict) and "_preloaded_dataset" in config:
            return config["_preloaded_dataset"]

        dataset = SpectroDataset(name=name)
        if name in self.cache:
            x_train, y_train, m_train, train_headers, m_train_headers, train_unit, x_test, y_test, m_test, test_headers, m_test_headers, test_unit = self.cache[name]
        else:
            # Try to load train data
            try:
                x_train, y_train, m_train, train_headers, m_train_headers, train_unit = handle_data(config, "train")
            except (ValueError, FileNotFoundError) as e:
                if "x_path is None" in str(e) or "train_x" in str(e):
                    x_train, y_train, m_train, train_headers, m_train_headers, train_unit = None, None, None, None, None, None
                else:
                    raise

            # Try to load test data
            try:
                x_test, y_test, m_test, test_headers, m_test_headers, test_unit = handle_data(config, "test")
            except (ValueError, FileNotFoundError) as e:
                if "x_path is None" in str(e) or "test_x" in str(e):
                    x_test, y_test, m_test, test_headers, m_test_headers, test_unit = None, None, None, None, None, None
                else:
                    raise

            self.cache[name] = (x_train, y_train, m_train, train_headers, m_train_headers, train_unit, x_test, y_test, m_test, test_headers, m_test_headers, test_unit)

        # Add samples and targets only if they exist
        train_count = 0
        test_count = 0

        if x_train is not None:
            dataset.add_samples(x_train, {"partition": "train"}, headers=train_headers, header_unit=train_unit)
            train_count = len(x_train) if not isinstance(x_train, list) else len(x_train[0])
            if y_train is not None:
                dataset.add_targets(y_train)
            if m_train is not None:
                dataset.add_metadata(m_train, headers=m_train_headers)

        if x_test is not None:
            dataset.add_samples(x_test, {"partition": "test"}, headers=test_headers, header_unit=test_unit)
            test_count = len(x_test) if not isinstance(x_test, list) else len(x_test[0])
            if y_test is not None:
                dataset.add_targets(y_test)
            if m_test is not None:
                dataset.add_metadata(m_test, headers=m_test_headers)

        # print(f"📊 Loaded dataset '{dataset.name}' with {train_count} training and {test_count} test samples.")
        return dataset

    def get_dataset(self, config, name) -> SpectroDataset:
        """Get dataset by config and name (backward compatible).

        Note: When called directly, uses the first task_type (or 'auto' if single dataset).
        For proper per-dataset task_type handling, use iter_datasets() or get_dataset_at().
        """
        # Find the index of this config to get the right task_type
        for idx, (cfg, cfg_name) in enumerate(self.configs):
            if cfg_name == name:
                return self._get_dataset_with_task_type(config, name, self._task_types[idx])
        # Fallback: load without forced task_type
        return self._load_dataset(config, name)

    def get_dataset_at(self, index) -> SpectroDataset:
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Dataset index {index} out of range. Available datasets: 0 to {len(self.configs)-1}.")
        config, name = self.configs[index]
        return self._get_dataset_with_task_type(config, name, self._task_types[index])

    def get_datasets(self) -> List[SpectroDataset]:
        return list(self.iter_datasets())

