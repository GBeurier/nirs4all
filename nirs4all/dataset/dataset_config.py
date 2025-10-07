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
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.dataset.loader import handle_data
from nirs4all.dataset.dataset_config_parser import parse_config


class DatasetConfigs:

    def __init__(self, configurations: Union[Dict[str, Any], List[Dict[str, Any]], str, List[str]]):
        user_configs = configurations if isinstance(configurations, list) else [configurations]
        self.configs = []
        for config in user_configs:
            parsed_config, dataset_name = parse_config(config)
            if parsed_config is not None:
                self.configs.append((parsed_config, dataset_name))
            else:
                print(f"❌ Skipping invalid dataset config: {config}")

        self.cache: Dict[str, Any] = {}
        # print(f"✅ {len(self.configs)} dataset configuration(s).")

    def iter_datasets(self):
        for config, name in self.configs:
            dataset = self.get_dataset(config, name)
            yield dataset

    def get_dataset(self, config, name) -> SpectroDataset:
        dataset = SpectroDataset(name=name)
        if name in self.cache:
            x_train, y_train, train_headers, x_test, y_test, test_headers = self.cache[name]
        else:
            # Try to load train data
            try:
                x_train, y_train, train_headers = handle_data(config, "train")
            except (ValueError, FileNotFoundError) as e:
                if "x_path is None" in str(e) or "train_x" in str(e):
                    x_train, y_train, train_headers = None, None, None
                else:
                    raise

            # Try to load test data
            try:
                x_test, y_test, test_headers = handle_data(config, "test")
            except (ValueError, FileNotFoundError) as e:
                if "x_path is None" in str(e) or "test_x" in str(e):
                    x_test, y_test, test_headers = None, None, None
                else:
                    raise

            self.cache[name] = (x_train, y_train, train_headers, x_test, y_test, test_headers)

        # Add samples and targets only if they exist
        train_count = 0
        test_count = 0

        if x_train is not None:
            dataset.add_samples(x_train, {"partition": "train"}, headers=train_headers)
            train_count = len(x_train) if not isinstance(x_train, list) else len(x_train[0])
            if y_train is not None:
                dataset.add_targets(y_train)

        if x_test is not None:
            dataset.add_samples(x_test, {"partition": "test"}, headers=test_headers)
            test_count = len(x_test) if not isinstance(x_test, list) else len(x_test[0])
            if y_test is not None:
                dataset.add_targets(y_test)

        # print(f"📊 Loaded dataset '{dataset.name}' with {train_count} training and {test_count} test samples.")
        return dataset

    def get_dataset_at(self, index) -> SpectroDataset:
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Dataset index {index} out of range. Available datasets: 0 to {len(self.configs)-1}.")
        config, name = self.configs[index]
        dataset = self.get_dataset(config, name)
        return dataset

    def get_datasets(self) -> List[SpectroDataset]:
        return list(self.iter_datasets())
