"""
DatasetConfigs - Configuration and caching for dataset loading.

This module provides DatasetConfigs class that handles dataset configuration,
name resolution, loader calls, and caching to avoid reloading the same dataset.
"""

import copy
import json
import hashlib
from pathlib import Path
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
            x_train, y_train, x_test, y_test = self.cache[name]
        else:
            x_train, y_train = handle_data(config, "train")
            x_test, y_test = handle_data(config, "test")
            self.cache[name] = (x_train, y_train, x_test, y_test)

        dataset.add_samples(x_train, {"partition": "train"})
        dataset.add_samples(x_test, {"partition": "test"})
        dataset.add_targets(y_train)
        dataset.add_targets(y_test)
        print(f"✅ Loaded dataset '{dataset.name}' with {len(x_train)} training and {len(x_test)} test samples.")
        return dataset

    def get_dataset_at(self, index) -> SpectroDataset:
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Dataset index {index} out of range. Available datasets: 0 to {len(self.configs)-1}.")
        config, name = self.configs[index]
        dataset = self.get_dataset(config, name)
        return dataset

    def get_datasets(self) -> List[SpectroDataset]:
        return list(self.iter_datasets())
