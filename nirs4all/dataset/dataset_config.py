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
from nirs4all.dataset.loader import get_dataset
from nirs4all.dataset.config import parse_config


class DatasetConfigs:
    """
    Configuration class for datasets that handles loading, naming, and caching.

    This class can handle single or multiple dataset configurations, caches loaded
    datasets to avoid reloading, and provides a clean interface for dataset management.
    """

    def __init__(self, data_configs: Union[Dict[str, Any], List[Dict[str, Any]], str, List[str]]):
        """
        Initialize DatasetConfigs with one or more dataset configurations.

        Args:
            data_configs: Single dataset config (dict or str path) or list of configs
        """
        if not isinstance(data_configs, list):
            data_configs = [data_configs]

        self.data_configs = data_configs
        self.cache_key = None
        self.cache = (None, None, None, None)

    def get_dataset(self, config) -> SpectroDataset:
        key = self._hash_config(config)
        if key == self.cache_key and self.cache[0] is not None:
            # deepcopy to avoid external mutations
            x_train_src, y_train_src, x_test_src, y_test_src = self.cache
            x_train = copy.deepcopy(x_train_src)
            y_train = copy.deepcopy(y_train_src)
            x_test = copy.deepcopy(x_test_src)
            y_test = copy.deepcopy(y_test_src)
        else:
            self.cache_key = key
            x_train, y_train, x_test, y_test = get_dataset(config)
            self.cache = (copy.deepcopy(x_train), copy.deepcopy(y_train), copy.deepcopy(x_test), copy.deepcopy(y_test))

        dataset = SpectroDataset(name=self.folder_to_name(config))
        dataset.add_samples(x_train, {"partition": "train"})
        dataset.add_samples(x_test, {"partition": "test"})
        dataset.add_targets(y_train)
        dataset.add_targets(y_test)
        print(f"✅ Loaded dataset '{dataset.name}' with {len(x_train)} training and {len(x_test)} test samples.")
        return dataset

    def _hash_config(self, config: Union[Dict[str, Any], str]) -> str:
        """
        Generate a hash for the dataset configuration.

        Args:
            config: Dataset configuration (dict or path string)

        Returns:
            MD5 hash string of the parsed config
        """
        # Parse the config to get the normalized dict
        parsed_config, _ = parse_config(config)

        if parsed_config is None:
            # Fallback for unparseable configs
            config_str = str(config)
        else:
            config_str = json.dumps(parsed_config, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()

    @property
    def names(self) -> List[str]:
        """
        Get the names of all configured datasets.

        Returns:
            List of dataset names
        """
        names = []
        for config in self.data_configs:
            _, name = parse_config(config)
            names.append(name)
        return names

    @property
    def name(self) -> str:
        """
        Get the name of the first dataset.

        Returns:
            Dataset name
        """
        return self.names[0]


    @staticmethod
    def folder_to_name(folder_path):
        path = Path(folder_path)
        for part in reversed(path.parts):
            clean_part = ''.join(c if c.isalnum() else '_' for c in part)
            if clean_part:
                return clean_part.lower()
        return "Unknown_dataset"

