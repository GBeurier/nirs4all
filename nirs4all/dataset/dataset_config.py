"""
DatasetConfigs - Configuration and caching for dataset loading.

This module provides DatasetConfigs class that handles dataset configuration,
name resolution, loader calls, and caching to avoid reloading the same dataset.
"""

import json
import hashlib
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
        self.cache: Dict[str, SpectroDataset] = {}  # hash -> dataset

    def get_datasets(self) -> List[SpectroDataset]:
        """
        Get the list of datasets, loading and caching as needed.

        Returns:
            List of SpectroDataset instances
        """
        datasets = []

        for config in self.data_configs:
            key = self._hash_config(config)

            if key in self.cache:
                dataset = self.cache[key]
            else:
                # Clear cache when loading a new dataset (as per requirements)
                self.cache.clear()
                dataset = get_dataset(config)
                self.cache[key] = dataset

            datasets.append(dataset)

        return datasets

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

    def clear_cache(self):
        """Clear the dataset cache."""
        self.cache.clear()
