"""
DatasetConfigs - Configuration and caching for dataset loading.

This module provides DatasetConfigs class that handles dataset configuration,
name resolution, loader calls, and caching to avoid reloading the same dataset.
"""

import copy
import hashlib
import json
from typing import Any, Optional, Union

from nirs4all.core.logging import get_logger
from nirs4all.data.dataset import SpectroDataset

logger = get_logger(__name__)
from nirs4all.data.config_parser import parse_config
from nirs4all.data.loaders.loader import handle_data
from nirs4all.data.signal_type import SignalType, SignalTypeInput, normalize_signal_type


class DatasetConfigs:

    def __init__(
        self,
        configurations: dict[str, Any] | list[dict[str, Any]] | str | list[str],
        task_type: str | list[str] = "auto",
        signal_type: SignalTypeInput | list[SignalTypeInput] | None = None,
        repetition: str | list[str | None] | None = None,
        aggregate_method: str | list[str] | None = None,
        aggregate_exclude_outliers: bool | list[bool] | None = None
    ):
        '''Initialize dataset configurations.

        Args:
            configurations: Dataset configuration(s) as path(s), dict(s), or list of either.
                Configuration dicts can include:
                - 'task_type': "regression", "binary_classification", "multiclass_classification", "auto"
                - 'aggregate': Sample aggregation column name, True for y-based, or None
                - 'train_x_params' or 'global_params' with:
                    - header_unit: "cm-1", "nm", "none", "text", "index"
                    - signal_type: "absorbance", "reflectance", "reflectance%", "transmittance", etc.
                    - delimiter, decimal_separator, has_header, na_policy, etc.

            task_type: Force task type. Can be:
                - A single string applied to all datasets
                - A list of strings (one per dataset)
                Valid values per dataset:
                - 'auto': Use config value or automatic detection (default)
                - 'regression': Force regression task
                - 'binary_classification': Force binary classification task
                - 'multiclass_classification': Force multiclass classification task
                Note: Constructor parameter overrides config dict value (except 'auto').

            signal_type: Override signal type for spectral data (applied after config loading).
                - A single value applied to all datasets/sources
                - A list of values (one per dataset)
                - None: Use config value or auto-detect (default)
                This parameter overrides any signal_type specified in the config.
                Valid values: 'absorbance', 'reflectance', 'reflectance%',
                'transmittance', 'transmittance%', 'auto', etc.

            repetition: Column name identifying sample repetitions (multiple spectral
                measurements of the same physical sample).
                - None (default): No repetition grouping
                - str: Metadata column name (e.g., 'Sample_ID', 'ID')
                - list: Per-dataset settings (must match number of datasets)
                When set, splits will automatically group by this column to prevent
                data leakage. This is the preferred way to handle repeated measurements.

            aggregate_method: Aggregation method for combining predictions.
                - None (default): Use 'mean' for regression, 'vote' for classification
                - 'mean': Average predictions within each group
                - 'median': Median prediction within each group
                - 'vote': Majority voting (for classification)
                - list: Per-dataset methods (must match number of datasets)

            aggregate_exclude_outliers: Whether to exclude outliers before aggregation.
                Uses Hotelling's T² statistic to identify and exclude outlier
                measurements within each sample group before averaging.
                - None/False (default): No outlier exclusion
                - True: Exclude outliers using T² with 0.95 confidence
                - list: Per-dataset settings (must match number of datasets)

        Example:
            # Via config dict (preferred for per-source control):
            config = {
                "train_x": "/path/to/data.csv",
                "task_type": "regression",
                "repetition": "sample_id",
                "train_x_params": {
                    "header_unit": "nm",
                    "signal_type": "reflectance",
                    "delimiter": ";"
                }
            }
            configs = DatasetConfigs(config)

            # Or via constructor parameter (overrides config):
            configs = DatasetConfigs("/path/to/folder", task_type="regression")
            configs = DatasetConfigs("/path/to/folder", signal_type="absorbance")
            configs = DatasetConfigs("/path/to/folder", repetition="sample_id")
        '''
        user_configs = configurations if isinstance(configurations, list) else [configurations]
        self.configs = []
        self._config_task_types: list[str | None] = []  # task_type from config dicts
        self._config_aggregates: list[str | bool | None] = []  # aggregate from config dicts
        self._config_aggregate_methods: list[str | None] = []  # aggregate_method from config dicts
        self._config_aggregate_exclude_outliers: list[bool | None] = []  # aggregate_exclude_outliers from config dicts
        self._config_repetitions: list[str | None] = []  # repetition from config dicts
        for config in user_configs:
            parsed_config, dataset_name = parse_config(config)
            if parsed_config is not None:
                self.configs.append((parsed_config, dataset_name))
                # Extract task_type from config dict if present
                config_task_type = None
                config_aggregate = None
                config_aggregate_method = None
                config_aggregate_exclude_outliers = None
                config_repetition = None
                if isinstance(parsed_config, dict):
                    config_task_type = parsed_config.get("task_type")
                    config_aggregate = parsed_config.get("aggregate")
                    config_aggregate_method = parsed_config.get("aggregate_method")
                    config_aggregate_exclude_outliers = parsed_config.get("aggregate_exclude_outliers")
                    config_repetition = parsed_config.get("repetition")
                    # If repetition not in config but aggregate is a string, use it as repetition
                    if config_repetition is None and isinstance(config_aggregate, str):
                        config_repetition = config_aggregate
                self._config_task_types.append(config_task_type)
                self._config_aggregates.append(config_aggregate)
                self._config_aggregate_methods.append(config_aggregate_method)
                self._config_aggregate_exclude_outliers.append(config_aggregate_exclude_outliers)
                self._config_repetitions.append(config_repetition)
            else:
                logger.error(f"Skipping invalid dataset config: {config}")

        self.max_cache_bytes: int = 2 * 1024 ** 3  # 2 GB default
        self.cache: dict[str, Any] = {}
        self._cache_sizes: dict[str, int] = {}  # byte size per cache key
        self._cache_total_bytes: int = 0

        # Normalize task_type to a list matching configs length
        # Priority: constructor parameter > config dict > "auto"
        if isinstance(task_type, str):
            if task_type == "auto":
                # Use config-level task_type if available, otherwise "auto"
                self._task_types = [
                    cfg_tt if cfg_tt is not None else "auto"
                    for cfg_tt in self._config_task_types
                ]
            else:
                # Constructor parameter overrides config
                self._task_types = [task_type] * len(self.configs)
        else:
            if len(task_type) != len(self.configs):
                raise ValueError(
                    f"task_type list length ({len(task_type)}) must match "
                    f"number of datasets ({len(self.configs)})"
                )
            # Per-dataset constructor parameters override config
            self._task_types = [
                tt if tt != "auto" else (self._config_task_types[i] or "auto")
                for i, tt in enumerate(task_type)
            ]

        # Normalize signal_type override to a list matching configs length
        # Note: This is an override; config-level signal_type is handled during loading
        if signal_type is None:
            self._signal_type_overrides: list[SignalType | None] = [None] * len(self.configs)
        elif isinstance(signal_type, (str, SignalType)):
            normalized = normalize_signal_type(signal_type)
            self._signal_type_overrides = [normalized] * len(self.configs)
        else:
            if len(signal_type) != len(self.configs):
                raise ValueError(
                    f"signal_type list length ({len(signal_type)}) must match "
                    f"number of datasets ({len(self.configs)})"
                )
            self._signal_type_overrides = [
                normalize_signal_type(st) if st is not None else None
                for st in signal_type
            ]

        # Use config-level aggregate if available, otherwise None
        self._aggregates: list[str | bool | None] = list(self._config_aggregates)

        # Normalize aggregate_method to a list matching configs length
        if aggregate_method is None:
            self._aggregate_methods: list[str | None] = list(self._config_aggregate_methods)
        elif isinstance(aggregate_method, str):
            self._aggregate_methods = [aggregate_method] * len(self.configs)
        else:
            if len(aggregate_method) != len(self.configs):
                raise ValueError(
                    f"aggregate_method list length ({len(aggregate_method)}) must match "
                    f"number of datasets ({len(self.configs)})"
                )
            self._aggregate_methods = [
                meth if meth is not None else self._config_aggregate_methods[i]
                for i, meth in enumerate(aggregate_method)
            ]

        # Normalize aggregate_exclude_outliers to a list matching configs length
        if aggregate_exclude_outliers is None:
            self._aggregate_exclude_outliers: list[bool] = [
                val if val is not None else False
                for val in self._config_aggregate_exclude_outliers
            ]
        elif isinstance(aggregate_exclude_outliers, bool):
            self._aggregate_exclude_outliers = [aggregate_exclude_outliers] * len(self.configs)
        else:
            if len(aggregate_exclude_outliers) != len(self.configs):
                raise ValueError(
                    f"aggregate_exclude_outliers list length ({len(aggregate_exclude_outliers)}) must match "
                    f"number of datasets ({len(self.configs)})"
                )
            self._aggregate_exclude_outliers = [
                val if val is not None else (self._config_aggregate_exclude_outliers[i] if self._config_aggregate_exclude_outliers[i] is not None else False)
                for i, val in enumerate(aggregate_exclude_outliers)
            ]

        # Normalize repetition to a list matching configs length
        # Priority: constructor parameter > config dict > None
        if repetition is None:
            # Use config-level repetition if available
            self._repetitions: list[str | None] = list(self._config_repetitions)
        elif isinstance(repetition, str):
            # Constructor parameter overrides config for all datasets
            self._repetitions = [repetition] * len(self.configs)
        else:
            # List of per-dataset repetition settings
            if len(repetition) != len(self.configs):
                raise ValueError(
                    f"repetition list length ({len(repetition)}) must match "
                    f"number of datasets ({len(self.configs)})"
                )
            # Per-dataset: constructor parameter overrides config when not None
            self._repetitions = [
                rep if rep is not None else self._config_repetitions[i]
                for i, rep in enumerate(repetition)
            ]

    @classmethod
    def from_spectrodataset(cls, dataset: "SpectroDataset") -> "DatasetConfigs":
        """Create a DatasetConfigs wrapping a single pre-loaded SpectroDataset."""
        return cls.from_spectrodatasets([dataset])

    @classmethod
    def from_spectrodatasets(cls, datasets: list["SpectroDataset"]) -> "DatasetConfigs":
        """Create a DatasetConfigs wrapping a list of pre-loaded SpectroDatasets."""
        n = len(datasets)
        configs = cls.__new__(cls)
        configs.configs = [({"_preloaded_dataset": ds}, ds.name) for ds in datasets]
        configs.cache = {}
        configs.max_cache_bytes = 2 * 1024 ** 3
        configs._cache_sizes = {}
        configs._cache_total_bytes = 0
        configs._task_types = ["auto"] * n
        configs._signal_type_overrides = [None] * n
        configs._aggregates = [None] * n
        configs._aggregate_methods = [None] * n
        configs._aggregate_exclude_outliers = [False] * n
        configs._config_task_types = [None] * n
        configs._config_aggregates = [None] * n
        configs._config_aggregate_methods = [None] * n
        configs._config_aggregate_exclude_outliers = [None] * n
        configs._config_repetitions = [None] * n
        configs._repetitions = [None] * n
        return configs

    def iter_datasets(self):
        for idx, (config, name) in enumerate(self.configs):
            dataset = self._get_dataset_with_types(
                config, name, self._task_types[idx], self._signal_type_overrides[idx],
                self._aggregates[idx], self._aggregate_methods[idx],
                self._aggregate_exclude_outliers[idx], self._repetitions[idx]
            )
            yield dataset

    def _get_dataset_with_types(
        self,
        config,
        name,
        task_type: str,
        signal_type_override: SignalType | None,
        aggregate: str | bool | None = None,
        aggregate_method: str | None = None,
        aggregate_exclude_outliers: bool = False,
        repetition: str | None = None
    ) -> SpectroDataset:
        """Internal method to get dataset with specific task and signal types.

        Args:
            config: Dataset configuration dict
            name: Dataset name
            task_type: Task type to force ('auto' for no override)
            signal_type_override: Signal type override from constructor (None for no override)
            aggregate: Aggregation setting (None, True, or column name)
            aggregate_method: Aggregation method ('mean', 'median', 'vote')
            aggregate_exclude_outliers: Whether to exclude outliers using T² before aggregation
            repetition: Column name identifying sample repetitions
        """
        dataset = self._load_dataset(config, name)

        # Apply forced task type if specified (not 'auto')
        if task_type != "auto":
            dataset.set_task_type(task_type, forced=True)

        # Apply constructor-level signal type override if specified
        # This overrides any signal_type from config params
        if signal_type_override is not None and signal_type_override != SignalType.AUTO:
            for src in range(dataset.n_sources):
                dataset.set_signal_type(signal_type_override, src=src, forced=True)

        # Apply repetition setting if specified
        if repetition is not None:
            dataset.set_repetition(repetition)

        # Apply aggregation settings if specified
        if aggregate is not None:
            dataset.set_aggregate(aggregate)
        if aggregate_method is not None:
            dataset.set_aggregate_method(aggregate_method)
        if aggregate_exclude_outliers:
            dataset.set_aggregate_exclude_outliers(True)

        return dataset

    @staticmethod
    def _make_cache_key(name: str, config: Any) -> str:
        """Create a cache key from dataset name and config parameters.

        Includes config parameters in the key to prevent collisions when the
        same dataset name is loaded with different configuration parameters.

        Args:
            name: Dataset name.
            config: Dataset configuration dict or path string.

        Returns:
            MD5-based cache key string.
        """
        key_parts = [name]
        if isinstance(config, dict):
            # Exclude non-serializable runtime objects from the key
            serializable = {
                k: str(v) for k, v in sorted(config.items())
                if k != "_preloaded_dataset"
            }
            key_parts.append(json.dumps(serializable, sort_keys=True, default=str))
        elif isinstance(config, str):
            key_parts.append(config)
        key_data = "|".join(key_parts)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cache_set(self, key: str, entry: Any) -> None:
        """Insert into cache with FIFO eviction when byte budget is exceeded."""
        from nirs4all.utils.memory import estimate_cache_entry_bytes

        entry_size = estimate_cache_entry_bytes(entry)

        # Skip caching if single entry exceeds budget
        if entry_size > self.max_cache_bytes:
            return

        # FIFO eviction: remove oldest entries until we have space
        while self._cache_total_bytes + entry_size > self.max_cache_bytes and self.cache:
            oldest_key = next(iter(self.cache))
            self._cache_remove(oldest_key)

        self.cache[key] = entry
        self._cache_sizes[key] = entry_size
        self._cache_total_bytes += entry_size

    def _cache_remove(self, key: str) -> None:
        """Remove an entry from cache and update bookkeeping."""
        self.cache.pop(key, None)
        size = self._cache_sizes.pop(key, 0)
        self._cache_total_bytes -= size

    def _load_dataset(self, config, name) -> SpectroDataset:
        """Load dataset from config, applying config-level signal_type if specified."""
        # Handle preloaded datasets - return a deep copy to avoid mutation across pipeline variants
        if isinstance(config, dict) and "_preloaded_dataset" in config:
            dataset: SpectroDataset = copy.deepcopy(config["_preloaded_dataset"])
            return dataset

        cache_key = self._make_cache_key(name, config)

        dataset = SpectroDataset(name=name)
        if cache_key in self.cache:
            (x_train, y_train, m_train, train_headers, m_train_headers, train_unit, train_signal_type,
             x_test, y_test, m_test, test_headers, m_test_headers, test_unit, test_signal_type) = self.cache[cache_key]
        else:
            # Try to load train data
            try:
                x_train, y_train, m_train, train_headers, m_train_headers, train_unit, train_signal_type = handle_data(config, "train")
            except (ValueError, FileNotFoundError) as e:
                if "x_path is None" in str(e) or "train_x" in str(e):
                    x_train, y_train, m_train, train_headers, m_train_headers, train_unit, train_signal_type = None, None, None, None, None, None, None
                else:
                    raise

            # Try to load test data
            try:
                x_test, y_test, m_test, test_headers, m_test_headers, test_unit, test_signal_type = handle_data(config, "test")
            except (ValueError, FileNotFoundError) as e:
                if "x_path is None" in str(e) or "test_x" in str(e):
                    x_test, y_test, m_test, test_headers, m_test_headers, test_unit, test_signal_type = None, None, None, None, None, None, None
                else:
                    raise

            self._cache_set(cache_key, (
                x_train, y_train, m_train, train_headers, m_train_headers, train_unit, train_signal_type,
                x_test, y_test, m_test, test_headers, m_test_headers, test_unit, test_signal_type
            ))

        # Add samples and targets only if they exist
        if x_train is not None:
            dataset.add_samples(x_train, {"partition": "train"}, headers=train_headers, header_unit=train_unit)

            # Apply signal types from config (per source if multi-source)
            if train_signal_type is not None:
                if isinstance(train_signal_type, list):
                    for src, sig_type in enumerate(train_signal_type):
                        if sig_type is not None:
                            dataset.set_signal_type(sig_type, src=src, forced=False)
                else:
                    dataset.set_signal_type(train_signal_type, src=0, forced=False)

            if y_train is not None:
                dataset.add_targets(y_train)
            if m_train is not None:
                dataset.add_metadata(m_train, headers=m_train_headers)

        if x_test is not None:
            dataset.add_samples(x_test, {"partition": "test"}, headers=test_headers, header_unit=test_unit)

            # Apply signal types from config (per source if multi-source)
            # Note: test data adds to existing sources, so signal types should already match
            if test_signal_type is not None and x_train is None:
                # Only apply if we didn't load train data (test-only case)
                if isinstance(test_signal_type, list):
                    for src, sig_type in enumerate(test_signal_type):
                        if sig_type is not None:
                            dataset.set_signal_type(sig_type, src=src, forced=False)
                else:
                    dataset.set_signal_type(test_signal_type, src=0, forced=False)

            if y_test is not None:
                dataset.add_targets(y_test)
            if m_test is not None:
                dataset.add_metadata(m_test, headers=m_test_headers)

        # Set NaN tracking flag based on loaded data
        dataset._may_contain_nan = dataset.has_nan

        return dataset

    def get_dataset(self, config, name) -> SpectroDataset:
        """Get dataset by config and name (backward compatible).

        Note: When called directly, uses the first task_type (or 'auto' if single dataset).
        For proper per-dataset task_type handling, use iter_datasets() or get_dataset_at().
        """
        # Find the index of this config to get the right task_type, signal_type, and aggregate
        for idx, (_cfg, cfg_name) in enumerate(self.configs):
            if cfg_name == name:
                return self._get_dataset_with_types(
                    config, name, self._task_types[idx], self._signal_type_overrides[idx],
                    self._aggregates[idx], self._aggregate_methods[idx],
                    self._aggregate_exclude_outliers[idx], self._repetitions[idx]
                )
        # Fallback: load without forced types
        return self._load_dataset(config, name)

    def get_dataset_at(self, index) -> SpectroDataset:
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Dataset index {index} out of range. Available datasets: 0 to {len(self.configs)-1}.")
        config, name = self.configs[index]
        return self._get_dataset_with_types(
            config, name, self._task_types[index], self._signal_type_overrides[index],
            self._aggregates[index], self._aggregate_methods[index],
            self._aggregate_exclude_outliers[index], self._repetitions[index]
        )

    @property
    def repetition(self) -> str | None:
        """Get the repetition column name for the first dataset.

        For multi-dataset configurations, use repetitions property instead.

        Returns:
            Column name identifying sample repetitions, or None.
        """
        return self._repetitions[0] if self._repetitions else None

    @property
    def repetitions(self) -> list[str | None]:
        """Get repetition column names for all datasets.

        Returns:
            List of column names (or None) for each dataset.
        """
        return self._repetitions

    def get_datasets(self) -> list[SpectroDataset]:
        return list(self.iter_datasets())

