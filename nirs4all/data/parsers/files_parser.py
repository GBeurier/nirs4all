"""
Files parser for dataset configuration.

This parser handles the new 'files' syntax defined in the specification.
Implemented in Phase 4 to support partition assignment.

The files syntax allows specifying multiple files with column/row selection
and partition assignment within a single configuration.

Example:
    files:
      - path: data/measurements.csv
        partition: train
        columns:
          features: "2:-1"
          targets: -1
          metadata: [0, 1]

    # Or with complex partition:
    files:
      - path: data/all_data.csv
        partition:
          column: "split"
          train_values: ["train"]
          test_values: ["test"]

The sources syntax (Phase 6) allows specifying multiple feature sources
for sensor fusion or multi-instrument datasets:

Example:
    sources:
      - name: "NIR"
        files:
          - path: data/NIR_train.csv
            partition: train
          - path: data/NIR_test.csv
            partition: test
        params:
          header_unit: nm
          signal_type: absorbance
      - name: "MIR"
        train_x: data/MIR_train.csv
        test_x: data/MIR_test.csv
        params:
          header_unit: cm-1
          signal_type: absorbance

    targets:
      path: data/targets.csv
      link_by: sample_id
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseParser, ParserResult
from ..schema import (
    DatasetConfigSchema,
    FileConfig,
    ColumnConfig,
    PartitionConfig,
    LoadingParams,
    PartitionType,
    SourceConfig,
    SourceFileConfig,
    SharedTargetsConfig,
    SharedMetadataConfig,
    VariationConfig,
    VariationFileConfig,
    PreprocessingApplied,
    VariationMode,
)


class FilesParser(BaseParser):
    """Parser for new 'files' syntax configuration.

    The files syntax provides:
    - Flexible column selection (by index, name, regex, range)
    - Row selection and filtering
    - Partition assignment per file or via partition config
    - Key-based sample linking across files
    """

    def can_parse(self, input_data: Any) -> bool:
        """Check if this is a files-format configuration.

        Args:
            input_data: The input to check.

        Returns:
            True if input has 'files' key with non-empty list.
        """
        if not isinstance(input_data, dict):
            return False

        files = input_data.get('files')
        if files is None:
            return False

        return isinstance(files, list) and len(files) > 0

    def parse(self, input_data: Dict[str, Any]) -> ParserResult:
        """Parse a files-format configuration.

        Args:
            input_data: Dictionary configuration to parse.

        Returns:
            ParserResult with parsed configuration.
        """
        files_list = input_data.get('files', [])
        errors = []
        warnings = []
        parsed_files = []

        # Parse global settings
        name = input_data.get('name')
        description = input_data.get('description')
        task_type = input_data.get('task_type')
        signal_type = input_data.get('signal_type')
        global_params = input_data.get('global_params')

        # Parse global partition config (applies if file doesn't specify partition)
        global_partition = input_data.get('partition')

        # Parse each file configuration
        for idx, file_config in enumerate(files_list):
            try:
                parsed_file = self._parse_single_file(
                    file_config, idx, global_partition
                )
                parsed_files.append(parsed_file)
            except Exception as e:
                errors.append(f"Error parsing file {idx}: {e}")

        if not parsed_files and not errors:
            errors.append("No valid files found in 'files' configuration.")

        # Organize files by partition
        train_files = []
        test_files = []
        predict_files = []

        for pf in parsed_files:
            partition = pf.get('_resolved_partition', 'train')
            if partition == 'train':
                train_files.append(pf)
            elif partition == 'test':
                test_files.append(pf)
            elif partition == 'predict':
                predict_files.append(pf)
            else:
                warnings.append(
                    f"Unknown partition '{partition}' for file {pf.get('path')}, "
                    f"defaulting to 'train'."
                )
                train_files.append(pf)

        # Build config schema data
        config_data = {}

        if name:
            config_data['name'] = name
        if description:
            config_data['description'] = description
        if task_type:
            config_data['task_type'] = task_type
        if global_params:
            if isinstance(global_params, dict):
                config_data['global_params'] = LoadingParams(**global_params)
            else:
                config_data['global_params'] = global_params

        # Convert file lists to train_x/test_x format for now
        # (for backward compatibility with existing loaders)
        if train_files:
            if len(train_files) == 1:
                config_data['train_x'] = train_files[0].get('path')
            else:
                config_data['train_x'] = [f.get('path') for f in train_files]

        if test_files:
            if len(test_files) == 1:
                config_data['test_x'] = test_files[0].get('path')
            else:
                config_data['test_x'] = [f.get('path') for f in test_files]

        # Store parsed files for advanced processing
        config_data['files'] = [
            self._to_file_config(pf) for pf in parsed_files
        ]

        return ParserResult(
            success=len(errors) == 0,
            config=DatasetConfigSchema(**config_data) if not errors else None,
            errors=errors,
            warnings=warnings,
            source_type="files"
        )

    def _parse_single_file(
        self,
        file_config: Union[str, Dict[str, Any]],
        index: int,
        global_partition: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Parse a single file configuration.

        Args:
            file_config: File configuration (path string or dict).
            index: File index in list.
            global_partition: Global partition config to use if file doesn't specify one.

        Returns:
            Parsed file configuration dict.
        """
        # Handle simple string path
        if isinstance(file_config, str):
            resolved_partition = self._resolve_partition_from_path(
                file_config, global_partition
            )
            return {
                'path': file_config,
                '_resolved_partition': resolved_partition,
            }

        # Handle full dict configuration
        if not isinstance(file_config, dict):
            raise ValueError(
                f"File config must be string or dict, got {type(file_config)}"
            )

        path = file_config.get('path')
        if not path:
            raise ValueError(f"File config at index {index} missing 'path' key")

        # Parse partition
        file_partition = file_config.get('partition')
        resolved_partition = self._resolve_partition(
            file_partition, path, global_partition
        )

        # Parse columns
        columns = file_config.get('columns')
        parsed_columns = None
        if columns:
            if isinstance(columns, dict):
                parsed_columns = ColumnConfig(**columns)
            else:
                parsed_columns = columns

        # Parse params
        params = file_config.get('params')
        parsed_params = None
        if params:
            if isinstance(params, dict):
                parsed_params = LoadingParams(**params)
            else:
                parsed_params = params

        return {
            'path': path,
            'partition': file_partition,
            '_resolved_partition': resolved_partition,
            'columns': parsed_columns,
            'params': parsed_params,
            'link_by': file_config.get('link_by'),
            'rows': file_config.get('rows'),
        }

    def _resolve_partition(
        self,
        file_partition: Optional[Union[str, Dict[str, Any]]],
        path: str,
        global_partition: Optional[Union[str, Dict[str, Any]]],
    ) -> str:
        """Resolve partition assignment for a file.

        Priority: file_partition > global_partition > path inference.
        """
        # If file has explicit partition
        if file_partition is not None:
            if isinstance(file_partition, str):
                return file_partition.lower()
            elif isinstance(file_partition, dict):
                # Dict partition means column-based or complex - defer to loader
                return 'mixed'
            elif isinstance(file_partition, PartitionType):
                return file_partition.value

        # If global partition is set
        if global_partition is not None:
            if isinstance(global_partition, str):
                return global_partition.lower()
            elif isinstance(global_partition, dict):
                return 'mixed'

        # Infer from path
        return self._resolve_partition_from_path(path, None)

    def _resolve_partition_from_path(
        self,
        path: str,
        fallback: Optional[str],
    ) -> str:
        """Infer partition from file path naming convention.

        Args:
            path: File path.
            fallback: Fallback partition if cannot infer.

        Returns:
            Partition name ('train', 'test', or 'predict').
        """
        path_lower = Path(path).stem.lower()

        # Training patterns
        train_patterns = ('train', 'cal', 'calibration', 'xcal', 'xtrain')
        for pattern in train_patterns:
            if pattern in path_lower:
                return 'train'

        # Test patterns
        test_patterns = ('test', 'val', 'validation', 'xval', 'xtest')
        for pattern in test_patterns:
            if pattern in path_lower:
                return 'test'

        # Predict patterns
        predict_patterns = ('predict', 'unknown', 'new')
        for pattern in predict_patterns:
            if pattern in path_lower:
                return 'predict'

        # Default to train if cannot infer
        return fallback if fallback else 'train'

    def _to_file_config(self, parsed: Dict[str, Any]) -> FileConfig:
        """Convert parsed file dict to FileConfig model."""
        partition_type = None
        partition_str = parsed.get('_resolved_partition')
        if partition_str and partition_str != 'mixed':
            try:
                partition_type = PartitionType(partition_str)
            except ValueError:
                pass

        return FileConfig(
            path=parsed['path'],
            partition=partition_type,
            columns=parsed.get('columns'),
            params=parsed.get('params'),
            link_by=parsed.get('link_by'),
        )


class SourcesParser(BaseParser):
    """Parser for multi-source 'sources' syntax configuration.

    The sources syntax provides:
    - Named feature sources (e.g., NIR, MIR spectrometers)
    - Per-source loading parameters
    - Automatic source alignment by sample key
    - Shared targets and metadata across sources

    Example configuration:
        sources:
          - name: "NIR"
            files:
              - path: data/NIR_train.csv
                partition: train
              - path: data/NIR_test.csv
                partition: test
            params:
              header_unit: nm
              signal_type: absorbance
          - name: "MIR"
            train_x: data/MIR_train.csv
            test_x: data/MIR_test.csv
            params:
              header_unit: cm-1

        targets:
          path: data/targets.csv
          link_by: sample_id

        metadata:
          path: data/metadata.csv
          link_by: sample_id
    """

    def can_parse(self, input_data: Any) -> bool:
        """Check if this is a sources-format configuration.

        Args:
            input_data: The input to check.

        Returns:
            True if input has 'sources' key with non-empty list.
        """
        if not isinstance(input_data, dict):
            return False

        sources = input_data.get('sources')
        if sources is None:
            return False

        return isinstance(sources, list) and len(sources) > 0

    def parse(self, input_data: Dict[str, Any]) -> ParserResult:
        """Parse a sources-format configuration.

        Converts the sources syntax to a DatasetConfigSchema that can be
        further converted to legacy format for backward compatibility.

        Args:
            input_data: Dictionary configuration to parse.

        Returns:
            ParserResult with parsed configuration.
        """
        sources_list = input_data.get('sources', [])
        errors = []
        warnings = []
        parsed_sources = []

        # Parse global settings
        name = input_data.get('name')
        description = input_data.get('description')
        task_type = input_data.get('task_type')
        global_params = input_data.get('global_params')

        # Parse each source configuration
        for idx, source_config in enumerate(sources_list):
            try:
                parsed_source = self._parse_single_source(source_config, idx, global_params)
                parsed_sources.append(parsed_source)
            except Exception as e:
                errors.append(f"Error parsing source {idx}: {e}")

        if not parsed_sources and not errors:
            errors.append("No valid sources found in 'sources' configuration.")

        # Validate source names are unique
        source_names = [s.name for s in parsed_sources]
        if len(source_names) != len(set(source_names)):
            errors.append(
                f"Duplicate source names found: {source_names}. "
                f"Each source must have a unique name."
            )

        # Parse shared targets
        shared_targets = None
        targets_config = input_data.get('targets')
        if targets_config:
            try:
                shared_targets = self._parse_shared_targets(targets_config)
            except Exception as e:
                errors.append(f"Error parsing targets: {e}")

        # Parse shared metadata
        shared_metadata = None
        metadata_config = input_data.get('metadata')
        if metadata_config:
            try:
                shared_metadata = self._parse_shared_metadata(metadata_config)
            except Exception as e:
                errors.append(f"Error parsing metadata: {e}")

        if errors:
            return ParserResult(
                success=False,
                errors=errors,
                warnings=warnings,
                source_type="sources"
            )

        # Build config schema data
        config_data = {
            'sources': parsed_sources,
        }

        if name:
            config_data['name'] = name
        if description:
            config_data['description'] = description
        if task_type:
            config_data['task_type'] = task_type
        if global_params:
            if isinstance(global_params, dict):
                config_data['global_params'] = LoadingParams(**global_params)
            else:
                config_data['global_params'] = global_params
        if shared_targets:
            config_data['shared_targets'] = shared_targets
        if shared_metadata:
            config_data['shared_metadata'] = shared_metadata

        # Create schema object
        try:
            schema = DatasetConfigSchema(**config_data)
        except Exception as e:
            return ParserResult(
                success=False,
                errors=[f"Failed to create config schema: {e}"],
                warnings=warnings,
                source_type="sources"
            )

        # Extract dataset name
        dataset_name = name
        if not dataset_name:
            # Use first source name as dataset name
            if parsed_sources:
                dataset_name = f"multisource_{parsed_sources[0].name}"
            else:
                dataset_name = "multisource_dataset"

        # Add warning about multi-source
        warnings.append(
            f"Multi-source dataset with {len(parsed_sources)} source(s): "
            f"{[s.name for s in parsed_sources]}"
        )

        return ParserResult(
            success=True,
            config=schema,
            dataset_name=dataset_name,
            errors=[],
            warnings=warnings,
            source_type="sources"
        )

    def _parse_single_source(
        self,
        source_config: Dict[str, Any],
        index: int,
        global_params: Optional[Dict[str, Any]] = None,
    ) -> SourceConfig:
        """Parse a single source configuration.

        Args:
            source_config: Source configuration dict.
            index: Source index in list.
            global_params: Global parameters to merge with source params.

        Returns:
            Parsed SourceConfig.

        Raises:
            ValueError: If source configuration is invalid.
        """
        if not isinstance(source_config, dict):
            raise ValueError(
                f"Source config must be a dict, got {type(source_config)}"
            )

        # Name is required
        name = source_config.get('name')
        if not name:
            name = f"source_{index}"

        # Parse source params
        source_params = source_config.get('params')
        if source_params and isinstance(source_params, dict):
            # Merge with global params
            if global_params:
                merged = global_params.copy()
                merged.update(source_params)
                source_params = LoadingParams(**merged)
            else:
                source_params = LoadingParams(**source_params)
        elif global_params:
            source_params = LoadingParams(**global_params)
        else:
            source_params = None

        # Parse files list
        files_list = source_config.get('files')
        parsed_files = None
        if files_list:
            parsed_files = []
            for f in files_list:
                if isinstance(f, str):
                    parsed_files.append(f)
                elif isinstance(f, dict):
                    parsed_files.append(SourceFileConfig(**f))
                else:
                    parsed_files.append(f)

        # Get direct paths
        train_x = source_config.get('train_x')
        test_x = source_config.get('test_x')

        # Get link_by
        link_by = source_config.get('link_by')

        return SourceConfig(
            name=name,
            files=parsed_files,
            train_x=train_x,
            test_x=test_x,
            params=source_params,
            link_by=link_by,
        )

    def _parse_shared_targets(
        self,
        targets_config: Union[str, Dict[str, Any], List[Any]],
    ) -> Union[SharedTargetsConfig, List[SharedTargetsConfig]]:
        """Parse shared targets configuration.

        Args:
            targets_config: Targets configuration (path, dict, or list).

        Returns:
            Parsed SharedTargetsConfig or list of them.
        """
        if isinstance(targets_config, str):
            return SharedTargetsConfig(path=targets_config)

        if isinstance(targets_config, dict):
            return SharedTargetsConfig(**targets_config)

        if isinstance(targets_config, list):
            return [
                SharedTargetsConfig(**t) if isinstance(t, dict)
                else SharedTargetsConfig(path=t) if isinstance(t, str)
                else t
                for t in targets_config
            ]

        raise ValueError(f"Invalid targets config type: {type(targets_config)}")

    def _parse_shared_metadata(
        self,
        metadata_config: Union[str, Dict[str, Any], List[Any]],
    ) -> Union[SharedMetadataConfig, List[SharedMetadataConfig]]:
        """Parse shared metadata configuration.

        Args:
            metadata_config: Metadata configuration (path, dict, or list).

        Returns:
            Parsed SharedMetadataConfig or list of them.
        """
        if isinstance(metadata_config, str):
            return SharedMetadataConfig(path=metadata_config)

        if isinstance(metadata_config, dict):
            return SharedMetadataConfig(**metadata_config)

        if isinstance(metadata_config, list):
            return [
                SharedMetadataConfig(**m) if isinstance(m, dict)
                else SharedMetadataConfig(path=m) if isinstance(m, str)
                else m
                for m in metadata_config
            ]

        raise ValueError(f"Invalid metadata config type: {type(metadata_config)}")


class VariationsParser(BaseParser):
    """Parser for feature variations 'variations' syntax configuration.

    The variations syntax provides:
    - Named feature variations (e.g., raw, snv, derivative)
    - Per-variation loading parameters
    - Preprocessing provenance tracking
    - Multiple variation modes (separate, concat, select, compare)

    Example configuration:
        variations:
          - name: "raw"
            files:
              - path: data/spectra_raw.csv
                partition: train
              - path: data/spectra_raw_test.csv
                partition: test

          - name: "snv"
            description: "SNV preprocessed spectra"
            preprocessing_applied:
              - type: "SNV"
                software: "OPUS 8.0"
            train_x: data/spectra_snv_train.csv
            test_x: data/spectra_snv_test.csv

        variation_mode: separate  # or concat, select, compare
        variation_select: ["raw", "snv"]  # only for mode=select

        targets:
          path: data/targets.csv
          link_by: sample_id
    """

    def can_parse(self, input_data: Any) -> bool:
        """Check if this is a variations-format configuration.

        Args:
            input_data: The input to check.

        Returns:
            True if input has 'variations' key with non-empty list.
        """
        if not isinstance(input_data, dict):
            return False

        variations = input_data.get('variations')
        if variations is None:
            return False

        return isinstance(variations, list) and len(variations) > 0

    def parse(self, input_data: Dict[str, Any]) -> ParserResult:
        """Parse a variations-format configuration.

        Converts the variations syntax to a DatasetConfigSchema that can be
        further converted to legacy format for backward compatibility.

        Args:
            input_data: Dictionary configuration to parse.

        Returns:
            ParserResult with parsed configuration.
        """
        variations_list = input_data.get('variations', [])
        errors = []
        warnings = []
        parsed_variations = []

        # Parse global settings
        name = input_data.get('name')
        description = input_data.get('description')
        task_type = input_data.get('task_type')
        signal_type = input_data.get('signal_type')
        global_params = input_data.get('global_params')
        variation_mode = input_data.get('variation_mode', 'separate')
        variation_select = input_data.get('variation_select')
        variation_prefix = input_data.get('variation_prefix')

        # Parse each variation configuration
        for idx, variation_config in enumerate(variations_list):
            try:
                parsed_variation = self._parse_single_variation(
                    variation_config, idx, global_params
                )
                parsed_variations.append(parsed_variation)
            except Exception as e:
                errors.append(f"Error parsing variation {idx}: {e}")

        if not parsed_variations and not errors:
            errors.append("No valid variations found in 'variations' configuration.")

        # Validate variation names are unique
        variation_names = [v.name for v in parsed_variations]
        if len(variation_names) != len(set(variation_names)):
            errors.append(
                f"Duplicate variation names found: {variation_names}. "
                f"Each variation must have a unique name."
            )

        # Parse shared targets
        shared_targets = None
        targets_config = input_data.get('targets')
        if targets_config:
            try:
                shared_targets = self._parse_shared_targets(targets_config)
            except Exception as e:
                errors.append(f"Error parsing targets: {e}")

        # Parse shared metadata
        shared_metadata = None
        metadata_config = input_data.get('metadata')
        if metadata_config:
            try:
                shared_metadata = self._parse_shared_metadata(metadata_config)
            except Exception as e:
                errors.append(f"Error parsing metadata: {e}")

        if errors:
            return ParserResult(
                success=False,
                errors=errors,
                warnings=warnings,
                source_type="variations"
            )

        # Build config schema data
        config_data = {
            'variations': parsed_variations,
        }

        if name:
            config_data['name'] = name
        if description:
            config_data['description'] = description
        if task_type:
            config_data['task_type'] = task_type
        if signal_type:
            config_data['signal_type'] = signal_type
        if global_params:
            if isinstance(global_params, dict):
                config_data['global_params'] = LoadingParams(**global_params)
            else:
                config_data['global_params'] = global_params
        if variation_mode:
            config_data['variation_mode'] = variation_mode
        if variation_select:
            config_data['variation_select'] = variation_select
        if variation_prefix is not None:
            config_data['variation_prefix'] = variation_prefix
        if shared_targets:
            config_data['shared_targets'] = shared_targets
        if shared_metadata:
            config_data['shared_metadata'] = shared_metadata

        # Create schema object
        try:
            schema = DatasetConfigSchema(**config_data)
        except Exception as e:
            return ParserResult(
                success=False,
                errors=[f"Failed to create config schema: {e}"],
                warnings=warnings,
                source_type="variations"
            )

        # Extract dataset name
        dataset_name = name
        if not dataset_name:
            # Use first variation name as dataset name
            if parsed_variations:
                dataset_name = f"variations_{parsed_variations[0].name}"
            else:
                dataset_name = "variations_dataset"

        # Add info about variations
        mode_str = variation_mode if isinstance(variation_mode, str) else variation_mode.value
        warnings.append(
            f"Variation dataset with {len(parsed_variations)} variation(s): "
            f"{[v.name for v in parsed_variations]}, mode: {mode_str}"
        )

        return ParserResult(
            success=True,
            config=schema,
            dataset_name=dataset_name,
            errors=[],
            warnings=warnings,
            source_type="variations"
        )

    def _parse_single_variation(
        self,
        variation_config: Dict[str, Any],
        index: int,
        global_params: Optional[Dict[str, Any]] = None,
    ) -> VariationConfig:
        """Parse a single variation configuration.

        Args:
            variation_config: Variation configuration dict.
            index: Variation index in list.
            global_params: Global parameters to merge with variation params.

        Returns:
            Parsed VariationConfig.

        Raises:
            ValueError: If variation configuration is invalid.
        """
        if not isinstance(variation_config, dict):
            raise ValueError(
                f"Variation config must be a dict, got {type(variation_config)}"
            )

        # Name is required
        name = variation_config.get('name')
        if not name:
            name = f"variation_{index}"

        # Description is optional
        description = variation_config.get('description')

        # Parse variation params
        variation_params = variation_config.get('params')
        if variation_params and isinstance(variation_params, dict):
            # Merge with global params
            if global_params:
                merged = global_params.copy()
                merged.update(variation_params)
                variation_params = LoadingParams(**merged)
            else:
                variation_params = LoadingParams(**variation_params)
        elif global_params:
            variation_params = LoadingParams(**global_params)
        else:
            variation_params = None

        # Parse files list
        files_list = variation_config.get('files')
        parsed_files = None
        if files_list:
            parsed_files = []
            for f in files_list:
                if isinstance(f, str):
                    parsed_files.append(f)
                elif isinstance(f, dict):
                    parsed_files.append(VariationFileConfig(**f))
                else:
                    parsed_files.append(f)

        # Get direct paths
        train_x = variation_config.get('train_x')
        test_x = variation_config.get('test_x')

        # Parse preprocessing_applied
        preprocessing_applied = None
        preprocessing_list = variation_config.get('preprocessing_applied')
        if preprocessing_list:
            preprocessing_applied = []
            for p in preprocessing_list:
                if isinstance(p, dict):
                    preprocessing_applied.append(PreprocessingApplied(**p))
                elif isinstance(p, PreprocessingApplied):
                    preprocessing_applied.append(p)

        return VariationConfig(
            name=name,
            description=description,
            files=parsed_files,
            train_x=train_x,
            test_x=test_x,
            params=variation_params,
            preprocessing_applied=preprocessing_applied,
        )

    def _parse_shared_targets(
        self,
        targets_config: Union[str, Dict[str, Any], List[Any]],
    ) -> Union[SharedTargetsConfig, List[SharedTargetsConfig]]:
        """Parse shared targets configuration.

        Args:
            targets_config: Targets configuration (path, dict, or list).

        Returns:
            Parsed SharedTargetsConfig or list of them.
        """
        if isinstance(targets_config, str):
            return SharedTargetsConfig(path=targets_config)

        if isinstance(targets_config, dict):
            return SharedTargetsConfig(**targets_config)

        if isinstance(targets_config, list):
            return [
                SharedTargetsConfig(**t) if isinstance(t, dict)
                else SharedTargetsConfig(path=t) if isinstance(t, str)
                else t
                for t in targets_config
            ]

        raise ValueError(f"Invalid targets config type: {type(targets_config)}")

    def _parse_shared_metadata(
        self,
        metadata_config: Union[str, Dict[str, Any], List[Any]],
    ) -> Union[SharedMetadataConfig, List[SharedMetadataConfig]]:
        """Parse shared metadata configuration.

        Args:
            metadata_config: Metadata configuration (path, dict, or list).

        Returns:
            Parsed SharedMetadataConfig or list of them.
        """
        if isinstance(metadata_config, str):
            return SharedMetadataConfig(path=metadata_config)

        if isinstance(metadata_config, dict):
            return SharedMetadataConfig(**metadata_config)

        if isinstance(metadata_config, list):
            return [
                SharedMetadataConfig(**m) if isinstance(m, dict)
                else SharedMetadataConfig(path=m) if isinstance(m, str)
                else m
                for m in metadata_config
            ]

        raise ValueError(f"Invalid metadata config type: {type(metadata_config)}")

