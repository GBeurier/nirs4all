"""
Folder parser for dataset configuration.

This parser handles folder paths, scanning for data files matching
standard naming conventions (Xcal, Xval, etc.).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseParser, ParserResult


# File naming patterns for auto-detection
FILE_PATTERNS = {
    "train_x": [
        "Xcal", "X_cal", "Cal_X", "calX",
        "train_X", "trainX", "X_train", "Xtrain"
    ],
    "test_x": [
        "Xval", "X_val", "val_X", "valX",
        "Xtest", "X_test", "test_X", "testX"
    ],
    "train_y": [
        "Ycal", "Y_cal", "Cal_Y", "calY",
        "train_Y", "trainY", "Y_train", "Ytrain"
    ],
    "test_y": [
        "Ytest", "Y_test", "test_Y", "testY",
        "Yval", "Y_val", "val_Y", "valY"
    ],
    "train_group": [
        "Mcal", "M_cal", "Cal_M", "calM",
        "train_M", "trainM", "M_train", "Mtrain",
        "Metacal", "Meta_cal", "Cal_Meta", "calMeta",
        "train_Meta", "trainMeta", "Meta_train", "Metatrain",
        "metadatacal", "metadata_cal", "Cal_metadata", "calMetadata",
        "train_metadata", "trainMetadata", "metadata_train", "metadatatrain"
    ],
    "test_group": [
        "Mtest", "M_test", "test_M", "testM",
        "Mval", "M_val", "val_M", "valM",
        "Metatest", "Meta_test", "test_Meta", "testMeta",
        "Metaval", "Meta_val", "val_Meta", "valMeta",
        "metadatatest", "metadata_test", "test_metadata", "testMetadata",
        "metadataval", "metadata_val", "val_metadata", "valMetadata"
    ],
}


class FolderParser(BaseParser):
    """Parser for folder-based dataset configuration.

    This parser scans a folder for data files matching standard naming
    conventions and creates a configuration dictionary.

    Supported file formats:
    - CSV files (.csv)
    - Compressed CSV files (.csv.gz, .csv.zip)

    Multi-source detection:
    - If multiple files match the same pattern (e.g., Xcal_NIR.csv, Xcal_MIR.csv),
      they are treated as multi-source data.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.csv', '.csv.gz', '.csv.zip', '.gz', '.zip'}

    def can_parse(self, input_data: Any) -> bool:
        """Check if input is a folder path.

        Args:
            input_data: The input to check.

        Returns:
            True if input is a string path to an existing directory.
        """
        if isinstance(input_data, str):
            # Check if it looks like a file path (has JSON/YAML extension)
            lower_path = input_data.lower()
            if lower_path.endswith(('.json', '.yaml', '.yml')):
                return False
            # Check if it's a directory
            return Path(input_data).is_dir()

        if isinstance(input_data, Path):
            return input_data.is_dir()

        # Also handle dict with 'folder' key
        if isinstance(input_data, dict):
            return 'folder' in input_data

        return False

    def parse(self, input_data: Any) -> ParserResult:
        """Parse a folder path into a configuration.

        Args:
            input_data: Folder path (str, Path) or dict with 'folder' key.

        Returns:
            ParserResult with configuration from scanned files.
        """
        # Extract folder path and optional params
        if isinstance(input_data, dict):
            folder_path = input_data.get('folder')
            global_params = input_data.get('params') or input_data.get('global_params')
        else:
            folder_path = input_data
            global_params = None

        if folder_path is None:
            return ParserResult(
                success=False,
                errors=["No folder path provided"],
                source_type="folder"
            )

        path = Path(folder_path)

        if not path.exists():
            return ParserResult(
                success=False,
                errors=[f"Folder does not exist: {folder_path}"],
                source_type="folder"
            )

        if not path.is_dir():
            return ParserResult(
                success=False,
                errors=[f"Path is not a directory: {folder_path}"],
                source_type="folder"
            )

        # Scan folder for data files
        config, warnings = self._scan_folder(path, global_params)

        # Check if any data was found
        has_train = config.get('train_x') is not None
        has_test = config.get('test_x') is not None

        if not has_train and not has_test:
            return ParserResult(
                success=False,
                errors=[
                    f"No data files found in folder: {folder_path}. "
                    "Expected files matching patterns like Xcal.csv, Xval.csv, etc."
                ],
                source_type="folder"
            )

        # Extract dataset name
        dataset_name = self._extract_name_from_path(path)

        return ParserResult(
            success=True,
            config=config,
            dataset_name=dataset_name,
            warnings=warnings,
            source_type="folder"
        )

    def _scan_folder(
        self,
        folder: Path,
        global_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Scan folder for data files.

        Args:
            folder: Path to folder to scan.
            global_params: Optional global loading parameters.

        Returns:
            Tuple of (config_dict, warnings_list).
        """
        config = {
            "train_x": None,
            "train_x_filter": None,
            "train_x_params": None,
            "train_y": None,
            "train_y_filter": None,
            "train_y_params": None,
            "train_group": None,
            "train_group_filter": None,
            "train_group_params": None,
            "train_params": None,
            "test_x": None,
            "test_x_filter": None,
            "test_x_params": None,
            "test_y": None,
            "test_y_filter": None,
            "test_y_params": None,
            "test_group": None,
            "test_group_filter": None,
            "test_group_params": None,
            "test_params": None,
            "global_params": global_params
        }

        warnings = []

        # Get all files in folder
        all_files = list(folder.glob("*"))

        # Match files to patterns
        for key, patterns in FILE_PATTERNS.items():
            matched_files = []

            for pattern in patterns:
                pattern_lower = pattern.lower()
                for file_path in all_files:
                    if not file_path.is_file():
                        continue

                    # Check if pattern matches (case-insensitive)
                    if pattern_lower in file_path.name.lower():
                        # Check file extension
                        if self._has_supported_extension(file_path):
                            file_posix = file_path.as_posix()
                            if file_posix not in matched_files:
                                matched_files.append(file_posix)

            # Assign matches to config
            if len(matched_files) == 1:
                config[key] = matched_files[0]
            elif len(matched_files) > 1:
                # Multi-source detected
                warnings.append(
                    f"Multiple files matched for {key}: {len(matched_files)} sources detected."
                )
                config[key] = matched_files

        return config, warnings

    def _has_supported_extension(self, path: Path) -> bool:
        """Check if file has a supported extension.

        Args:
            path: Path to check.

        Returns:
            True if file extension is supported.
        """
        name = path.name.lower()

        # Check for compound extensions first
        if name.endswith('.csv.gz') or name.endswith('.csv.zip'):
            return True

        # Check simple extensions
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
