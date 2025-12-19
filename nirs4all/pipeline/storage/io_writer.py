"""Pipeline file writer - handles file I/O within pipeline directories."""
import json
import warnings
from pathlib import Path
from typing import Any, Optional, Union


class PipelineWriter:
    """Writes files within a pipeline directory.

    Focused responsibility: File I/O operations for a single pipeline.
    """

    def __init__(self, pipeline_dir: Path, save_charts: bool = True):
        """Initialize writer for a specific pipeline directory.

        Args:
            pipeline_dir: Directory for this pipeline
            save_charts: Whether to save charts and visual outputs
        """
        self.pipeline_dir = Path(pipeline_dir)
        self.save_charts = save_charts

    def save_file(
        self,
        filename: str,
        content: str,
        overwrite: bool = True,
        encoding: str = 'utf-8',
        warn_on_overwrite: bool = True
    ) -> Path:
        """Save a text file to the pipeline directory.

        Args:
            filename: Name of file to create
            content: Text content to write
            overwrite: Whether to overwrite existing files
            encoding: Text encoding (default: utf-8)
            warn_on_overwrite: Whether to warn when overwriting

        Returns:
            Path to saved file

        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        filepath = self.pipeline_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Use overwrite=True to replace."
            )

        if filepath.exists() and warn_on_overwrite:
            warnings.warn(f"Overwriting existing file: {filename}")

        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)

        return filepath

    def save_json(
        self,
        filename: str,
        data: Any,
        overwrite: bool = True,
        indent: Optional[int] = 2
    ) -> Path:
        """Save data as JSON file.

        Args:
            filename: Name of file (will add .json if missing)
            data: Data to serialize as JSON
            overwrite: Whether to overwrite existing files
            indent: JSON indentation (None for compact)

        Returns:
            Path to saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'

        json_content = json.dumps(data, indent=indent, default=str)
        return self.save_file(filename, json_content, overwrite, warn_on_overwrite=False)

    def save_output(
        self,
        name: str,
        data: Union[bytes, str],
        extension: str = ".png"
    ) -> Optional[Path]:
        """Save a human-readable output file (chart, report, etc.).

        Args:
            name: Output name (e.g., "2D_Chart")
            data: Binary or text data to save
            extension: File extension (e.g., ".png", ".csv", ".txt")

        Returns:
            Path to saved file, or None if save_charts=False
        """
        if not self.save_charts:
            return None

        # Create filename
        if not name.endswith(extension):
            filename = f"{name}{extension}"
        else:
            filename = name

        filepath = self.pipeline_dir / filename

        # Save the file
        if isinstance(data, bytes):
            filepath.write_bytes(data)
        elif isinstance(data, str):
            filepath.write_text(data, encoding="utf-8")
        else:
            raise TypeError(f"Data must be bytes or str, got {type(data)}")

        return filepath

    def list_files(self) -> list[str]:
        """List all files in the pipeline directory.

        Returns:
            List of filenames
        """
        return [f.name for f in self.pipeline_dir.glob("*") if f.is_file()]

    def get_path(self) -> Path:
        """Get the pipeline directory path.

        Returns:
            Path to pipeline directory
        """
        return self.pipeline_dir
