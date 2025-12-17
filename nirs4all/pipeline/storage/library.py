"""Pipeline library manager - save and load reusable pipeline templates.

The library allows users to:
- Save successful pipeline configurations as templates
- Load templates for reuse across datasets
- Organize templates by category (preprocessing, modeling, full_pipeline, etc.)
- Search and filter templates by metadata
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class PipelineLibrary:
    """Manages reusable pipeline templates in the workspace library.

    Templates are stored in: workspace/library/{category}/{template_name}/
    Each template contains:
    - pipeline.json: The pipeline configuration
    - metadata.json: Description, tags, performance metrics, etc.
    - README.md: Human-readable documentation
    """

    def __init__(self, workspace_path: Path):
        """Initialize library manager.

        Args:
            workspace_path: Root workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.library_path = self.workspace_path / "library"
        self.library_path.mkdir(exist_ok=True)

    def save_template(
        self,
        pipeline_config: Dict[str, Any],
        name: str,
        category: str = "general",
        description: str = "",
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        notes: str = "",
        overwrite: bool = False
    ) -> Path:
        """Save a pipeline configuration as a reusable template.

        Args:
            pipeline_config: Pipeline configuration dictionary
            name: Template name (will be sanitized for filesystem)
            category: Category/folder (e.g., "preprocessing", "modeling", "full_pipeline")
            description: Short description of the template
            tags: List of tags for filtering (e.g., ["classification", "neural_network"])
            metrics: Performance metrics (e.g., {"accuracy": 0.95, "f1": 0.93})
            notes: Additional notes or usage instructions
            overwrite: Whether to overwrite existing template

        Returns:
            Path to saved template directory

        Raises:
            FileExistsError: If template exists and overwrite=False
            ValueError: If name contains invalid characters
        """
        # Sanitize name
        safe_name = self._sanitize_name(name)
        if not safe_name:
            raise ValueError(f"Invalid template name: {name}")

        # Create category directory
        category_path = self.library_path / category
        category_path.mkdir(exist_ok=True)

        # Create template directory
        template_path = category_path / safe_name
        if template_path.exists() and not overwrite:
            raise FileExistsError(
                f"Template '{name}' already exists in category '{category}'. "
                f"Use overwrite=True to replace."
            )

        template_path.mkdir(exist_ok=True)

        # Save pipeline configuration
        pipeline_file = template_path / "pipeline.json"
        with open(pipeline_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_config, f, indent=2, default=str)

        # Create metadata
        metadata = {
            "name": name,
            "category": category,
            "description": description,
            "tags": tags or [],
            "metrics": metrics or {},
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "version": "1.0"
        }

        # Save metadata
        metadata_file = template_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Create README
        readme_content = self._generate_readme(name, description, tags, metrics, notes)
        readme_file = template_path / "README.md"
        readme_file.write_text(readme_content, encoding='utf-8')

        logger.success(f"Template '{name}' saved to library/{category}/{safe_name}")
        return template_path

    def load_template(self, name: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Load a pipeline template by name.

        Args:
            name: Template name
            category: Optional category to search in (searches all if None)

        Returns:
            Pipeline configuration dictionary

        Raises:
            FileNotFoundError: If template not found
        """
        template_path = self._find_template(name, category)
        if not template_path:
            cat_msg = f" in category '{category}'" if category else ""
            raise FileNotFoundError(f"Template '{name}' not found{cat_msg}")

        pipeline_file = template_path / "pipeline.json"
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.success(f"Loaded template '{name}' from library")
        return config

    def get_template_metadata(
        self,
        name: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metadata for a template.

        Args:
            name: Template name
            category: Optional category to search in

        Returns:
            Metadata dictionary
        """
        template_path = self._find_template(name, category)
        if not template_path:
            cat_msg = f" in category '{category}'" if category else ""
            raise FileNotFoundError(f"Template '{name}' not found{cat_msg}")

        metadata_file = template_path / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all templates, optionally filtered by category and tags.

        Args:
            category: Optional category filter
            tags: Optional list of tags to filter by (matches any)

        Returns:
            List of template metadata dictionaries
        """
        templates = []

        # Determine which categories to search
        if category:
            categories = [category]
        else:
            categories = [d.name for d in self.library_path.iterdir() if d.is_dir()]

        # Search each category
        for cat in categories:
            cat_path = self.library_path / cat
            if not cat_path.exists():
                continue

            for template_dir in cat_path.iterdir():
                if not template_dir.is_dir():
                    continue

                metadata_file = template_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Filter by tags if specified
                    if tags:
                        template_tags = set(metadata.get('tags', []))
                        if not any(tag in template_tags for tag in tags):
                            continue

                    templates.append(metadata)
                except Exception:
                    continue

        return templates

    def delete_template(self, name: str, category: Optional[str] = None) -> None:
        """Delete a template from the library.

        Args:
            name: Template name
            category: Optional category to search in

        Raises:
            FileNotFoundError: If template not found
        """
        template_path = self._find_template(name, category)
        if not template_path:
            cat_msg = f" in category '{category}'" if category else ""
            raise FileNotFoundError(f"Template '{name}' not found{cat_msg}")

        shutil.rmtree(template_path)
        logger.success(f"Template '{name}' deleted from library")

    def copy_from_pipeline(
        self,
        pipeline_dir: Path,
        name: str,
        category: str = "general",
        description: str = "",
        tags: Optional[List[str]] = None,
        extract_metrics: bool = True
    ) -> Path:
        """Copy a successful pipeline to the library as a template.

        Args:
            pipeline_dir: Path to pipeline directory (e.g., workspace/runs/.../0001_hash/)
            name: Template name
            category: Category for the template
            description: Description of the template
            tags: Tags for filtering
            extract_metrics: Whether to extract metrics from manifest

        Returns:
            Path to saved template
        """
        pipeline_dir = Path(pipeline_dir)

        # Load pipeline config
        pipeline_json = pipeline_dir / "pipeline.json"
        if not pipeline_json.exists():
            raise FileNotFoundError(f"No pipeline.json found in {pipeline_dir}")

        with open(pipeline_json, 'r', encoding='utf-8') as f:
            pipeline_config = json.load(f)

        # Extract metrics from manifest if requested
        metrics = {}
        if extract_metrics:
            manifest_file = pipeline_dir / "manifest.yaml"
            if manifest_file.exists():
                import yaml
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                    # Extract any metrics stored in manifest
                    metrics = manifest.get('metrics', {})

        # Save as template
        return self.save_template(
            pipeline_config=pipeline_config,
            name=name,
            category=category,
            description=description,
            tags=tags,
            metrics=metrics
        )

    def export_template(
        self,
        name: str,
        export_path: Union[str, Path],
        category: Optional[str] = None
    ) -> Path:
        """Export a template to a standalone directory.

        Args:
            name: Template name
            export_path: Destination directory
            category: Optional category to search in

        Returns:
            Path to exported template
        """
        template_path = self._find_template(name, category)
        if not template_path:
            cat_msg = f" in category '{category}'" if category else ""
            raise FileNotFoundError(f"Template '{name}' not found{cat_msg}")

        export_path = Path(export_path)
        shutil.copytree(template_path, export_path, dirs_exist_ok=True)
        logger.success(f"Template '{name}' exported to {export_path}")
        return export_path

    def import_template(
        self,
        import_path: Union[str, Path],
        category: str = "general",
        overwrite: bool = False
    ) -> Path:
        """Import a template from an external directory.

        Args:
            import_path: Path to template directory
            category: Category to import into
            overwrite: Whether to overwrite existing template

        Returns:
            Path to imported template in library
        """
        import_path = Path(import_path)

        # Load metadata to get name
        metadata_file = import_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.json found in {import_path}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        name = metadata.get('name', import_path.name)
        safe_name = self._sanitize_name(name)

        # Create destination
        category_path = self.library_path / category
        category_path.mkdir(exist_ok=True)
        template_path = category_path / safe_name

        if template_path.exists() and not overwrite:
            raise FileExistsError(
                f"Template '{name}' already exists. Use overwrite=True to replace."
            )

        # Copy template
        shutil.copytree(import_path, template_path, dirs_exist_ok=True)
        logger.success(f"Template '{name}' imported to library/{category}/{safe_name}")
        return template_path

    def _find_template(self, name: str, category: Optional[str] = None) -> Optional[Path]:
        """Find a template by name, optionally in a specific category.

        Returns:
            Path to template directory, or None if not found
        """
        safe_name = self._sanitize_name(name)

        # Search in specific category
        if category:
            template_path = self.library_path / category / safe_name
            if template_path.exists():
                return template_path
            return None

        # Search all categories
        for cat_dir in self.library_path.iterdir():
            if not cat_dir.is_dir():
                continue

            template_path = cat_dir / safe_name
            if template_path.exists():
                return template_path

        return None

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a template name for filesystem use.

        Args:
            name: Template name

        Returns:
            Sanitized name safe for filesystem
        """
        # Replace spaces with underscores
        safe_name = name.replace(' ', '_')

        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '')

        # Convert to lowercase for consistency
        safe_name = safe_name.lower()

        return safe_name

    def _generate_readme(
        self,
        name: str,
        description: str,
        tags: Optional[List[str]],
        metrics: Optional[Dict[str, float]],
        notes: str
    ) -> str:
        """Generate README content for a template.

        Returns:
            Markdown formatted README content
        """
        readme = f"# {name}\n\n"

        if description:
            readme += f"{description}\n\n"

        if tags:
            readme += f"**Tags**: {', '.join(tags)}\n\n"

        if metrics:
            readme += "## Performance Metrics\n\n"
            for metric, value in metrics.items():
                readme += f"- **{metric}**: {value}\n"
            readme += "\n"

        readme += "## Usage\n\n"
        readme += "```python\n"
        readme += "from nirs4all.pipeline import PipelineRunner, PipelineLibrary\n\n"
        readme += "# Load template from library\n"
        readme += f"library = PipelineLibrary(workspace_path)\n"
        readme += f"pipeline_config = library.load_template('{name}')\n\n"
        readme += "# Run with your dataset\n"
        readme += "runner = PipelineRunner(workspace_path=workspace_path)\n"
        readme += "results = runner.run(pipeline_config, dataset)\n"
        readme += "```\n\n"

        if notes:
            readme += "## Notes\n\n"
            readme += f"{notes}\n\n"

        readme += f"---\n\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        return readme
