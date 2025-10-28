"""
LibraryManager for managing saved pipeline templates and trained models.

Manages three types of saves:
- templates: config-only (no trained artifacts)
- filtered: config + metrics only
- pipeline: full pipeline with binaries
- fullrun: entire run directory
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class LibraryManager:
    """Manage library of saved pipelines."""

    def __init__(self, library_dir: Path):
        """Initialize library manager.

        Args:
            library_dir: Root library directory (typically workspace/library)
        """
        self.library_dir = Path(library_dir)
        self.templates_dir = self.library_dir / "templates"
        self.trained_dir = self.library_dir / "trained"

        # Three types of trained pipelines
        self.filtered_dir = self.trained_dir / "filtered"
        self.pipeline_dir = self.trained_dir / "pipeline"
        self.fullrun_dir = self.trained_dir / "fullrun"

        # Initialize directories
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        self.fullrun_dir.mkdir(parents=True, exist_ok=True)

    def save_template(
        self,
        pipeline_config: Dict,
        name: str,
        description: str = ""
    ) -> Path:
        """Save pipeline template (config only, no trained artifacts).

        Args:
            pipeline_config: Pipeline configuration dictionary
            name: Template name
            description: Optional description

        Returns:
            Path to saved template file
        """
        template_file = self.templates_dir / f"{name}.json"

        template = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "config": pipeline_config,
            "type": "template"
        }

        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)

        return template_file

    def save_filtered(
        self,
        pipeline_dir: Path,
        name: str,
        description: str = ""
    ) -> Path:
        """Save filtered pipeline (config + metrics only).

        Useful for tracking experiments and comparing configurations.

        Args:
            pipeline_dir: Source pipeline directory
            name: Save name
            description: Optional description

        Returns:
            Path to saved filtered pipeline
        """
        dest_dir = self.filtered_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy only JSON files
        if (pipeline_dir / "pipeline.json").exists():
            shutil.copy2(pipeline_dir / "pipeline.json", dest_dir)
        if (pipeline_dir / "metrics.json").exists():
            shutil.copy2(pipeline_dir / "metrics.json", dest_dir)

        # Extract n_features from manifest if available
        n_features = self._extract_n_features(pipeline_dir)

        # Add metadata
        metadata = {
            "name": name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "type": "filtered",
            "source": str(pipeline_dir),
            "n_features": n_features
        }
        with open(dest_dir / "library_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_dir

    def save_pipeline_full(
        self,
        run_dir: Path,
        pipeline_dir: Path,
        name: str,
        description: str = ""
    ) -> Path:
        """Save full pipeline (all files + binaries).

        Useful for deployment, retraining, and full reproducibility.

        Args:
            run_dir: Parent run directory (contains _binaries/)
            pipeline_dir: Pipeline directory
            name: Save name
            description: Optional description

        Returns:
            Path to saved pipeline
        """
        dest_dir = self.pipeline_dir / name

        # Copy entire pipeline folder
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(pipeline_dir, dest_dir)

        # Copy referenced binaries from run's _binaries/
        binaries_src = run_dir / "_binaries"
        binaries_dest = dest_dir / "_binaries"
        binaries_dest.mkdir(exist_ok=True)

        # Parse pipeline.json to find referenced artifacts
        pipeline_json = pipeline_dir / "pipeline.json"
        if pipeline_json.exists():
            with open(pipeline_json) as f:
                pipeline_config = json.load(f)

            if "artifacts" in pipeline_config:
                for artifact_ref in pipeline_config["artifacts"]:
                    artifact_path = artifact_ref["path"]  # e.g., "../_binaries/scaler_a1b2c3.pkl"
                    artifact_filename = Path(artifact_path).name
                    src_file = binaries_src / artifact_filename
                    if src_file.exists():
                        shutil.copy2(src_file, binaries_dest)

        # Extract n_features from manifest
        n_features = self._extract_n_features(pipeline_dir)

        # Add metadata
        metadata = {
            "name": name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "type": "pipeline",
            "source": str(pipeline_dir),
            "n_features": n_features
        }
        with open(dest_dir / "library_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_dir

    def save_fullrun(
        self,
        run_dir: Path,
        name: str,
        description: str = ""
    ) -> Path:
        """Save entire run (all pipelines + binaries + data).

        Useful for complete experiment archiving and cross-dataset comparison.

        Args:
            run_dir: Run directory to save
            name: Save name
            description: Optional description

        Returns:
            Path to saved run
        """
        dest_dir = self.fullrun_dir / name

        # Copy entire run folder
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(run_dir, dest_dir)

        # Add metadata
        metadata = {
            "name": name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "type": "fullrun",
            "source": str(run_dir)
        }
        with open(dest_dir / "library_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_dir

    def list_templates(self) -> List[Dict]:
        """List all available templates.

        Returns:
            List of template dictionaries
        """
        templates = []
        for file in self.templates_dir.glob("*.json"):
            with open(file) as f:
                templates.append(json.load(f))
        return templates

    def load_template(self, name: str) -> Dict:
        """Load a template by name.

        Args:
            name: Template name

        Returns:
            Template dictionary
        """
        template_file = self.templates_dir / f"{name}.json"
        with open(template_file) as f:
            return json.load(f)

    def list_filtered(self) -> List[Dict]:
        """List all filtered pipelines.

        Returns:
            List of metadata dictionaries
        """
        filtered = []
        for metadata_file in self.filtered_dir.glob("*/library_metadata.json"):
            with open(metadata_file) as f:
                filtered.append(json.load(f))
        return filtered

    def list_pipelines(self) -> List[Dict]:
        """List all full pipelines.

        Returns:
            List of metadata dictionaries
        """
        pipelines = []
        for metadata_file in self.pipeline_dir.glob("*/library_metadata.json"):
            with open(metadata_file) as f:
                pipelines.append(json.load(f))
        return pipelines

    def list_fullruns(self) -> List[Dict]:
        """List all saved full runs.

        Returns:
            List of metadata dictionaries
        """
        fullruns = []
        for metadata_file in self.fullrun_dir.glob("*/library_metadata.json"):
            with open(metadata_file) as f:
                fullruns.append(json.load(f))
        return fullruns

    def _extract_n_features(self, pipeline_dir: Path) -> Optional[int]:
        """Extract number of features from pipeline directory.

        Checks pipeline.json or manifest files for n_features information.

        Args:
            pipeline_dir: Pipeline directory path

        Returns:
            Number of features, or None if not found
        """
        pipeline_dir = Path(pipeline_dir)

        # Try to get from pipeline.json
        pipeline_json = pipeline_dir / "pipeline.json"
        if pipeline_json.exists():
            try:
                with open(pipeline_json, 'r') as f:
                    data = json.load(f)
                    if 'n_features' in data:
                        return data['n_features']
            except:
                pass

        # Try to get from manifest.yaml
        manifest_file = pipeline_dir / "manifest.yaml"
        if manifest_file.exists():
            try:
                import yaml
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                    if 'n_features' in manifest:
                        return manifest['n_features']
                    if 'dataset' in manifest and 'n_features' in manifest['dataset']:
                        return manifest['dataset']['n_features']
            except:
                pass

        # Try to extract from folds CSV if it exists
        for csv_file in pipeline_dir.glob("fold*.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, nrows=1)
                # Exclude target and metadata columns
                exclude_cols = ['target', 'y', 'fold', 'split', 'partition']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                if feature_cols:
                    return len(feature_cols)
            except:
                pass

        return None
