#!/usr/bin/env python
"""
Export Pipeline Samples to Canonical Serialized Format
=======================================================

This script loads all 10 pipeline sample definitions and outputs their
canonical serialized format that nirs4all uses internally. This is useful
for webapp integration testing.

The canonical format:
- All class references use internal module paths (e.g., sklearn.preprocessing._data.StandardScaler)
- Only non-default parameters are included
- Tuples are converted to lists for JSON/YAML compatibility
- Comments are stripped

Usage:
    cd examples/pipeline_samples
    python export_canonical.py

    # Export to specific format
    python export_canonical.py --format json
    python export_canonical.py --format yaml

    # Export a specific pipeline
    python export_canonical.py -p 01_basic_regression.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add nirs4all to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.config.component_serialization import serialize_component

def load_pipeline_file(filepath: Path) -> dict:
    """Load pipeline definition from JSON or YAML file."""
    suffix = filepath.suffix.lower()
    with open(filepath, 'r') as f:
        if suffix == '.json':
            data = json.load(f)
        elif suffix in ('.yaml', '.yml'):
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    return data

def filter_comments(pipeline: list) -> list:
    """Remove _comment steps from pipeline."""
    filtered = []
    for step in pipeline:
        if isinstance(step, dict):
            # Skip comment-only steps
            if set(step.keys()) == {"_comment"}:
                continue
            # Remove _comment key from steps
            step = {k: v for k, v in step.items() if k != "_comment"}
        filtered.append(step)
    return filtered

def get_canonical_pipeline(filepath: Path) -> dict:
    """
    Load a pipeline file and return its canonical serialized form.

    The canonical form is what nirs4all uses internally:
    - Fully resolved class paths
    - Serialized with only non-default params
    - No comments
    """
    data = load_pipeline_file(filepath)

    # Extract pipeline steps
    if isinstance(data, dict):
        if "pipeline" in data:
            steps = data["pipeline"]
            name = data.get("name", filepath.stem)
            description = data.get("description", "")
        else:
            raise ValueError("Dictionary must have 'pipeline' key")
    elif isinstance(data, list):
        steps = data
        name = filepath.stem
        description = ""
    else:
        raise ValueError("Pipeline must be list or dict with 'pipeline' key")

    # Filter comments
    steps = filter_comments(steps)

    # Get canonical serialized form using nirs4all's serialization
    # This matches what PipelineConfigs does internally
    from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

    try:
        # Create PipelineConfigs to get the serialized form
        config = PipelineConfigs(steps, name=name, description=description)

        # Get the first (or only) configuration
        canonical_steps = config.steps[0] if config.steps else []

        return {
            "name": name,
            "description": description,
            "pipeline": canonical_steps,
            "has_generators": config.has_configurations,
            "num_configurations": len(config.steps),
        }
    except Exception as e:
        # Fall back to direct serialization if full config fails
        print(f"Warning: Could not create full config for {filepath.name}: {e}")
        canonical_steps = serialize_component(steps)
        return {
            "name": name,
            "description": description,
            "pipeline": canonical_steps,
            "has_generators": False,
            "num_configurations": 1,
            "error": str(e),
        }

def export_canonical(output_format: str = "json", specific_pipeline: str = None):
    """Export all pipeline samples to canonical format."""
    script_dir = Path(__file__).resolve().parent

    # Find pipeline files
    pipeline_files = sorted(
        list(script_dir.glob("*.json")) +
        list(script_dir.glob("*.yaml"))
    )

    # Exclude this script and test files
    pipeline_files = [
        f for f in pipeline_files
        if f.stem not in ('export_canonical', 'test_all_pipelines')
    ]

    # Filter if specific pipeline requested
    if specific_pipeline:
        pipeline_files = [f for f in pipeline_files if specific_pipeline in f.name]

    results = {}

    for filepath in pipeline_files:
        print(f"Processing: {filepath.name}...")
        try:
            canonical = get_canonical_pipeline(filepath)
            results[filepath.name] = canonical
            print(f"  ✓ {canonical.get('num_configurations', 1)} configuration(s)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[filepath.name] = {"error": str(e)}

    # Output
    output_dir = script_dir / "canonical"
    output_dir.mkdir(exist_ok=True)

    for filename, canonical in results.items():
        if "error" in canonical and "pipeline" not in canonical:
            continue

        base_name = Path(filename).stem
        if output_format == "json":
            output_file = output_dir / f"{base_name}_canonical.json"
            with open(output_file, 'w') as f:
                json.dump(canonical, f, indent=2)
        else:
            output_file = output_dir / f"{base_name}_canonical.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(canonical, f, default_flow_style=False, sort_keys=False)

        print(f"  → {output_file.name}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Export pipeline samples to canonical format")
    parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--pipeline", "-p",
        type=str,
        default=None,
        help="Process specific pipeline only (partial match)"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print canonical output to stdout instead of files"
    )
    args = parser.parse_args()

    results = export_canonical(args.format, args.pipeline)

    if args.print:
        print("\n" + "=" * 60)
        print("CANONICAL PIPELINE DEFINITIONS")
        print("=" * 60 + "\n")
        for filename, canonical in results.items():
            print(f"\n### {filename} ###\n")
            print(json.dumps(canonical, indent=2))
            print()

if __name__ == "__main__":
    main()
