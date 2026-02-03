#!/usr/bin/env python3
"""
Link all test datasets to the nirs4all webapp.

This script reads all YAML configs from sample_configs/datasets/ and
links each dataset to the current workspace via the webapp API.

Usage:
    python scripts/link_test_datasets.py [--api-url URL] [--dry-run]

Requirements:
    - The webapp backend must be running (default: http://localhost:8000)
    - A workspace must be selected in the webapp
"""

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


def load_config(config_path: Path) -> dict:
    """Load and parse a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_config(yaml_config: dict, base_path: Path) -> dict:
    """Convert YAML config to webapp LinkDatasetRequest format."""
    config = {}

    # Copy relevant fields
    if "global_params" in yaml_config:
        gp = yaml_config["global_params"]
        config["delimiter"] = gp.get("delimiter", ";")
        config["decimal_separator"] = gp.get("decimal_separator", ".")
        config["has_header"] = gp.get("has_header", True)
        config["header_unit"] = gp.get("header_unit", "nm")
        if "encoding" in gp:
            config["encoding"] = gp["encoding"]

    if "signal_type" in yaml_config:
        config["signal_type"] = yaml_config["signal_type"]

    if "task_type" in yaml_config:
        config["task_type"] = yaml_config["task_type"]

    # Map file paths
    if "train_x" in yaml_config:
        config["train_x"] = str(base_path / yaml_config["train_x"])
    if "train_y" in yaml_config:
        config["train_y"] = str(base_path / yaml_config["train_y"])
    if "test_x" in yaml_config:
        config["test_x"] = str(base_path / yaml_config["test_x"])
    if "test_y" in yaml_config:
        config["test_y"] = str(base_path / yaml_config["test_y"])
    if "train_group" in yaml_config:
        config["train_group"] = str(base_path / yaml_config["train_group"])
    if "test_group" in yaml_config:
        config["test_group"] = str(base_path / yaml_config["test_group"])

    return config


def link_dataset(api_url: str, dataset_path: str, config: dict) -> dict:
    """Link a dataset via the webapp API."""
    url = f"{api_url}/api/datasets/link"
    payload = {
        "path": dataset_path,
        "config": config,
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def check_workspace(api_url: str) -> bool:
    """Check if a workspace is selected."""
    try:
        response = requests.get(f"{api_url}/api/workspace", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("workspace") is not None
    except Exception as e:
        print(f"Error checking workspace: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Link test datasets to webapp")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Webapp API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making API calls",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only link configs matching this pattern (e.g., 'A*' or 'B01')",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    examples_dir = script_dir.parent
    configs_dir = examples_dir / "sample_configs" / "datasets"

    if not configs_dir.exists():
        print(f"Error: Config directory not found: {configs_dir}")
        sys.exit(1)

    # Get all YAML configs
    config_files = sorted(configs_dir.glob("*.yaml"))

    if args.filter:
        import fnmatch
        config_files = [f for f in config_files if fnmatch.fnmatch(f.stem, args.filter)]

    if not config_files:
        print("No config files found matching criteria")
        sys.exit(1)

    print(f"Found {len(config_files)} dataset configs")

    # Check workspace (skip in dry-run mode)
    if not args.dry_run:
        print(f"\nChecking workspace at {args.api_url}...")
        if not check_workspace(args.api_url):
            print("Error: No workspace is selected. Please select a workspace in the webapp first.")
            sys.exit(1)
        print("Workspace OK")

    # Process each config
    results = {"success": [], "failed": [], "skipped": []}

    print(f"\n{'=' * 60}")
    print("Linking datasets...")
    print("=" * 60)

    for config_file in config_files:
        name = config_file.stem
        print(f"\n[{name}] ", end="")

        try:
            # Load YAML config
            yaml_config = load_config(config_file)

            # Build webapp config
            webapp_config = build_dataset_config(yaml_config, examples_dir)

            # Determine dataset path (folder containing the files)
            if "train_x" in yaml_config:
                rel_path = yaml_config["train_x"]
                dataset_path = str(examples_dir / Path(rel_path).parent)
            elif "sources" in yaml_config and yaml_config["sources"]:
                # Multi-source: use first source's train_x
                first_source = yaml_config["sources"][0]
                if "train_x" in first_source:
                    rel_path = first_source["train_x"]
                    dataset_path = str(examples_dir / Path(rel_path).parent)
                else:
                    print("SKIP (no train_x in sources)")
                    results["skipped"].append(name)
                    continue
            elif "variations" in yaml_config and yaml_config["variations"]:
                # Variations format: use first variation's train_x
                first_var = yaml_config["variations"][0]
                if "train_x" in first_var:
                    rel_path = first_var["train_x"]
                    dataset_path = str(examples_dir / Path(rel_path).parent)
                else:
                    print("SKIP (no train_x in variations)")
                    results["skipped"].append(name)
                    continue
            elif "files" in yaml_config:
                # Files array format
                x_files = [f for f in yaml_config["files"] if f.get("type") == "X"]
                if x_files:
                    rel_path = x_files[0]["path"]
                    dataset_path = str(examples_dir / Path(rel_path).parent)
                else:
                    print("SKIP (no X files)")
                    results["skipped"].append(name)
                    continue
            else:
                print("SKIP (no train_x)")
                results["skipped"].append(name)
                continue

            if args.dry_run:
                print(f"DRY-RUN: Would link {dataset_path}")
                print(f"  Config: {json.dumps(webapp_config, indent=2)[:200]}...")
                results["success"].append(name)
                continue

            # Link the dataset
            result = link_dataset(args.api_url, dataset_path, webapp_config)
            print(f"OK - ID: {result.get('id', 'N/A')}")
            results["success"].append(name)

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            print(f"FAIL - {error_detail}")
            results["failed"].append((name, error_detail))

        except Exception as e:
            print(f"FAIL - {e}")
            results["failed"].append((name, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"  Linked:  {len(results['success'])}")
    print(f"  Failed:  {len(results['failed'])}")
    print(f"  Skipped: {len(results['skipped'])}")

    if results["failed"]:
        print("\nFailed datasets:")
        for name, error in results["failed"]:
            print(f"  - {name}: {error[:80]}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
