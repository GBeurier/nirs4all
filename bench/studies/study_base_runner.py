"""
Study Base Runner - Core Logic for Study Execution
==================================================
Base class containing all logic for running training and reporting pipelines.
Not meant to be used directly - inherit from this in your config file.

This class supports two execution modes:
1. CLI mode: Passes parameters via command line arguments
2. Direct mode: Passes Python objects directly to functions (for complex configs)
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union


class StudyRunner:
    """Base runner class with all execution logic."""

    def __init__(self):
        """Initialize base runner."""
        # Dataset Configuration
        self.folder_list: List[str] = []
        self.aggregation_key_list: List[str] = []

        # Training Configuration
        self.test_mode: bool = False
        self.transfer_pp_preset: Optional[str] = None  # "fast", "balanced", "comprehensive"
        self.transfer_pp_selected: int = 10
        self.transfer_pp_config: Optional[Dict[str, Any]] = None  # Advanced: TransferPreprocessingSelector kwargs

        self.pls_pp_count: int = 40
        self.pls_pp_top_selected: int = 10
        self.pls_trials: int = 20
        self.opls_trials: int = 30
        self.test_lwpls: bool = False
        self.ridge_trials: int = 20

        self.tabpfn_trials: int = 10
        self.tabpfn_model_variants: List[str] = ['default', 'real', 'low-skew', 'small-samples']
        self.tabpfn_pp_max_count: int = 20
        self.tabpfn_pp_max_size: int = 3
        self.tabpfn_pp: Optional[List[Any]] = None  # Advanced: List of preprocessing pipelines

        self.global_pp: Optional[Dict[str, Any]] = None  # Advanced: Global preprocessing spec

        # Task type override: forces task type and disables automatic detection
        # Can be a single string (applied to all datasets) or a list (one per dataset)
        # Valid values: 'regression', 'binary_classification', 'multiclass_classification', 'auto' (default)
        self.task_type: Union[str, List[str]] = "auto"

        self.device: str = "cuda"
        self.verbose: int = 1
        self.show_plots: bool = False

        # Reporting Configuration
        self.workspace_path: str = "wk"
        self.output_dir: str = "reports"
        self.report_mode: str = "aggregated"  # 'raw', 'aggregated', or 'both'
        self.include_dataset_viz: bool = True
        self.report_aggregation_key: Optional[str] = None  # Override aggregation key for report (None = use first from list)
        self.report_exclude_models: Optional[List[str]] = None  # Models to exclude from report
        self.report_model_rename_map: Optional[Dict[str, str]] = None  # Model renaming map

        # Execution Control
        self.skip_training: bool = False
        self.skip_reporting: bool = False

    def _log(self, msg: str, symbol: str = "ℹ️") -> None:
        """Print a log message.

        Args:
            msg: Message to print
            symbol: Emoji/symbol prefix
        """
        print(f"{symbol} {msg}")

    def _validate_config(self) -> bool:
        """Validate configuration.

        Returns:
            True if valid, False otherwise
        """
        if not self.folder_list:
            self._log("Error: No datasets configured in folder_list", "❌")
            return False

        if len(self.folder_list) != len(self.aggregation_key_list):
            self._log(
                f"Error: {len(self.folder_list)} datasets but "
                f"{len(self.aggregation_key_list)} aggregation keys. "
                "They must match.",
                "❌"
            )
            return False

        return True

    def _run_command(self, cmd: List[str], description: str) -> int:
        """Run a command and handle errors.

        Args:
            cmd: Command to run as list of strings
            description: Description of what's being run

        Returns:
            Return code from command
        """
        print(f"\n{'='*70}")
        print(f"🚀 {description}")
        print(f"{'='*70}")
        if self.verbose >= 2:
            print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=Path(__file__).parent)

        if result.returncode != 0:
            self._log(f"{description} failed with code {result.returncode}", "❌")
        else:
            self._log(f"{description} completed successfully", "✅")

        return result.returncode

    def _find_generated_parquets(self, workspace: Path, datasets: List[str]) -> List[str]:
        """Find generated meta.parquet files for datasets.

        Args:
            workspace: Workspace directory path
            datasets: List of dataset folder names

        Returns:
            List of found parquet filenames (without extension)
        """
        found = []

        for dataset_path in datasets:
            # Use lowercase to match folder_to_name() convention in config_parser.py
            dataset_name = Path(dataset_path).name.lower()
            parquet_path = workspace / f"{dataset_name}.meta.parquet"

            if parquet_path.exists():
                found.append(dataset_name)
                if self.verbose >= 1:
                    print(f"  ✓ Found: {parquet_path}")
            else:
                if self.verbose >= 1:
                    print(f"  ⚠ Not found: {parquet_path}")

        return found

    def _use_advanced_mode(self) -> bool:
        """Determine if advanced mode should be used.

        Advanced mode uses direct function calls instead of CLI.

        Returns:
            True if any advanced features are configured
        """
        return (
            self.global_pp is not None or
            self.tabpfn_pp is not None or
            self.transfer_pp_config is not None
        )

    def _run_training_cli(self) -> int:
        """Run training via CLI (simple mode).

        Returns:
            Return code
        """
        cmd = [
            sys.executable,
            "study_training.py",
            "--device", self.device,
            "--verbose", str(self.verbose),
        ]

        if self.test_mode:
            cmd.append("--test-mode")

        cmd.extend(["--datasets"] + self.folder_list)
        cmd.extend(["--aggregation-keys"] + self.aggregation_key_list)

        cmd.extend([
            "--transfer-preset", self.transfer_pp_preset,
            "--transfer-selected", str(self.transfer_pp_selected),
            "--pls-pp-count", str(self.pls_pp_count),
            "--pls-pp-top", str(self.pls_pp_top_selected),
            "--pls-trials", str(self.pls_trials),
            "--opls-trials", str(self.opls_trials),
            "--ridge-trials", str(self.ridge_trials),
            "--tabpfn-trials", str(self.tabpfn_trials),
            "--tabpfn-pp-max-count", str(self.tabpfn_pp_max_count),
            "--tabpfn-pp-max-size", str(self.tabpfn_pp_max_size),
            "--workspace", self.workspace_path,
        ])

        cmd.extend(["--tabpfn-variants"] + self.tabpfn_model_variants)

        # Handle task_type: can be string or list
        if isinstance(self.task_type, list):
            # Pass as comma-separated for CLI
            if not all(t == "auto" for t in self.task_type):
                cmd.extend(["--task-type"] + self.task_type)
        elif self.task_type != "auto":
            cmd.extend(["--task-type", self.task_type])

        if self.test_lwpls:
            cmd.append("--test-lwpls")

        if self.show_plots:
            cmd.append("--show")

        return self._run_command(cmd, "Training Phase (CLI mode)")

    def _run_training_advanced(self) -> int:
        """Run training via direct function call (advanced mode).

        Returns:
            Return code (0 for success, 1 for failure)
        """
        try:
            # Import here to avoid circular imports
            import study_trainer as runner_module

            self._log("Training Phase (Advanced mode - direct function call)", "🚀")

            # Prepare config
            config = {
                'folder_list': self.folder_list,
                'aggregation_key_list': self.aggregation_key_list,
                'test_mode': self.test_mode,
                'workspace_path': self.workspace_path,
                'task_type': self.task_type,
                'transfer_pp_preset': self.transfer_pp_preset,
                'transfer_pp_selected': self.transfer_pp_selected,
                'transfer_pp_config': self.transfer_pp_config,
                'pls_pp_count': self.pls_pp_count,
                'pls_pp_top_selected': self.pls_pp_top_selected,
                'pls_trials': self.pls_trials,
                'opls_trials': self.opls_trials,
                'test_lwpls': self.test_lwpls,
                'ridge_trials': self.ridge_trials,
                'tabpfn_trials': self.tabpfn_trials,
                'tabpfn_model_variants': self.tabpfn_model_variants,
                'tabpfn_pp_max_count': self.tabpfn_pp_max_count,
                'tabpfn_pp_max_size': self.tabpfn_pp_max_size,
                'tabpfn_pp': self.tabpfn_pp,
                'global_pp': self.global_pp,
                'device': self.device,
                'verbose': self.verbose,
                'show_plots': self.show_plots,
            }

            runner_module.run_study(config)
            self._log("Training completed successfully", "✅")
            return 0

        except Exception as e:
            self._log(f"Training failed: {e}", "❌")
            if self.verbose >= 2:
                import traceback
                traceback.print_exc()
            return 1

    def _run_training(self) -> int:
        """Run training phase.

        Automatically chooses CLI or advanced mode based on configuration.

        Returns:
            Return code
        """
        if self._use_advanced_mode():
            return self._run_training_advanced()
        else:
            return self._run_training_cli()

    def _run_reporting(self) -> int:
        """Run reporting phase.

        Returns:
            Return code
        """
        workspace_path = Path(self.workspace_path)

        print(f"\n{'='*70}")
        print("📂 Looking for generated parquet files...")
        print("=" * 70)

        filenames = self._find_generated_parquets(workspace_path, self.folder_list)

        if not filenames:
            self._log(
                "No parquet files found. Skipping reporting phase.\n"
                "    (This is normal if training was skipped or failed)",
                "⚠️"
            )
            return 0

        if self.verbose >= 1:
            print(f"\nFound {len(filenames)} datasets to report on")

        cmd = [
            sys.executable,
            "study_report.py",
            "--workspace", self.workspace_path,
            "--output", self.output_dir,
            "--mode", self.report_mode,
        ]

        cmd.extend(["--filenames"] + filenames)

        # Determine aggregation key: use explicit report key, or first from list if unique
        agg_key = self.report_aggregation_key
        if agg_key is None and self.aggregation_key_list and len(set(self.aggregation_key_list)) == 1:
            agg_key = self.aggregation_key_list[0]
        if agg_key:
            cmd.extend(["--aggregation-key", agg_key])

        if self.include_dataset_viz and self.folder_list:
            cmd.extend(["--dataset-folder", self.folder_list[0]])

        # Pass model filtering options
        if self.report_exclude_models:
            cmd.extend(["--exclude-models"] + self.report_exclude_models)

        if self.report_model_rename_map:
            # Pass as key=value pairs
            rename_pairs = [f"{k}={v}" for k, v in self.report_model_rename_map.items()]
            cmd.extend(["--rename-models"] + rename_pairs)

        return self._run_command(cmd, "Reporting Phase")

    def run(self) -> int:
        """Execute the complete study pipeline.

        This is the main entry point. Call this method to run training and reporting.

        Returns:
            0 for success, 1 for failure
        """
        # Validate configuration
        if not self._validate_config():
            return 1

        # Print header
        print("=" * 70)
        print("NIRS4ALL UNIFIED STUDY RUNNER")
        print("=" * 70)
        print(f"Mode: {'Advanced (Python objects)' if self._use_advanced_mode() else 'Simple (CLI)'}")
        print(f"Test mode: {self.test_mode}")
        print(f"Device: {self.device}")
        print(f"Datasets: {len(self.folder_list)}")
        print()

        # Phase 1: Training
        if not self.skip_training:
            result = self._run_training()
            if result != 0:
                self._log("Training phase failed. Aborting.", "❌")
                return 1
        else:
            self._log("Skipping training phase", "⏭️")

        # Phase 2: Reporting
        if not self.skip_reporting:
            result = self._run_reporting()
            if result != 0:
                self._log("Reporting phase completed with errors", "⚠️")
                return 1
        else:
            self._log("Skipping reporting phase", "⏭️")

        # Success
        print("\n" + "=" * 70)
        print("✅ STUDY PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Reports available in: {self.output_dir}/")
        print("=" * 70)

        return 0
