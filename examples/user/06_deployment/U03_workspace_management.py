"""
U03 - Workspace Management: Sessions and Organization
======================================================

Manage runs, sessions, and workspace organization for experiments.

This tutorial covers:

* Session context manager for multiple runs
* Workspace structure and navigation
* DuckDB-backed prediction storage
* Library management for pipeline templates

Prerequisites
-------------
Complete :ref:`U02_export_bundle` first.

Next Steps
----------
See :ref:`U04_sklearn_integration` for sklearn compatibility.

Duration: ~5 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
import shutil
from pathlib import Path

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U03 Workspace Management Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Why Sessions and Workspace Management?
# =============================================================================
print("\n" + "=" * 60)
print("U03 - Workspace Management: Sessions and Organization")
print("=" * 60)

print("""
Sessions and workspace management help you:

  📊 ORGANIZE EXPERIMENTS
     - Consistent workspace structure
     - Track all predictions across runs
     - Easy comparison and archival

  ⚡ EFFICIENT EXECUTION
     - Share configuration across runs
     - Reduced overhead for multiple runs
     - Automatic resource cleanup

  📁 WORKSPACE STRUCTURE
     workspace/
     ├── store.duckdb    # All structured data (runs, predictions, etc.)
     ├── artifacts/      # Content-addressed binary artifacts
     ├── exports/        # Exported best results
     └── library/        # Saved pipeline templates
""")

# =============================================================================
# Section 2: Basic Session Usage
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Basic Session Usage")
print("-" * 60)

print("""
Sessions share configuration across multiple runs.
""")

# Without session - each run is independent
print("Without session (independent runs):")
result_no_session = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
    dataset="sample_data/regression",
    name="NoSession_PLS",
    verbose=0,
    plots_visible=False
)
print(f"  RMSE: {result_no_session.best_rmse:.4f}")

# With session - runs share configuration
print("\nWith session (shared configuration):")
with nirs4all.session(verbose=0, save_artifacts=True, plots_visible=False) as s:
    # Run 1: PLS
    result1 = nirs4all.run(
        pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
        dataset="sample_data/regression",
        name="Session_PLS",
        session=s
    )
    print(f"  PLS: RMSE = {result1.best_rmse:.4f}")

    # Run 2: Ridge
    result2 = nirs4all.run(
        pipeline=[MinMaxScaler(), Ridge(alpha=1.0)],
        dataset="sample_data/regression",
        name="Session_Ridge",
        session=s
    )
    print(f"  Ridge: RMSE = {result2.best_rmse:.4f}")

print("Session closed - resources cleaned up")

# =============================================================================
# Section 3: Preprocessing Comparison with Session
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Preprocessing Comparison with Session")
print("-" * 60)

print("""
Compare preprocessing methods efficiently using a session.
""")

# Define preprocessing methods
preprocessing_configs = {
    "Baseline": [],
    "MinMax": [MinMaxScaler()],
    "SNV": [MinMaxScaler(), StandardNormalVariate()],
    "SavGol": [MinMaxScaler(), SavitzkyGolay(window_length=11)],
}

results = {}

with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
    print("Running preprocessing comparison...")

    for name, preproc in preprocessing_configs.items():
        pipeline = preproc + [
            ShuffleSplit(n_splits=3, test_size=0.25),
            {"model": PLSRegression(n_components=10)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset="sample_data/regression",
            name=f"Preproc_{name}",
            session=s
        )

        results[name] = result.best_rmse
        print(f"  {name:15s} RMSE: {result.best_rmse:.4f}")

# Find best
best_name = min(results, key=results.get)
print(f"\nBest preprocessing: {best_name}")

# =============================================================================
# Section 4: Hyperparameter Sweep with Session
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Hyperparameter Sweep with Session")
print("-" * 60)

print("""
Explore hyperparameters systematically within a session.
""")

n_components_range = [3, 5, 10, 15, 20]
sweep_results = {}

with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
    print("PLS n_components sweep:")

    for n_comp in n_components_range:
        result = nirs4all.run(
            pipeline=[
                MinMaxScaler(),
                StandardNormalVariate(),
                ShuffleSplit(n_splits=3, test_size=0.25),
                {"model": PLSRegression(n_components=n_comp)}
            ],
            dataset="sample_data/regression",
            name=f"PLS_{n_comp}",
            session=s
        )

        sweep_results[n_comp] = result.best_rmse
        print(f"  n_components={n_comp:2d}: RMSE = {result.best_rmse:.4f}")

# Find optimal
best_n = min(sweep_results, key=sweep_results.get)
print(f"\nOptimal n_components: {best_n} (RMSE = {sweep_results[best_n]:.4f})")

# =============================================================================
# Section 5: Workspace Structure
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Workspace Structure")
print("-" * 60)

print("""
The workspace organizes all experiment outputs:
""")

# Create temporary workspace for demonstration
demo_workspace = Path("workspace_demo")
if demo_workspace.exists():
    shutil.rmtree(demo_workspace)

# Run with custom workspace
with nirs4all.session(
    workspace_path=str(demo_workspace),
    verbose=0,
    save_artifacts=True,
    plots_visible=False
) as s:
    result = nirs4all.run(
        pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
        dataset="sample_data/regression",
        name="WorkspaceDemo",
        session=s
    )

# Show structure
print("\nWorkspace structure:")
print(f"{demo_workspace}/")

if demo_workspace.exists():
    for item in sorted(demo_workspace.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(demo_workspace)
            depth = len(rel_path.parts) - 1
            indent = "  " * depth
            print(f"  {indent}📄 {item.name}")
        elif item.is_dir() and not any(p.startswith('_') for p in item.parts):
            rel_path = item.relative_to(demo_workspace)
            depth = len(rel_path.parts) - 1
            indent = "  " * depth
            print(f"  {indent}📁 {item.name}/")

# Cleanup
if demo_workspace.exists():
    shutil.rmtree(demo_workspace)

# =============================================================================
# Section 6: Session Best Practices
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Session Best Practices")
print("-" * 60)

print("""
Best practices for session usage:
""")

# Pattern 1: Collect results
print("\nPattern 1: Collect results in a list")
all_results = []

with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
    for n in [5, 10, 15]:
        result = nirs4all.run(
            pipeline=[MinMaxScaler(), PLSRegression(n)],
            dataset="sample_data/regression",
            name=f"Collect_{n}",
            session=s
        )
        all_results.append({
            'n_components': n,
            'rmse': result.best_rmse,
            'r2': result.best_r2
        })

# Process collected results
best_result = min(all_results, key=lambda x: x['rmse'])
print(f"  Best: n={best_result['n_components']} (RMSE={best_result['rmse']:.4f})")

# Pattern 2: Override settings per run
print("\nPattern 2: Override settings per run")
with nirs4all.session(verbose=0, plots_visible=False) as s:
    # Quiet run
    result_quiet = nirs4all.run(
        pipeline=[MinMaxScaler(), PLSRegression(5)],
        dataset="sample_data/regression",
        name="Quiet",
        session=s,
        verbose=0  # Override
    )

    # Verbose run (override session)
    result_verbose = nirs4all.run(
        pipeline=[MinMaxScaler(), PLSRegression(10)],
        dataset="sample_data/regression",
        name="Verbose",
        session=s,
        verbose=1  # More verbose
    )

print("  Per-run verbose levels applied")

# =============================================================================
# Section 7: When to Use Sessions
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: When to Use Sessions")
print("-" * 60)

print("""
✓ USE SESSIONS FOR:
  - Comparing multiple preprocessing methods
  - Hyperparameter sweeps
  - Model architecture comparisons
  - Systematic experiments

✗ DON'T USE SESSIONS FOR:
  - Single one-off runs
  - Independent experiments
  - Quick tests
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Session and Workspace Workflow:

  1. BASIC SESSION:
     with nirs4all.session(verbose=1) as s:
         result = nirs4all.run(..., session=s)

  2. SESSION WITH CONFIGURATION:
     with nirs4all.session(
         verbose=1,
         save_artifacts=True,
         workspace_path="custom/path"
     ) as s:
         result1 = nirs4all.run(..., session=s)
         result2 = nirs4all.run(..., session=s)

  3. COLLECT RESULTS:
     all_results = []
     with nirs4all.session(...) as s:
         for config in configs:
             result = nirs4all.run(..., session=s)
             all_results.append(result)
     best = min(all_results, key=lambda x: x.best_rmse)

  4. OVERRIDE PER RUN:
     with nirs4all.session(verbose=0) as s:
         quiet = nirs4all.run(..., verbose=0)
         loud = nirs4all.run(..., verbose=2)

Workspace Structure:
  workspace/
  ├── store.duckdb    # All structured data (runs, predictions, etc.)
  ├── artifacts/      # Content-addressed binary artifacts
  ├── exports/        # Exported best results (.n4a bundles)
  └── library/        # Saved pipeline templates (.json)

Next: U04_sklearn_integration.py - sklearn Pipeline compatibility
""")
