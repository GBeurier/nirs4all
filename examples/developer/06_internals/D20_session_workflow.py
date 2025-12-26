"""
D20 - Session Workflow: Managing Pipeline Sessions
===================================================

nirs4all.Session provides stateful session management for
complex workflows, model persistence, and iterative development.

This tutorial covers:

* Session creation and lifecycle
* Session methods: run, predict, retrain
* Session persistence (save/load)
* Session introspection
* Multi-session workflows

Prerequisites
-------------
- U02_basic_regression for pipeline basics

Next Steps
----------
See D21_custom_controllers for extending the controller system.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D20 Session Workflow Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D20 - Session Workflow: Managing Pipeline Sessions")
print("=" * 60)

print("""
nirs4all provides two main APIs:

1. Functional API (stateless):
    result = nirs4all.run(pipeline, dataset)

2. Session API (stateful):
    session = nirs4all.Session(pipeline=pipeline, name="MyModel")
    result = session.run(dataset)
    predictions = session.predict(new_data)
    session.save("model.n4a")

Session API is useful for:
  - Model persistence
  - Iterative development
  - Prediction on new data
  - Complex workflows
""")


# =============================================================================
# Section 1: Creating a Session
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Creating a Session")
print("-" * 60)

print("""
Create a session from a pipeline:

    session = nirs4all.Session(
        pipeline=pipeline,
        name="MyModel",
        verbose=1
    )
""")

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    {"y_processing": StandardScaler()},
    PLSRegression(n_components=10),
]

session = nirs4all.Session(
    pipeline=pipeline,
    name="SessionDemo",
    verbose=1
)

print(f"\nSession created: {session.name}")
print(f"  Pipeline steps: {len(session.pipeline)}")
print(f"  Status: {session.status}")


# =============================================================================
# Section 2: Running a Session
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Running a Session")
print("-" * 60)

print("""
Run the session to train the pipeline:

    result = session.run(dataset)
""")

result = session.run(
    dataset="sample_data/regression",
    plots_visible=args.plots
)

print(f"\nSession run complete")
print(f"  Status: {session.status}")
print(f"  Predictions: {result.num_predictions}")
print(f"  Best RMSE: {result.best_rmse:.4f}")


# =============================================================================
# Section 3: Making Predictions
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Making Predictions")
print("-" * 60)

print("""
Use trained session for predictions:

    predictions = session.predict(new_data)

The session applies all trained preprocessing steps
before passing data to the model.
""")

# Predict on new data (using same dataset for demo)
predictions = session.predict(
    dataset="sample_data/regression"
)

print(f"\nPredictions on new data:")
print(f"  Model: {predictions.model_name}")
print(f"  Predictions shape: {predictions.shape}")


# =============================================================================
# Section 4: Session Introspection
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Session Introspection")
print("-" * 60)

print("""
Inspect session state:

    session.status        # 'initialized', 'trained', 'error'
    session.pipeline      # Pipeline definition
    session.is_trained    # Whether training completed
    session.history       # Run history
""")

print(f"\nSession introspection:")
print(f"  Name: {session.name}")
print(f"  Status: {session.status}")
print(f"  Pipeline steps: {len(session.pipeline)}")
print(f"  Is trained: {session.is_trained}")
print(f"  History entries: {len(session.history)}")


# =============================================================================
# Section 5: Saving a Session
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Saving a Session")
print("-" * 60)

print("""
Save session for later use:

    session.save("models/my_model.n4a")

Saves:
  - Pipeline definition
  - Trained step states
  - Configuration
  - Metadata
""")

save_path = Path("workspace/sessions/demo_session.n4a")
save_path.parent.mkdir(parents=True, exist_ok=True)

session.save(str(save_path))
print(f"\nSession saved to: {save_path}")
print(f"  File size: {save_path.stat().st_size / 1024:.1f} KB")


# =============================================================================
# Section 6: Loading a Session
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Loading a Session")
print("-" * 60)

print("""
Load a saved session:

    session = nirs4all.load_session("models/my_model.n4a")
    predictions = session.predict(new_data)
""")

loaded_session = nirs4all.load_session(str(save_path))

print(f"\nSession loaded: {loaded_session.name}")
print(f"  Status: {loaded_session.status}")
print(f"  Is trained: {loaded_session.is_trained}")


# =============================================================================
# Section 7: Session Retraining
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Session Retraining")
print("-" * 60)

print("""
Retrain a session on new data:

    result = session.retrain(
        dataset=new_dataset,
        mode='transfer'  # or 'full', 'finetune'
    )
""")

retrain_result = session.retrain(
    dataset="sample_data/regression",
    mode='transfer'
)

print(f"\nSession retrained:")
print(f"  Mode: transfer")
print(f"  Predictions: {retrain_result.num_predictions}")
print(f"  Best RMSE: {retrain_result.best_rmse:.4f}")


# =============================================================================
# Section 8: Context Manager Usage
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Context Manager for Resource Sharing")
print("-" * 60)

print("""
Use session as context manager for multiple runs:

    with nirs4all.session(verbose=1, save_artifacts=True) as s:
        r1 = nirs4all.run(pipeline1, data, session=s)
        r2 = nirs4all.run(pipeline2, data, session=s)
        # Both share the same runner and workspace
""")

with nirs4all.session(verbose=0, save_artifacts=False) as s:
    # Multiple runs share the same runner
    r1 = nirs4all.run(
        pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=2, random_state=42),
                  PLSRegression(n_components=5)],
        dataset="sample_data/regression",
        session=s
    )
    r2 = nirs4all.run(
        pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=2, random_state=42),
                  PLSRegression(n_components=10)],
        dataset="sample_data/regression",
        session=s
    )
    print(f"\nContext manager comparison:")
    print(f"  PLS(5):  RMSE = {r1.best_rmse:.4f}")
    print(f"  PLS(10): RMSE = {r2.best_rmse:.4f}")


# =============================================================================
# Section 9: Session Lifecycle
# =============================================================================
print("\n" + "-" * 60)
print("Example 9: Session Lifecycle")
print("-" * 60)

print("""
📋 Session Lifecycle:

    ┌─────────────┐
    │ initialized │ ← session = nirs4all.Session(pipeline=...)
    └──────┬──────┘
           │ session.run(dataset)
           ▼
    ┌─────────────┐
    │   trained   │ ← Ready for predict/retrain/save
    └──────┬──────┘
           │ session.save() / session.retrain()
           ▼
    ┌─────────────┐
    │   saved/    │
    │  updated    │
    └─────────────┘

Methods available by state:
  initialized: run()
  trained: predict(), retrain(), save(), run()
  loaded: predict(), retrain(), save(), run()
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. nirs4all.Session() creates stateful pipeline manager
2. session.run() trains the pipeline
3. session.predict() makes predictions
4. session.save()/load_session() for persistence
5. session.retrain() for model updates
6. Context manager for resource sharing

Key methods:
    session = nirs4all.Session(pipeline=..., name="...")
    result = session.run(dataset)
    predictions = session.predict(new_data)
    session.save("model.n4a")
    session = nirs4all.load_session("model.n4a")
    session.retrain(new_data, mode='transfer')

Session vs Functional API:
- Use Session for: persistence, iterative work, production
- Use run() for: quick experiments, one-off analysis

Next: D21_custom_controllers.py - Extending the controller system
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
