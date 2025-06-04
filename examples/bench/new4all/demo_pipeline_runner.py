"""
Demo script showing the PipelineRunner in action vs regular Pipeline

This demonstrates the key differences:
1. Direct config interpretation without building PipelineOperations
2. Visible branch management and context updates
3. Explicit parallelization control
"""
import numpy as np
from pathlib import Path

# Mock imports for demo (these would be real in the actual environment)
try:
    from PipelineRunner import PipelineRunner
    from Pipeline import Pipeline
    from PipelineConfig import PipelineConfig
    from SpectraDataset import SpectraDataset
    from sample import config as sample_config
except ImportError as e:
    print(f"Import error (expected in demo): {e}")
    print("This is a demonstration of the PipelineRunner concept")


def demo_pipeline_runner_concept():
    """Demonstrate the PipelineRunner concept and philosophy"""

    print("=" * 60)
    print("PIPELINE RUNNER DEMONSTRATION")
    print("=" * 60)

    # Sample configuration similar to sample.py
    demo_config = {
        "experiment": {
            "action": "classification",
            "dataset": "sample_data.csv"
        },
        "pipeline": [
            "PlotData",
            {"feature_augmentation": ["None", "SG", ["SNV", "GS"]]},
            "ShuffleSplit()",
            {"cluster": "KMeans(n_clusters=5)"},
            "RepeatedStratifiedKFold()",
            "uncluster",
            {
                "dispatch": [
                    # Branch 1: Simple RF model
                    [
                        "MinMaxScaler()",
                        {"model": "RandomForestClassifier()"}
                    ],
                    # Branch 2: SVM with optimization
                    {
                        "model": "SVC()",
                        "finetune_params": {"C": [0.1, 1.0, 10.0]}
                    },
                    # Branch 3: Stacking ensemble
                    {
                        "stack": {
                            "model": "RandomForestClassifier()",
                            "base_learners": [
                                {"model": "GradientBoostingClassifier()"},
                                {"model": "DecisionTreeClassifier()"}
                            ]
                        }
                    }
                ]
            },
            "PlotModelPerformance"
        ]
    }

    print("\n1. PIPELINE RUNNER APPROACH:")
    print("-" * 30)

    # Simulated PipelineRunner execution
    print("🚀 PipelineRunner starting...")
    print("📋 Direct config interpretation:")

    current_step = 0
    current_filters = {}

    for step in demo_config["pipeline"]:
        current_step += 1

        if isinstance(step, str):
            print(f"  Step {current_step}: String '{step}'")
            if step == "uncluster":
                if 'group' in current_filters:
                    del current_filters['group']
                print(f"    🔓 Updated filters: {current_filters}")

        elif isinstance(step, dict):
            keys = list(step.keys())
            print(f"  Step {current_step}: Dict {keys}")

            if "feature_augmentation" in step:
                print(f"    🔧 Feature augmentation with {len(step['feature_augmentation'])} methods")

            elif "cluster" in step:
                current_filters['group'] = True
                print("    🎯 Clustering applied")
                print(f"    🏷️  Updated filters: {current_filters}")

            elif "dispatch" in step:
                branches = step["dispatch"]
                print(f"    🌳 Dispatch: {len(branches)} parallel branches")

                for i, branch in enumerate(branches):
                    print(f"      Branch {i+1}:")
                    if isinstance(branch, list):
                        print(f"        📋 Sub-pipeline with {len(branch)} steps")
                        for sub_step in branch:
                            print(f"          - {sub_step}")
                    else:
                        print(f"        🔧 {branch}")

                print("    ✅ All branches executed")

    print("\n2. TRADITIONAL PIPELINE APPROACH:")
    print("-" * 30)
    print("📦 Pipeline building...")
    print("  - Converting config to PipelineOperations")
    print("  - Creating operation objects")
    print("  - Building execution chain")
    print("🔄 Pipeline executing...")
    print("  - Sequential operation execution")
    print("  - Operations handle branching internally")
    print("  - Context updates hidden in operations")

    print("\n3. KEY DIFFERENCES:")
    print("-" * 30)
    print("PipelineRunner Philosophy:")
    print("  ✅ Direct config interpretation")
    print("  ✅ Visible context management")
    print("  ✅ Explicit branch parallelization")
    print("  ✅ Clear execution flow")
    print("  ✅ No intermediate operation objects")
    print("  ✅ Better debugging and monitoring")

    print("\nTraditional Pipeline:")
    print("  ❌ Config -> Operations -> Execution")
    print("  ❌ Hidden context in operations")
    print("  ❌ Operations manage sub-pipelines")
    print("  ❌ Less visible execution flow")
    print("  ❌ Harder to debug complex branches")

    print("\n4. CONTEXT TRACKING EXAMPLE:")
    print("-" * 30)
    print("Initial context filters: {}")
    print("After clustering: {'group': True}")
    print("After uncluster: {}")
    print("Branch 1 context: {'branch': 1}")
    print("Branch 2 context: {'branch': 2}")
    print("Merged context: predictions from all branches")

    print("\n5. PARALLEL EXECUTION EXAMPLE:")
    print("-" * 30)
    print("Sequential mode:")
    print("  Branch 1 → Branch 2 → Branch 3")
    print("Parallel mode (with ThreadPoolExecutor):")
    print("  Branch 1 ∥ Branch 2 ∥ Branch 3")
    print("  → Merge results")

    print("\n6. CONFIGURATION VISIBILITY:")
    print("-" * 30)
    print("The pipeline structure is preserved and visible:")
    print("- Augmentation steps are clear")
    print("- Branching is explicit")
    print("- Context updates are logged")
    print("- Parallel vs sequential choice is clear")

    print("\n" + "=" * 60)
    print("CONCLUSION: PipelineRunner provides better visibility,")
    print("control, and understanding of complex pipeline execution")
    print("=" * 60)


def demo_api_integration():
    """Show how PipelineRunner integrates with current API"""

    print("\n" + "=" * 60)
    print("API INTEGRATION EXAMPLE")
    print("=" * 60)

    print("\nUsage with current API:")
    print("-" * 30)

    code_example = '''
# Create runner with parallel support
runner = PipelineRunner(max_workers=4, continue_on_error=True)

# Load dataset (using current SpectraDataset API)
dataset = SpectraDataset.from_csv("data/sample_data.csv")

# Execute with different config types
result1 = runner.run_pipeline(sample_config, dataset)  # Python config
result2 = runner.run_pipeline("config.yaml", dataset)   # YAML config
result3 = runner.run_pipeline(pipeline_list, dataset)   # Direct list

# Get execution summary
summary = runner.get_execution_summary()
print(f"Executed {summary['total_steps']} steps")
print(f"Success rate: {summary['successful_steps']}/{summary['total_steps']}")

# Access predictions
predictions = summary['predictions']
for model_name, preds in predictions.items():
    print(f"Model {model_name}: {preds.shape}")
'''

    print(code_example)

    print("\nIntegration with existing operations:")
    print("-" * 30)
    print("- TransformationOperation: Used for preprocessing")
    print("- ClusteringOperation: Used for clustering steps")
    print("- ModelOperation: Used for model training")
    print("- StackOperation: Used for ensemble methods")
    print("- OptimizationOperation: Used for hyperparameter tuning")

    print("\nContext management integration:")
    print("-" * 30)
    print("- PipelineContext: Enhanced for branch tracking")
    print("- Filter updates: Visible and logged")
    print("- Prediction storage: Centralized in context")
    print("- Branch isolation: Parallel-safe execution")


if __name__ == "__main__":
    demo_pipeline_runner_concept()
    demo_api_integration()
