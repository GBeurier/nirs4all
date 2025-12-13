"""
Q_meta_stacking.py - Comprehensive Meta-Model Stacking Examples for nirs4all

This example demonstrates how to use MetaModel for stacked generalization:
1. Train multiple base models (PLS, RandomForest, XGBoost, etc.)
2. Use their out-of-fold (OOF) predictions as features for a meta-learner
3. The meta-model learns to optimally combine base model predictions

Key concepts:
- Out-of-fold (OOF) predictions prevent data leakage during training
- Source models can be selected explicitly, automatically, or by criteria
- Coverage strategies handle incomplete fold predictions gracefully
- Test aggregation methods combine fold predictions for test set
- Branch scope controls which branches contribute source models

Phase 7 Advanced Features:
- Multi-Level Stacking: Stack meta-models on top of meta-models
- Cross-Branch Stacking: Use models from all branches with feature alignment
- Finetune Integration: Optuna hyperparameter optimization support

Examples included:
1. Basic stacking - All previous models as sources
2. Explicit source selection - Choose specific models
3. Top-K source selection - Use best N models by validation score
4. Custom stacking config - Handle edge cases and aggregation
5. Stacking with branches - Use models from preprocessing branches
6. Stacking with diverse estimators - Mix different model types
7. Save and reload - Persist and reuse stacking models
8. Multi-level stacking - Hierarchical stacking with StackingLevel
9. Cross-branch stacking - ALL_BRANCHES scope with alignment
10. Finetune integration - Optuna optimization for meta-models
11. Advanced multi-level - Circular dependency detection
"""

import numpy as np
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import nirs4all components
from nirs4all.data.dataset import DatasetConfigs
from nirs4all.pipeline import PipelineRunner
from nirs4all.pipeline.storage.workspace import Workspace

# Import meta-model stacking components
from nirs4all.operators.models import (
    MetaModel,
    StackingConfig,
    CoverageStrategy,
    TestAggregation,
    BranchScope,
)
from nirs4all.operators.preprocessing import DerivativeTransform


def example_basic_stacking():
    """
    Example 1: Basic meta-model stacking with all previous models.

    This is the simplest form of stacking:
    - All models before the MetaModel are used as source models
    - OOF predictions are automatically collected from cross-validation
    - Default config uses strict coverage and mean aggregation
    """
    print("=" * 70)
    print("Example 1: Basic Meta-Model Stacking")
    print("=" * 70)

    # Load sample dataset
    dataset = DatasetConfigs("examples/sample_data/")

    # Pipeline with base models and meta-learner
    pipeline = [
        # Step 1: Preprocessing
        MinMaxScaler(),

        # Step 2: Cross-validation splitter (required for OOF collection)
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Steps 3-4: Base models (Level 0)
        PLSRegression(n_components=5),
        RandomForestRegressor(n_estimators=50, random_state=42),

        # Step 5: Meta-learner (Level 1) - stacks all previous models
        {"model": MetaModel(model=Ridge(alpha=1.0))},
    ]

    # Run training
    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_stacking_basic")

    # Get and display results
    predictions = runner.predictions
    top_models = predictions.top(n=5, rank_partition="val")

    print("\nTop 5 models by validation score:")
    print("-" * 50)
    for pred in top_models:
        score = pred.get('val_score', 0)
        print(f"  {pred['model_name']:40s}: {score:.4f}")

    # Compare meta-model with base models
    pls_score = predictions.filter(model_name_contains="PLS")[0]['val_score']
    rf_score = predictions.filter(model_name_contains="RandomForest")[0]['val_score']
    meta_score = predictions.filter(model_name_contains="MetaModel")[0]['val_score']

    print("\n📊 Score Comparison:")
    print(f"  PLS:         {pls_score:.4f}")
    print(f"  RF:          {rf_score:.4f}")
    print(f"  MetaModel:   {meta_score:.4f}")

    improvement = (meta_score - max(pls_score, rf_score)) / abs(max(pls_score, rf_score)) * 100
    print(f"  Improvement: {improvement:+.2f}%")


def example_explicit_source_selection():
    """
    Example 2: Meta-model with explicit source model selection.

    Use specific models by name instead of all previous models.
    Useful when you want to combine only certain model types.
    """
    print("\n" + "=" * 70)
    print("Example 2: Explicit Source Model Selection")
    print("=" * 70)

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Multiple PLS models with different components
        {"model": PLSRegression(n_components=3), "name": "PLS_3"},
        {"model": PLSRegression(n_components=5), "name": "PLS_5"},
        {"model": PLSRegression(n_components=10), "name": "PLS_10"},

        # Random Forest (will NOT be used by meta-model)
        RandomForestRegressor(n_estimators=100, random_state=42),

        # Meta-learner using ONLY PLS models (explicit selection)
        {"model": MetaModel(
            model=Ridge(alpha=0.5),
            source_models=["PLS_3", "PLS_5", "PLS_10"],  # Only these 3
        ), "name": "PLS_Ensemble"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_explicit")

    predictions = runner.predictions
    print("\nModel performances:")
    print("-" * 50)

    for name in ["PLS_3", "PLS_5", "PLS_10", "RandomForest", "PLS_Ensemble"]:
        preds = predictions.filter(model_name_contains=name, partition="val")
        if preds:
            print(f"  {name:30s}: {preds[0]['val_score']:.4f}")


def example_topk_source_selection():
    """
    Example 3: Top-K source selection by validation score.

    Automatically select the best N models based on their validation
    performance. This is useful when you have many base models.
    """
    print("\n" + "=" * 70)
    print("Example 3: Top-K Source Selection")
    print("=" * 70)

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Many base models
        {"model": PLSRegression(n_components=3), "name": "PLS_3"},
        {"model": PLSRegression(n_components=5), "name": "PLS_5"},
        {"model": PLSRegression(n_components=10), "name": "PLS_10"},
        {"model": PLSRegression(n_components=15), "name": "PLS_15"},
        RandomForestRegressor(n_estimators=50, random_state=42),
        GradientBoostingRegressor(n_estimators=50, random_state=42),
        KNeighborsRegressor(n_neighbors=5),

        # Meta-learner using top 3 models by validation score
        {"model": MetaModel(
            model=Ridge(alpha=1.0),
            source_models={"top_k": 3, "metric": "r2"},  # Select best 3
        ), "name": "Top3_Meta"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_topk")

    predictions = runner.predictions

    # Show all model scores
    all_preds = predictions.filter(partition="val")
    all_preds = sorted(all_preds, key=lambda p: p.get('val_score', 0), reverse=True)

    print("\nAll models ranked by validation score:")
    print("-" * 50)
    for pred in all_preds:
        is_meta = "MetaModel" in pred['model_name'] or "Top3" in pred['model_name']
        marker = "⭐" if is_meta else "  "
        print(f"  {marker} {pred['model_name']:35s}: {pred['val_score']:.4f}")


def example_custom_stacking_config():
    """
    Example 4: Custom stacking configuration.

    Configure how OOF predictions are handled:
    - coverage_strategy: How to handle missing predictions
    - test_aggregation: How to combine fold predictions for test set
    - min_coverage_ratio: Minimum required sample coverage
    """
    print("\n" + "=" * 70)
    print("Example 4: Custom Stacking Configuration")
    print("=" * 70)

    dataset = DatasetConfigs("examples/sample_data/")

    # Custom stacking config for robust handling
    stacking_config = StackingConfig(
        # IMPUTE_MEAN: Fill missing OOF predictions with column mean
        # Other options: STRICT (fail), DROP_INCOMPLETE (mask), IMPUTE_ZERO
        coverage_strategy=CoverageStrategy.IMPUTE_MEAN,

        # WEIGHTED_MEAN: Weight test predictions by validation scores
        # Other options: MEAN (simple average), BEST_FOLD (use best fold only)
        test_aggregation=TestAggregation.WEIGHTED_MEAN,

        # Minimum coverage ratio (0.8 = 80% of samples must have predictions)
        min_coverage_ratio=0.8,
    )

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Base models
        PLSRegression(n_components=5),
        RandomForestRegressor(n_estimators=50, random_state=42),

        # Meta-learner with custom config
        {"model": MetaModel(
            model=Ridge(alpha=1.0),
            stacking_config=stacking_config,
        )},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_config")

    predictions = runner.predictions

    print("\n📋 Stacking Configuration Used:")
    print(f"  Coverage Strategy: {stacking_config.coverage_strategy.value}")
    print(f"  Test Aggregation:  {stacking_config.test_aggregation.value}")
    print(f"  Min Coverage:      {stacking_config.min_coverage_ratio:.0%}")

    print("\nModel performances:")
    print("-" * 50)
    top_models = predictions.top(n=5, rank_partition="val")
    for pred in top_models:
        print(f"  {pred['model_name']:40s}: {pred['val_score']:.4f}")


def example_stacking_with_branches():
    """
    Example 5: Stacking with preprocessing branches.

    Use models from different preprocessing branches as sources.
    The branch_scope parameter controls which branches contribute.
    """
    print("\n" + "=" * 70)
    print("Example 5: Stacking with Preprocessing Branches")
    print("=" * 70)

    from nirs4all.operators.control.branching import Branch

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Branching with different preprocessing
        Branch(
            branches=[
                # Branch 0: Raw spectra
                [
                    {"model": PLSRegression(n_components=5), "name": "PLS_Raw"},
                    RandomForestRegressor(n_estimators=30, random_state=42),
                ],
                # Branch 1: First derivative
                [
                    DerivativeTransform(order=1),
                    {"model": PLSRegression(n_components=5), "name": "PLS_D1"},
                    RandomForestRegressor(n_estimators=30, random_state=42),
                ],
                # Branch 2: Second derivative
                [
                    DerivativeTransform(order=2),
                    {"model": PLSRegression(n_components=5), "name": "PLS_D2"},
                ],
            ],
            merge_predictions=True,  # Merge predictions for stacking
        ),

        # Meta-model after branch merge - uses ALL branch models
        {"model": MetaModel(
            model=Ridge(alpha=1.0),
            stacking_config=StackingConfig(
                branch_scope=BranchScope.ALL_BRANCHES,  # Use all branches
            ),
        ), "name": "BranchMeta"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_branches")

    predictions = runner.predictions

    print("\nModels from all branches:")
    print("-" * 50)
    all_preds = predictions.filter(partition="val")
    for pred in sorted(all_preds, key=lambda p: p.get('val_score', 0), reverse=True):
        print(f"  {pred['model_name']:40s}: {pred['val_score']:.4f}")


def example_diverse_estimators():
    """
    Example 6: Stacking diverse estimator types.

    Combine very different model types for diversity:
    - Linear models (PLS, Ridge)
    - Tree-based (RF, GradientBoosting)
    - Distance-based (KNN)
    """
    print("\n" + "=" * 70)
    print("Example 6: Stacking Diverse Estimators")
    print("=" * 70)

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        StandardScaler(),  # Better for diverse models
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Diverse base models
        PLSRegression(n_components=5),                          # Linear
        Ridge(alpha=1.0),                                        # Linear
        RandomForestRegressor(n_estimators=50, random_state=42),  # Tree
        GradientBoostingRegressor(n_estimators=50, random_state=42),  # Boosted
        KNeighborsRegressor(n_neighbors=5),                      # Distance

        # Meta-learner combines all
        {"model": MetaModel(
            model=ElasticNet(alpha=0.5, l1_ratio=0.5),  # Regularized meta
        ), "name": "DiverseStack"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_diverse")

    predictions = runner.predictions

    print("\nDiverse ensemble results:")
    print("-" * 50)
    top_models = predictions.top(n=10, rank_partition="val")
    for pred in top_models:
        print(f"  {pred['model_name']:45s}: {pred['val_score']:.4f}")


def example_save_and_reload():
    """
    Example 7: Save and reload stacking models.

    Demonstrates persistence and prediction mode:
    1. Train and save a stacking pipeline
    2. Reload from workspace
    3. Make predictions on new data
    """
    print("\n" + "=" * 70)
    print("Example 7: Save and Reload Stacking Model")
    print("=" * 70)

    dataset = DatasetConfigs("examples/sample_data/")
    workspace_path = "workspace/meta_persistence"

    # --- TRAINING PHASE ---
    print("\n📝 Training Phase:")
    print("-" * 50)

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),
        PLSRegression(n_components=5),
        RandomForestRegressor(n_estimators=50, random_state=42),
        {"model": MetaModel(model=Ridge(alpha=1.0))},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest=workspace_path)

    # Store original test predictions
    original_preds = runner.predictions.filter(
        model_name_contains="MetaModel", partition="test"
    )
    original_score = original_preds[0]['val_score'] if original_preds else None

    print("  Training complete")
    print(f"  MetaModel test score: {original_score:.4f}" if original_score else "")

    # --- RELOAD PHASE ---
    print("\n🔄 Reload Phase:")
    print("-" * 50)

    workspace = Workspace(workspace_path)

    # List saved artifacts
    artifacts = workspace.list_artifacts()
    meta_artifacts = [a for a in artifacts if "MetaModel" in str(a.get('class_name', ''))]
    print(f"  Found {len(artifacts)} artifacts")
    print(f"  Meta-model artifacts: {len(meta_artifacts)}")

    # Load pipeline for prediction
    loaded_pipeline = workspace.load_pipeline()
    if loaded_pipeline:
        print("  Pipeline loaded successfully")

        # --- PREDICTION PHASE ---
        print("\n🔮 Prediction Phase:")
        print("-" * 50)

        runner2 = PipelineRunner()
        runner2.run(dataset, loaded_pipeline, dest=workspace_path, mode="predict")

        new_preds = runner2.predictions.filter(
            model_name_contains="MetaModel", partition="test"
        )

        if new_preds and original_preds:
            # Compare predictions
            orig_y = original_preds[0].get('y_pred')
            new_y = new_preds[0].get('y_pred')

            if orig_y is not None and new_y is not None:
                diff = np.abs(orig_y - new_y).max()
                print(f"  Max prediction difference: {diff:.2e}")
                print(f"  ✅ Predictions {'match' if diff < 1e-6 else 'differ'}")


def example_multi_level_stacking():
    """
    Example 8: Multi-level stacking (stacking of stackers).

    Advanced: Create a hierarchy of meta-models.
    Level 0: Base models
    Level 1: First meta-model
    Level 2: Second meta-model (uses Level 0 + Level 1)

    Phase 7 Feature: Multi-Level Stacking with explicit StackingLevel
    - StackingLevel.AUTO: Automatically detect level based on sources
    - StackingLevel.LEVEL_1: First stacking level (base models as sources)
    - StackingLevel.LEVEL_2: Second level (can use Level 1 meta-models)
    - StackingLevel.LEVEL_3: Third level (can use Level 2 meta-models)

    The allow_meta_sources flag controls whether meta-models can be
    used as sources for other meta-models.
    """
    print("\n" + "=" * 70)
    print("Example 8: Multi-Level Stacking")
    print("=" * 70)

    from nirs4all.operators.models.meta import StackingLevel

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Level 0: Base models
        {"model": PLSRegression(n_components=3), "name": "PLS_L0"},
        {"model": PLSRegression(n_components=10), "name": "PLS10_L0"},
        RandomForestRegressor(n_estimators=50, random_state=42),

        # Level 1: First meta-model (stacks Level 0)
        # Explicit level=1 for clarity
        {"model": MetaModel(
            model=Ridge(alpha=1.0),
            source_models=["PLS_L0", "PLS10_L0"],  # Only PLS models
            stacking_config=StackingConfig(level=StackingLevel.LEVEL_1),
        ), "name": "Meta_L1_PLS"},

        # Level 2: Second meta-model (stacks all including L1)
        # Enable allow_meta_sources to use the Level 1 meta-model
        {"model": MetaModel(
            model=Lasso(alpha=0.1),
            stacking_config=StackingConfig(
                level=StackingLevel.LEVEL_2,
                allow_meta_sources=True,  # Enable stacking of meta-models
            ),
            # Uses: RF from L0 + Meta_L1_PLS from L1
        ), "name": "Meta_L2_Final"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_multilevel")

    predictions = runner.predictions

    print("\nMulti-level stacking results:")
    print("-" * 50)
    print("  Level 0 (Base):")
    for name in ["PLS_L0", "PLS10_L0", "RandomForest"]:
        preds = predictions.filter(model_name_contains=name, partition="val")
        if preds:
            print(f"    {name:30s}: {preds[0]['val_score']:.4f}")

    print("\n  Level 1 (First Stack):")
    meta_l1 = predictions.filter(model_name_contains="Meta_L1_PLS", partition="val")
    if meta_l1:
        print(f"    {'Meta_L1_PLS':30s}: {meta_l1[0]['val_score']:.4f}")

    print("\n  Level 2 (Final Stack):")
    meta_l2 = predictions.filter(model_name_contains="Meta_L2_Final", partition="val")
    if meta_l2:
        print(f"    {'Meta_L2_Final':30s}: {meta_l2[0]['val_score']:.4f}")


def example_cross_branch_stacking():
    """
    Example 9: Cross-branch stacking with ALL_BRANCHES scope.

    Phase 7 Feature: Cross-Branch Stacking
    - BranchScope.CURRENT_ONLY: Only use models from current branch (default)
    - BranchScope.ALL_BRANCHES: Use models from all branches
    - BranchScope.SPECIFIED: Use models from specific branches

    This example demonstrates using models from all preprocessing branches
    as sources for a single meta-model after a Branch merge.
    """
    print("\n" + "=" * 70)
    print("Example 9: Cross-Branch Stacking")
    print("=" * 70)

    from nirs4all.operators.control.branching import Branch

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Branching with different preprocessing
        Branch(
            branches=[
                # Branch 0: Raw spectra processing
                [
                    {"model": PLSRegression(n_components=5), "name": "PLS_Raw"},
                    {"model": Ridge(alpha=1.0), "name": "Ridge_Raw"},
                ],
                # Branch 1: First derivative processing
                [
                    DerivativeTransform(order=1),
                    {"model": PLSRegression(n_components=5), "name": "PLS_D1"},
                    {"model": Ridge(alpha=1.0), "name": "Ridge_D1"},
                ],
                # Branch 2: Second derivative processing
                [
                    DerivativeTransform(order=2),
                    {"model": PLSRegression(n_components=3), "name": "PLS_D2"},
                ],
            ],
            merge_predictions=True,
        ),

        # Cross-branch meta-model - uses models from ALL branches
        {"model": MetaModel(
            model=ElasticNet(alpha=0.5, l1_ratio=0.5),
            stacking_config=StackingConfig(
                branch_scope=BranchScope.ALL_BRANCHES,  # Cross-branch stacking
                coverage_strategy=CoverageStrategy.IMPUTE_MEAN,  # Handle alignment
            ),
        ), "name": "CrossBranch_Meta"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_crossbranch")

    predictions = runner.predictions

    print("\nModels from all branches:")
    print("-" * 50)
    branch_models = ["PLS_Raw", "Ridge_Raw", "PLS_D1", "Ridge_D1", "PLS_D2"]
    for name in branch_models:
        preds = predictions.filter(model_name_contains=name, partition="val")
        if preds:
            print(f"  {name:35s}: {preds[0]['val_score']:.4f}")

    print("\nCross-branch meta-model:")
    meta = predictions.filter(model_name_contains="CrossBranch_Meta", partition="val")
    if meta:
        print(f"  {'CrossBranch_Meta':35s}: {meta[0]['val_score']:.4f}")


def example_finetune_integration():
    """
    Example 10: Finetune integration for meta-model hyperparameters.

    Phase 7 Feature: Finetune Integration
    - finetune_space parameter defines hyperparameter search space
    - get_finetune_params() returns Optuna-compatible configuration
    - Supports n_trials, approach, and eval_mode settings

    This example shows how to define a finetune space for the meta-model
    that can be used with Optuna optimization.
    """
    print("\n" + "=" * 70)
    print("Example 10: Finetune Integration")
    print("=" * 70)

    dataset = DatasetConfigs("examples/sample_data/")

    # Define finetune space for the meta-model
    finetune_space = {
        # Model hyperparameters (use model__ prefix)
        "model__alpha": (0.001, 100.0),  # Log-uniform range for Ridge alpha

        # Optuna settings
        "n_trials": 50,        # Number of optimization trials
        "approach": "grouped",  # Use grouped cross-validation
        "eval_mode": "best",   # Select best trial
    }

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Base models
        PLSRegression(n_components=5),
        RandomForestRegressor(n_estimators=50, random_state=42),

        # Meta-model with finetune space
        {"model": MetaModel(
            model=Ridge(alpha=1.0),  # Default value (will be optimized)
            finetune_space=finetune_space,  # Optuna search space
        ), "name": "Tunable_Meta"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_finetune")

    predictions = runner.predictions

    print("\n📋 Finetune Configuration:")
    print("-" * 50)
    print("  Parameter: model__alpha")
    print("  Range:     (0.001, 100.0) log-uniform")
    print(f"  Trials:    {finetune_space['n_trials']}")
    print(f"  Approach:  {finetune_space['approach']}")

    # Access finetune params programmatically
    meta = MetaModel(model=Ridge(), finetune_space=finetune_space)
    params = meta.get_finetune_params()
    if params:
        print("\n  get_finetune_params() output:")
        for key, value in params.items():
            print(f"    {key}: {value}")

    print("\nModel performances:")
    print("-" * 50)
    top_models = predictions.top(n=5, rank_partition="val")
    for pred in top_models:
        print(f"  {pred['model_name']:40s}: {pred['val_score']:.4f}")


def example_advanced_multi_level():
    """
    Example 11: Advanced multi-level with circular dependency detection.

    Phase 7 Feature: Multi-Level Stacking with Validation
    - Automatic circular dependency detection
    - Level validation to prevent meta-models stacking higher levels
    - max_level configuration to limit stacking depth

    This example demonstrates the safety features of multi-level stacking.
    """
    print("\n" + "=" * 70)
    print("Example 11: Advanced Multi-Level with Validation")
    print("=" * 70)

    from nirs4all.operators.models.meta import StackingLevel

    dataset = DatasetConfigs("examples/sample_data/")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Level 0: Diverse base models
        {"model": PLSRegression(n_components=5), "name": "PLS_Base"},
        {"model": Ridge(alpha=1.0), "name": "Ridge_Base"},
        RandomForestRegressor(n_estimators=30, random_state=42),

        # Level 1: Specialized meta-models
        {"model": MetaModel(
            model=Ridge(alpha=0.5),
            source_models=["PLS_Base", "Ridge_Base"],  # Linear ensemble
            stacking_config=StackingConfig(
                level=StackingLevel.LEVEL_1,
            ),
        ), "name": "Meta_Linear"},

        {"model": MetaModel(
            model=Ridge(alpha=0.5),
            source_models=["RandomForest"],  # Tree ensemble
            stacking_config=StackingConfig(
                level=StackingLevel.LEVEL_1,
            ),
        ), "name": "Meta_Tree"},

        # Level 2: Final meta-model combining Level 1 outputs
        {"model": MetaModel(
            model=Lasso(alpha=0.01),
            source_models=["Meta_Linear", "Meta_Tree"],  # Stack the stackers
            stacking_config=StackingConfig(
                level=StackingLevel.LEVEL_2,
                allow_meta_sources=True,  # Required for stacking meta-models
                max_level=3,  # Limit depth
            ),
        ), "name": "Meta_Final"},
    ]

    runner = PipelineRunner()
    runner.run(dataset, pipeline, dest="workspace/meta_advanced_multilevel")

    predictions = runner.predictions

    print("\nHierarchical stacking structure:")
    print("-" * 50)
    print("  Level 0 (Base Models):")
    for name in ["PLS_Base", "Ridge_Base", "RandomForest"]:
        preds = predictions.filter(model_name_contains=name, partition="val")
        if preds:
            print(f"    └─ {name:28s}: {preds[0]['val_score']:.4f}")

    print("\n  Level 1 (First Stack):")
    for name in ["Meta_Linear", "Meta_Tree"]:
        preds = predictions.filter(model_name_contains=name, partition="val")
        if preds:
            print(f"    └─ {name:28s}: {preds[0]['val_score']:.4f}")

    print("\n  Level 2 (Final Stack):")
    final = predictions.filter(model_name_contains="Meta_Final", partition="val")
    if final:
        print(f"    └─ {'Meta_Final':28s}: {final[0]['val_score']:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("   Meta-Model Stacking Examples for nirs4all")
    print("=" * 70)

    print("""
These examples demonstrate stacked generalization (stacking) where a
meta-learner combines predictions from multiple base models.

📌 Key Benefits:
   • Automatic out-of-fold prediction handling (no data leakage)
   • Flexible source model selection (all, explicit, top-K)
   • Configurable coverage strategies for robustness
   • Supports preprocessing branches
   • Full persistence and reload support
   • Works with any sklearn-compatible estimator

📁 Basic Examples (1-7):
   1. Basic Stacking        - Simple all-previous-models stacking
   2. Explicit Selection    - Choose specific models by name
   3. Top-K Selection       - Auto-select best N models
   4. Custom Config         - Coverage strategies & aggregation
   5. Branch Stacking       - Use models from different branches
   6. Diverse Estimators    - Mix model types for diversity
   7. Save & Reload         - Persistence and prediction mode

📁 Phase 7 Advanced Examples (8-11):
   8. Multi-Level           - Hierarchical stacking with StackingLevel
   9. Cross-Branch          - ALL_BRANCHES scope with alignment
   10. Finetune Integration - Optuna optimization for meta-models
   11. Advanced Multi-Level - Circular dependency detection

Run individual examples by uncommenting below.
""")

    # Uncomment to run examples:
    example_basic_stacking()
    # example_explicit_source_selection()
    # example_topk_source_selection()
    # example_custom_stacking_config()
    # example_stacking_with_branches()
    # example_diverse_estimators()
    # example_save_and_reload()
    # example_multi_level_stacking()
    # example_cross_branch_stacking()
    # example_finetune_integration()
    # example_advanced_multi_level()
