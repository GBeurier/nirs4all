"""
Folds Operation - Creates cross-validation folds for train partition

This operation:
1. Creates folds on train partition samples
2. Saves fold assignments in dataset
3. Supports stratified and regular K-fold
4. Can work with groups (cluster-based folding)
"""
import numpy as np
from typing import Optional, List, Union, Dict
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class OperationFolds(PipelineOperation):
    """Pipeline operation that creates cross-validation folds"""

    def __init__(self,
                 n_splits: int = 5,
                 fold_strategy: str = "kfold",
                 stratified: bool = False,
                 use_groups: bool = False,
                 random_state: int = 42,
                 target_partition: str = "train",
                 operation_name: Optional[str] = None):
        """
        Initialize folds operation

        Args:
            n_splits: number of folds to create
            fold_strategy: folding strategy ("kfold", "stratified", "group")
            stratified: whether to use stratified folding (for classification)
            use_groups: whether to use group-based folding (cluster-aware)
            random_state: random state for reproducibility
            target_partition: partition to create folds for (default: "train")
            operation_name: custom name for this operation
        """
        self.n_splits = n_splits
        self.fold_strategy = fold_strategy
        self.stratified = stratified
        self.use_groups = use_groups
        self.random_state = random_state
        self.target_partition = target_partition
        self.operation_name = operation_name
        self.fold_assignments = {}  # {sample_id: fold_id}

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute fold creation: assign fold indices to train samples"""
        print(f"🔄 Executing {self.get_name()}")

        # Get target partition data
        target_view = dataset.select(partition=self.target_partition, **context.current_filters)
        if len(target_view) == 0:
            raise ValueError(f"No data found in partition '{self.target_partition}' for fold creation")

        sample_ids = target_view.get_sample_ids()
        n_samples = len(sample_ids)

        print(f"  📊 Creating {self.n_splits} folds for {n_samples} samples in '{self.target_partition}' partition")
        print(f"  🔧 Strategy: {self.fold_strategy}, Stratified: {self.stratified}, Use groups: {self.use_groups}")

        # Prepare data for folding
        X_dummy = np.arange(n_samples).reshape(-1, 1)  # Dummy features for sklearn folding

        # Get targets and groups if needed
        y = None
        groups = None

        if self.stratified or self.fold_strategy == "stratified":
            try:
                y = target_view.get_targets("auto")
                print(f"    📈 Using targets for stratified folding")
            except:
                print(f"    ⚠️ No targets found, falling back to regular k-fold")
                self.stratified = False

        if self.use_groups or self.fold_strategy == "group":
            try:
                groups = target_view.group_ids
                if groups is not None and len(set(groups)) > 1:
                    print(f"    👥 Using {len(set(groups))} groups for group-based folding")
                else:
                    print(f"    ⚠️ No groups found, falling back to regular k-fold")
                    self.use_groups = False
                    groups = None
            except:
                print(f"    ⚠️ Groups not available, falling back to regular k-fold")
                self.use_groups = False
                groups = None

        # Create appropriate folder
        if self.use_groups and groups is not None:
            folder = GroupKFold(n_splits=self.n_splits)
            fold_iterator = folder.split(X_dummy, y, groups)
        elif self.stratified and y is not None:
            folder = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            fold_iterator = folder.split(X_dummy, y)
        else:
            folder = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            fold_iterator = folder.split(X_dummy)

        # Assign folds
        fold_assignment = np.full(n_samples, -1)  # Initialize with -1 (no fold)

        for fold_idx, (train_indices, val_indices) in enumerate(fold_iterator):
            # Mark validation samples with their fold number
            fold_assignment[val_indices] = fold_idx        # Store fold assignments
        self.fold_assignments = {}
        for i, sample_id in enumerate(sample_ids):
            fold_id = int(fold_assignment[i])
            self.fold_assignments[sample_id] = fold_id

        # Create fold definitions for the dataset
        fold_definitions = []
        for fold_idx in range(self.n_splits):
            fold_samples = [sid for sid, fid in self.fold_assignments.items() if fid == fold_idx]
            fold_definitions.append({
                "fold_id": fold_idx,
                "samples": fold_samples,
                "strategy": self.fold_strategy,
                "n_splits": self.n_splits
            })

        dataset.add_folds(fold_definitions)

        # Verify fold distribution
        fold_counts = {}
        for fold_id in self.fold_assignments.values():
            fold_counts[fold_id] = fold_counts.get(fold_id, 0) + 1

        print("  ✅ Folds created successfully:")
        for fold_id, count in sorted(fold_counts.items()):
            print(f"    Fold {fold_id}: {count} samples")

    def get_fold_assignments(self) -> Dict[str, int]:
        """Get mapping of sample_id -> fold_id"""
        return self.fold_assignments.copy()

    def get_fold_samples(self, fold_id: int) -> List[str]:
        """Get list of sample IDs in specific fold"""
        return [sample_id for sample_id, fid in self.fold_assignments.items() if fid == fold_id]

    def get_name(self) -> str:
        """Get operation name"""
        if self.operation_name:
            return self.operation_name
        strategy_str = "Stratified" if self.stratified else "Group" if self.use_groups else "K"
        return f"Folds({strategy_str}-{self.n_splits})"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if fold creation can be executed"""
        target_view = dataset.select(partition=self.target_partition, **context.current_filters)
        return len(target_view) >= self.n_splits  # Need at least n_splits samples
