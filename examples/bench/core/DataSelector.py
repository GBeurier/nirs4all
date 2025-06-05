"""
DataSelector - Operation scoping and data selection rules

Defines default scoping behaviors for different types of operations:
- TransformerMixin: fit on train, transform on all
- ClusterMixin: fit on train, predict on all
- Models: train on train, validate on validation, predict on test
- etc.
"""
from typing import Dict, List, Optional, Any, Set
from abc import ABC, abstractmethod


class ScopeRule(ABC):
    """Base class for operation scoping rules."""

    @abstractmethod
    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        """Get filters for fitting the operation."""
        pass

    @abstractmethod
    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        """Get filters for transforming data."""
        pass

    @abstractmethod
    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        """Get filters for prediction."""
        pass


class StandardTransformerRule(ScopeRule):
    """Standard transformer: fit on train, transform on all data."""

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return context.current_filters.copy()

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return context.current_filters.copy()


class ClusterRule(ScopeRule):
    """Clustering: fit on train, predict on all."""

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return context.current_filters.copy()

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return context.current_filters.copy()


class ModelRule(ScopeRule):
    """Model: train on train, validate on validation, predict on test."""

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        # For models, transform usually means validate
        return {**context.current_filters, "partition": "validation"}

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "test"}


class FoldRule(ScopeRule):
    """Folds: only applied to train data."""

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}


class SplitRule(ScopeRule):
    """Split: applied to train, creates test partition."""

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": "train"}


class AugmentationRule(ScopeRule):
    """Augmentation: applied to specific data subsets."""

    def __init__(self, target_partition: str = "train"):
        self.target_partition = target_partition

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": self.target_partition}

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": self.target_partition}

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return {**context.current_filters, "partition": self.target_partition}


class AdvancedScopeRule(ScopeRule):
    """Advanced scope rule with configurable behavior and context awareness."""

    def __init__(self, fit_scope: str = "train", transform_scope: str = "all",
                 predict_scope: str = "test", context_aware: bool = True):
        self.fit_scope = fit_scope
        self.transform_scope = transform_scope
        self.predict_scope = predict_scope
        self.context_aware = context_aware

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        base_filters = context.current_filters.copy() if self.context_aware else {}

        if self.fit_scope == "train":
            base_filters["partition"] = "train"
        elif self.fit_scope == "all":
            pass  # No additional partition filter
        elif self.fit_scope == "current_branch":
            if hasattr(context, 'current_branch'):
                base_filters["branch"] = context.current_branch
        elif self.fit_scope == "centroids_only":
            base_filters["centroid"] = True
        elif self.fit_scope == "non_augmented":
            base_filters["augmented"] = False

        return base_filters

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        base_filters = context.current_filters.copy() if self.context_aware else {}

        if self.transform_scope == "all":
            pass  # No additional filters
        elif self.transform_scope == "train":
            base_filters["partition"] = "train"
        elif self.transform_scope == "test":
            base_filters["partition"] = "test"
        elif self.transform_scope == "current_processing_level":
            if hasattr(context, 'current_processing_level'):
                base_filters["processing"] = str(context.current_processing_level)

        return base_filters

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        base_filters = context.current_filters.copy() if self.context_aware else {}

        if self.predict_scope == "test":
            base_filters["partition"] = "test"
        elif self.predict_scope == "all":
            pass  # No additional filters
        elif self.predict_scope == "validation":
            base_filters["partition"] = "validation"

        return base_filters


class SourceAwareRule(ScopeRule):
    """Source-aware scoping for multi-source operations."""

    def __init__(self, source_strategy: str = "all", merge_mode: str = "concatenate"):
        self.source_strategy = source_strategy  # all, primary, specific, adaptive
        self.merge_mode = merge_mode  # concatenate, ensemble, weighted

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        filters = {**context.current_filters, "partition": "train"}

        if self.source_strategy == "primary":
            filters["source_type"] = "primary"
        elif self.source_strategy == "specific":
            if hasattr(context, 'active_sources'):
                filters["active_sources"] = context.active_sources

        filters["source_merge_mode"] = self.merge_mode
        return filters

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        filters = context.current_filters.copy()
        filters["source_merge_mode"] = self.merge_mode

        if self.source_strategy == "primary":
            filters["source_type"] = "primary"
        elif self.source_strategy == "specific":
            if hasattr(context, 'active_sources'):
                filters["active_sources"] = context.active_sources

        return filters

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return self.get_transform_filters(context)


class AugmentationAwareRule(ScopeRule):
    """Augmentation-aware scoping for operations that need to handle augmented data."""

    def __init__(self, fit_augmented: bool = False, transform_augmented: bool = True):
        self.fit_augmented = fit_augmented
        self.transform_augmented = transform_augmented

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        filters = {**context.current_filters, "partition": "train"}

        if not self.fit_augmented:
            filters["augmented"] = False

        return filters

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        filters = context.current_filters.copy()

        if not self.transform_augmented:
            filters["augmented"] = False

        return filters

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return self.get_transform_filters(context)


class ClusterAwareRule(ScopeRule):
    """Cluster-aware scoping for operations that work with clustering results."""

    def __init__(self, use_centroids: bool = False, cluster_specific: bool = False):
        self.use_centroids = use_centroids
        self.cluster_specific = cluster_specific

    def get_fit_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        filters = {**context.current_filters, "partition": "train"}

        if self.use_centroids:
            filters["centroid"] = True

        if self.cluster_specific and hasattr(context, 'current_cluster'):
            filters["cluster"] = context.current_cluster

        return filters

    def get_transform_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        filters = context.current_filters.copy()

        if self.cluster_specific and hasattr(context, 'current_cluster'):
            filters["cluster"] = context.current_cluster

        return filters

    def get_predict_filters(self, context: 'PipelineContext') -> Dict[str, Any]:
        return self.get_transform_filters(context)


class DataSelector:
    """
    Central manager for operation data selection and scoping.

    Determines which subset of data each operation should work on based on:
    - Operation type (TransformerMixin, ClusterMixin, Model, etc.)
    - Current pipeline context (filters, branch, scope stack)
    - Operation-specific configuration
    """

    def __init__(self):
        # Enhanced rule registry with more sophisticated scoping
        self.rules: Dict[str, ScopeRule] = {
            "transformer": StandardTransformerRule(),
            "cluster": ClusterRule(),
            "model": ModelRule(),
            "fold": FoldRule(),
            "split": SplitRule(),
            "sample_augmentation": AugmentationRule("train"),
            "feature_augmentation": AugmentationRule("train"),

            # Advanced rules for complex operations
            "source_aware_transformer": SourceAwareRule("all", "concatenate"),
            "source_ensemble": SourceAwareRule("all", "ensemble"),
            "primary_source_only": SourceAwareRule("primary", "concatenate"),
            "augmentation_aware": AugmentationAwareRule(fit_augmented=False, transform_augmented=True),
            "cluster_centroid": ClusterAwareRule(use_centroids=True, cluster_specific=False),
            "cluster_specific": ClusterAwareRule(use_centroids=False, cluster_specific=True),

            # Context-aware advanced rules
            "adaptive_scope": AdvancedScopeRule("train", "all", "test", context_aware=True),
            "branch_local": AdvancedScopeRule("current_branch", "current_branch", "current_branch"),
            "centroid_based": AdvancedScopeRule("centroids_only", "all", "test"),
            "non_augmented_fit": AdvancedScopeRule("non_augmented", "all", "test"),
        }

        # Enhanced operation type mappings with more granular detection
        self.operation_types = {
            # Standard sklearn modules
            "sklearn.preprocessing": "transformer",
            "sklearn.decomposition": "transformer",
            "sklearn.feature_selection": "transformer",
            "sklearn.cluster": "cluster",
            "sklearn.ensemble": "model",
            "sklearn.linear_model": "model",
            "sklearn.tree": "model",
            "sklearn.svm": "model",
            "sklearn.neighbors": "model",
            "sklearn.naive_bayes": "model",
            "sklearn.neural_network": "model",

            # Specialized modules
            "sklearn.model_selection": "fold",
            "sklearn.pipeline": "transformer",
            "sklearn.compose": "transformer",

            # Custom operation types
            "custom.augmentation": "sample_augmentation",
            "custom.feature_augmentation": "feature_augmentation",
            "custom.source_dispatch": "source_aware_transformer",
            "custom.clustering": "cluster_centroid",
        }

        # Operation-specific rule overrides
        self.specific_rules: Dict[str, str] = {
            "StandardScaler": "transformer",
            "MinMaxScaler": "transformer",
            "RobustScaler": "transformer",
            "PCA": "source_aware_transformer",  # PCA often benefits from source awareness
            "KMeans": "cluster_centroid",
            "AgglomerativeClustering": "cluster_specific",
            "RandomForestClassifier": "model",
            "SVC": "non_augmented_fit",  # SVM often works better without augmented data
        }

    def get_operation_type(self, operation: Any) -> str:
        """Enhanced operation type detection with hierarchical rules."""

        # Check specific class name first
        class_name = operation.__class__.__name__
        if class_name in self.specific_rules:
            return self.specific_rules[class_name]

        # Check module hierarchy
        if hasattr(operation, '__module__'):
            module = operation.__module__
            for module_prefix, op_type in self.operation_types.items():
                if module.startswith(module_prefix):
                    return op_type

        # Advanced operation detection based on interface
        return self._detect_operation_interface(operation)

    def _detect_operation_interface(self, operation: Any) -> str:
        """Detect operation type based on interface patterns."""

        # Check for clustering operations
        if (hasattr(operation, 'fit') and
            (hasattr(operation, 'cluster_centers_') or
             hasattr(operation, 'labels_') or
             'cluster' in operation.__class__.__name__.lower())):
            return "cluster"

        # Check for augmentation operations
        if (hasattr(operation, 'augment') or
            'augment' in operation.__class__.__name__.lower()):
            if hasattr(operation, 'feature_augment'):
                return "feature_augmentation"
            else:
                return "sample_augmentation"

        # Check for source-aware operations
        if (hasattr(operation, 'source_dispatch') or
            hasattr(operation, 'multi_source') or
            'source' in operation.__class__.__name__.lower()):
            return "source_aware_transformer"

        # Check for transformers vs models
        if hasattr(operation, 'fit') and hasattr(operation, 'transform'):
            return "transformer"
        elif hasattr(operation, 'fit') and hasattr(operation, 'predict'):
            return "model"
        elif hasattr(operation, 'split'):
            return "fold"

        # Default fallback
        return "transformer"

    def get_enhanced_scope(self, operation: Any, context: 'PipelineContext',
                         phase: str = "fit", custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get enhanced scope with custom configuration support."""

        # Get base operation type and rule
        op_type = self.get_operation_type(operation)
        rule = self.rules.get(op_type, self.rules["transformer"])

        # Get base scope from rule
        if phase == "fit":
            scope = rule.get_fit_filters(context)
        elif phase == "transform":
            scope = rule.get_transform_filters(context)
        elif phase == "predict":
            scope = rule.get_predict_filters(context)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Apply custom configuration overrides
        if custom_config:
            scope = self._apply_custom_config(scope, custom_config, context)

        return scope

    def _apply_custom_config(self, base_scope: Dict[str, Any],
                           custom_config: Dict[str, Any],
                           context: 'PipelineContext') -> Dict[str, Any]:
        """Apply custom configuration to base scope."""

        scope = base_scope.copy()

        # Handle scope overrides
        if "scope_override" in custom_config:
            scope.update(custom_config["scope_override"])

        # Handle source-specific configuration
        if "source_config" in custom_config:
            source_config = custom_config["source_config"]
            if "active_sources" in source_config:
                scope["active_sources"] = source_config["active_sources"]
            if "merge_mode" in source_config:
                scope["source_merge_mode"] = source_config["merge_mode"]

        # Handle augmentation configuration
        if "augmentation_config" in custom_config:
            aug_config = custom_config["augmentation_config"]
            if "exclude_augmented" in aug_config:
                scope["augmented"] = not aug_config["exclude_augmented"]

        # Handle cluster configuration
        if "cluster_config" in custom_config:
            cluster_config = custom_config["cluster_config"]
            if "use_centroids" in cluster_config:
                scope["centroid"] = cluster_config["use_centroids"]
            if "cluster_filter" in cluster_config:
                scope["cluster"] = cluster_config["cluster_filter"]

        return scope

    def register_custom_rule(self, operation_identifier: str, rule: ScopeRule,
                           rule_type: str = "class_name"):
        """Register custom scoping rule for specific operations."""

        if rule_type == "class_name":
            self.specific_rules[operation_identifier] = rule
        elif rule_type == "module":
            self.operation_types[operation_identifier] = rule
        elif rule_type == "rule_name":
            self.rules[operation_identifier] = rule
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    def get_scope_diagnostics(self, operation: Any, context: 'PipelineContext') -> Dict[str, Any]:
        """Get diagnostic information about scoping decisions."""

        op_type = self.get_operation_type(operation)
        rule = self.rules.get(op_type, self.rules["transformer"])

        return {
            "operation_class": operation.__class__.__name__,
            "operation_module": getattr(operation, '__module__', 'unknown'),
            "detected_type": op_type,
            "rule_class": rule.__class__.__name__,
            "fit_scope": rule.get_fit_filters(context),
            "transform_scope": rule.get_transform_filters(context),
            "predict_scope": rule.get_predict_filters(context),
        }
