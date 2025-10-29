"""Dataset component accessors for better separation of concerns."""

from nirs4all.data.dataset_components.feature_accessor import FeatureAccessor
from nirs4all.data.dataset_components.target_accessor import TargetAccessor
from nirs4all.data.dataset_components.metadata_accessor import MetadataAccessor

__all__ = [
    "FeatureAccessor",
    "TargetAccessor",
    "MetadataAccessor",
]
