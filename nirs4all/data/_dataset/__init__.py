"""Dataset component accessors for better separation of concerns."""

from nirs4all.data._dataset.feature_accessor import FeatureAccessor
from nirs4all.data._dataset.target_accessor import TargetAccessor
from nirs4all.data._dataset.metadata_accessor import MetadataAccessor

__all__ = [
    "FeatureAccessor",
    "TargetAccessor",
    "MetadataAccessor",
]
