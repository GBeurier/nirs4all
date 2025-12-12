"""
Analysis utilities for visualization.
"""
from nirs4all.visualization.analysis.transfer import PreprocPCAEvaluator
from nirs4all.visualization.analysis.branch import BranchAnalyzer, BranchSummary

# Import ShapAnalyzer conditionally to handle numpy 2.x compatibility
try:
    from nirs4all.visualization.analysis.shap import ShapAnalyzer
    __all__ = ['PreprocPCAEvaluator', 'ShapAnalyzer', 'BranchAnalyzer', 'BranchSummary']
except (ImportError, AttributeError):
    # SHAP not available or incompatible with current numpy version
    __all__ = ['PreprocPCAEvaluator', 'BranchAnalyzer', 'BranchSummary']
