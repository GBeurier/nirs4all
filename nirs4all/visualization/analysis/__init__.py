"""
Analysis utilities for visualization.
"""
from nirs4all.visualization.analysis.transfer import PreprocPCAEvaluator

# Import ShapAnalyzer conditionally to handle numpy 2.x compatibility
try:
    from nirs4all.visualization.analysis.shap import ShapAnalyzer
    __all__ = ['PreprocPCAEvaluator', 'ShapAnalyzer']
except (ImportError, AttributeError):
    # SHAP not available or incompatible with current numpy version
    __all__ = ['PreprocPCAEvaluator']
