"""
Visualization tools for NIRS data analysis.
"""
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.visualization.pipeline_diagram import (
    PipelineDiagram,
    plot_pipeline_diagram,
)
# Backward compatibility - branch_diagram is now an alias for pipeline_diagram
from nirs4all.visualization.branch_diagram import BranchDiagram, plot_branch_diagram
from nirs4all.visualization.analysis.branch import BranchAnalyzer, BranchSummary

__all__ = [
    'PredictionAnalyzer',
    'PipelineDiagram',
    'plot_pipeline_diagram',
    # Backward compatibility
    'BranchDiagram',
    'plot_branch_diagram',
    'BranchAnalyzer',
    'BranchSummary',
]
