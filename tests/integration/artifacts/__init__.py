"""
Integration tests for the artifacts system.

These tests verify the complete artifact lifecycle in real pipeline scenarios:
- Training + Predict flow (test_artifact_flow.py)
- Branch-specific artifacts (test_branching_artifacts.py - in pipeline/)
- Stacking and meta-model persistence (test_stacking_artifacts.py)
- Cross-run deduplication (test_deduplication.py)
- CV fold models (test_fold_models.py)
- Cleanup utilities (test_cleanup.py)
"""
