"""Canonical bare default estimators remain model nodes for refit selection."""

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.pipeline.analysis.topology import analyze_topology
from nirs4all.pipeline.config import PipelineConfigs


def test_default_bare_model_and_splitter_survive_canonical_topology():
    config = PipelineConfigs([KFold(3), Ridge()])
    topology = analyze_topology(config.steps[0])
    assert topology.splitter_step_index == 0
    assert len(topology.model_nodes) == 1
    assert topology.model_nodes[0].model_class.endswith(".Ridge")
    assert topology.model_nodes[0].step_index == 1
