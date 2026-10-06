"""Selection and constructor-contract regressions from the October audit."""

from types import SimpleNamespace

import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from nirs4all.operators.models.meta import MetaModel
from nirs4all.operators.models.selection import DiversitySelector, ModelCandidate, TopKByMetricSelector


@pytest.mark.parametrize("metric,scores,winner", [("r2", [0.0, 0.8], "b"), ("rmse", [0.0, 0.8], "a")])
@pytest.mark.parametrize("selector", [TopKByMetricSelector(k=1), DiversitySelector(max_per_class=1)])
def test_model_selection_respects_zero_scores_and_metric_direction(metric, scores, winner, selector):
    candidates = [ModelCandidate(name, "Ridge", 1, val_score=score, metric=metric)
                  for name, score in zip(("a", "b"), scores, strict=True)]
    context = SimpleNamespace(state=SimpleNamespace(step_number=2), selector=SimpleNamespace(branch_id=None))
    if isinstance(selector, TopKByMetricSelector):
        selector.metric = metric
    assert [item.model_name for item in selector.select(candidates, context, None)] == [winner]


def test_meta_name_is_settable_and_cloneable():
    model = MetaModel(Ridge(), name="old")
    model.set_params(name="new")
    assert model.name == "new"
    assert clone(model).name == "new"
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.set_params(level=4)
