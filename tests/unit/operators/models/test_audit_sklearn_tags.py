"""Estimator tags must survive supported sklearn >= 1.6 dispatch."""

import importlib

import pytest
from sklearn.base import clone, is_classifier, is_regressor


@pytest.mark.parametrize("module,name", [
    ("simpls", "SIMPLS"), ("plsda", "PLSDA"), ("oplsda", "OPLSDA"), ("opls", "OPLS"),
    ("mbpls", "MBPLS"), ("dipls", "DiPLS"), ("sparsepls", "SparsePLS"), ("lwpls", "LWPLS"),
    ("ipls", "IntervalPLS"), ("robust_pls", "RobustPLS"), ("recursive_pls", "RecursivePLS"),
    ("kopls", "KOPLS"), ("nlpls", "KernelPLS"), ("oklmpls", "OKLMPLS"), ("fckpls", "FCKPLS"),
    ("pcr", "PCR"), ("tabpfn_nirs", "TabPFNNIRSRegressor"),
])
def test_estimator_dispatch_and_clone_preserve_kind(module, name):
    cls = getattr(importlib.import_module(f"nirs4all.operators.models.sklearn.{module}"), name)
    model = cls()
    classifier = name in {"PLSDA", "OPLSDA"}
    for estimator in (model, clone(model)):
        assert is_classifier(estimator) is classifier
        assert is_regressor(estimator) is not classifier
