"""Focused dag-ml pipeline step parsing regressions."""

from __future__ import annotations

from sklearn.model_selection import KFold

from nirs4all.pipeline.dagml.steps import _expand_operator_generators, _split_pipeline


def test_split_pipeline_does_not_treat_string_class_paths_as_splitters() -> None:
    splitter = KFold(n_splits=3)
    steps, parsed = _split_pipeline(
        [
            "nirs4all.operators.transforms.StandardNormalVariate",
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3}},
            splitter,
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]
    )

    assert parsed is splitter
    assert steps == [
        "nirs4all.operators.transforms.StandardNormalVariate",
        {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3}},
        {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
    ]


def test_generator_free_canonical_pipeline_deserializes_studio_steps() -> None:
    variants = _expand_operator_generators(
        [
            "nirs4all.operators.transforms.StandardNormalVariate",
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3}},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 5}}},
        ]
    )

    assert len(variants) == 1
    steps, splitter = _split_pipeline(variants[0])

    assert type(steps[0]).__name__ == "StandardNormalVariate"
    assert type(splitter).__name__ == "KFold"
    assert steps[1]["model"].__class__.__name__ == "PLSRegression"
