"""EXHAUSTIVE dual-engine GENERATION conformance cases (locks C/#23 + B/#33).

The generator DSL is the most combinatorially complex part of the pipeline
surface: a single declaration fans out into a *set* of variants the runner
trains, ranks, and refits. Both engines must enumerate the SAME variant set in
the SAME order and select the SAME winner, or selection + bundle replay diverge.

This module is the comprehensive parity ORACLE that the upcoming NATIVE generator
work must preserve byte-for-byte. Today the dag-ml backend HOST-EXPANDS operator
generators (it reuses ``expand_spec`` + ``apply_all_constraints``); these cases
LOCK that currently-correct behavior so the native rewrite cannot regress it
silently. ``cases_generators.py`` already covers each base keyword in isolation
(``_or_`` / ``_range_`` / ``_log_range_`` / ``_grid_`` / ``_zip_`` / ``_chain_``
/ ``_sample_`` / ``_cartesian_`` / ``finetune_params``); this module adds the
COMBINATIONS, the CONSTRAINT surface, the SEEDED-vs-unseeded split, the EDGE
cases, and the multi-model selection shape they did not reach.

Verified generator semantics (measured against ``nirs4all.pipeline.config
._generator``) that shape these cases:

* Constraints (``_mutex_`` / ``_requires_`` / ``_exclude_``) only filter when a
  generator produces LIST variants — i.e. a ``_or_`` with ``pick``/``arrange``
  (each variant a combination) or a ``_cartesian_`` (each variant a pipeline
  list). They are read ONLY off a PURE generator node (``OrStrategy`` /
  ``CartesianStrategy``); a constraint key on a MIXED ``_or_`` step dict would
  land in the base dict and break expansion, so every constraint case here puts
  the constraint on a pure node that becomes a preprocessing step.
* ``count_combinations`` does NOT apply constraints (it returns the pre-filter
  count); only ``expand_spec`` prunes — and the orchestrator runs ``expand_spec``,
  so the variant count (and ``num_predictions``) reflects the PRUNED set.
* ``_depends_on_`` is declared in ``CONSTRAINT_KEYWORDS`` but the constraint
  engine (``parse_constraints`` / ``apply_all_constraints``) NEVER consults it —
  it is a DEAD keyword with no filter semantics. Adding it to a top-level
  generator node even BREAKS expansion (it is absent from the pure-node key sets,
  so the node leaves its strategy path). It is documented by
  ``test_generators_conformance_extra.test_depends_on_is_inert_in_constraint_engine``
  rather than a pipeline case, because no runnable pipeline shape can carry it.
* A single-variant GENERATOR (degenerate ``_or_`` of one, a ``_range_`` of one
  point, or a constraint that prunes to one) yields an EMPTY ``config_name`` on
  the dag-ml side (no selection among multiple), while legacy populates it. This
  does NOT affect any standard parity assertion (``num_predictions`` /
  ``best_score`` / the selected-metric name / the top-n model SET / per-sample
  ``y_pred`` all match), so these cases are GREEN; the conformance test only calls
  ``assert_same_winner`` (which reads ``config_name``) for the relaxed-tolerance
  allowlist, never for these.

INTENTIONAL num_predictions divergence recorded here (ADR-17 1c): a multi-model
``{"model": {"_or_": [...]}}`` over DISTINCT model classes diverges in
``num_predictions`` — legacy refits EVERY model variant and stores a
``(model, *, final)`` row per model (34 entries), while dag-ml's operator-SELECT
refits ONLY the selected winner (32 entries, the correct SELECT semantic). The
winner identity, ``best_score``, ``best_rmse``, the selected-metric name, the
top-n model set, and the winner's per-sample ``y_pred`` are all identical; the
gap is purely the one loser's stored ``(train, final)`` + ``(test, final)`` refit
rows. dag-ml is RIGHT, so this is NOT a strict-xfail — it is a PASSING parity-note
in ``test_conformance_dual_engine.NUM_PREDICTIONS_DIVERGENCE``: the conformance
body asserts winner + full metric/contract parity + winner-scoped ``y_pred``
parity, and pins the EXACT documented 34-legacy / 32-dag-ml counts (only the
measured +2 loser-final-row delta passes; any other count FAILS).

Additional coverage (sections 7-14) for genuine exhaustiveness:

* second-order selection ``then_pick`` / ``then_arrange``;
* the remaining ``_sample_`` distributions (uniform / normal / choice), each
  ``_seed_``-pinned so both engines draw the identical set -> strict parity;
* the DICT spellings of ``_range_`` / ``_log_range_``;
* nested generator VALUES inside ``_grid_`` / ``_zip_`` param maps, and ``_zip_``
  with UNEQUAL-length lists (truncate-to-shortest);
* ``_cartesian_`` with ``pick`` (selecting pairs of complete pipelines);
* TWO constraint types composed on ONE node (``_mutex_`` + ``_exclude_``);
* the INERT annotation keywords ``_tags_`` / ``_metadata_`` (clean no-ops that, unlike
  ``_depends_on_``, do NOT break expansion and do NOT change the variant set);
* ``count`` capping with ``_seed_``.

The EXACT surviving variant set (members, not just count) for every constraint case
is locked at the DSL level in ``test_generators_conformance_extra`` so a wrong-prune
(right count, wrong members) fails; the engine-level winner identity is locked via
``SAME_WINNER_CASES`` in the dual-engine test for the multi-variant cases.

The ``_or_`` count-sampling path is deterministic when ``_seed_`` is present,
including weighted sampling via ``_weights_``. These are live parity cases, not
registry skips: the Python expander threads the node-local seed and weights into
``OrStrategy``'s ``sample_with_seed`` cap, and dag-ml consumes the same expanded
variant set.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.operators.transforms import Detrend, FirstDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

# Shared tags. `slow` gates these under the slow marker (each runs twice — legacy
# + dag-ml); `generator` groups the generation oracle for targeted `-k` runs.
_GEN = frozenset({"generator", "slow"})
_CAPS = (
    "preprocessing_transform",
    "cross_validator",
    "sklearn_model",
    "regression_model",
    "generator",
)


# =============================================================================
# 1) SEEDED vs UNSEEDED _sample_  (deterministic -> strict parity)
# =============================================================================
# `cases_generators.generator_sample_log_uniform_alpha` is the UNSEEDED partner
# (no `_seed_`): genuinely stochastic across engines -> strict-xfail. This is its
# SEEDED twin: `_seed_` pins the sampler, so both engines draw the IDENTICAL alpha
# set, train the same variants, and select the same winner -> strict PARITY.


def _factory_sample_seeded_alpha() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_sample_": {"distribution": "log_uniform", "from": 1e-4, "to": 1e0, "num": 5},
            "_seed_": 123,
            "param": "alpha",
            "model": Ridge,
        },
    ]


register(
    PipelineCase(
        name="generator_sample_seeded_alpha",
        description="`_sample_` Ridge.alpha log-uniform WITH `_seed_`: deterministic across engines "
        "-> strict parity (the seeded twin of the unseeded `generator_sample_log_uniform_alpha` xfail).",
        keywords=("_sample_", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_sample_seeded_alpha,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


# =============================================================================
# 2) GENERATOR x other workflow keywords (combinations)
# =============================================================================


def _factory_or_pp_yprocessing() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend]},
        {"y_processing": MinMaxScaler()},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_preprocessing_yprocessing",
        description="`_or_` preprocessing combined with a `y_processing` target scaler — "
        "generator expansion threaded through an inverse-transform path. 3 variants x 3 folds.",
        keywords=("_or_", "y_processing", "model"),
        capabilities=(*_CAPS, "y_processing_transform"),
        dataset_key="regression",
        pipeline_factory=_factory_or_pp_yprocessing,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_or_pp_kfold() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend]},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_preprocessing_kfold",
        description="`_or_` preprocessing under a KFold splitter (vs the ShuffleSplit baselines) — "
        "asserts generator x splitter-variety parity. 3 variants x 3 folds.",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_pp_kfold,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_mixed_or_and_param_range() -> list[Any]:
    # Operator-generator (preprocessing `_or_`) AND a model-param sweep (`_range_`
    # over n_components) in the SAME pipeline: 2 preproc x 3 n_components = 6.
    return [
        {"_or_": [SNV, MSC]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"_range_": [5, 15, 5], "param": "n_components", "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_mixed_or_and_param_range",
        description="Operator `_or_` (preprocessing) AND a model `_range_` param sweep in one pipeline — "
        "2 preproc x 3 n_components = 6 variants x 3 folds.",
        keywords=("_or_", "_range_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_mixed_or_and_param_range,
        expected_min_predictions=90,
        tags=_GEN,
    )
)


# =============================================================================
# 3) STRUCTURAL NESTING
# =============================================================================


def _factory_cartesian_with_param_range() -> list[Any]:
    # `_cartesian_` of two `_or_` stages (2 x 2 = 4 preprocessing pipelines) PLUS a
    # model `_range_` sweep (3 n_components) -> 4 x 3 = 12 variants.
    return [
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"_range_": [5, 15, 5], "param": "n_components", "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_cartesian_with_param_range",
        description="`_cartesian_` of two preprocessing `_or_`s (4 combos) x a model `_range_` sweep "
        "(3 n_components) = 12 variants x 3 folds — deep nesting + a param sweep.",
        keywords=("_cartesian_", "_or_", "_range_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_cartesian_with_param_range,
        expected_min_predictions=180,
        tags=_GEN,
    )
)


def _factory_or_multistep_branch() -> list[Any]:
    # Each `_or_` choice is a MULTI-STEP list (a 2-preprocessor branch), not a
    # single class — `_or_` over branches of differing length/shape.
    return [
        {"_or_": [[SNV, FirstDerivative], [MSC, Detrend]]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_multistep_branch",
        description="`_or_` whose choices are MULTI-STEP preprocessing branches "
        "([SNV, 1stDer] vs [MSC, Detrend]) — generator over branch-shaped choices. 2 variants x 3 folds.",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_multistep_branch,
        expected_min_predictions=30,
        tags=_GEN,
    )
)


def _factory_chain_param_n_components() -> list[Any]:
    # `_chain_` over an explicit list of model param values (n_components),
    # sibling-form sweep — sequential ordered iteration of 3 values.
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_chain_": [5, 10, 15], "param": "n_components", "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_chain_param_n_components",
        description="`_chain_` over PLSRegression.n_components values [5, 10, 15] via the sibling-form "
        "`param` sweep — ordered sequential iteration. 3 variants x 3 folds.",
        keywords=("_chain_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_chain_param_n_components,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_or_arrange_ordered() -> list[Any]:
    # `arrange` is the ORDERED partner of `pick`: permutations, not combinations,
    # so SNV>MSC and MSC>SNV are DISTINCT preprocessing sequences. P(3,2) = 6.
    return [
        {"_or_": [SNV, MSC, Detrend], "arrange": 2},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_arrange_ordered",
        description="`_or_` arrange=2 over 3 preprocessors — ORDERED permutations P(3,2)=6 (SNV>MSC vs "
        "MSC>SNV are distinct sequences), the ordered partner of `pick`. 6 variants x 3 folds.",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_arrange_ordered,
        expected_min_predictions=90,
        tags=_GEN,
    )
)


# =============================================================================
# 4) GENERATOR CONSTRAINTS (item B surface) — _mutex_ / _requires_ / _exclude_
# =============================================================================
# Constraints prune the expanded variant SET; the parity claim is that the engines
# train+rank the SAME pruned set and select the SAME winner. The constraint sits
# on a PURE `_or_` (with `pick`, so variants are combinations) or a PURE
# `_cartesian_` (variants are pipeline lists) — the only nodes the constraint
# engine reads off. C(4,2) = 6 before pruning.


def _factory_or_pick_mutex() -> list[Any]:
    # `_mutex_` [[SNV, MSC]]: SNV and MSC cannot co-occur -> the {SNV, MSC} combo
    # is removed -> 6 - 1 = 5 variants.
    return [
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_mutex_": [[SNV, MSC]]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_pick_mutex",
        description="`_or_` pick=2 over 4 preprocessors with `_mutex_` [[SNV, MSC]] — SNV+MSC pruned, "
        "C(4,2)=6 -> 5 variants x 3 folds. Locks the mutex-pruned set + winner.",
        keywords=("_or_", "_mutex_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_or_pick_mutex,
        expected_min_predictions=75,
        tags=_GEN,
    )
)


def _factory_or_pick_mutex3() -> list[Any]:
    # SIZE-3 `_mutex_` [[SNV, MSC, Detrend]] — this is where the legacy "not all co-occur"
    # (`_satisfies_mutex`: `mutex_set.issubset(combo_set)`) semantic DIVERGES from a naive
    # "at most one present" reading. pick=3 over 4 ops -> C(4,3)=4 combos; legacy forbids ONLY
    # the single combo where ALL THREE mutex members co-occur ({SNV,MSC,Detrend}) and keeps every
    # combo carrying just two of them -> 4 - 1 = 3 survivors. A "<=1 present" rule would instead
    # forbid every combo with 2+ of the group (all four) -> 0 survivors, so this case pins the
    # >2 mutex CONTRACT (the dag-ml native rule item 1a must later match). Routes via Python-expand
    # on BOTH engines today (constraints not native-routed yet), so it locks the LEGACY semantic.
    return [
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 3, "_mutex_": [[SNV, MSC, Detrend]]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_pick_mutex3",
        description="`_or_` pick=3 over 4 preprocessors with a SIZE-3 `_mutex_` [[SNV, MSC, Detrend]] — "
        "legacy forbids ONLY the all-three-present combo ({SNV,MSC,Detrend}), keeping the three combos "
        "with just two of them: C(4,3)=4 -> 3 variants x 3 folds. Locks the >2 mutex 'not all co-occur' "
        "contract (where '<=1 present' would wrongly prune to 0) + the same winner across engines.",
        keywords=("_or_", "_mutex_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_or_pick_mutex3,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_or_pick_requires() -> list[Any]:
    # `_requires_` [[SNV, MSC]]: if SNV is picked, MSC must also be picked. Of the
    # 6 pairs, those containing SNV-without-MSC are removed: {SNV,Detrend},
    # {SNV,1stDer} drop -> 6 - 2 = 4 variants.
    return [
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_requires_": [[SNV, MSC]]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_pick_requires",
        description="`_or_` pick=2 with `_requires_` [[SNV, MSC]] — SNV requires MSC, dropping "
        "SNV-without-MSC pairs -> 6 -> 4 variants x 3 folds. Locks the requires-pruned set + winner.",
        keywords=("_or_", "_requires_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_or_pick_requires,
        expected_min_predictions=60,
        tags=_GEN,
    )
)


def _factory_or_pick_exclude() -> list[Any]:
    # `_exclude_` [[SNV, Detrend]]: the specific {SNV, Detrend} combo is removed
    # -> 6 - 1 = 5 variants.
    return [
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_exclude_": [[SNV, Detrend]]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_pick_exclude",
        description="`_or_` pick=2 with `_exclude_` [[SNV, Detrend]] — that exact pair dropped "
        "-> 6 -> 5 variants x 3 folds. Locks the exclude-pruned set + winner.",
        keywords=("_or_", "_exclude_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_or_pick_exclude,
        expected_min_predictions=75,
        tags=_GEN,
    )
)


def _factory_cartesian_exclude() -> list[Any]:
    # `_cartesian_` of 2x2 stages = 4 pipeline lists; `_exclude_` [[SNV, Detrend]]
    # removes the pipeline whose stages are exactly {SNV, Detrend} -> 3 variants.
    return [
        {
            "_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}],
            "_exclude_": [[SNV, Detrend]],
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_cartesian_exclude",
        description="`_cartesian_` 2x2 (4 pipelines) with `_exclude_` [[SNV, Detrend]] — that pipeline "
        "pruned -> 3 variants x 3 folds. Locks cartesian-level constraint pruning + winner.",
        keywords=("_cartesian_", "_or_", "_exclude_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_cartesian_exclude,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


# =============================================================================
# 5) EDGE CASES
# =============================================================================


def _factory_or_single_variant() -> list[Any]:
    # Degenerate `_or_` of ONE choice -> a single-variant generator. Locks that a
    # generator that expands to one is byte-identical to the no-generator baseline
    # in scores, despite dag-ml emitting an empty config_name for it.
    return [
        {"_or_": [SNV]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_single_variant",
        description="Degenerate `_or_` of ONE choice — single-variant generator. Locks score/"
        "num_predictions parity for the one-variant edge (dag-ml returns an empty config_name here).",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_single_variant,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_constraint_prunes_to_one() -> list[Any]:
    # Two mutex pairs prune C(3,2)=3 down to the single surviving combo
    # {MSC, Detrend} — empty-after-constraint reduced to one variant.
    return [
        {
            "_or_": [SNV, MSC, Detrend],
            "pick": 2,
            "_mutex_": [[SNV, MSC], [SNV, Detrend]],
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_constraint_prunes_to_one",
        description="`_or_` pick=2 with two `_mutex_` pairs that prune C(3,2)=3 down to ONE surviving "
        "combo {MSC, Detrend} — the heavily-pruned edge. Locks the pruned set + score parity.",
        keywords=("_or_", "_mutex_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_constraint_prunes_to_one,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_grid_twelve_variants() -> list[Any]:
    # Large grid: 4 n_components x 3 scale = 12 variants. Asserts num_predictions
    # scales with the variant count and the CV-best winner is selected.
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_grid_": {"n_components": [5, 8, 11, 14], "scale": [True, False, True]},
            "model": PLSRegression,
        },
    ]


register(
    PipelineCase(
        name="generator_grid_twelve_variants",
        description="Large `_grid_` PLSRegression(n_components[4] x scale[3]) = 12 variants x 3 folds — "
        "asserts num_predictions scales with the variant count and the true CV-best wins.",
        keywords=("_grid_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_grid_twelve_variants,
        expected_min_predictions=180,
        tags=_GEN,
    )
)


# =============================================================================
# 6) MULTI-MODEL _or_ over model classes  (INTENTIONAL num_predictions divergence — ADR-17 1c)
# =============================================================================
# A `{"model": {"_or_": [...]}}` over DISTINCT model classes selects a winner the
# same way on both engines (identical winner, best_score, best_rmse, winner
# y_pred Δ=0.0), but legacy refits EVERY model variant and stores a (model, *,
# final) row per model (34 prediction entries) while dag-ml's operator-SELECT
# refits ONLY the selected winner (32, the correct SELECT semantic). The 2-entry
# gap is purely the LOSER's stored refit rows. dag-ml is RIGHT, so this is an
# INTENTIONAL native-vs-legacy delta — recorded in
# test_conformance_dual_engine.NUM_PREDICTIONS_DIVERGENCE as a PASSING parity-note
# (winner + metric + winner-y_pred parity, num_predictions EXEMPT), NOT a
# strict-xfail that would wrongly chase the legacy refit-all (34) count.


def _factory_or_models_pls_ridge() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": {"_or_": [PLSRegression(n_components=10), Ridge(alpha=1.0)]}},
    ]


register(
    PipelineCase(
        name="generator_or_models_pls_ridge",
        description="Multi-model `{model: {_or_: [PLSR, Ridge]}}` selection over distinct model classes. "
        "Winner/best_score/best_rmse/winner-y_pred match, but num_predictions diverges (legacy refits "
        "every model -> loser final rows: 34; dag-ml operator-SELECT refits the winner only: 32) -> "
        "INTENTIONAL native-vs-legacy divergence (ADR-17 1c), a PASSING parity-note (num_predictions exempt).",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_models_pls_ridge,
        expected_min_predictions=32,
        tags=_GEN,
    )
)


# =============================================================================
# 7) SECOND-ORDER SELECTION — then_pick / then_arrange
# =============================================================================
# `then_*` runs a SECOND selection over the result of the first: pick 1-of-3
# (3 singletons), then pick/arrange 2 of THOSE -> each variant is a 2-step
# preprocessing branch. Both forms run NATIVE at parity.


def _factory_then_pick() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend], "pick": 1, "then_pick": 2},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_then_pick",
        description="`_or_` pick=1 then_pick=2 — second-order UNORDERED selection: pick 3 singletons, "
        "then choose 2 of them -> C(3,2)=3 two-step preprocessing branches x 3 folds.",
        keywords=("_or_", "then_pick", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_then_pick,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_then_arrange() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend], "pick": 1, "then_arrange": 2},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_then_arrange",
        description="`_or_` pick=1 then_arrange=2 — second-order ORDERED selection: pick 3 singletons, "
        "then arrange 2 of them -> P(3,2)=6 ordered two-step branches x 3 folds.",
        keywords=("_or_", "then_arrange", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_then_arrange,
        expected_min_predictions=90,
        tags=_GEN,
    )
)


# =============================================================================
# 8) SAMPLING DISTRIBUTIONS (each SEEDED -> deterministic parity)
# =============================================================================
# `cases_generators` covers seeded log_uniform `_sample_` (and the unseeded xfail);
# `generator_sample_seeded_alpha` above is the log_uniform seeded twin. Here are
# the OTHER three distributions, each `_seed_`-pinned so both engines draw the
# identical value set. uniform/normal target the float `alpha` (Ridge); choice
# targets int `n_components` (PLSRegression) so the drawn values are valid.


def _factory_sample_uniform_alpha() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_sample_": {"distribution": "uniform", "from": 0.1, "to": 5.0, "num": 5}, "_seed_": 42, "param": "alpha", "model": Ridge},
    ]


register(
    PipelineCase(
        name="generator_sample_uniform_alpha",
        description="`_sample_` uniform Ridge.alpha in [0.1, 5.0], num=5, SEEDED — deterministic across "
        "engines -> strict parity. 5 variants x 3 folds.",
        keywords=("_sample_", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_sample_uniform_alpha,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_sample_normal_alpha() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_sample_": {"distribution": "normal", "mean": 1.0, "std": 0.3, "num": 5}, "_seed_": 42, "param": "alpha", "model": Ridge},
    ]


register(
    PipelineCase(
        name="generator_sample_normal_alpha",
        description="`_sample_` normal Ridge.alpha (mean=1.0, std=0.3), num=5, SEEDED — deterministic "
        "across engines -> strict parity. 5 variants x 3 folds.",
        keywords=("_sample_", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_sample_normal_alpha,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_sample_choice_ncomp() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_sample_": {"distribution": "choice", "values": [5, 10, 15, 20], "num": 5}, "_seed_": 42, "param": "n_components", "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_sample_choice_ncomp",
        description="`_sample_` choice over PLSRegression.n_components values [5,10,15,20], num=5, SEEDED "
        "— deterministic across engines -> strict parity. 5 draws x 3 folds.",
        keywords=("_sample_", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_sample_choice_ncomp,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


# =============================================================================
# 9) DICT FORMS for _range_ / _log_range_
# =============================================================================
# The list form (`_range_: [5, 25, 5]`) is covered in `cases_generators`; the
# DICT form (`{from, to, step}` / `{from, to, num}`) is the alternative spelling.


def _factory_range_dict() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_range_": {"from": 5, "to": 25, "step": 5}, "param": "n_components", "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_range_dict_form",
        description="`_range_` DICT form {from:5, to:25, step:5} over PLSRegression.n_components — the "
        "dict spelling of the list form. 5 variants x 3 folds.",
        keywords=("_range_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_range_dict,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_log_range_dict() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_log_range_": {"from": 1e-4, "to": 1e0, "num": 5}, "param": "alpha", "model": Ridge},
    ]


register(
    PipelineCase(
        name="generator_log_range_dict_form",
        description="`_log_range_` DICT form {from:1e-4, to:1e0, num:5} over Ridge.alpha — the dict "
        "spelling of the list form. 5 variants x 3 folds.",
        keywords=("_log_range_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_log_range_dict,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


# =============================================================================
# 10) NESTED GENERATOR VALUES inside _grid_ / _zip_, and unequal _zip_
# =============================================================================
# A generator VALUE inside a `_grid_`/`_zip_` param map (`n_components: {_range_:...}`)
# is recursively expanded before the product/zip. `_zip_` with UNEQUAL-length
# lists truncates to the shortest — an important edge.


def _factory_grid_nested() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_grid_": {"n_components": {"_range_": [5, 15, 5]}, "scale": [True, False]}, "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_grid_nested_range",
        description="`_grid_` with a NESTED `_range_` value (n_components:{_range_:[5,15,5]}) x scale[2] "
        "-> 3 x 2 = 6 variants x 3 folds. Nested generator expanded before the cartesian product.",
        keywords=("_grid_", "_range_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_grid_nested,
        expected_min_predictions=90,
        tags=_GEN,
    )
)


def _factory_zip_nested() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_zip_": {"n_components": {"_range_": [5, 15, 5]}, "scale": [True, False, True]}, "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_zip_nested_range",
        description="`_zip_` with a NESTED `_range_` value (n_components:{_range_:[5,15,5]}) zipped with "
        "scale[3] -> 3 paired variants x 3 folds. Nested generator expanded before the zip.",
        keywords=("_zip_", "_range_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_zip_nested,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_zip_unequal() -> list[Any]:
    # Unequal lengths: n_components has 3, scale has 2 -> zip truncates to 2.
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_zip_": {"n_components": [5, 10, 15], "scale": [True, False]}, "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_zip_unequal_lengths",
        description="`_zip_` with UNEQUAL-length lists (n_components[3] vs scale[2]) — truncates to the "
        "shortest -> 2 paired variants x 3 folds. Locks the zip-truncation edge.",
        keywords=("_zip_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_zip_unequal,
        expected_min_predictions=30,
        tags=_GEN,
    )
)


# =============================================================================
# 11) _cartesian_ with pick + COMBINED constraints on one node
# =============================================================================


def _factory_cartesian_pick() -> list[Any]:
    # 2x2 cartesian (4 pipelines), then pick 2 of the complete pipelines -> C(4,2)=6.
    # Each variant is a PAIR of preprocessing pipelines (a multi-branch step).
    return [
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "pick": 2},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_cartesian_pick",
        description="`_cartesian_` 2x2 (4 pipelines) with pick=2 — selects C(4,2)=6 PAIRS of complete "
        "preprocessing pipelines. Locks cartesian-level pick selection. 6 variants x 3 folds.",
        keywords=("_cartesian_", "_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_cartesian_pick,
        expected_min_predictions=90,
        tags=_GEN,
    )
)


def _factory_combined_constraints() -> list[Any]:
    # TWO constraint types on ONE node: mutex [SNV,MSC] AND exclude {Detrend,1stDer}
    # -> C(4,2)=6 minus {SNV,MSC} minus {Detrend,1stDer} = 4 survivors.
    return [
        {
            "_or_": [SNV, MSC, Detrend, FirstDerivative],
            "pick": 2,
            "_mutex_": [[SNV, MSC]],
            "_exclude_": [[Detrend, FirstDerivative]],
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_combined_constraints",
        description="`_or_` pick=2 with `_mutex_` [[SNV,MSC]] AND `_exclude_` [[Detrend,1stDer]] on ONE "
        "node — both constraint types compose: 6 - 1 - 1 = 4 survivors x 3 folds. Survivor set locked in "
        "test_generators_conformance_extra.",
        keywords=("_or_", "_mutex_", "_exclude_", "model"),
        capabilities=(*_CAPS, "filter"),
        dataset_key="regression",
        pipeline_factory=_factory_combined_constraints,
        expected_min_predictions=60,
        tags=_GEN,
    )
)


# =============================================================================
# 12) ANNOTATION keywords (_tags_ / _metadata_) — inert, do not change the set
# =============================================================================
# Unlike `_depends_on_` (which breaks expansion), `_tags_` / `_metadata_` are
# CLEAN no-ops: they annotate a generator node without changing the variant set.
# Locked as parity cases identical to the bare `_or_` baseline.


def _factory_tags_noop() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend], "_tags_": ["baseline"]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_tags_annotation",
        description="`_or_` with a `_tags_` annotation — inert metadata that does NOT change the variant "
        "set (3 variants, same as the bare `_or_`). Both engines at parity. 3 variants x 3 folds.",
        keywords=("_or_", "_tags_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_tags_noop,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_metadata_noop() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend], "_metadata_": {"author": "conformance"}},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_metadata_annotation",
        description="`_or_` with a `_metadata_` annotation — inert metadata that does NOT change the "
        "variant set (3 variants). Both engines at parity. 3 variants x 3 folds.",
        keywords=("_or_", "_metadata_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_metadata_noop,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


# =============================================================================
# 13) count CAP (with _seed_)
# =============================================================================
# `count` caps the variant set to N, sampling with the `_seed_` RNG. `_cartesian_`
# + count + _seed_ is DETERMINISTIC across engines (the cartesian count-subsample
# RNG matches). Section 14 below locks the `_or_` pick+count and `_weights_`+count
# forms as deterministic Python-expand parity cases too.


def _factory_cartesian_count_seed() -> list[Any]:
    # 2x2 cartesian (4 pipelines), capped to 2 with seed 7 -> deterministic subset.
    return [
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "count": 2, "_seed_": 7},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_cartesian_count_seed",
        description="`_cartesian_` 2x2 capped to `count`=2 with `_seed_`=7 — the seeded count-subsample is "
        "DETERMINISTIC across engines (cartesian) -> strict parity. 2 variants x 3 folds.",
        keywords=("_cartesian_", "_or_", "count", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_cartesian_count_seed,
        expected_min_predictions=30,
        tags=_GEN,
    )
)


# =============================================================================
# 14) count / _weights_ subsampling on _or_  (seeded deterministic parity)
# =============================================================================
# `_seed_` stabilizes the `count` / `_weights_` subsample on a `_or_` node. These
# live parity cases exercise the Python expansion path that dag-ml consumes before
# concrete native execution, matching the deterministic `_cartesian_` count path.


def _factory_or_count_seed() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "count": 3, "_seed_": 7},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_count_seed",
        description="`_or_` pick=2 capped to `count`=3 with `_seed_`=7 — exercises the `_or_`+count cap. "
        "The node-local `_seed_` makes the subsample deterministic.",
        keywords=("_or_", "count", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_count_seed,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_weights_count_seed() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend], "_weights_": [0.7, 0.2, 0.1], "count": 2, "_seed_": 7},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_weights_count_seed",
        description="`_or_` with `_weights_` [0.7,0.2,0.1] + `count`=2 + `_seed_`=7 — exercises weighted "
        "random selection with deterministic node-local `_seed_`.",
        keywords=("_or_", "_weights_", "count", "_seed_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_weights_count_seed,
        expected_min_predictions=30,
        tags=_GEN,
    )
)


# =============================================================================
# 15) MULTI-STEP & NESTED operator `_or_` choices (ADR-17 item 5 slice D)
# =============================================================================
# SUB-PART 1 — a bare `_or_` whose choices are MULTI-STEP sub-pipelines
# (`[SNV, Detrend]`) routes NATIVE operator-SELECT: dag-ml already lowers a
# list-valued `_or_` choice to a MULTI-TRANSFORM branch (compat
# `lower_pipeline_fragment` → one transform per element) and
# `expand_or_generator_sequences` carries the whole branch as ONE survivor, so
# each survivor is exactly that one choice's flat operator list — NO pick/arrange
# recombination, NO nested-list survivor. The cross-language `variant_label`
# fingerprints the multi-op sub-sequence, so the content-keyed config map aligns
# and parity is FULL (member-exact survivors + same winner + num_predictions).
#
# SUB-PART 2 — a NESTED operator-generator choice (`_or_: [{_or_: [A,B]}, C]`)
# DEMOTES fail-closed: dag-ml rejects a generator nested inside another
# generator's branch (`find_nested_generator`), and native support is a large
# engine change (the nested generator namespaces in the OUTER's already-namespaced
# context — see generation.rs `collect_operator_generator_steps`). Legacy
# `expand_spec` FLATTENS it into the parent's flat choice space (`A, B, C`), so the
# DEMOTED case runs the proven Python-expand on dag-ml and matches legacy exactly.
# A multi-step choice COMBINED with `pick` (`generator_or_multistep_pick`) likewise
# DEMOTES — `pick` over multi-step choices makes NESTED-list survivors
# (`[[SNV,Detrend], MSC]`) the native single-op-per-branch fingerprint cannot key.


def _factory_or_multistep_choices() -> list[Any]:
    # Each choice is a multi-step sub-pipeline OR a single op; NO selector → each survivor is exactly
    # one choice's flat operator list. 4 survivors x 3 folds; NATIVE operator-SELECT at full parity.
    return [
        {"_or_": [[SNV, FirstDerivative], [MSC, Detrend], SNV, MSC]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_multistep_choices",
        description="bare `_or_` of MULTI-STEP choices [[SNV,1stDer],[MSC,Detrend],SNV,MSC] — ADR-17 item 5 "
        "slice D: routes NATIVE operator-SELECT (each survivor = one choice's flat op list). 4 variants x 3 folds.",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_multistep_choices,
        expected_min_predictions=60,
        tags=_GEN,
    )
)


def _factory_or_multistep_pick() -> list[Any]:
    # MULTI-STEP choices + pick=2 → NESTED-list survivors (`[[SNV,1stDer], MSC]`) the single-op-per-branch
    # native fingerprint cannot key; DEMOTES to the proven Python-expand (still dag-ml engine).
    return [
        {"_or_": [[SNV, FirstDerivative], MSC, Detrend], "pick": 2},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_multistep_pick",
        description="MULTI-STEP `_or_` choices + pick=2 — slice D DEMOTE: pick over multi-step choices makes "
        "nested-list survivors the native fingerprint cannot key, so it runs Python-expand on dag-ml. C(3,2)=3 x 3 folds.",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_multistep_pick,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_or_nested() -> list[Any]:
    # A NESTED operator-generator choice. dag-ml rejects nested generators (large engine change), so this
    # DEMOTES fail-closed; legacy FLATTENS `{_or_: [SNV,MSC]}` into the parent → survivors SNV, MSC, Detrend.
    return [
        {"_or_": [{"_or_": [SNV, MSC]}, Detrend]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_nested",
        description="NESTED operator-generator `_or_: [{_or_: [SNV,MSC]}, Detrend]` — slice D DEMOTE: dag-ml "
        "rejects nested generators (large engine change), so it runs Python-expand on dag-ml; legacy flattens to 3 survivors x 3 folds.",
        keywords=("_or_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_or_nested,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


# =============================================================================
# 16) _zip_ + _chain_ generators (ADR-17 item 5 slice F)
# =============================================================================
# `_zip_` (PAIRED param iteration) and `_chain_` (SEQUENTIAL ordered configs) are
# the LAST item-5 surface. CLASSIFICATION (verified end-to-end):
#
# * param-level `_zip_` ({"_zip_": {p: [...], q: [...]}, "model": M}) and
#   operator-level `_chain_` ({"_chain_": [cfg0, cfg1, ...]}) BOTH run NATIVE on the
#   dag-ml engine at FULL parity — but via the proven host VARIANT-EXPAND path
#   (`_expand_operator_generators` → each variant through dag-ml's single-variant
#   CV+refit), NOT the dedicated in-engine generation paths (`_run_native_generation`
#   for param sweeps / `_run_native_operator_generation` for operator-SELECT). The
#   host bridge DELIBERATELY raises NotImplementedError for both shapes
#   (`dagml_bridge._step_to_dsl`: a `_zip_` model param-generator and a `_chain_`
#   operator generator both fall to the Python-expand path), exactly as slices B/C/D
#   demote richer `_or_`/`_cartesian_` shapes — so they are dag-ml-NATIVE via expand,
#   never bubble to legacy. The base keyword cases (`generator_zip_paired`,
#   `generator_chain_sequential`) + the nested/unequal `_zip_` edges already lock
#   this; these add the with-MODEL and multi-step/multi-model surface.
#
# WHY NOT the dedicated in-engine paths (the LARGE engine change, DEMOTED/DEFERRED):
#   dag-ml DOES carry native `_zip_` (compat_zip_variants → operator-step `variants`)
#   and `_chain_` (lowered as an `Or` branch set) primitives, but routing the HOST to
#   them would (a) need a NEW `variants`-emission form in `_step_to_dsl` (the host only
#   emits the `param_generators`/`generators` form today) + new winner-config keying,
#   (b) dag-ml's `compat_zip_variants` REJECTS unequal-length zip (legacy TRUNCATES) and
#   does NOT expand nested-generator zip values (legacy DOES), and (c) for operator
#   `_chain_` it would FLIP num_predictions to winner-only-refit — a NEW divergence the
#   current expand path does NOT have. Net: a large host integration for ZERO (or
#   negative) parity gain. So the dedicated-path promotion is DEMOTED fail-closed and
#   left for a future slice; the variant-expand route is the parity-faithful one.


def _factory_zip_single_param() -> list[Any]:
    # param-level `_zip_` over ONE model param (n_components paired across 3 values),
    # with-model — the minimal paired-iteration form. 3 variants x 3 folds, FULL parity.
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"_zip_": {"n_components": [3, 7, 11]}, "model": PLSRegression},
    ]


register(
    PipelineCase(
        name="generator_zip_single_param",
        description="param-level `_zip_` over ONE model param (n_components [3,7,11]) with-model — paired "
        "iteration, 3 variants x 3 folds. NATIVE-via-expand at FULL parity (slice F).",
        keywords=("_zip_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_zip_single_param,
        expected_min_predictions=45,
        tags=_GEN,
    )
)


def _factory_chain_multistep_configs() -> list[Any]:
    # operator-level `_chain_` over MULTI-STEP sub-pipelines (the `[config0, config1]`
    # list-of-step-lists form) — sequential ordered, each survivor a 2-op sub-pipeline.
    # 2 variants x 3 folds, FULL parity (single concrete downstream model).
    return [
        {"_chain_": [[SNV(), FirstDerivative()], [MSC(), Detrend()]]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_chain_multistep_configs",
        description="operator-level `_chain_` over MULTI-STEP sub-pipelines [[SNV,1stDer],[MSC,Detrend]] — "
        "sequential ordered, 2 variants x 3 folds. NATIVE-via-expand at FULL parity (slice F).",
        keywords=("_chain_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_chain_multistep_configs,
        expected_min_predictions=30,
        tags=_GEN,
    )
)


def _factory_chain_model_configs() -> list[Any]:
    # operator-level `_chain_` over MODEL-config dicts (the canonical `[{model: A},
    # {model: B}, ...]` with-model form) over DISTINCT model classes (PLS, PLS, Ridge).
    # Sequential ordered, 3 variants x 3 folds. Multi-model selection → the SAME
    # num_predictions divergence as `generator_or_models_pls_ridge`: legacy refits every
    # model (49, loser final rows) while dag-ml operator-SELECT refits the winner only
    # (47) — winner/best_rmse/winner-y_pred all match (config_53e81da4_refit, the PLS
    # winner). An INTENTIONAL native-vs-legacy divergence (ADR-17 1c), a PASSING parity-note.
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_chain_": [
                {"model": PLSRegression(n_components=5)},
                {"model": PLSRegression(n_components=10)},
                {"model": Ridge(alpha=1.0)},
            ]
        },
    ]


register(
    PipelineCase(
        name="generator_chain_model_configs",
        description="operator-level `_chain_` over MODEL configs [{PLSR(5)},{PLSR(10)},{Ridge}] (with-model, "
        "distinct classes) — sequential ordered, 3 variants x 3 folds. Winner/best_rmse/winner-y_pred match, "
        "but num_predictions diverges (legacy refits every model: 49; dag-ml operator-SELECT refits the winner "
        "only: 47) → INTENTIONAL native-vs-legacy divergence (ADR-17 1c), a PASSING parity-note (slice F).",
        keywords=("_chain_", "model"),
        capabilities=_CAPS,
        dataset_key="regression",
        pipeline_factory=_factory_chain_model_configs,
        expected_min_predictions=47,
        tags=_GEN,
    )
)
