"""The dual-engine winner lock tolerates only a split between natively co-best variants."""

from types import SimpleNamespace

import pytest

from . import _conformance_helpers as H

_CASE = SimpleNamespace(name="tie_case")


def _result(winner: str, scores: dict[str, float] | None = None) -> SimpleNamespace:
    rows = [{"config_name": name, "val_score": score, "partition": "val", "fold_id": "avg"} for name, score in (scores or {}).items()]

    def filter_predictions(**filters):
        assert filters == {"partition": "val", "fold_id": "avg", "load_arrays": False}
        return rows

    return SimpleNamespace(best={"config_name": winner}, predictions=SimpleNamespace(filter_predictions=filter_predictions))


# The native selection scores measured for generator_cartesian_pick on the reference machine.
_NATIVE = {"config_8ee9444f": 14.449551413807683, "config_ad018d2d": 14.449556790020203, "config_bffe6a98": 14.507468503562434}


def test_identical_winners_are_not_a_split():
    assert H.native_winner_split_is_a_tie(_result("config_8ee9444f_refit"), _result("config_8ee9444f_refit", _NATIVE), _CASE) is False


def test_co_best_native_variants_are_a_tie_whichever_the_native_engine_selects():
    hosted = {**_NATIVE, "config_ad018d2d": 14.449551413807683 - 4e-6}
    assert H.native_winner_split_is_a_tie(_result("config_8ee9444f_refit"), _result("config_ad018d2d_refit", hosted), _CASE) is True


def test_a_separated_native_winner_is_a_real_selection_split():
    legacy, dagml = _result("config_bffe6a98_refit"), _result("config_8ee9444f_refit", _NATIVE)
    assert H.native_winner_split_is_a_tie(legacy, dagml, _CASE) is False
    with pytest.raises(AssertionError, match="DIFFERENT winners"):
        H.assert_same_winner(legacy, dagml, _CASE)


def test_a_split_without_native_scores_is_never_excused():
    assert H.native_winner_split_is_a_tie(_result("config_8ee9444f_refit"), _result("config_ad018d2d_refit"), _CASE) is False


def test_the_native_winner_must_hold_the_best_native_score():
    with pytest.raises(AssertionError, match="not the best native selection score"):
        H.native_winner_split_is_a_tie(_result("config_8ee9444f_refit"), _result("config_bffe6a98_refit", _NATIVE), _CASE)
