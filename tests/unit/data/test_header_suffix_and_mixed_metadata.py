"""Regression tests for two real-world robustness fixes:

1. spectral headers carrying a unit suffix (e.g. ``"852.78_nm"``) must parse to a numeric axis;
2. metadata with a *mixed* object column (strings + floats) must materialize via polars.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.metadata import Metadata
from nirs4all.utils.header_units import get_x_values_and_label, parse_numeric_headers


class TestParseNumericHeaders:
    """The shared unit-suffix-tolerant header parser."""

    def test_plain_numeric(self):
        np.testing.assert_allclose(parse_numeric_headers(["1100", "1102.5", "2498"]), [1100.0, 1102.5, 2498.0])

    def test_nm_suffix_variants(self):
        np.testing.assert_allclose(parse_numeric_headers(["852.78_nm", "853.34 nm", "2502.37nm"]), [852.78, 853.34, 2502.37])

    def test_cm1_suffix_variants(self):
        np.testing.assert_allclose(parse_numeric_headers(["4000_cm-1", "8000cm-1", "12000 cm-1"]), [4000.0, 8000.0, 12000.0])

    def test_rejects_arbitrary_text(self):
        # Anchored: a number buried in text must NOT be mined out.
        assert parse_numeric_headers(["feature_852.78_nm"]) is None
        assert parse_numeric_headers(["850 (raw)"]) is None
        assert parse_numeric_headers(["abc", "def"]) is None

    def test_empty_or_none(self):
        assert parse_numeric_headers(None) is None
        assert parse_numeric_headers([]) is None

    def test_get_x_values_suffixed(self):
        x, label = get_x_values_and_label(["852.78_nm", "853.34_nm"], "nm", 2)
        np.testing.assert_allclose(x, [852.78, 853.34])
        assert label == "Wavelength (nm)"

    def test_get_x_values_text_falls_back_to_indices(self):
        x, label = get_x_values_and_label(["feat_a", "feat_b", "feat_c"], "nm", 3)
        np.testing.assert_array_equal(x, [0, 1, 2])


class TestWavelengthsWithSuffixedHeaders:
    """`wavelengths_nm`/`wavelengths_cm1` tolerate unit-suffixed headers, still raise on real text."""

    def _dataset(self, headers, unit):
        ds = SpectroDataset(name="t")
        ds.add_samples(np.random.rand(4, len(headers)), headers=headers)
        ds._features.sources[0].set_headers(headers, unit=unit)
        return ds

    def test_wavelengths_nm_with_nm_suffix(self):
        ds = self._dataset(["852.78_nm", "853.34_nm", "2502.37_nm"], "nm")
        np.testing.assert_allclose(ds.wavelengths_nm(0), [852.78, 853.34, 2502.37])

    def test_wavelengths_cm1_from_nm_suffix(self):
        ds = self._dataset(["1000_nm", "2000_nm"], "nm")
        np.testing.assert_allclose(ds.wavelengths_cm1(0), [10_000_000.0 / 1000.0, 10_000_000.0 / 2000.0])

    def test_nonnumeric_headers_still_raise(self):
        ds = self._dataset(["alpha", "beta", "gamma"], "nm")
        with pytest.raises(ValueError):
            ds.wavelengths_nm(0)


class TestMixedMetadataColumn:
    """Metadata with a mixed str/float object column must load (no pyarrow ArrowTypeError)."""

    def test_mixed_object_column_loads_and_preserves_nulls(self):
        meta = Metadata()
        meta.add_metadata(pd.DataFrame({"Year": ["2020", 2021.0, None], "site": ["a", "b", "c"]}))
        assert meta.num_rows == 3
        year = meta.df["Year"]
        assert year.dtype == pl.String  # mixed column coerced to string
        assert year.null_count() == 1  # null preserved (not turned into "nan"/"None")

    def test_pure_numeric_column_stays_numeric(self):
        meta = Metadata()
        meta.add_metadata(pd.DataFrame({"temp": [20.0, 21.5, 22.0]}))
        assert meta.df["temp"].dtype in (pl.Float64, pl.Float32)
