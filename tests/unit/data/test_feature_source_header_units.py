"""
Test header unit metadata in FeatureSource - Step 1
"""

import pytest
import numpy as np
from nirs4all.data.feature_source import FeatureSource


class TestFeatureSourceHeaderUnits:
    """Test header unit metadata storage in FeatureSource"""

    def test_default_header_unit(self):
        """Test that default header unit is cm-1"""
        source = FeatureSource()
        assert source.header_unit == "cm-1"

    def test_set_headers_with_default_unit(self):
        """Test setting headers without specifying unit (defaults to cm-1)"""
        source = FeatureSource()
        headers = ["1000.0", "1001.0", "1002.0"]

        source.set_headers(headers)

        assert source.headers == headers
        assert source.header_unit == "cm-1"

    def test_set_headers_with_cm1_unit(self):
        """Test setting headers with cm-1 unit explicitly"""
        source = FeatureSource()
        headers = ["4000.0", "4500.0", "5000.0"]

        source.set_headers(headers, unit="cm-1")

        assert source.headers == headers
        assert source.header_unit == "cm-1"

    def test_set_headers_with_nm_unit(self):
        """Test setting headers with nm unit"""
        source = FeatureSource()
        headers = ["780", "800", "850"]

        source.set_headers(headers, unit="nm")

        assert source.headers == headers
        assert source.header_unit == "nm"

    def test_set_headers_with_none_unit(self):
        """Test setting headers with none unit (no headers in CSV)"""
        source = FeatureSource()
        headers = ["0", "1", "2"]

        source.set_headers(headers, unit="none")

        assert source.headers == headers
        assert source.header_unit == "none"

    def test_set_headers_with_text_unit(self):
        """Test setting headers with text unit (non-numeric headers)"""
        source = FeatureSource()
        headers = ["feature_1", "feature_2", "feature_3"]

        source.set_headers(headers, unit="text")

        assert source.headers == headers
        assert source.header_unit == "text"

    def test_set_headers_with_index_unit(self):
        """Test setting headers with index unit"""
        source = FeatureSource()
        headers = ["0", "1", "2"]

        source.set_headers(headers, unit="index")

        assert source.headers == headers
        assert source.header_unit == "index"

    def test_header_unit_persists_after_add_samples(self):
        """Test that header unit persists after adding samples"""
        source = FeatureSource()
        headers = ["780", "800", "850"]

        # Set headers with nm unit
        source.set_headers(headers, unit="nm")

        # Add some samples
        samples = np.random.rand(10, 3)
        source.add_samples(samples, headers=headers)

        # Unit should still be nm
        assert source.header_unit == "nm"
        assert source.headers == headers

    def test_update_headers_changes_unit(self):
        """Test that calling set_headers again updates the unit"""
        source = FeatureSource()

        # First set with cm-1
        source.set_headers(["4000", "5000"], unit="cm-1")
        assert source.header_unit == "cm-1"

        # Update to nm
        source.set_headers(["780", "800"], unit="nm")
        assert source.header_unit == "nm"

    def test_headers_none_preserves_unit(self):
        """Test setting headers to None preserves the unit"""
        source = FeatureSource()
        source.set_headers(["1000", "2000"], unit="nm")

        # Set headers to None
        source.set_headers(None, unit="nm")

        assert source.headers is None
        assert source.header_unit == "nm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
