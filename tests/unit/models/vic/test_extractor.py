"""Tests for VIC result extractor."""
from __future__ import annotations

import pytest


class TestVICResultExtractor:
    """Tests for VIC result extractor."""

    def test_extractor_can_be_imported(self):
        from symfluence.models.vic.extractor import VICResultExtractor
        assert VICResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        import symfluence.models.vic  # noqa: F401 — trigger registration
        from symfluence.core.registries import R
        assert 'VIC' in R.result_extractors

    def test_output_file_patterns(self):
        from symfluence.models.vic.extractor import VICResultExtractor
        extractor = VICResultExtractor('VIC')
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns
        assert 'et' in patterns
        assert 'snow' in patterns

    def test_variable_names_streamflow(self):
        from symfluence.models.vic.extractor import VICResultExtractor
        extractor = VICResultExtractor('VIC')
        names = extractor.get_variable_names('streamflow')
        assert 'OUT_RUNOFF' in names
        assert 'OUT_BASEFLOW' in names

    def test_variable_names_et(self):
        from symfluence.models.vic.extractor import VICResultExtractor
        extractor = VICResultExtractor('VIC')
        names = extractor.get_variable_names('et')
        assert 'OUT_EVAP' in names

    def test_requires_unit_conversion(self):
        from symfluence.models.vic.extractor import VICResultExtractor
        extractor = VICResultExtractor('VIC')
        assert extractor.requires_unit_conversion('streamflow') is True
        assert extractor.requires_unit_conversion('snow') is False

    def test_spatial_aggregation_method(self):
        from symfluence.models.vic.extractor import VICResultExtractor
        extractor = VICResultExtractor('VIC')
        assert extractor.get_spatial_aggregation_method('streamflow') == 'mean'
        assert extractor.get_spatial_aggregation_method('soil_moisture') == 'mean'
