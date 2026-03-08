"""Tests for dRoute network adapter."""

import pytest


class TestDRouteNetworkAdapter:
    """Tests for dRoute network adapter."""

    def test_adapter_can_be_imported(self):
        from droute.network_adapter import DRouteNetworkAdapter
        assert DRouteNetworkAdapter is not None

    def test_adapter_instantiation(self):
        import logging

        from droute.network_adapter import DRouteNetworkAdapter
        logger = logging.getLogger('test')
        adapter = DRouteNetworkAdapter(logger)
        assert adapter is not None


class TestDRouteResultExtractor:
    """Tests for dRoute result extractor."""

    def test_extractor_can_be_imported(self):
        from droute.extractor import DRouteResultExtractor
        assert DRouteResultExtractor is not None

    def test_extractor_registered(self):
        import droute  # noqa: F401 — trigger registration

        from symfluence.core.registries import R
        assert 'DROUTE' in R.result_extractors

    def test_output_file_patterns(self):
        from droute.extractor import DRouteResultExtractor
        extractor = DRouteResultExtractor()
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns
