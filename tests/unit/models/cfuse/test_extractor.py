"""Tests for cFUSE result extractor."""

import pytest


class TestCFUSEResultExtractorImport:
    """Tests for cFUSE result extractor import and registration."""

    def test_extractor_can_be_imported(self):
        from cfuse.extractor import CFUSEResultExtractor
        assert CFUSEResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        import cfuse  # noqa: F401 — trigger registration

        from symfluence.core.registries import R
        assert 'CFUSE' in R.result_extractors


class TestCFUSEOutputPatterns:
    """Tests for cFUSE output file patterns."""

    def test_output_file_patterns(self):
        from cfuse.extractor import CFUSEResultExtractor
        extractor = CFUSEResultExtractor()
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns

    def test_streamflow_patterns_include_nc(self):
        from cfuse.extractor import CFUSEResultExtractor
        extractor = CFUSEResultExtractor()
        patterns = extractor.get_output_file_patterns()['streamflow']
        assert any('.nc' in p for p in patterns)


class TestCFUSEVariableNames:
    """Tests for cFUSE variable name mappings."""

    def test_streamflow_variable_names(self):
        from cfuse.extractor import CFUSEResultExtractor
        extractor = CFUSEResultExtractor()
        names = extractor.get_variable_names('streamflow')
        assert len(names) > 0
        assert any('streamflow' in n.lower() or 'q' in n.lower() or 'discharge' in n.lower()
                    for n in names)

    def test_runoff_variable_names(self):
        from cfuse.extractor import CFUSEResultExtractor
        extractor = CFUSEResultExtractor()
        names = extractor.get_variable_names('runoff')
        assert len(names) > 0
