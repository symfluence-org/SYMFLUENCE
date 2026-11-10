from __future__ import annotations

from symfluence.core.registries import R

"""Tests for Noah-MP postprocessor."""


class TestNoahMPPostProcessor:
    """Tests for Noah-MP postprocessor."""

    def test_postprocessor_can_be_imported(self):
        from symfluence.models.noahmp.postprocessor import NoahMPPostProcessor
        assert NoahMPPostProcessor is not None

    def test_postprocessor_registered_with_registry(self):
        import symfluence.models.noahmp  # noqa: F401
        assert 'NOAHMP' in R.postprocessors

    def test_model_name(self):
        from symfluence.models.noahmp.postprocessor import NoahMPPostProcessor
        assert NoahMPPostProcessor.model_name == "NOAHMP"

    def test_streamflow_unit(self):
        from symfluence.models.noahmp.postprocessor import NoahMPPostProcessor
        assert NoahMPPostProcessor.streamflow_unit == "cms"

    def test_streamflow_variable(self):
        from symfluence.models.noahmp.postprocessor import NoahMPPostProcessor
        assert NoahMPPostProcessor.streamflow_variable == "total_runoff"
