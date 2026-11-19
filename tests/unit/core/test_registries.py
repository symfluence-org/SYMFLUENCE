"""Tests for the Registries facade."""

from __future__ import annotations

import pytest

from symfluence.core.registries import R, Registries
from symfluence.core.registry import Registry
from symfluence.testing import registry_snapshot

# ======================================================================
# Fixtures — isolated registries (clear before and after each test)
# ======================================================================


@pytest.fixture(autouse=True)
def _clean_registries():
    """Save and restore all registries around each test.

    The seed-then-snapshot ordering this depends on is subtle enough that it now
    lives in ``symfluence.testing.registry_snapshot`` — a supported surface, so
    the extracted models repo does not have to reimplement it against
    ``Registry``'s private attributes. See that function for why spending the
    lazy population first is load-bearing: it is what stopped
    ``test_lowercase_delineation`` and ``test_lowercase_data_registries`` from
    having the real ``LumpedWatershedDelineator`` replace their fake between the
    ``add`` and the ``get``.
    """
    with registry_snapshot(clear=True):
        yield


class _FakePreprocessor:
    pass


class _FakeRunner:
    pass


class _FakePostProcessor:
    pass


class _FakePlotter:
    pass


class _FakeConfigAdapter:
    pass


class _FakeExtractor:
    pass


# ======================================================================
# Basic registry existence
# ======================================================================


class TestRegistryInstances:
    def test_all_registries_returns_dict(self):
        regs = Registries.all_registries()
        assert isinstance(regs, dict)
        assert len(regs) > 20  # we have ~30 registries

    def test_all_registries_values_are_registry(self):
        for name, reg in Registries.all_registries().items():
            assert isinstance(reg, Registry), f"{name} is not a Registry"

    def test_convenience_alias(self):
        assert R is Registries

    def test_key_registries_exist(self):
        expected = [
            "preprocessors", "runners", "postprocessors", "visualizers",
            "config_adapters", "config_schemas", "config_defaults",
            "config_transformers", "config_validators",
            "result_extractors", "output_adapters",
            "optimizers", "workers", "parameter_managers",
            "calibration_targets", "objectives",
            "acquisition_handlers", "dataset_handlers", "observation_handlers",
            "evaluators", "sensitivity_analyzers", "decision_analyzers",
            "koopman_analyzers", "metrics",
            "plotters", "visualization_funcs",
            "delineation_strategies",
            "bmi_adapters",
            "forcing_adapters", "build_instructions", "presets",
        ]
        regs = Registries.all_registries()
        for name in expected:
            assert name in regs, f"Missing registry: {name}"


# ======================================================================
# Normalization conventions
# ======================================================================


class TestNormalization:
    def test_uppercase_default(self):
        R.runners.add("summa", _FakeRunner)
        assert R.runners.get("SUMMA") is _FakeRunner

    def test_lowercase_data_registries(self):
        R.acquisition_handlers.add("ERA5", _FakePreprocessor)
        assert R.acquisition_handlers.get("era5") is _FakePreprocessor

    def test_lowercase_delineation(self):
        R.delineation_strategies.add("Lumped", _FakePreprocessor)
        assert R.delineation_strategies.get("lumped") is _FakePreprocessor


# ======================================================================
# Cross-cutting methods
# ======================================================================


class TestCrossCutting:
    def test_registered_models_empty(self):
        assert Registries.registered_models() == []

    def test_registered_models_union(self):
        R.preprocessors.add("SUMMA", _FakePreprocessor)
        R.runners.add("FUSE", _FakeRunner)
        models = Registries.registered_models()
        assert "SUMMA" in models
        assert "FUSE" in models

    def test_for_model(self):
        R.preprocessors.add("SUMMA", _FakePreprocessor)
        R.runners.add("SUMMA", _FakeRunner)
        R.plotters.add("SUMMA", _FakePlotter)

        result = Registries.for_model("SUMMA")
        assert result["preprocessors"] is _FakePreprocessor
        assert result["runners"] is _FakeRunner
        assert result["plotters"] is _FakePlotter
        assert "postprocessors" not in result  # not registered

    def test_for_model_empty(self):
        assert Registries.for_model("NONEXISTENT") == {}

    def test_summary(self):
        R.runners.add("SUMMA", _FakeRunner)
        s = Registries.summary()
        assert "runners" in s
        assert s["runners"]["entries"] == 1

    def test_validate_model_valid(self):
        R.preprocessors.add("TEST", _FakePreprocessor)
        R.runners.add("TEST", _FakeRunner)
        R.postprocessors.add("TEST", _FakePostProcessor)

        v = Registries.validate_model("TEST")
        assert v["valid"] is True
        assert v["missing"] == []

    def test_validate_model_missing_required(self):
        R.runners.add("TEST", _FakeRunner)

        v = Registries.validate_model("TEST")
        assert v["valid"] is False
        assert "preprocessors" in v["missing"]
        assert "postprocessors" in v["missing"]

    def test_validate_model_optional(self):
        R.preprocessors.add("TEST", _FakePreprocessor)
        R.runners.add("TEST", _FakeRunner)
        R.postprocessors.add("TEST", _FakePostProcessor)

        v = Registries.validate_model("TEST")
        assert "visualizers" in v["optional_missing"]


# ======================================================================
# Calibration target convenience
# ======================================================================


class TestCalibrationTargets:
    def test_get_calibration_target(self):
        class _Target:
            pass

        R.calibration_targets.add("SUMMA_STREAMFLOW", _Target)
        assert Registries.get_calibration_target("SUMMA", "streamflow") is _Target

    def test_get_calibration_target_miss(self):
        assert Registries.get_calibration_target("SUMMA", "snow") is None


# ======================================================================
# Model-name normalization (separators in config spellings)
# ======================================================================


class TestModelNameNormalization:
    """Model-keyed registries strip word separators so config spellings like
    'WRF-Hydro' resolve to the registered 'WRFHYDRO' entry, while underscores in
    intentional compound keys (e.g. 'CFUSE_ROUTED') are preserved.
    """

    def test_hyphenated_spelling_resolves(self):
        class _Runner:
            pass

        R.runners.add("WRFHYDRO", _Runner)
        assert R.runners.get("WRF-Hydro") is _Runner
        assert R.runners.get("wrf hydro") is _Runner
        assert R.runners.get("WRFHYDRO") is _Runner

    def test_underscore_compound_key_is_preserved(self):
        class _PostA:
            pass

        class _PostB:
            pass

        # Two distinct underscore-compound models must not collapse together.
        R.postprocessors.add("CFUSE_ROUTED", _PostA)
        R.postprocessors.add("HBV_ROUTED", _PostB)
        assert R.postprocessors.get("CFUSE_ROUTED") is _PostA
        assert R.postprocessors.get("HBV_ROUTED") is _PostB
