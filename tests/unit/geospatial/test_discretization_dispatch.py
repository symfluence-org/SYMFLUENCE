import logging
from pathlib import Path

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.geospatial.discretization import DomainDiscretizer
from symfluence.geospatial.discretization.attributes import combined, elevation


def _base_config(tmp_path, discretization):
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "DEM_NAME": "default",
        "DEM_PATH": "default",
        "DOMAIN_DEFINITION_METHOD": "delineate",
        "CATCHMENT_PATH": "default",
        "CATCHMENT_SHP_NAME": "default",
        "CATCHMENT_SHP_GRUID": "GRU_ID",
        "CATCHMENT_SHP_HRUID": "HRU_ID",
        "SUB_GRID_DISCRETIZATION": discretization,
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
    }
    return SymfluenceConfig(**config_dict)


def test_discretize_domain_dispatches_single_attribute(tmp_path, monkeypatch):
    config = _base_config(tmp_path, "elevation")
    logger = logging.getLogger("test_discretize_domain_dispatches_single_attribute")

    called = {}

    def fake_discretize(self):
        called["method"] = "elevation"

    monkeypatch.setattr(elevation, "discretize", fake_discretize)

    expected = Path(tmp_path / "sorted.shp")
    monkeypatch.setattr(
        DomainDiscretizer, "sort_catchment_shape", lambda self: expected
    )

    discretizer = DomainDiscretizer(config, logger)
    result = discretizer.discretize_domain()

    assert called["method"] == "elevation"
    assert result == expected


def test_discretize_domain_dispatches_combined_attributes(tmp_path, monkeypatch):
    config = _base_config(tmp_path, "elevation, landclass")
    logger = logging.getLogger("test_discretize_domain_dispatches_combined_attributes")

    captured = {}

    def fake_combined(self, attrs):
        captured["attrs"] = attrs

    monkeypatch.setattr(combined, "discretize", fake_combined)

    expected = Path(tmp_path / "sorted_combined.shp")
    monkeypatch.setattr(
        DomainDiscretizer, "sort_catchment_shape", lambda self: expected
    )

    discretizer = DomainDiscretizer(config, logger)
    result = discretizer.discretize_domain()

    assert captured["attrs"] == ["elevation", "landclass"]
    assert result == expected


def test_discretize_domain_rejects_unknown_method(tmp_path):
    config = _base_config(tmp_path, "not_a_method")
    logger = logging.getLogger("test_discretize_domain_rejects_unknown_method")

    discretizer = DomainDiscretizer(config, logger)

    with pytest.raises(ValueError):
        discretizer.discretize_domain()


def _base_config_with_shp(tmp_path, discretization, shp_name):
    """Variant of _base_config that sets a custom CATCHMENT_SHP_NAME at construction.

    `SymfluenceConfig.paths` is a frozen pydantic model, so we must pass the
    non-default value through the constructor rather than mutate post-hoc.
    """
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "DEM_NAME": "default",
        "DEM_PATH": "default",
        "DOMAIN_DEFINITION_METHOD": "delineate",
        "CATCHMENT_PATH": "default",
        "CATCHMENT_SHP_NAME": shp_name,
        "CATCHMENT_SHP_GRUID": "GRU_ID",
        "CATCHMENT_SHP_HRUID": "HRU_ID",
        "SUB_GRID_DISCRETIZATION": discretization,
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
    }
    return SymfluenceConfig(**config_dict)


def test_discretize_domain_errors_when_byo_shapefile_missing(tmp_path):
    """Custom CATCHMENT_SHP_NAME pointing at a non-existent file must fail
    with an actionable message, not a deep FileNotFoundError from geopandas.

    Reported by NB: setting CATCHMENT_SHP_NAME to the file that the
    discretization step was supposed to produce caused the code to skip
    generation and then crash inside sort_catchment_shape() reading the
    missing file. This test pins the improved diagnostic at discretize_domain
    boundary so users see the cause and the two ways to fix it.
    """
    config = _base_config_with_shp(
        tmp_path, "elevation", "test_domain_HRUs_elevation.shp"
    )
    logger = logging.getLogger("test_discretize_domain_errors_when_byo_shapefile_missing")

    discretizer = DomainDiscretizer(config, logger)

    with pytest.raises(FileNotFoundError) as exc:
        discretizer.discretize_domain()

    msg = str(exc.value)
    assert "test_domain_HRUs_elevation.shp" in msg
    assert "CATCHMENT_SHP_NAME" in msg
    assert "default" in msg


def test_discretize_domain_uses_byo_shapefile_when_it_exists(tmp_path, monkeypatch):
    """When CATCHMENT_SHP_NAME points at an existing file, skip discretization
    and just call sort_catchment_shape — that is the legitimate BYO path."""
    byo_name = "my_custom_hrus.shp"
    config = _base_config_with_shp(tmp_path, "elevation", byo_name)

    discretizer = DomainDiscretizer(
        config, logging.getLogger("test_discretize_domain_uses_byo_shapefile_when_it_exists")
    )
    byo_subpath = discretizer._get_catchment_subpath(byo_name)
    byo_path = discretizer.project_dir / byo_subpath / byo_name
    byo_path.parent.mkdir(parents=True, exist_ok=True)
    byo_path.write_text("dummy")

    sentinel = Path(tmp_path / "sorted_byo.shp")
    monkeypatch.setattr(
        DomainDiscretizer, "sort_catchment_shape", lambda self: sentinel
    )

    result = discretizer.discretize_domain()
    assert result == sentinel
