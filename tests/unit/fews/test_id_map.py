"""Tests for FEWS variable ID mapper."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig, IDMapEntry
from symfluence.fews.exceptions import IDMappingError
from symfluence.fews.id_map import IDMapper


class TestIDMapperForward:
    def test_explicit_mapping(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        name, factor, offset = mapper.fews_to_symfluence("P.obs")
        assert name == "pptrate"
        assert factor == 1.0
        assert offset == 0.0

    def test_auto_detect(self):
        cfg = FEWSConfig(auto_id_map=True)
        mapper = IDMapper(cfg)
        name, _, _ = mapper.fews_to_symfluence("P.obs")
        assert name == "pptrate"

    def test_pass_through(self):
        cfg = FEWSConfig(auto_id_map=False)
        mapper = IDMapper(cfg)
        name, factor, offset = mapper.fews_to_symfluence("UnknownVar")
        assert name == "UnknownVar"
        assert factor == 1.0

    def test_conversion_factor(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        name, factor, offset = mapper.fews_to_symfluence("P.obs")
        assert name == "pptrate"
        assert pytest.approx(factor) == 1.0 / 3600.0

    def test_conversion_offset(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        name, factor, offset = mapper.fews_to_symfluence("T.obs")
        assert name == "airtemp"
        assert pytest.approx(offset) == 273.15


class TestIDMapperReverse:
    def test_reverse_mapping(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        name, inv_factor, inv_offset = mapper.symfluence_to_fews("pptrate")
        assert name == "P.obs"
        assert inv_factor == 1.0

    def test_reverse_with_factor(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        name, inv_factor, inv_offset = mapper.symfluence_to_fews("pptrate")
        assert name == "P.obs"
        assert pytest.approx(inv_factor) == 3600.0

    def test_export_needs_an_explicit_entry(self):
        """No guessing on export: an unmapped variable is an error, not a
        pass-through under its SYMFLUENCE name."""
        mapper = IDMapper(FEWSConfig(auto_id_map=False))
        with pytest.raises(IDMappingError, match="No explicit ID-map entry"):
            mapper.symfluence_to_fews("custom_var")

    def test_auto_map_is_not_used_for_export(self):
        """The auto-map listed Q.obs before Q.sim, so simulated discharge was
        exported under the *observation* id."""
        mapper = IDMapper(FEWSConfig(auto_id_map=True))
        assert not mapper.has_export_mapping("discharge")
        ds = xr.Dataset({"discharge": ("time", [1.0, 2.0])})
        with pytest.raises(IDMappingError, match="explicit ID-map export entry"):
            mapper.rename_dataset_sym_to_fews(ds)

    def test_explicit_export_entry(self):
        cfg = FEWSConfig(id_map=[
            IDMapEntry(fews_id="Q.obs", symfluence_id="discharge", direction="import"),
            IDMapEntry(fews_id="Q.sim", symfluence_id="discharge", direction="export",
                       fews_unit="m3/s"),
        ])
        mapper = IDMapper(cfg)
        ds = xr.Dataset({
            "discharge": ("time", [1.0, 2.0], {"units": "m3/s"}),
            "unmapped": ("time", [0.0, 0.0]),
        })
        out = mapper.rename_dataset_sym_to_fews(ds)
        assert list(out.data_vars) == ["Q.sim"]  # unmapped variable left out
        assert out["Q.sim"].attrs["units"] == "m3/s"
        # ... and the import direction still reads observations
        assert mapper.fews_to_symfluence("Q.obs")[0] == "discharge"

    def test_ambiguous_export_is_an_error_when_used(self):
        cfg = FEWSConfig(auto_id_map=False, id_map=[
            IDMapEntry(fews_id="P.obs", symfluence_id="pptrate"),
            IDMapEntry(fews_id="P.forecast", symfluence_id="pptrate"),
        ])
        mapper = IDMapper(cfg)  # fine for import
        assert mapper.fews_to_symfluence("P.forecast")[0] == "pptrate"
        with pytest.raises(IDMappingError, match="several export mappings"):
            mapper.symfluence_to_fews("pptrate")


class TestIDMapperYAML:
    def test_load_yaml(self, tmp_path):
        yaml_content = """
- fews_id: WL.obs
  symfluence_id: water_level
  fews_unit: m
  symfluence_unit: m
- fews_id: Q.sim
  symfluence_id: discharge
"""
        yaml_path = tmp_path / "id_map.yaml"
        yaml_path.write_text(yaml_content)

        cfg = FEWSConfig(id_map_file=str(yaml_path), auto_id_map=False)
        mapper = IDMapper(cfg)
        name, _, _ = mapper.fews_to_symfluence("WL.obs")
        assert name == "water_level"

    def test_missing_yaml(self, tmp_path):
        cfg = FEWSConfig(id_map_file=str(tmp_path / "nonexistent.yaml"), auto_id_map=False)
        # Should warn but not raise
        mapper = IDMapper(cfg)
        assert mapper is not None

    def test_invalid_yaml_format(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("key: value")  # Not a list
        cfg = FEWSConfig(id_map_file=str(yaml_path), auto_id_map=False)
        with pytest.raises(IDMappingError, match="must be a list"):
            IDMapper(cfg)


class TestIDMapperDataset:
    def test_rename_fews_to_sym(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        ds = xr.Dataset({
            "P.obs": ("time", [1.0, 2.0]),
            "T.obs": ("time", [280.0, 281.0]),
        })
        result = mapper.rename_dataset_fews_to_sym(ds)
        assert "pptrate" in result.data_vars
        assert "airtemp" in result.data_vars

    def test_rename_sym_to_fews(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        ds = xr.Dataset({
            "pptrate": ("time", [1.0, 2.0]),
            "airtemp": ("time", [280.0, 281.0]),
        })
        result = mapper.rename_dataset_sym_to_fews(ds)
        assert "P.obs" in result.data_vars
        assert "T.obs" in result.data_vars

    def test_conversion_applied(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        ds = xr.Dataset({
            "T.obs": ("time", [0.0, 10.0]),  # degC
        })
        result = mapper.rename_dataset_fews_to_sym(ds)
        # Should be K after conversion
        np.testing.assert_allclose(result["airtemp"].values, [273.15, 283.15], rtol=1e-5)

    def test_units_follow_the_conversion(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        ds = xr.Dataset({"T.obs": ("time", [0.0, 10.0], {"units": "degC"})})
        result = mapper.rename_dataset_fews_to_sym(ds)
        assert result["airtemp"].attrs["units"] == "K"


class TestUnitSafety:
    """The auto-map renames without converting, so it must not accept data in
    other units (mm/h precipitation as mm/s is a 3600-fold error)."""

    @pytest.mark.parametrize("fews_id,units", [
        ("P.obs", "mm/h"), ("P.obs", "mm"), ("T.obs", "degC"), ("SP", "hPa"),
    ])
    def test_auto_map_refuses_other_units(self, fews_id, units):
        mapper = IDMapper(FEWSConfig(auto_id_map=True))
        ds = xr.Dataset({fews_id: ("time", [1.0, 2.0], {"units": units})})
        with pytest.raises(IDMappingError, match="applies no conversion"):
            mapper.rename_dataset_fews_to_sym(ds)

    @pytest.mark.parametrize("fews_id,units,sym", [
        ("P.obs", "kg m-2 s-1", "pptrate"), ("P.obs", "mm/s", "pptrate"),
        ("T.obs", "K", "airtemp"), ("SW", "W m-2", "SWRadAtm"),
    ])
    def test_auto_map_accepts_matching_units(self, fews_id, units, sym):
        mapper = IDMapper(FEWSConfig(auto_id_map=True))
        ds = xr.Dataset({fews_id: ("time", [1.0, 2.0], {"units": units})})
        result = mapper.rename_dataset_fews_to_sym(ds)
        np.testing.assert_allclose(result[sym].values, [1.0, 2.0])

    def test_explicit_factor_overrides_auto_map(self):
        cfg = FEWSConfig(id_map=[IDMapEntry(
            fews_id="P.obs", symfluence_id="pptrate", fews_unit="mm/h",
            symfluence_unit="mm/s", conversion_factor=1.0 / 3600.0)])
        ds = xr.Dataset({"P.obs": ("time", [3600.0], {"units": "mm/h"})})
        result = IDMapper(cfg).rename_dataset_fews_to_sym(ds)
        np.testing.assert_allclose(result["pptrate"].values, [1.0])

    def test_explicit_entry_with_unit_change_needs_a_conversion(self):
        cfg = FEWSConfig(id_map=[IDMapEntry(
            fews_id="T.obs", symfluence_id="airtemp", fews_unit="degC", symfluence_unit="K")])
        with pytest.raises(IDMappingError, match="no conversion_factor"):
            IDMapper(cfg)

    def test_data_not_in_declared_unit_is_refused(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        ds = xr.Dataset({"T.obs": ("time", [280.0], {"units": "K"})})
        with pytest.raises(IDMappingError, match="declares fews_unit"):
            mapper.rename_dataset_fews_to_sym(ds)

    def test_two_inputs_for_one_variable_is_a_clear_error(self):
        """P.obs + P.forecast used to crash inside xarray with a rename conflict."""
        mapper = IDMapper(FEWSConfig(auto_id_map=True))
        ds = xr.Dataset({
            "P.obs": ("time", [1.0, 2.0], {"units": "mm/s"}),
            "P.forecast": ("time", [1.0, 2.0], {"units": "mm/s"}),
        })
        with pytest.raises(IDMappingError, match="map to the same SYMFLUENCE variable"):
            mapper.rename_dataset_fews_to_sym(ds)
