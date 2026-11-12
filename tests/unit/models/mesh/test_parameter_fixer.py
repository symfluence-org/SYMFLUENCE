"""
Tests for MESH parameter fixer.

Tests cover run options fixes, GRU count mismatch handling,
DDB operations, CLASS file operations, and safe forcing creation.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forcing_dir(tmp_path):
    """Create a temporary forcing directory."""
    d = tmp_path / "forcing"
    d.mkdir()
    return d


@pytest.fixture
def setup_dir(tmp_path):
    """Create a temporary settings directory."""
    d = tmp_path / "settings"
    d.mkdir()
    return d


@pytest.fixture
def fixer(forcing_dir, setup_dir):
    """Create a MESHParameterFixer with default config."""
    from symfluence.models.mesh.preprocessing.parameter_fixer import MESHParameterFixer

    return MESHParameterFixer(
        forcing_dir=forcing_dir,
        setup_dir=setup_dir,
        config={
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_SPATIAL_MODE": "distributed",
            "MESH_SPINUP_DAYS": 365,
        },
    )


@pytest.fixture
def run_options_content():
    """Sample MESH_input_run_options.ini content."""
    return """\
##### Global settings #####
 15 # Number of control flags
----# Control flags
BASINFORCINGFLAG      nc_subbasin
SHDFILEFLAG           nc_subbasin pad_outlets
OUTFILESFLAG         daily
OUTFIELDSFLAG        none
STREAMFLOWOUTFLAG     csv
BASINAVGWBFILEFLAG    daily
RUNMODE               runrte
FROZENSOILINFILFLAG   0
FREZTH                0.0
SWELIM                800.0
SNDENLIM              600.0
PBSMFLAG              off
METRICSSPINUP         730
PRINTSIMSTATUS        date_monthly
DIAGNOSEMODE          off
#####
name_var=SWRadAtm
name_var=spechum
name_var=airtemp
name_var=windspd
name_var=pptrate
name_var=airpres
name_var=LWRadAtm
"""


@pytest.fixture
def class_ini_content():
    """Sample MESH_parameters_CLASS.ini content with 2 GRU blocks."""
    return """\
  51.0  -116.0  1.0  1.0  0.1  1.0  0.0  1  2  04 DEGLAT/DEGLON/ZBLDGRD/ZRFHGRD/ZRFMGRD/GCGRD/FAREROT/NL/NM
 0.0  0.0  1.0  0.0  0.0  3.5  1.0  0.0  0.0  1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0  05 5xFCAN/4xLAMX/3xLNZ0/SDEP/XSLP/XDRAINH/MANN/KSAT/MID
 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  06 SANDG/CLAYG/ORGM/...
 0.1  0.2  0.3  0.4  0.5  07 CMIDROT/...
 0.1  0.2  0.3  0.4  08 ROOT/...
 0.0  0.0  1.0  0.0  0.0  2.0  1.0  0.0  0.0  1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0  05 5xFCAN/4xLAMX/3xLNZ0/SDEP/XSLP/XDRAINH/MANN/KSAT/MID
 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  06 SANDG/CLAYG/ORGM/...
 0.1  0.2  0.3  0.4  0.5  07 CMIDROT/...
 0.1  0.2  0.3  0.4  08 ROOT/...
 0  0  0  0  20 IORGC
 0.0 0.0  21 RSMLNa/RSMLNb
 0.0 0.0  22 INITIALS
"""


@pytest.fixture
def ddb_dataset():
    """Create a minimal DDB xarray Dataset for testing."""
    n_sub = 1
    n_gru = 3
    gru_data = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)

    return xr.Dataset({
        "GRU": (["subbasin", "NGRU"], gru_data),
        "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        "GridArea": (["subbasin"], np.array([1e8])),
        "lat": (["subbasin"], np.array([51.0])),
        "lon": (["subbasin"], np.array([-116.0])),
    })


# ---------------------------------------------------------------------------
# TestRunOptionsVarNames
# ---------------------------------------------------------------------------

class TestRunOptionsVarNames:
    """Test fix_run_options_var_names."""

    def test_replaces_variable_names(self, fixer, run_options_content):
        """Old-style variable names should be replaced with MESH names."""
        fixer.run_options_path.write_text(run_options_content)
        fixer.fix_run_options_var_names()

        content = fixer.run_options_path.read_text()
        assert "name_var=FSIN" in content
        assert "name_var=QA" in content
        assert "name_var=TA" in content
        assert "name_var=UV" in content
        assert "name_var=PRE" in content
        assert "name_var=PRES" in content
        assert "name_var=FLIN" in content
        # Old names should be gone
        assert "name_var=SWRadAtm" not in content
        assert "name_var=spechum" not in content

    def test_idempotent(self, fixer, run_options_content):
        """Running twice should produce the same output."""
        fixer.run_options_path.write_text(run_options_content)
        fixer.fix_run_options_var_names()
        content1 = fixer.run_options_path.read_text()
        fixer.fix_run_options_var_names()
        content2 = fixer.run_options_path.read_text()
        assert content1 == content2

    def test_missing_file_no_error(self, fixer):
        """Should silently return if run_options file doesn't exist."""
        fixer.fix_run_options_var_names()  # No exception


class TestRunOptionsSnowParams:
    """Test fix_run_options_snow_params."""

    def test_single_cell_forces_noroute(self, fixer, run_options_content, ddb_dataset):
        """Single-cell domain should force RUNMODE=noroute."""
        # Create single-cell DDB
        ddb_dataset.to_netcdf(fixer.ddb_path)
        fixer.run_options_path.write_text(run_options_content)

        fixer.fix_run_options_snow_params()

        content = fixer.run_options_path.read_text()
        assert "noroute" in content

    def test_frozen_soil_flag(self, forcing_dir, setup_dir, run_options_content):
        """MESH_ENABLE_FROZEN_SOIL=True should set FROZENSOILINFILFLAG=1."""
        from symfluence.models.mesh.preprocessing.parameter_fixer import MESHParameterFixer

        fixer = MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_ENABLE_FROZEN_SOIL": True,
                "MESH_SPINUP_DAYS": 365,
            },
        )
        fixer.run_options_path.write_text(run_options_content)

        fixer.fix_run_options_snow_params()

        content = fixer.run_options_path.read_text()
        assert "FROZENSOILINFILFLAG   1" in content

    def test_missing_file_no_error(self, fixer):
        """Should silently return if run_options file doesn't exist."""
        fixer.fix_run_options_snow_params()


class TestUpdateControlFlagCount:
    """Test _update_control_flag_count."""

    def test_counts_flags_correctly(self, fixer, run_options_content):
        """Should count non-comment, non-empty lines in the flags section."""
        fixer.run_options_path.write_text(run_options_content)
        fixer._update_control_flag_count()
        content = fixer.run_options_path.read_text()
        # Should have the correct count of flags
        lines = content.split("\n")
        for line in lines:
            if "Number of control flags" in line:
                # Extract the number
                import re
                match = re.search(r"(\d+)", line)
                if match:
                    count = int(match.group(1))
                    assert count > 0
                break


# ---------------------------------------------------------------------------
# TestGRUCountMismatch
# ---------------------------------------------------------------------------

class TestGRUCountMismatch:
    """Test fix_gru_count_mismatch orchestration."""

    def test_collapse_to_single_gru(self, forcing_dir, setup_dir, ddb_dataset, class_ini_content):
        """MESH_FORCE_SINGLE_GRU should collapse to 1 active GRU."""
        from symfluence.models.mesh.preprocessing.parameter_fixer import MESHParameterFixer

        fixer = MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_FORCE_SINGLE_GRU": True,
            },
        )
        ddb_dataset.to_netcdf(fixer.ddb_path)
        fixer.class_file_path.write_text(class_ini_content)

        fixer.fix_gru_count_mismatch()

        with xr.open_dataset(fixer.ddb_path) as ds:
            assert ds.sizes["NGRU"] == 2  # MESH off-by-one: 2 cols → reads 1

    def test_idempotent_when_aligned(self, fixer, class_ini_content):
        """When already aligned, no changes should be made."""
        # Create DDB with 3 GRU cols → MESH reads 2 → needs 2 CLASS blocks
        gru_data = np.array([[0.7, 0.3, 0.0]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.array([1e8])),
            "lat": (["subbasin"], np.array([51.0])),
            "lon": (["subbasin"], np.array([-116.0])),
        })
        ds.to_netcdf(fixer.ddb_path)
        fixer.class_file_path.write_text(class_ini_content)

        fixer.fix_gru_count_mismatch()

        # Should not crash, file should still exist
        assert fixer.ddb_path.exists()
        assert fixer.class_file_path.exists()


# ---------------------------------------------------------------------------
# TestDDBOperations
# ---------------------------------------------------------------------------

class TestDDBOperations:
    """Test DDB-related methods on MESHParameterFixer."""

    def test_get_ddb_gru_count(self, fixer, ddb_dataset):
        """Should return NGRU dimension size."""
        ddb_dataset.to_netcdf(fixer.ddb_path)
        assert fixer._get_ddb_gru_count() == 3

    def test_get_ddb_gru_count_missing_file(self, fixer):
        """Should return None when DDB doesn't exist."""
        assert fixer._get_ddb_gru_count() is None

    def test_trim_to_active_grus(self, fixer, ddb_dataset):
        """Should trim DDB to target count and renormalize."""
        ddb_dataset.to_netcdf(fixer.ddb_path)

        fixer._trim_ddb_to_active_grus(2)

        with xr.open_dataset(fixer.ddb_path) as ds:
            assert ds.sizes["NGRU"] == 2
            # Fractions should be renormalized to sum to 1
            gru_sum = ds["GRU"].sum("NGRU").values[0]
            assert gru_sum == pytest.approx(1.0, abs=0.01)

    def test_trim_no_op_when_at_target(self, fixer, ddb_dataset):
        """Should do nothing when already at target count."""
        ddb_dataset.to_netcdf(fixer.ddb_path)

        fixer._trim_ddb_to_active_grus(5)  # target > current (3)

        with xr.open_dataset(fixer.ddb_path) as ds:
            assert ds.sizes["NGRU"] == 3  # Unchanged

    def test_ensure_gru_normalization(self, fixer):
        """Should normalize GRU fractions to sum to 1.0."""
        gru_data = np.array([[0.5, 0.3]], dtype=np.float64)  # sum = 0.8
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._ensure_gru_normalization()

        with xr.open_dataset(fixer.ddb_path) as ds:
            gru_sum = ds["GRU"].sum("NGRU").values[0]
            assert gru_sum == pytest.approx(1.0, abs=0.001)

    def test_renormalize_mesh_active_grus(self, fixer):
        """Should renormalize only the first N active GRU columns."""
        gru_data = np.array([[0.4, 0.3, 0.1]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._renormalize_mesh_active_grus(2)  # Only first 2 cols

        with xr.open_dataset(fixer.ddb_path) as ds:
            gru = ds["GRU"].values[0]
            # First 2 should sum to 1.0
            assert gru[:2].sum() == pytest.approx(1.0, abs=0.001)
            # Third column should be zeroed
            assert gru[2] == pytest.approx(0.0, abs=0.001)

    def test_off_by_one_ngru_count(self, fixer):
        """MESH reads NGRU-1: 3 cols → reads 2 active GRUs."""
        gru_data = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        count = fixer._get_mesh_active_gru_count()
        assert count == 2  # NGRU=3, MESH reads 3-1=2

    def test_get_num_cells(self, fixer, ddb_dataset):
        """Should return number of subbasins in DDB."""
        ddb_dataset.to_netcdf(fixer.ddb_path)
        assert fixer._get_num_cells() == 1

    def test_get_num_cells_missing_file(self, fixer):
        """Should return 1 when DDB doesn't exist."""
        assert fixer._get_num_cells() == 1

    def test_get_spatial_dim(self, fixer):
        """Should detect subbasin or N dimension."""
        ds1 = xr.Dataset({"x": (["subbasin"], [1])})
        assert fixer._get_spatial_dim(ds1) == "subbasin"

        ds2 = xr.Dataset({"x": (["N"], [1])})
        assert fixer._get_spatial_dim(ds2) == "N"

        ds3 = xr.Dataset({"x": (["gridcell"], [1])})
        assert fixer._get_spatial_dim(ds3) is None


# ---------------------------------------------------------------------------
# TestCLASSFileOperations
# ---------------------------------------------------------------------------

class TestCLASSFileOperations:
    """Test CLASS .ini file operations."""

    def test_get_class_block_count(self, fixer, class_ini_content):
        """Should count blocks via XSLP/XDRAINH/MANN/KSAT/MID marker."""
        fixer.class_file_path.write_text(class_ini_content)
        assert fixer._get_class_block_count() == 2

    def test_get_class_block_count_missing_file(self, fixer):
        """Should return None when file doesn't exist."""
        assert fixer._get_class_block_count() is None

    def test_read_nm_from_lines_legacy(self, fixer):
        """Should parse NM from legacy format (9th column of line 04)."""
        lines = [
            "  51.0  -116.0  1.0  1.0  0.1  1.0  0.0  1  3  04 DEGLAT/DEGLON/ZBLDGRD/ZRFHGRD/ZRFMGRD/GCGRD/FAREROT/NL/NM"
        ]
        assert fixer._read_nm_from_lines(lines) == 3

    def test_read_nm_from_lines_ini_style(self, fixer):
        """Should parse NM from ini-style 'NM x' format."""
        lines = ["NM 5    ! number of landcover classes (GRUs)"]
        assert fixer._read_nm_from_lines(lines) == 5

    def test_update_class_nm_legacy(self, fixer, class_ini_content):
        """Should update NM in legacy CLASS format."""
        fixer.class_file_path.write_text(class_ini_content)
        fixer._update_class_nm(5)

        content = fixer.class_file_path.read_text()
        lines = content.split("\n")
        for line in lines:
            if "04 DEGLAT" in line:
                parts = line.split()
                assert parts[8] == "5"
                break

    def test_update_class_nm_ini_style(self, fixer):
        """Should update NM in ini-style format."""
        fixer.class_file_path.write_text("NM 2    ! number of landcover classes (GRUs)\n")
        fixer._update_class_nm(7)

        content = fixer.class_file_path.read_text()
        assert "NM 7" in content

    def test_trim_class_to_count(self, fixer, class_ini_content):
        """Should keep only the first N CLASS blocks."""
        fixer.class_file_path.write_text(class_ini_content)
        fixer._trim_class_to_count(1)

        content = fixer.class_file_path.read_text()
        # Only 1 block marker should remain
        assert content.count("05 5xFCAN/4xLAMX") == 1


# ---------------------------------------------------------------------------
# TestSafeForcing
# ---------------------------------------------------------------------------

class TestSafeForcing:
    """Test create_safe_forcing method."""

    def test_creates_trimmed_forcing(self, fixer):
        """Should create a safe forcing file trimmed to simulation period."""
        from datetime import datetime

        import pandas as pd

        # Create minimal forcing file
        times = pd.date_range("2019-06-01", "2021-06-30", freq="h")
        ds = xr.Dataset(
            {
                "FSIN": (["subbasin", "time"], np.random.rand(1, len(times))),
                "PRE": (["subbasin", "time"], np.random.rand(1, len(times))),
            },
            coords={"time": times, "subbasin": [1]},
        )
        ds["time"].encoding["units"] = "hours since 1900-01-01"
        ds["time"].encoding["calendar"] = "standard"
        forcing_path = fixer.forcing_dir / "MESH_forcing.nc"
        ds.to_netcdf(forcing_path)

        # Return datetime objects, not strings — create_safe_forcing does timedelta arithmetic
        fixer.get_simulation_time_window = lambda: (
            datetime(2020, 1, 1), datetime(2020, 12, 31)
        )
        fixer.create_safe_forcing()

        safe_path = fixer.forcing_dir / "MESH_forcing_safe.nc"
        assert safe_path.exists()

        with xr.open_dataset(safe_path) as ds_safe:
            # Should be trimmed to roughly the simulation period (with spinup)
            assert len(ds_safe.time) < len(times)

    def test_no_crash_missing_forcing(self, fixer):
        """Should not crash when forcing file doesn't exist."""
        fixer.get_simulation_time_window = lambda: ("2020-01-01", "2020-12-31")
        fixer.create_safe_forcing()  # Should log warning and return


# ---------------------------------------------------------------------------
# TestElevationBandBlocks
# ---------------------------------------------------------------------------

class TestElevationBandBlocks:
    """Test create_elevation_band_class_blocks."""

    def test_creates_correct_block_count(self, fixer, class_ini_content, ddb_dataset):
        """Should create one CLASS block per elevation band."""
        fixer.class_file_path.write_text(class_ini_content)
        ddb_dataset.to_netcdf(fixer.ddb_path)

        elevation_info = [
            {"elevation": 1500.0, "fraction": 0.3},
            {"elevation": 2000.0, "fraction": 0.4},
            {"elevation": 2500.0, "fraction": 0.3},
        ]

        fixer.create_elevation_band_class_blocks(elevation_info)

        content = fixer.class_file_path.read_text()
        # Should have exactly 3 blocks
        block_count = content.count("05 5xFCAN/4xLAMX") + content.count("[GRU_")
        assert block_count == 3

    def test_missing_class_file_no_error(self, fixer, ddb_dataset):
        """Should not crash when CLASS file doesn't exist."""
        ddb_dataset.to_netcdf(fixer.ddb_path)
        elevation_info = [{"elevation": 1500.0, "fraction": 1.0}]
        fixer.create_elevation_band_class_blocks(elevation_info)


# ---------------------------------------------------------------------------
# TestRemoveSmallGRUs
# ---------------------------------------------------------------------------

class TestRemoveSmallGRUs:
    """Test _remove_small_grus method."""

    def test_removes_below_threshold(self, fixer, class_ini_content):
        """GRUs below 5% threshold should be removed."""
        gru_data = np.array([[0.7, 0.27, 0.03]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)
        fixer.class_file_path.write_text(class_ini_content)
        fixer._config_dict_override = {"MESH_GRU_MIN_TOTAL": 0.05}

        fixer._remove_small_grus()

        with xr.open_dataset(fixer.ddb_path) as ds_out:
            # Third GRU (3%) should be removed
            assert ds_out.sizes["NGRU"] == 2

    def test_keeps_all_above_threshold(self, fixer):
        """No GRUs should be removed when all above threshold."""
        gru_data = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._remove_small_grus()

        with xr.open_dataset(fixer.ddb_path) as ds_out:
            assert ds_out.sizes["NGRU"] == 3

    def test_keeps_largest_when_all_below(self, fixer):
        """When all GRUs below threshold, keep the largest."""
        gru_data = np.array([[0.02, 0.04, 0.01]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._remove_small_grus()

        with xr.open_dataset(fixer.ddb_path) as ds_out:
            # Should keep at least one GRU (the largest)
            assert ds_out.sizes["NGRU"] >= 1


# ---------------------------------------------------------------------------
# TestClassFieldOverrides
# ---------------------------------------------------------------------------

# A realistic single-GRU CLASS.ini block as produced by meshflow + fixes.
# Values here are the *flashy-regime* meshflow-derived defaults that the
# config overrides must be able to replace.
_MESHFLOW_CLASS_INI = """\
  MESH Model                                                                 01 TITLE
  MESHFlow                                                                   02 NAME
  University of Calgary, Canada                                              03 PLACE
   51.36   -116.00      10.0      2.0       50.0   -1.0    1    1    1       04 DEGLAT/DEGLON/ZRFM/ZRFH/ZBLD/GC/ILW/NL/NM

   0.000   0.000   0.000   1.000   0.000    3.15   0.000   0.000   1.500 05 5xFCAN/4xLAMX
   0.000   0.000   0.000  -4.605   0.000   0.000   0.000   0.000   1.500     06 5xLNZ0/4xLAMN
   0.000   0.000   0.000   0.050   0.000   0.000   0.000   0.000   0.200 07 5xALVC/4xCMAS
   0.000   0.000   0.000   0.290   0.000    1.00   0.000   0.000   0.100 08 5xALIC/4xROOT
   450.0   0.000   0.000 100.000   0.000   0.000   0.000  30.000 09 4xRSMN/4xQA50
   0.000   0.000   0.000   0.500   0.000   0.000   0.000   1.000 10 4xVPDA/4xVPDB
   0.000   0.000   0.000 100.000           0.000   0.000   0.000   5.000     11 4xPSGA/4xPSGB
    1.00   0.810   1.000   2.960 12 DRN/SDEP/FARE/DD
   0.155   0.505   0.022   32.20      10 Temp_sub-_gras 13 XSLP/XDRAINH/MANN/KSAT/MID
  23.000  25.000  33.000                                                     14 3xSAND (or more)
  11.000  12.000  15.000                                                     15 3xCLAY (or more)
   1.000   1.000   0.470                                                     16 3xORGM (or more)
  5.000  5.000  5.000  -5.000  -10.000   0.000  17 3xTBAR (or more)/TCAN/TSNO/TPND
   0.200   0.200   0.200   0.000   0.000   0.000   0.000                     18 3xTHLQ (or more)/3xTHIC (or more)/ZPND
   0.000   0.000   100.0   0.75   250.0   1.000  19 RCAN/SCAN/SNO/ALBS/RHOS/GRO

   0       0       0       0                                                 20 (not used, but 4x integer values are required)
"""

_JUNE_OVERRIDES = {
    'sand': [50.0, 50.0, 50.0],
    'clay': [20.0, 20.0, 20.0],
    'orgm': [0.0, 0.0, 0.0],
    'dd': 50.0,
    'mid': 100,
    'tbar': [4.0, 2.0, 1.0],
    'thlq': [0.25, 0.15, 0.04],
    'cmas': 4.5,
    'qa50': 36.0,
    'vpda': 0.8,
    'vpdb': 1.05,
}


class TestClassFieldOverrides:
    """Test CLASSFileManager.apply_field_overrides (config-driven CLASS fields)."""

    def _mgr(self, path):
        from symfluence.models.mesh.preprocessing.class_file_manager import (
            CLASSFileManager,
        )
        return CLASSFileManager(path, Mock())

    def test_overrides_all_regime_fields(self, forcing_dir):
        """All configured fields should replace the meshflow-derived values."""
        p = forcing_dir / "MESH_parameters_CLASS.ini"
        p.write_text(_MESHFLOW_CLASS_INI)
        self._mgr(p).apply_field_overrides(dict(_JUNE_OVERRIDES))

        lines = p.read_text().splitlines()

        def line_with(marker):
            return next(ln for ln in lines if marker in ln)

        sand = line_with('3xSAND').split()
        assert sand[:3] == ['50.00', '50.00', '50.00']
        clay = line_with('3xCLAY').split()
        assert clay[:3] == ['20.00', '20.00', '20.00']
        # DD is the 4th value on the DRN/SDEP/FARE/DD line
        dd_line = line_with('DRN/SDEP').split()
        assert dd_line[3] == '50.00'
        # MID is the 5th value on the XSLP line; the veg label must be preserved
        xslp_line = line_with('XSLP/XDRAINH')
        assert xslp_line.split()[4] == '100'
        assert 'Temp_sub-_gras' in xslp_line
        # TBAR first three values, TCAN/TSNO preserved
        tbar_line = line_with('3xTBAR').split()
        assert tbar_line[:3] == ['4.00', '2.00', '1.00']
        assert tbar_line[3] == '-5.000'
        thlq_line = line_with('3xTHLQ').split()
        assert thlq_line[:3] == ['0.250', '0.150', '0.040']
        # Veg scalars applied only to the active (non-zero) column, mirroring
        # meshflow's single-active-category layout (inactive columns stay 0).
        cmas_line = line_with('5xALVC/4xCMAS').split()
        assert cmas_line[5:9] == ['0.000', '0.000', '0.000', '4.50']
        qa50_line = line_with('4xRSMN/4xQA50').split()
        assert qa50_line[4:8] == ['0.000', '0.000', '0.000', '36.00']
        vpd_line = line_with('4xVPDA/4xVPDB').split()
        assert vpd_line[0:4] == ['0.000', '0.000', '0.000', '0.800']
        assert vpd_line[4:8] == ['0.000', '0.000', '0.000', '1.05']

    def test_none_values_are_noop(self, forcing_dir):
        """Unset (None) override keys must not modify the file."""
        p = forcing_dir / "MESH_parameters_CLASS.ini"
        p.write_text(_MESHFLOW_CLASS_INI)
        self._mgr(p).apply_field_overrides({'sand': None, 'dd': None})
        assert p.read_text() == _MESHFLOW_CLASS_INI

    def test_empty_overrides_noop(self, forcing_dir):
        """An empty override dict must not modify the file."""
        p = forcing_dir / "MESH_parameters_CLASS.ini"
        p.write_text(_MESHFLOW_CLASS_INI)
        self._mgr(p).apply_field_overrides({})
        assert p.read_text() == _MESHFLOW_CLASS_INI

    def test_partial_override_leaves_others(self, forcing_dir):
        """Only the specified field changes; siblings stay meshflow-derived."""
        p = forcing_dir / "MESH_parameters_CLASS.ini"
        p.write_text(_MESHFLOW_CLASS_INI)
        self._mgr(p).apply_field_overrides({'dd': 50.0})
        lines = p.read_text().splitlines()
        dd_line = next(ln for ln in lines if 'DRN/SDEP' in ln).split()
        assert dd_line[3] == '50.00'
        # SAND untouched -> still the flashy-regime meshflow values
        sand = next(ln for ln in lines if '3xSAND' in ln).split()
        assert sand[:3] == ['23.000', '25.000', '33.000']

    def test_missing_file_no_error(self, forcing_dir):
        """Should silently return if the CLASS file doesn't exist."""
        self._mgr(forcing_dir / "does_not_exist.ini").apply_field_overrides(
            dict(_JUNE_OVERRIDES)
        )

    def test_veg_override_targets_only_active_column(self, forcing_dir):
        """Inactive (zero) veg columns must stay zero to avoid CLASS crashes."""
        p = forcing_dir / "MESH_parameters_CLASS.ini"
        p.write_text(_MESHFLOW_CLASS_INI)
        self._mgr(p).apply_field_overrides({'cmas': 4.5})
        cmas = next(
            ln for ln in p.read_text().splitlines() if '5xALVC/4xCMAS' in ln
        ).split()
        # Only the active (position 8) column changes; 5-7 remain zero.
        assert cmas[5:9] == ['0.000', '0.000', '0.000', '4.50']

    def test_veg_override_all_zero_fallback(self, forcing_dir):
        """When the 4x group is all-zero, override the last column."""
        p = forcing_dir / "MESH_parameters_CLASS.ini"
        # CMAS group (positions 5-8) all zero.
        p.write_text(
            "   0.000   0.000   0.000   0.050   0.000   0.000   0.000   0.000"
            "   0.000 07 5xALVC/4xCMAS\n"
        )
        self._mgr(p).apply_field_overrides({'cmas': 4.5})
        cmas = p.read_text().split()
        assert cmas[5:9] == ['0.000', '0.000', '0.000', '4.50']

    def test_fixer_reads_config_overrides(self, forcing_dir, setup_dir):
        """MESHParameterFixer.apply_class_field_overrides should honor config."""
        from symfluence.models.mesh.preprocessing.parameter_fixer import (
            MESHParameterFixer,
        )
        fixer = MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_SPINUP_DAYS": 365,
                "MESH_SOIL_SAND": [50.0, 50.0, 50.0],
                "MESH_DD": 50.0,
                "MESH_MID": 100,
                "MESH_VEG_CMAS": 4.5,
            },
        )
        fixer.class_file_path.write_text(_MESHFLOW_CLASS_INI)
        fixer.apply_class_field_overrides()

        lines = fixer.class_file_path.read_text().splitlines()
        sand = next(ln for ln in lines if '3xSAND' in ln).split()
        assert sand[:3] == ['50.00', '50.00', '50.00']
        dd_line = next(ln for ln in lines if 'DRN/SDEP' in ln).split()
        assert dd_line[3] == '50.00'


class TestHydrologyIwfOverride:
    """Test MESHParameterFixer.apply_hydrology_field_overrides (IWF flag)."""

    def _fixer(self, forcing_dir, setup_dir, iwf):
        from symfluence.models.mesh.preprocessing.parameter_fixer import (
            MESHParameterFixer,
        )
        return MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_SPINUP_DAYS": 365,
                "MESH_IWF": iwf,
            },
        )

    def test_sets_iwf_off(self, forcing_dir, setup_dir):
        """MESH_IWF=0 should turn interflow off in hydrology.ini."""
        fixer = self._fixer(forcing_dir, setup_dir, 0)
        fixer.hydro_path.write_text("IWF   1\nRCHARG  0.2\n")
        fixer.apply_hydrology_field_overrides()
        content = fixer.hydro_path.read_text()
        assert "IWF   0" in content
        assert "RCHARG  0.2" in content

    def test_preserves_comment(self, forcing_dir, setup_dir):
        """Trailing comments on the IWF line should be preserved."""
        fixer = self._fixer(forcing_dir, setup_dir, 0)
        fixer.hydro_path.write_text("IWF   1  # interflow flag\n")
        fixer.apply_hydrology_field_overrides()
        content = fixer.hydro_path.read_text()
        assert "IWF   0" in content
        assert "# interflow flag" in content

    def test_unset_is_noop(self, forcing_dir, setup_dir):
        """No MESH_IWF -> hydrology file unchanged."""
        from symfluence.models.mesh.preprocessing.parameter_fixer import (
            MESHParameterFixer,
        )
        fixer = MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={"HYDROLOGICAL_MODEL": "MESH", "MESH_SPINUP_DAYS": 365},
        )
        original = "IWF   1\n"
        fixer.hydro_path.write_text(original)
        fixer.apply_hydrology_field_overrides()
        assert fixer.hydro_path.read_text() == original


# ---------------------------------------------------------------------------
# TestRunModeSelection  (bug B: LZS baseflow gating via RUNMODE)
# ---------------------------------------------------------------------------

class TestRunModeSelection:
    """RUNMODE selection in RunOptionsConfigBuilder.fix_snow_params.

    A single cell has no channel network, so WATROUTE routing (and the wf_lzs
    lower-zone baseflow store) is auto-disabled ('noroute'). Multi-cell domains
    route. An explicit MESH_RUNMODE must always be honoured — including keeping
    'runrte' on a single cell so FLZ/PWR/RCHARG stay live. Previously the config
    key was silently ignored (passed as the typed accessor, no dict_key) and
    every single-cell domain was forced to 'noroute'.
    """

    import logging as _logging

    def _build(self, tmp_path, config):
        from symfluence.models.mesh.preprocessing.run_options_builder import (
            RunOptionsConfigBuilder,
        )
        p = tmp_path / "MESH_input_run_options.ini"
        p.write_text(
            "  3 # flags\n"
            "RUNMODE               runrte\n"
            "STREAMFLOWOUTFLAG     csv\n"
            "OUTFILESFLAG         daily\n"
            "OUTFIELDSFLAG        none\n"
            "BASINAVGWBFILEFLAG    daily\n"
            "FROZENSOILINFILFLAG   0\n"
            "FREZTH                0.0\n"
            "SWELIM                800.0\n"
            "SNDENLIM              600.0\n"
            "PBSMFLAG              off\n"
            "METRICSSPINUP         730\n"
            "DIAGNOSEMODE          off\n"
            "PRINTSIMSTATUS        date_monthly\n"
            "SHDFILEFLAG           nc_subbasin pad_outlets\n"
            "BASINFORCINGFLAG      nc_subbasin\n"
        )
        import logging
        return RunOptionsConfigBuilder(p, config, logging.getLogger("test")), p

    def _runmode(self, path):
        import re
        m = re.search(r'RUNMODE\s+(\w+)', path.read_text())
        return m.group(1)

    def test_single_cell_unset_defaults_to_noroute(self, tmp_path):
        builder, p = self._build(tmp_path, {"HYDROLOGICAL_MODEL": "MESH"})
        builder.fix_snow_params(lambda: 1)
        assert self._runmode(p) == "noroute"

    def test_multi_cell_unset_defaults_to_runrte(self, tmp_path):
        builder, p = self._build(tmp_path, {"HYDROLOGICAL_MODEL": "MESH"})
        builder.fix_snow_params(lambda: 49)
        assert self._runmode(p) == "runrte"

    def test_explicit_runrte_honoured_on_single_cell(self, tmp_path):
        # The fix: an explicit request keeps the baseflow store live even on 1 cell
        builder, p = self._build(
            tmp_path, {"HYDROLOGICAL_MODEL": "MESH", "MESH_RUNMODE": "runrte"})
        builder.fix_snow_params(lambda: 1)
        assert self._runmode(p) == "runrte"

    def test_explicit_noroute_honoured_on_multi_cell(self, tmp_path):
        builder, p = self._build(
            tmp_path, {"HYDROLOGICAL_MODEL": "MESH", "MESH_RUNMODE": "noroute"})
        builder.fix_snow_params(lambda: 49)
        assert self._runmode(p) == "noroute"
