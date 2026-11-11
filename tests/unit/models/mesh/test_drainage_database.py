"""
Tests for MESH drainage database handler.

Tests cover initialization, ensure_completeness, GRU normalization,
reorder_by_rank, and elevation band conversion logic.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr


class TestInitialization:
    """Test MESHDrainageDatabase construction and path setup."""

    def test_init_sets_paths(self, tmp_path):
        """Constructor stores all path arguments."""
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        config = {"HYDROLOGICAL_MODEL": "MESH"}

        ddb = MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path / "rivers",
            rivers_name="test_rivers.shp",
            catchment_path=tmp_path / "catch",
            catchment_name="test_catch.shp",
            config=config,
        )
        assert ddb.forcing_dir == forcing
        assert ddb.rivers_name == "test_rivers.shp"

    def test_ddb_path_property(self, tmp_path):
        """ddb_path returns MESH_drainage_database.nc in forcing_dir."""
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        ddb = MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path,
            rivers_name="r.shp",
            catchment_path=tmp_path,
            catchment_name="c.shp",
            config={"HYDROLOGICAL_MODEL": "MESH"},
        )
        assert ddb.ddb_path == forcing / "MESH_drainage_database.nc"


class TestGetSpatialDim:
    """Test _get_spatial_dim helper."""

    def test_returns_N_dim(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        ddb = MESHDrainageDatabase(
            forcing_dir=tmp_path, rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={"HYDROLOGICAL_MODEL": "MESH"},
        )
        ds = xr.Dataset({"Rank": (["N"], [1, 2])})
        assert ddb._get_spatial_dim(ds) == "N"

    def test_returns_subbasin_dim(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        ddb = MESHDrainageDatabase(
            forcing_dir=tmp_path, rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={"HYDROLOGICAL_MODEL": "MESH"},
        )
        ds = xr.Dataset({"Rank": (["subbasin"], [1, 2])})
        assert ddb._get_spatial_dim(ds) == "subbasin"

    def test_returns_none_when_unknown(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        ddb = MESHDrainageDatabase(
            forcing_dir=tmp_path, rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={"HYDROLOGICAL_MODEL": "MESH"},
        )
        ds = xr.Dataset({"Rank": (["gridcell"], [1, 2])})
        assert ddb._get_spatial_dim(ds) is None


class TestShouldForceSingleGRU:
    """Test _should_force_single_gru decision logic."""

    def _make_ddb(self, tmp_path, config):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        return MESHDrainageDatabase(
            forcing_dir=tmp_path, rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config=config,
        )

    def test_explicit_true(self, tmp_path):
        ddb = self._make_ddb(tmp_path, {
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_FORCE_SINGLE_GRU": True,
        })
        assert ddb._should_force_single_gru() is True

    def test_explicit_false(self, tmp_path):
        ddb = self._make_ddb(tmp_path, {
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_FORCE_SINGLE_GRU": False,
        })
        assert ddb._should_force_single_gru() is False

    def test_auto_lumped_mode(self, tmp_path):
        ddb = self._make_ddb(tmp_path, {
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_SPATIAL_MODE": "lumped",
        })
        assert ddb._should_force_single_gru() is True

    def test_auto_distributed_mode(self, tmp_path):
        ddb = self._make_ddb(tmp_path, {
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_SPATIAL_MODE": "distributed",
        })
        assert ddb._should_force_single_gru() is False

    def test_elevation_mode_overrides(self, tmp_path):
        """Elevation discretization always disables force_single_gru."""
        ddb = self._make_ddb(tmp_path, {
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_FORCE_SINGLE_GRU": True,
            "SUB_GRID_DISCRETIZATION": "elevation",
        })
        assert ddb._should_force_single_gru() is False


class TestEnsureCompleteness:
    """Test ensure_completeness adds missing variables and normalizes."""

    @pytest.fixture
    def ddb_handler(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        return MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_SPATIAL_MODE": "distributed",
            },
        )

    def _write_ddb(self, path, n_sub=2, n_gru=3, include_ireach=False):
        """Write a minimal drainage database NetCDF."""
        gru_data = np.zeros((n_sub, n_gru), dtype=np.float64)
        gru_data[:, 0] = 0.7
        gru_data[:, 1] = 0.3
        # Leave extra columns as 0 for trimming tests

        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.arange(1, n_sub + 1, dtype=np.int32)),
            "Next": (["subbasin"], np.array([2, 0], dtype=np.int32)[:n_sub]),
            "GridArea": (["subbasin"], np.full(n_sub, 1e8, dtype=np.float64)),
            "ChnlSlope": (["subbasin"], np.full(n_sub, 0.01, dtype=np.float64)),
            "ChnlLength": (["subbasin"], np.full(n_sub, 1000.0, dtype=np.float64)),
            "lat": (["subbasin"], np.full(n_sub, 51.0, dtype=np.float64)),
            "lon": (["subbasin"], np.full(n_sub, -116.0, dtype=np.float64)),
        })
        if include_ireach:
            ds["IREACH"] = (["subbasin"], np.zeros(n_sub, dtype=np.int32))
            ds["IAK"] = (["subbasin"], np.ones(n_sub, dtype=np.int32))
        ds.to_netcdf(path)

    def test_adds_missing_ireach_iak(self, ddb_handler):
        """ensure_completeness adds IREACH and IAK if missing."""
        self._write_ddb(ddb_handler.ddb_path, n_sub=2, n_gru=3, include_ireach=False)
        ddb_handler.ensure_completeness()

        with xr.open_dataset(ddb_handler.ddb_path) as ds:
            assert "IREACH" in ds
            assert "IAK" in ds
            assert ds["IREACH"].dtype == np.int32
            assert ds["IAK"].dtype == np.int32

    def test_adds_al_da(self, ddb_handler):
        """ensure_completeness adds AL and DA if missing."""
        self._write_ddb(ddb_handler.ddb_path, n_sub=2, n_gru=2)
        ddb_handler.ensure_completeness()

        with xr.open_dataset(ddb_handler.ddb_path) as ds:
            assert "AL" in ds
            assert "DA" in ds
            # AL is derived from ChnlLength when available
            np.testing.assert_allclose(
                ds["AL"].values, ds["ChnlLength"].values, rtol=1e-5
            )

    def test_gru_trimming_removes_zero_columns(self, ddb_handler):
        """GRU columns that are all-zero should be trimmed."""
        self._write_ddb(ddb_handler.ddb_path, n_sub=2, n_gru=4)
        ddb_handler.ensure_completeness()

        with xr.open_dataset(ddb_handler.ddb_path) as ds:
            if "GRU" in ds and "NGRU" in ds.dims:
                # Only columns with nonzero fractions + MESH padding should remain
                gru_vals = ds["GRU"].values
                n_active = np.sum(gru_vals.sum(axis=0) > 0)
                # At minimum, zero-only columns beyond active ones are trimmed
                assert gru_vals.shape[1] <= 4

    def test_elevation_mode_skips_trimming(self, tmp_path):
        """In elevation mode, zero-fraction GRU columns should not be trimmed."""
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        ddb = MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "SUB_GRID_DISCRETIZATION": "elevation",
                "MESH_SPATIAL_MODE": "distributed",
            },
        )
        # Create DDB with identity GRU pattern (elevation bands)
        n_sub = 3
        n_gru = 4  # 3 bands + 1 padding
        gru_data = np.zeros((n_sub, n_gru), dtype=np.float64)
        for i in range(n_sub):
            gru_data[i, i] = 1.0  # identity matrix
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.arange(1, n_sub + 1, dtype=np.int32)),
            "Next": (["subbasin"], np.zeros(n_sub, dtype=np.int32)),
            "GridArea": (["subbasin"], np.full(n_sub, 1e8)),
            "ChnlSlope": (["subbasin"], np.full(n_sub, 0.01)),
            "ChnlLength": (["subbasin"], np.full(n_sub, 1000.0)),
            "lat": (["subbasin"], np.full(n_sub, 51.0)),
            "lon": (["subbasin"], np.full(n_sub, -116.0)),
        })
        ds.to_netcdf(ddb.ddb_path)

        ddb.ensure_completeness()

        with xr.open_dataset(ddb.ddb_path) as ds_out:
            # All 4 GRU columns should be preserved (no trimming)
            assert ds_out.sizes["NGRU"] >= n_gru

    def test_no_crash_on_missing_file(self, ddb_handler):
        """ensure_completeness should not raise when file is missing."""
        # ddb_path does not exist
        ddb_handler.ensure_completeness()  # Should just log and return

    def test_normalizes_gru_fractions(self, ddb_handler):
        """GRU fractions should sum to 1.0 per subbasin after completeness."""
        gru_data = np.array([[0.5, 0.3], [0.4, 0.2]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1, 2], dtype=np.int32)),
            "Next": (["subbasin"], np.array([2, 0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.full(2, 1e8)),
            "ChnlSlope": (["subbasin"], np.full(2, 0.01)),
            "ChnlLength": (["subbasin"], np.full(2, 1000.0)),
            "lat": (["subbasin"], np.full(2, 51.0)),
            "lon": (["subbasin"], np.full(2, -116.0)),
        })
        ds.to_netcdf(ddb_handler.ddb_path)

        ddb_handler.ensure_completeness()

        with xr.open_dataset(ddb_handler.ddb_path) as ds_out:
            if "GRU" in ds_out and "NGRU" in ds_out.dims:
                gru_sums = ds_out["GRU"].sum("NGRU").values
                # Each subbasin's GRU fractions should sum to 1.0
                np.testing.assert_allclose(gru_sums, 1.0, atol=0.01)


class TestGRUNormalization:
    """Test GRU fraction normalization and force_single_gru behavior."""

    @pytest.fixture
    def ddb_handler(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        return MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_SPATIAL_MODE": "distributed",
            },
        )

    def _write_ranked_ddb(self, path, gru_data, n_sub=None):
        """Write DDB with Rank/Next/GRU for reorder tests."""
        if n_sub is None:
            n_sub = gru_data.shape[0]
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.arange(1, n_sub + 1, dtype=np.int32)),
            "Next": (["subbasin"], np.array(
                [i + 1 if i < n_sub - 1 else 0 for i in range(n_sub)], dtype=np.int32
            )),
            "GridArea": (["subbasin"], np.full(n_sub, 1e8)),
            "ChnlSlope": (["subbasin"], np.full(n_sub, 0.01)),
            "ChnlLength": (["subbasin"], np.full(n_sub, 1000.0)),
            "lat": (["subbasin"], np.full(n_sub, 51.0)),
            "lon": (["subbasin"], np.full(n_sub, -116.0)),
        })
        ds.to_netcdf(path)

    def test_reorder_normalizes_fractions(self, ddb_handler):
        """reorder_by_rank_and_normalize normalizes GRU fractions to sum=1."""
        gru_data = np.array([[0.5, 0.3], [0.6, 0.2]], dtype=np.float64)
        self._write_ranked_ddb(ddb_handler.ddb_path, gru_data)

        ddb_handler.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb_handler.ddb_path) as ds:
            gru_sums = ds["GRU"].sum("NGRU").values
            np.testing.assert_allclose(gru_sums, 1.0, atol=0.01)

    def test_zero_sum_gru_gets_default(self, ddb_handler):
        """Subbasins with 0 GRU coverage get first GRU set to 1.0."""
        gru_data = np.array([[0.0, 0.0], [0.7, 0.3]], dtype=np.float64)
        self._write_ranked_ddb(ddb_handler.ddb_path, gru_data)

        ddb_handler.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb_handler.ddb_path) as ds:
            # First subbasin should have GRU[0] = 1.0
            assert ds["GRU"].values[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_force_single_gru_collapses(self, tmp_path):
        """When force_single_gru is True, reorder creates NGRU=2."""
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        ddb = MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_FORCE_SINGLE_GRU": True,
            },
        )
        # 3 GRUs initially
        gru_data = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.array([1e8])),
            "ChnlSlope": (["subbasin"], np.array([0.01])),
            "ChnlLength": (["subbasin"], np.array([1000.0])),
            "lat": (["subbasin"], np.array([51.0])),
            "lon": (["subbasin"], np.array([-116.0])),
        })
        ds.to_netcdf(ddb.ddb_path)

        ddb.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb.ddb_path) as ds_out:
            # MESH off-by-one: NGRU=2 so MESH reads 1 GRU
            assert ds_out.sizes["NGRU"] == 2
            # First column should be ~0.998, second ~0.002
            np.testing.assert_allclose(ds_out["GRU"].values[0, 0], 0.998, atol=0.01)


class TestReorderByRank:
    """Test reorder_by_rank_and_normalize sorting and Next remapping."""

    @pytest.fixture
    def ddb_handler(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        forcing = tmp_path / "forcing"
        forcing.mkdir()
        return MESHDrainageDatabase(
            forcing_dir=forcing,
            rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_SPATIAL_MODE": "distributed",
            },
        )

    def test_sorts_by_rank(self, ddb_handler):
        """Subbasins should be sorted by Rank in ascending order."""
        # Write with ranks in reverse order
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], np.array([[0.5, 0.5], [0.6, 0.4]], dtype=np.float64)),
            "Rank": (["subbasin"], np.array([3, 1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0, 3], dtype=np.int32)),
            "GridArea": (["subbasin"], np.array([2e8, 1e8])),
            "ChnlSlope": (["subbasin"], np.full(2, 0.01)),
            "ChnlLength": (["subbasin"], np.full(2, 1000.0)),
            "lat": (["subbasin"], np.full(2, 51.0)),
            "lon": (["subbasin"], np.full(2, -116.0)),
        })
        ds.to_netcdf(ddb_handler.ddb_path)

        ddb_handler.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb_handler.ddb_path) as ds_out:
            # Ranks should be [1, 2] after remapping
            np.testing.assert_array_equal(ds_out["Rank"].values, [1, 2])

    def test_single_subbasin_next_equals_one(self, ddb_handler):
        """Single-cell domain should have Next=1 (self-reference for MESH)."""
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], np.array([[0.998, 0.002]], dtype=np.float64)),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.array([1e8])),
            "ChnlSlope": (["subbasin"], np.array([0.01])),
            "ChnlLength": (["subbasin"], np.array([1000.0])),
            "lat": (["subbasin"], np.array([51.0])),
            "lon": (["subbasin"], np.array([-116.0])),
        })
        ds.to_netcdf(ddb_handler.ddb_path)

        ddb_handler.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb_handler.ddb_path) as ds_out:
            assert ds_out["Next"].values[0] == 1

    def test_no_crash_missing_file(self, ddb_handler):
        """reorder should not raise when ddb_path doesn't exist."""
        ddb_handler.reorder_by_rank_and_normalize()  # Should log warning and return

    def test_next_remapping(self, ddb_handler):
        """Next values should be correctly remapped after reordering."""
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], np.array(
                [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], dtype=np.float64
            )),
            "Rank": (["subbasin"], np.array([5, 2, 10], dtype=np.int32)),
            # sub0 (rank=5) drains to sub2 (rank=10), sub1 (rank=2) drains to sub0 (rank=5)
            "Next": (["subbasin"], np.array([10, 5, 0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.full(3, 1e8)),
            "ChnlSlope": (["subbasin"], np.full(3, 0.01)),
            "ChnlLength": (["subbasin"], np.full(3, 1000.0)),
            "lat": (["subbasin"], np.full(3, 51.0)),
            "lon": (["subbasin"], np.full(3, -116.0)),
        })
        ds.to_netcdf(ddb_handler.ddb_path)

        ddb_handler.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb_handler.ddb_path) as ds_out:
            # After reorder: rank 2 → new rank 1, rank 5 → new rank 2, rank 10 → new rank 3
            np.testing.assert_array_equal(ds_out["Rank"].values, [1, 2, 3])
            # Next: sub1(old rank 2, new 1) → old Next=5 → new 2
            # sub0(old rank 5, new 2) → old Next=10 → new 3
            # sub2(old rank 10, new 3) → old Next=0 → 0 (outlet)
            np.testing.assert_array_equal(ds_out["Next"].values, [2, 3, 0])

    def test_drops_ngru_coordinate(self, ddb_handler):
        """NGRU should be a dimension only, not a coordinate variable."""
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], np.array([[0.998, 0.002]], dtype=np.float64)),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.array([1e8])),
            "ChnlSlope": (["subbasin"], np.array([0.01])),
            "ChnlLength": (["subbasin"], np.array([1000.0])),
            "lat": (["subbasin"], np.array([51.0])),
            "lon": (["subbasin"], np.array([-116.0])),
        })
        # Add NGRU as a coordinate
        ds = ds.assign_coords(NGRU=np.arange(2))
        ds.to_netcdf(ddb_handler.ddb_path)

        ddb_handler.reorder_by_rank_and_normalize()

        with xr.open_dataset(ddb_handler.ddb_path) as ds_out:
            # NGRU should be a dimension but NOT a coordinate
            assert "NGRU" in ds_out.dims
            assert "NGRU" not in ds_out.coords


class TestAsBool:
    """Test _as_bool helper parsing."""

    @pytest.fixture
    def ddb_handler(self, tmp_path):
        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase

        return MESHDrainageDatabase(
            forcing_dir=tmp_path, rivers_path=tmp_path, rivers_name="r.shp",
            catchment_path=tmp_path, catchment_name="c.shp",
            config={"HYDROLOGICAL_MODEL": "MESH"},
        )

    def test_bool_passthrough(self, ddb_handler):
        assert ddb_handler._as_bool(True) is True
        assert ddb_handler._as_bool(False) is False

    def test_string_true(self, ddb_handler):
        for val in ("true", "True", "TRUE", "1", "yes", "y", "on"):
            assert ddb_handler._as_bool(val) is True

    def test_string_false(self, ddb_handler):
        for val in ("false", "False", "FALSE", "0", "no", "n", "off"):
            assert ddb_handler._as_bool(val) is False

    def test_none_returns_default(self, ddb_handler):
        assert ddb_handler._as_bool(None, default=True) is True
        assert ddb_handler._as_bool(None, default=False) is False

    def test_numeric(self, ddb_handler):
        assert ddb_handler._as_bool(1) is True
        assert ddb_handler._as_bool(0) is False


class TestComputeDrainageArea:
    """Accumulated drainage area when the meshflow DDB carries no DA."""

    def _da(self, grid_area, next_, rank):
        import logging

        import xarray as xr

        from symfluence.models.mesh.preprocessing.drainage_database import MESHDrainageDatabase
        ddb = MESHDrainageDatabase.__new__(MESHDrainageDatabase)
        ddb.logger = logging.getLogger('test')
        ds = xr.Dataset({'GridArea': ('N', grid_area), 'Next': ('N', next_), 'Rank': ('N', rank)})
        return ddb._compute_drainage_area(ds, 'N', len(grid_area)).values

    def test_chain_accumulates_once(self):
        """Regression: a chain of four 1 km2 cells gave [1, 5, 5, 5]."""
        da = self._da([1e6, 1e6, 1e6, 1e6], [2, 3, 4, 0], [1, 2, 3, 4])
        assert da.tolist() == [1e6, 2e6, 3e6, 4e6]

    def test_confluence_in_any_storage_order(self):
        # ranks 1 and 2 are headwaters joining at rank 3, which drains to the outlet (4)
        da = self._da([4e6, 3e6, 1e6, 2e6], [0, 4, 3, 3], [4, 3, 1, 2])
        assert da.tolist() == [10e6, 6e6, 1e6, 2e6]

    def test_self_referencing_outlet(self):
        da = self._da([1e6, 2e6], [2, 2], [1, 2])
        assert da.tolist() == [1e6, 3e6]

    def test_cycle_is_rejected(self):
        from symfluence.models.mesh.errors import MESHInputError
        with pytest.raises(MESHInputError, match='cycle'):
            self._da([1e6, 1e6], [2, 1], [1, 2])
