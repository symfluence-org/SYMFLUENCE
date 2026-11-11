"""Tests for FEWS pre-adapter."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig
from symfluence.fews.exceptions import FEWSAdapterError
from symfluence.fews.pi_diagnostics import DiagnosticsCollector
from symfluence.fews.pre_adapter import FEWSPreAdapter


class TestFEWSPreAdapter:
    def test_pre_adapter_netcdf(self, sample_run_info_xml, sample_netcdf, tmp_work_dir):
        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        config, run_info = adapter.run(diag=diag)

        assert run_info.start_time.year == 2023
        assert not diag.has_fatal
        # Translated forcing goes to the adapter's own directory; the file
        # FEWS exported to toModel/ is left as FEWS wrote it.
        assert adapter.forcing_path == tmp_work_dir / "symfluence" / "forcing.nc"
        with xr.open_dataset(adapter.forcing_path) as translated:
            assert {"pptrate", "airtemp"} <= set(translated.data_vars)
        with xr.open_dataset(tmp_work_dir / "toModel" / "forcing.nc") as exported:
            assert "P.obs" in exported.data_vars
        assert adapter.config_overrides["EXPERIMENT_TIME_START"] == "2023-01-01 00:00"

    def test_pre_adapter_pi_xml(self, sample_run_info_xml, sample_pi_xml, tmp_work_dir,
                                sample_fews_config_with_conversion):
        fews_cfg = sample_fews_config_with_conversion.model_copy(
            update={"data_format": "pi-xml"})
        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            fews_config=fews_cfg,
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        config, run_info = adapter.run(diag=diag)

        assert not diag.has_fatal
        with xr.open_dataset(adapter.forcing_path) as translated:
            # degC -> K through the explicit entry
            assert float(translated["airtemp"].values[0]) == pytest.approx(278.15)

    def test_pi_xml_in_other_units_needs_explicit_conversion(
        self, sample_run_info_xml, sample_pi_xml, tmp_work_dir
    ):
        """mm/h and degC input must not be auto-mapped to mm/s and K unchanged."""
        from symfluence.fews.exceptions import IDMappingError

        adapter = FEWSPreAdapter(run_info_path=sample_run_info_xml, data_format="pi-xml")
        with pytest.raises(IDMappingError, match="applies no conversion"):
            adapter.run()

    def test_files_named_in_run_file_are_used(self, real_run_info_xml, tmp_work_dir):
        """A real run file lists its inputs; only those are read."""
        times = np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
        xr.Dataset({"T.obs": ("time", np.full(len(times), 280.0), {"units": "K"})},
                   coords={"time": times}).to_netcdf(tmp_work_dir / "toModel" / "met.nc")
        # a stray file in toModel/ that the run file does not mention
        xr.Dataset({"junk": ("time", np.zeros(len(times)))},
                   coords={"time": times}).to_netcdf(tmp_work_dir / "toModel" / "other.nc")

        adapter = FEWSPreAdapter(run_info_path=real_run_info_xml, data_format="netcdf-cf")
        adapter.run()
        with xr.open_dataset(adapter.forcing_path) as translated:
            assert list(translated.data_vars) == ["airtemp"]

    def test_pre_adapter_no_forcing(self, sample_run_info_xml, tmp_work_dir):
        # Remove all files from toModel
        for f in (tmp_work_dir / "toModel").iterdir():
            f.unlink()

        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        with pytest.raises(FEWSAdapterError, match="No NetCDF files"):
            adapter.run(diag=diag)

    def test_diagnostics_on_failure(self, sample_run_info_xml, tmp_work_dir):
        # Remove all files from toModel
        for f in (tmp_work_dir / "toModel").iterdir():
            f.unlink()

        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        try:
            adapter.run(diag=diag)
        except FEWSAdapterError:
            pass

        # Diagnostics should have been populated
        assert len(diag.messages) > 0

    def test_pre_adapter_with_custom_config(self, sample_run_info_xml, sample_netcdf, tmp_work_dir):
        fews_cfg = FEWSConfig(
            work_dir=str(tmp_work_dir),
            data_format="netcdf-cf",
            auto_id_map=True,
        )
        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            fews_config=fews_cfg,
        )
        config, run_info = adapter.run()
        assert run_info is not None
