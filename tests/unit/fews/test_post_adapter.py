"""Tests for FEWS post-adapter."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig
from symfluence.fews.exceptions import FEWSAdapterError, IDMappingError
from symfluence.fews.pi_diagnostics import DiagnosticsCollector
from symfluence.fews.pi_xml import read_pi_xml_timeseries
from symfluence.fews.post_adapter import FEWSPostAdapter


def _adapter(run_info, config, fews_config, **kwargs):
    return FEWSPostAdapter(
        run_info_path=run_info, config=config, fews_config=fews_config, **kwargs
    )


class TestFEWSPostAdapter:
    def test_exports_workflow_results_netcdf(
        self, sample_run_info_xml, tmp_work_dir, model_project, export_fews_config
    ):
        config, _ = model_project
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        output_dir = _adapter(sample_run_info_xml, config, export_fews_config).run(diag=diag)

        assert not diag.has_fatal
        with xr.open_dataset(output_dir / "output.nc") as result:
            assert list(result.data_vars) == ["Q.sim"]
            np.testing.assert_allclose(result["Q.sim"].values[[0, -1]], [10.0, 20.0])

    def test_exports_pi_xml(
        self, sample_run_info_xml, tmp_work_dir, model_project, export_fews_config
    ):
        config, _ = model_project
        fews_cfg = export_fews_config.model_copy(update={"data_format": "pi-xml"})
        output_dir = _adapter(sample_run_info_xml, config, fews_cfg).run()

        ds = read_pi_xml_timeseries(output_dir / "timeseries.xml")
        assert list(ds.data_vars) == ["Q.sim"]
        assert ds["Q.sim"].attrs["units"] == "m3/s"
        # location defaults to the SYMFLUENCE domain
        assert ds["Q.sim"].attrs["location_id"] == "test_basin"

    def test_forcing_in_exchange_dirs_is_never_exported(
        self, sample_run_info_xml, sample_netcdf, tmp_work_dir, export_fews_config, tmp_path
    ):
        """The first *.nc in workDir / toModel is the forcing the pre-adapter
        wrote; it used to be exported to FEWS as model output with exit 0."""
        from types import SimpleNamespace

        (tmp_work_dir / "output.nc").write_bytes(sample_netcdf.read_bytes())
        config = SimpleNamespace(
            system=SimpleNamespace(data_dir=str(tmp_path / "empty_data")),
            domain=SimpleNamespace(name="test_basin", experiment_id="fews_run_1"),
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        with pytest.raises(FEWSAdapterError, match="No model output found"):
            _adapter(sample_run_info_xml, config, export_fews_config).run(diag=diag)
        assert not (tmp_work_dir / "toFews" / "output.nc").exists()

    def test_config_is_required(self, sample_run_info_xml, export_fews_config):
        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml, fews_config=export_fews_config
        )
        with pytest.raises(FEWSAdapterError, match="--config"):
            adapter.run()

    def test_results_outside_run_period(
        self, real_run_info_xml, tmp_work_dir, model_project, export_fews_config
    ):
        import pandas as pd

        config, results_csv = model_project
        frame = pd.read_csv(results_csv, index_col=0, parse_dates=True)
        frame.index = frame.index + pd.Timedelta(days=365)
        frame.to_csv(results_csv)
        with pytest.raises(FEWSAdapterError, match="for the FEWS run period"):
            _adapter(real_run_info_xml, config, export_fews_config).run()

    def test_output_is_clipped_to_run_period_and_written_where_fews_asked(
        self, real_run_info_xml, tmp_work_dir, model_project, export_fews_config
    ):
        config, _ = model_project
        _adapter(real_run_info_xml, config, export_fews_config).run()
        # run file: 2023-01-01 00:00 .. 2023-01-03 00:00 at GMT+1; results start
        # 2023-01-01 00:00 UTC, so the window is 00:00 .. 2023-01-02 23:00 UTC
        with xr.open_dataset(tmp_work_dir / "fromModel" / "q.nc") as result:
            assert result["time"].values[0] == np.datetime64("2023-01-01T00:00:00")
            assert result["time"].values[-1] == np.datetime64("2023-01-02T23:00:00")

    def test_export_without_explicit_mapping_fails(
        self, sample_run_info_xml, model_project, tmp_work_dir
    ):
        """discharge must not be exported under a guessed id (it went out as
        the observation id Q.obs via the auto-map)."""
        config, _ = model_project
        fews_cfg = FEWSConfig(work_dir=str(tmp_work_dir), auto_id_map=True)
        with pytest.raises(IDMappingError, match="explicit ID-map export entry"):
            _adapter(sample_run_info_xml, config, fews_cfg).run()
        assert not (tmp_work_dir / "toFews" / "output.nc").exists()

    def test_diagnostics_always_written(self, sample_run_info_xml, tmp_work_dir):
        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")

        try:
            adapter.run(diag=diag)
        except FEWSAdapterError as exc:
            diag.fatal(str(exc))
        finally:
            diag.write()

        assert (tmp_work_dir / "toFews" / "diag.xml").exists()
        assert diag.has_fatal
