"""End-to-end integration test for the FEWS adapter flow.

Synthetic pipeline:
  1. Create run_info.xml + forcing data in toModel/
  2. Run pre-adapter (import forcing, map variables)
  3. Skip actual model execution (write the workflow's standardized results)
  4. Run post-adapter (map back, export to toFews/)
  5. Verify toFews/ contains valid output + diagnostics
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig, IDMapEntry
from symfluence.fews.pi_diagnostics import DiagnosticsCollector
from symfluence.fews.post_adapter import FEWSPostAdapter
from symfluence.fews.pre_adapter import FEWSPreAdapter


@pytest.fixture
def fews_work_dir(tmp_path):
    """Set up a complete FEWS work directory."""
    (tmp_path / "toModel").mkdir()
    (tmp_path / "toFews").mkdir()
    return tmp_path


@pytest.fixture
def run_info_xml(fews_work_dir):
    """Write run_info.xml."""
    xml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Run>
            <workDir>{fews_work_dir}</workDir>
            <inputDir>{fews_work_dir}/toModel</inputDir>
            <outputDir>{fews_work_dir}/toFews</outputDir>
            <startDateTime>2023-06-01T00:00:00Z</startDateTime>
            <endDateTime>2023-06-10T00:00:00Z</endDateTime>
            <properties>
                <DOMAIN_NAME>integration_test</DOMAIN_NAME>
            </properties>
        </Run>
    """)
    path = fews_work_dir / "run_info.xml"
    path.write_text(xml)
    return path


@pytest.fixture
def forcing_netcdf(fews_work_dir):
    """Write synthetic forcing data."""
    times = np.arange("2023-06-01", "2023-06-10", dtype="datetime64[h]")
    ds = xr.Dataset(
        {
            "P.obs": ("time", np.random.rand(len(times)) * 5.0),
            "T.obs": ("time", np.random.rand(len(times)) * 15.0 + 5.0),
        },
        coords={"time": times},
    )
    path = fews_work_dir / "toModel" / "forcing.nc"
    ds.to_netcdf(path)
    return path


def _write_model_results(tmp_path, times, values):
    """What the workflow's postprocess_results step leaves behind, plus a
    config stub pointing at it (the post-adapter exports only this)."""
    from types import SimpleNamespace

    import pandas as pd

    data_dir = tmp_path / "symfluence_data"
    results_dir = data_dir / "domain_integration_test" / "results"
    results_dir.mkdir(parents=True)
    frame = pd.DataFrame({"GR_discharge_cms": values}, index=pd.DatetimeIndex(times))
    frame.index.name = "datetime"
    frame.to_csv(results_dir / "run_1_results.csv")
    return SimpleNamespace(
        system=SimpleNamespace(data_dir=str(data_dir)),
        domain=SimpleNamespace(name="integration_test", experiment_id="run_1"),
    )


class TestFEWSAdapterFlow:
    def test_end_to_end_netcdf(self, run_info_xml, forcing_netcdf, fews_work_dir):
        """Full pre -> (skip model) -> post flow with NetCDF-CF format."""
        fews_cfg = FEWSConfig(
            work_dir=str(fews_work_dir),
            data_format="netcdf-cf",
            id_map=[
                IDMapEntry(fews_id="P.obs", symfluence_id="pptrate"),
                IDMapEntry(fews_id="T.obs", symfluence_id="airtemp"),
                IDMapEntry(fews_id="Q.sim", symfluence_id="discharge", direction="export"),
            ],
            auto_id_map=False,
        )

        diag = DiagnosticsCollector(fews_work_dir / "toFews" / "diag.xml")

        # --- PRE-ADAPTER ---
        pre = FEWSPreAdapter(
            run_info_path=run_info_xml,
            fews_config=fews_cfg,
        )
        config, run_info = pre.run(diag=diag)

        # Verify forcing was processed
        assert pre.forcing_path.exists()

        # --- SIMULATE MODEL OUTPUT ---
        times = np.arange("2023-06-01", "2023-06-10", dtype="datetime64[h]")
        config = _write_model_results(
            fews_work_dir, times, np.random.rand(len(times)) * 50.0)

        # --- POST-ADAPTER ---
        post = FEWSPostAdapter(
            run_info_path=run_info_xml,
            fews_config=fews_cfg,
            config=config,
        )
        output_dir = post.run(diag=diag)

        # --- VERIFY ---
        assert output_dir.is_dir()
        assert (output_dir / "output.nc").exists()

        # Check diagnostics
        diag.write()
        assert (fews_work_dir / "toFews" / "diag.xml").exists()
        assert not diag.has_fatal

        # Read back the output
        result = xr.open_dataset(output_dir / "output.nc")
        assert list(result.data_vars) == ["Q.sim"]
        assert result.attrs["Conventions"] == "CF-1.6"
        result.close()

    def test_end_to_end_pi_xml(self, run_info_xml, fews_work_dir):
        """Full flow with PI-XML format."""
        # Write PI-XML forcing
        pi_xml = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <TimeSeries xmlns="http://www.wldelft.nl/fews/PI">
                <timeZone>0.0</timeZone>
                <series>
                    <header>
                        <type>instantaneous</type>
                        <locationId>loc1</locationId>
                        <parameterId>P.obs</parameterId>
                        <missVal>-999.0</missVal>
                        <units>mm/s</units>
                    </header>
                    <event date="2023-06-01" time="00:00:00" value="1.0"/>
                    <event date="2023-06-01" time="01:00:00" value="2.0"/>
                    <event date="2023-06-01" time="02:00:00" value="3.0"/>
                </series>
            </TimeSeries>
        """)
        (fews_work_dir / "toModel" / "timeseries.xml").write_text(pi_xml)

        fews_cfg = FEWSConfig(
            work_dir=str(fews_work_dir),
            data_format="pi-xml",
            auto_id_map=True,
            id_map=[
                IDMapEntry(fews_id="Q.sim", symfluence_id="discharge", direction="export"),
            ],
        )
        diag = DiagnosticsCollector(fews_work_dir / "toFews" / "diag.xml")

        # Pre-adapter
        pre = FEWSPreAdapter(run_info_path=run_info_xml, fews_config=fews_cfg)
        config, run_info = pre.run(diag=diag)

        # Synthetic output
        times = np.arange("2023-06-01", "2023-06-01T03:00:00", dtype="datetime64[h]")
        model_config = _write_model_results(fews_work_dir, times, [10.0, 20.0, 30.0])

        # Post-adapter (writes PI-XML)
        post = FEWSPostAdapter(run_info_path=run_info_xml, fews_config=fews_cfg,
                               config=model_config)
        output_dir = post.run(diag=diag)

        assert (output_dir / "timeseries.xml").exists()
        diag.write()
        assert not diag.has_fatal
