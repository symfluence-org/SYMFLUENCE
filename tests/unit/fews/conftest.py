"""Shared fixtures for FEWS adapter unit tests."""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig, IDMapEntry


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Create a FEWS-style work directory structure."""
    (tmp_path / "toModel").mkdir()
    (tmp_path / "toFews").mkdir()
    (tmp_path / "states_in").mkdir()
    (tmp_path / "states_out").mkdir()
    return tmp_path


@pytest.fixture
def sample_run_info_xml(tmp_work_dir):
    """Write a sample run_info.xml and return its path."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Run>
            <workDir>{work_dir}</workDir>
            <inputDir>{work_dir}/toModel</inputDir>
            <outputDir>{work_dir}/toFews</outputDir>
            <stateInputDir>{work_dir}/states_in</stateInputDir>
            <stateOutputDir>{work_dir}/states_out</stateOutputDir>
            <startDateTime>2023-01-01T00:00:00Z</startDateTime>
            <endDateTime>2023-01-10T00:00:00Z</endDateTime>
            <time0>2023-01-05T00:00:00Z</time0>
            <timeStep>3600</timeStep>
            <properties>
                <DOMAIN_NAME>test_basin</DOMAIN_NAME>
                <EXPERIMENT_ID>fews_run_1</EXPERIMENT_ID>
                <HYDROLOGICAL_MODEL>GR</HYDROLOGICAL_MODEL>
            </properties>
        </Run>
    """).format(work_dir=tmp_work_dir)

    path = tmp_work_dir / "run_info.xml"
    path.write_text(xml_content)
    return path


@pytest.fixture
def sample_pi_xml(tmp_work_dir):
    """Write a sample PI-XML timeseries file and return its path."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <TimeSeries xmlns="http://www.wldelft.nl/fews/PI">
            <timeZone>0.0</timeZone>
            <series>
                <header>
                    <type>instantaneous</type>
                    <locationId>station_01</locationId>
                    <parameterId>P.obs</parameterId>
                    <missVal>-999.0</missVal>
                    <units>mm/h</units>
                </header>
                <event date="2023-01-01" time="00:00:00" value="0.5"/>
                <event date="2023-01-01" time="01:00:00" value="1.2"/>
                <event date="2023-01-01" time="02:00:00" value="-999.0"/>
                <event date="2023-01-01" time="03:00:00" value="0.0"/>
            </series>
            <series>
                <header>
                    <type>instantaneous</type>
                    <locationId>station_01</locationId>
                    <parameterId>T.obs</parameterId>
                    <missVal>-999.0</missVal>
                    <units>degC</units>
                </header>
                <event date="2023-01-01" time="00:00:00" value="5.0"/>
                <event date="2023-01-01" time="01:00:00" value="4.5"/>
                <event date="2023-01-01" time="02:00:00" value="4.0"/>
                <event date="2023-01-01" time="03:00:00" value="3.5"/>
            </series>
        </TimeSeries>
    """)

    path = tmp_work_dir / "toModel" / "timeseries.xml"
    path.write_text(xml_content)
    return path


@pytest.fixture
def sample_netcdf(tmp_work_dir):
    """Write a sample NetCDF forcing file and return its path."""
    times = np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
    ds = xr.Dataset(
        {
            "P.obs": ("time", np.random.rand(len(times)) * 5.0),
            "T.obs": ("time", np.random.rand(len(times)) * 10.0 + 270.0),
        },
        coords={"time": times},
    )
    path = tmp_work_dir / "toModel" / "forcing.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def sample_fews_config():
    """Create a basic FEWSConfig."""
    return FEWSConfig(
        work_dir=".",
        data_format="netcdf-cf",
        id_map=[
            IDMapEntry(fews_id="P.obs", symfluence_id="pptrate"),
            IDMapEntry(fews_id="T.obs", symfluence_id="airtemp"),
        ],
    )


@pytest.fixture
def sample_fews_config_with_conversion():
    """Create a FEWSConfig with unit conversion."""
    return FEWSConfig(
        work_dir=".",
        data_format="netcdf-cf",
        id_map=[
            IDMapEntry(
                fews_id="P.obs",
                symfluence_id="pptrate",
                fews_unit="mm/h",
                symfluence_unit="kg m-2 s-1",
                conversion_factor=1.0 / 3600.0,
            ),
            IDMapEntry(
                fews_id="T.obs",
                symfluence_id="airtemp",
                fews_unit="degC",
                symfluence_unit="K",
                conversion_offset=273.15,
            ),
        ],
    )


@pytest.fixture
def sample_state_files(tmp_work_dir):
    """Create sample state NetCDF files in the FEWS state input directory."""
    states_dir = tmp_work_dir / "states_in"
    for name in ["state_snow.nc", "state_soil.nc"]:
        ds = xr.Dataset({"value": ("x", [1.0, 2.0, 3.0])})
        ds.to_netcdf(states_dir / name)
    return states_dir


@pytest.fixture
def real_run_info_xml(tmp_work_dir):
    """A run file as Delft-FEWS writes it (pi_run.xsd): PI namespace, date/time
    attributes, a timeZone, individually listed files, typed properties."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Run xmlns="http://www.wldelft.nl/fews/PI"
             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.5">
            <timeZone>1.0</timeZone>
            <startDateTime date="2023-01-01" time="00:00:00"/>
            <endDateTime date="2023-01-03" time="00:00:00"/>
            <time0 date="2023-01-02" time="00:00:00"/>
            <workDir>{work_dir}</workDir>
            <inputTimeSeriesFile>{work_dir}/toModel/in.xml</inputTimeSeriesFile>
            <inputNetcdfFile>{work_dir}/toModel/met.nc</inputNetcdfFile>
            <outputDiagnosticFile>{work_dir}/diagnostics/diag.xml</outputDiagnosticFile>
            <outputTimeSeriesFile>{work_dir}/fromModel/q.xml</outputTimeSeriesFile>
            <outputNetcdfFile>{work_dir}/fromModel/q.nc</outputNetcdfFile>
            <properties>
                <string key="DOMAIN_NAME" value="test_basin"/>
                <int key="N_MEMBERS" value="5"/>
                <dateTime key="ISSUE" date="2023-01-02" time="06:00:00"/>
            </properties>
        </Run>
    """).format(work_dir=tmp_work_dir)
    path = tmp_work_dir / "run_info.xml"
    path.write_text(xml_content)
    return path


@pytest.fixture
def model_project(tmp_path):
    """A SYMFLUENCE project with standardized results, and its (stub) config.

    Returns ``(config, results_csv)``. The results cover 2023-01-01..04 hourly,
    i.e. the run period of ``sample_run_info_xml`` (2023-01-01..10) only partly.
    """
    from types import SimpleNamespace

    import pandas as pd

    data_dir = tmp_path / "symfluence_data"
    results_dir = data_dir / "domain_test_basin" / "results"
    results_dir.mkdir(parents=True)
    times = pd.date_range("2023-01-01", "2023-01-04", freq="h")
    frame = pd.DataFrame(
        {"GR_discharge_cms": np.linspace(10.0, 20.0, len(times))}, index=times
    )
    frame.index.name = "datetime"
    results_csv = results_dir / "fews_run_1_results.csv"
    frame.to_csv(results_csv)
    config = SimpleNamespace(
        system=SimpleNamespace(data_dir=str(data_dir)),
        domain=SimpleNamespace(name="test_basin", experiment_id="fews_run_1"),
    )
    return config, results_csv


@pytest.fixture
def export_fews_config(tmp_work_dir):
    """FEWSConfig with the explicit export entry the post-adapter requires."""
    return FEWSConfig(
        work_dir=str(tmp_work_dir),
        data_format="netcdf-cf",
        id_map=[
            IDMapEntry(fews_id="Q.sim", symfluence_id="discharge",
                       fews_unit="m3/s", symfluence_unit="m3/s", direction="export"),
        ],
    )
