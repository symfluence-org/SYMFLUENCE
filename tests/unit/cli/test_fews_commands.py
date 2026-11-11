# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""`symfluence fews` command handlers: the model is really run with the FEWS
overrides, a missing --config is an error, and `launch` does not pretend."""
from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.cli.commands import fews_commands
from symfluence.cli.commands.fews_commands import FEWSCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli]

NS = {"pi": "http://www.wldelft.nl/fews/PI"}


@pytest.fixture
def fews_run(tmp_path):
    """A FEWS work dir (real PI run file), a base config and an id map."""
    work = tmp_path / "work"
    (work / "toModel").mkdir(parents=True)
    data_dir = tmp_path / "data"
    run_info = work / "run_info.xml"
    run_info.write_text(textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Run xmlns="http://www.wldelft.nl/fews/PI" version="1.5">
            <timeZone>0.0</timeZone>
            <startDateTime date="2023-01-01" time="00:00:00"/>
            <endDateTime date="2023-01-02" time="00:00:00"/>
            <time0 date="2023-01-01" time="12:00:00"/>
            <workDir>{work}</workDir>
            <inputNetcdfFile>{work}/toModel/forcing.nc</inputNetcdfFile>
            <outputDiagnosticFile>{work}/toFews/diag.xml</outputDiagnosticFile>
            <outputNetcdfFile>{work}/toFews/output.nc</outputNetcdfFile>
        </Run>
    """))
    times = pd.date_range("2023-01-01", "2023-01-02", freq="h")
    xr.Dataset({"T.obs": ("time", np.full(len(times), 280.0), {"units": "K"})},
               coords={"time": times}).to_netcdf(work / "toModel" / "forcing.nc")

    config = tmp_path / "config.yaml"
    config.write_text(textwrap.dedent(f"""\
        SYMFLUENCE_DATA_DIR: {data_dir}
        SYMFLUENCE_CODE_DIR: {tmp_path / 'code'}
        DOMAIN_NAME: fews_basin
        EXPERIMENT_ID: run_1
        EXPERIMENT_TIME_START: "2010-01-01 00:00"
        EXPERIMENT_TIME_END: "2010-12-31 23:00"
        DOMAIN_DEFINITION_METHOD: lumped
        SUB_GRID_DISCRETIZATION: lumped
        HYDROLOGICAL_MODEL: SUMMA
        FORCING_DATASET: ERA5
    """))
    id_map = tmp_path / "id_map.yaml"
    id_map.write_text(
        "- {fews_id: Q.sim, symfluence_id: discharge, direction: export}\n")

    def write_results():
        results = data_dir / "domain_fews_basin" / "results"
        results.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"SUMMA_discharge_cms": np.arange(len(times), dtype=float)},
                             index=times)
        frame.index.name = "datetime"
        frame.to_csv(results / "run_1_results.csv")

    return Namespace(work=work, run_info=run_info, config=config, id_map=id_map,
                     write_results=write_results)


def _args(fews_run, **overrides):
    values = dict(run_info=str(fews_run.run_info), config=str(fews_run.config),
                  format="netcdf-cf", id_map=str(fews_run.id_map), debug=False)
    values.update(overrides)
    return Namespace(**values)


def _diag_lines(fews_run):
    root = ET.parse(str(fews_run.work / "toFews" / "diag.xml")).getroot()
    return [(ln.get("level"), ln.get("description")) for ln in root.findall("pi:line", NS)]


def test_run_passes_fews_overrides_to_the_model_and_exports_its_results(fews_run):
    calls = []

    def fake_run_model(config_path, overrides, debug=False):
        calls.append((config_path, dict(overrides)))
        fews_run.write_results()
        return []

    with patch.object(fews_commands, "_run_model", fake_run_model):
        rc = FEWSCommands.run_full(_args(fews_run))

    assert rc == ExitCode.SUCCESS
    (config_path, overrides), = calls
    assert config_path == fews_run.config
    # the FEWS period, not the base config's 2010
    assert overrides["EXPERIMENT_TIME_START"] == "2023-01-01 00:00"
    assert overrides["EXPERIMENT_TIME_END"] == "2023-01-02 00:00"
    with xr.open_dataset(fews_run.work / "toFews" / "output.nc") as out:
        assert list(out.data_vars) == ["Q.sim"]
        assert out.sizes["time"] == 25
    lines = _diag_lines(fews_run)
    assert not any(level == "0" for level, _ in lines)
    # the user is told the FEWS forcing does not drive the model
    assert any(level == "2" and "not injected" in text for level, text in lines)


def test_run_without_config_is_an_error_and_runs_nothing(fews_run):
    with patch.object(fews_commands, "_run_model") as run_model:
        rc = FEWSCommands.run_full(_args(fews_run, config=None))
    assert rc == ExitCode.USAGE_ERROR
    run_model.assert_not_called()
    assert not (fews_run.work / "toFews" / "output.nc").exists()
    assert any(level == "0" and "--config" in text for level, text in _diag_lines(fews_run))


def test_run_fails_when_the_model_produced_no_output(fews_run):
    """Used to export the pre-adapter's own forcing and exit 0."""
    with patch.object(fews_commands, "_run_model", lambda *a, **k: []):
        rc = FEWSCommands.run_full(_args(fews_run))
    assert rc != ExitCode.SUCCESS
    assert not (fews_run.work / "toFews" / "output.nc").exists()
    assert any(level == "0" and "No model output found" in text
               for level, text in _diag_lines(fews_run))


def test_run_reports_failed_workflow_steps(fews_run):
    with patch.object(fews_commands, "_run_model", lambda *a, **k: ["run_model"]):
        rc = FEWSCommands.run_full(_args(fews_run))
    assert rc == ExitCode.WORKFLOW_ERROR
    assert any(level == "0" and "run_model" in text for level, text in _diag_lines(fews_run))


def test_launch_fails_clearly_without_a_fews_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("FEWS_HOME", str(tmp_path / "nothing-here"))
    monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path / "data"))
    with patch("subprocess.run") as run, \
            patch.object(FEWSCommands, "_console") as console:
        rc = FEWSCommands.launch(Namespace(config=None, port=8080))
    assert rc == ExitCode.BINARY_ERROR
    run.assert_not_called()
    assert "obtained from Deltares" in console.error.call_args.args[0]


def test_launch_ignores_the_generated_stub(tmp_path, monkeypatch):
    """The directory `binary install openfews` creates (launcher + template,
    no FEWS jars) is not a FEWS installation."""
    home = tmp_path / "openfews"
    (home / "bin").mkdir(parents=True)
    (home / "lib").mkdir()
    (home / "bin" / "fews.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("FEWS_HOME", str(home))
    monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path / "data"))
    with patch("subprocess.run") as run:
        rc = FEWSCommands.launch(Namespace(config=None, port=8080))
    assert rc == ExitCode.BINARY_ERROR
    run.assert_not_called()


def test_adapter_template_passes_config_and_install_text_is_honest():
    from symfluence.cli.external_tools_build_commands import OPENFEWS_BUILD_COMMAND

    assert "fews run --run-info %ROOT_DIR%/run_info.xml --config " in OPENFEWS_BUILD_COMMAND
    assert "FEWS_URL=" not in OPENFEWS_BUILD_COMMAND
    assert "is not installed by this command" in OPENFEWS_BUILD_COMMAND
    # the embedded template is well-formed XML
    start = OPENFEWS_BUILD_COMMAND.index("<?xml")
    end = OPENFEWS_BUILD_COMMAND.index("ADAPTER_EOF", start)
    ET.fromstring(OPENFEWS_BUILD_COMMAND[start:end].encode())
