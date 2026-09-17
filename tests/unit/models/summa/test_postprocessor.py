"""Regression coverage for native SUMMA outputs alongside restart snapshots."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.summa.postprocessor import SUMMAPostProcessor


@pytest.mark.parametrize("variable", ["averageRoutedRunoff", "scalarTotalRunoff"])
@pytest.mark.parametrize("experiment", ["workshop", "restart_workshop"])
def test_native_runoff_ignores_restart_file(tmp_path, variable, experiment):
    output = tmp_path / "simulations" / experiment / "SUMMA"
    output.mkdir(parents=True)
    xr.Dataset({"scalarSWE": ("hru", [10.0])}).to_netcdf(
        output / f"{experiment}_restart_2023083123.nc"
    )
    times = pd.date_range("2023-04-01", periods=48, freq="h")
    xr.Dataset(
        {variable: (("time", "hru"), (np.arange(1, 49) * 1e-6)[:, None]),
         "HRUarea": ("hru", [1000.0])},
        coords={"time": times, "hru": [1]},
    ).to_netcdf(output / f"{experiment}_timestep.nc")
    saved = tmp_path / "results.csv"
    processor = SimpleNamespace(
        project_dir=tmp_path, experiment_id=experiment, logger=Mock(),
        resample_frequency="D", save_streamflow_to_results=Mock(return_value=saved),
    )

    result = SUMMAPostProcessor._extract_native_summa_streamflow(processor)

    assert result == saved
    series = processor.save_streamflow_to_results.call_args.args[0]
    np.testing.assert_allclose(series.to_numpy(), [0.0125, 0.0365])
    assert series.index.equals(pd.date_range("2023-04-01", periods=2, freq="D"))
    processor.logger.error.assert_not_called()


def test_restart_only_is_not_treated_as_simulation_output(tmp_path):
    output = tmp_path / "simulations" / "workshop" / "SUMMA"
    output.mkdir(parents=True)
    (output / "workshop_restart_2023083123.nc").touch()
    processor = SimpleNamespace(
        project_dir=tmp_path, experiment_id="workshop", logger=Mock(),
        save_streamflow_to_results=Mock(),
    )
    assert SUMMAPostProcessor._extract_native_summa_streamflow(processor) is None
    processor.save_streamflow_to_results.assert_not_called()
