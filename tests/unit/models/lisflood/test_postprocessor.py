# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD model postprocessor."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.lisflood.postprocessor import LisfloodPostProcessor


class TestPostprocessorImport:
    """Basic import tests."""

    def test_postprocessor_can_be_imported(self):
        assert LisfloodPostProcessor is not None

    def test_model_name(self):
        assert LisfloodPostProcessor.model_name == "LISFLOOD"

    def test_streamflow_variable(self):
        assert LisfloodPostProcessor.streamflow_variable == "dis"

    def test_streamflow_unit(self):
        assert LisfloodPostProcessor.streamflow_unit == "cms"


class TestPostprocessorNetCDFOutput:
    """Tests for reading NetCDF discharge output."""

    def test_reads_netcdf_discharge(self, tmp_path, lisflood_config, mock_logger, setup_lisflood_directories):
        """Create a sample NetCDF discharge file and verify extraction."""
        output_dir = setup_lisflood_directories["simulations_dir"]

        # Create a minimal discharge NetCDF
        times = pd.date_range("2005-01-01", "2005-03-01", freq="D")
        dis_data = np.random.uniform(0.5, 10.0, (len(times), 3, 3))
        ds = xr.Dataset(
            {
                "dis": xr.DataArray(
                    dis_data,
                    dims=["time", "y", "x"],
                    coords={"time": times, "y": [2, 1, 0], "x": [0, 1, 2]},
                ),
            }
        )
        ds.to_netcdf(output_dir / "dis.nc")

        pp = LisfloodPostProcessor(lisflood_config, mock_logger)

        # Patch _get_output_dir to point to our test output
        with patch.object(pp, "_get_output_dir", return_value=output_dir):
            with patch.object(pp, "save_streamflow_to_results", return_value=output_dir / "results.csv") as mock_save:
                result = pp.extract_streamflow()

                assert mock_save.called
                saved_series = mock_save.call_args[0][0]
                assert len(saved_series) == len(times)


class TestPostprocessorTSSOutput:
    """Tests for reading TSS text file output."""

    def test_read_tss_file_parses_correctly(self, tmp_path):
        """Test _read_tss_file with a sample TSS file."""
        tss_content = """timeseries noheader
2
station 1
1   0.500
2   1.200
3   2.800
4   3.100
5   4.500
"""
        tss_path = tmp_path / "dis.tss"
        tss_path.write_text(tss_content)

        # Create a minimal postprocessor to call _read_tss_file
        # The method needs _get_simulation_dates and _get_config_value
        pp_mock = Mock(spec=LisfloodPostProcessor)
        pp_mock._read_tss_file = LisfloodPostProcessor._read_tss_file.__get__(pp_mock)

        # Mock the methods it calls
        import datetime

        pp_mock._get_simulation_dates.return_value = (
            datetime.datetime(2005, 1, 1),
            datetime.datetime(2005, 1, 6),
        )
        pp_mock._get_config_value.return_value = 86400

        result = pp_mock._read_tss_file(tss_path)

        assert isinstance(result, pd.Series)
        # Header lines ("2", "station 1") are not data rows
        assert len(result) == 5
        assert result.iloc[-1] == pytest.approx(4.5)
        assert result.index[0] == pd.Timestamp("2005-01-01")
        assert result.index[-1] == pd.Timestamp("2005-01-05")

    def test_read_tss_file_handles_header_lines(self, tmp_path):
        """Test TSS parsing with various header formats."""
        tss_content = """timeseries
1
station_name
1   10.0
2   20.0
3   30.0
"""
        tss_path = tmp_path / "test.tss"
        tss_path.write_text(tss_content)

        pp_mock = Mock(spec=LisfloodPostProcessor)
        pp_mock._read_tss_file = LisfloodPostProcessor._read_tss_file.__get__(pp_mock)

        import datetime

        pp_mock._get_simulation_dates.return_value = (
            datetime.datetime(2005, 1, 1),
            datetime.datetime(2005, 1, 3),
        )
        pp_mock._get_config_value.return_value = 86400

        result = pp_mock._read_tss_file(tss_path)
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(30.0)


class TestPostprocessorFallback:
    """Tests for output fallback behavior."""

    def test_returns_none_when_no_output(self, lisflood_config, mock_logger, setup_lisflood_directories):
        """When no discharge files exist, extract_streamflow returns None."""
        output_dir = setup_lisflood_directories["simulations_dir"]
        pp = LisfloodPostProcessor(lisflood_config, mock_logger)

        with patch.object(pp, "_get_output_dir", return_value=output_dir):
            result = pp.extract_streamflow()
            assert result is None
