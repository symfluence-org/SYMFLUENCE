"""
Tests for OutputFileLocator - file pattern matching for model outputs.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.modeling.output_file_locator import OutputFileLocator, get_output_file_locator


class TestOutputFileLocatorPatterns:
    """Test that glob patterns include recursive variants."""

    def test_netcdf_patterns_have_recursive_variants(self):
        """Verify NETCDF_PATTERNS include recursive **/* variants."""
        locator = OutputFileLocator()

        assert '**/*_day.nc' in locator.NETCDF_PATTERNS['daily']
        assert '**/*_daily.nc' in locator.NETCDF_PATTERNS['daily']
        assert '**/*timestep.nc' in locator.NETCDF_PATTERNS['timestep']
        assert '**/*_streamflow.nc' in locator.NETCDF_PATTERNS['streamflow']
        assert '**/*.nc' in locator.NETCDF_PATTERNS['generic']

    def test_model_patterns_have_recursive_variants(self):
        """Verify MODEL_PATTERNS include recursive variants where needed."""
        locator = OutputFileLocator()

        # RHESSys should have recursive pattern
        assert '**/*_basin.daily' in locator.MODEL_PATTERNS['RHESSys']['patterns']

        # mizuRoute should have recursive .h.*.nc pattern
        assert '**/*.h.*.nc' in locator.MODEL_PATTERNS['mizuRoute']['patterns']

        # JFUSE and CFUSE should have recursive patterns
        assert '**/*_jfuse_output.nc' in locator.MODEL_PATTERNS['JFUSE']['patterns']
        assert '**/*_cfuse_output.nc' in locator.MODEL_PATTERNS['CFUSE']['patterns']


class TestOutputFileLocatorFileSearch:
    """Test file searching functionality."""

    def test_find_output_files_nonexistent_directory(self, mock_logger):
        """Test handling of non-existent directory."""
        locator = OutputFileLocator(mock_logger)
        result = locator.find_output_files('/nonexistent/path')
        assert result == []

    def test_find_output_files_empty_directory(self, tmp_path, mock_logger):
        """Test handling of empty directory."""
        locator = OutputFileLocator(mock_logger)
        result = locator.find_output_files(tmp_path)
        assert result == []

    def test_find_output_files_with_daily_nc(self, tmp_path, mock_logger):
        """Test finding daily NetCDF files."""
        # Create a daily file
        daily_file = tmp_path / 'test_day.nc'
        daily_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_output_files(tmp_path, output_type='et')

        assert len(result) == 1
        assert result[0] == daily_file

    def test_find_output_files_recursive_search(self, tmp_path, mock_logger):
        """Test recursive file search in subdirectories."""
        # Create nested directory structure
        subdir = tmp_path / 'nested' / 'output'
        subdir.mkdir(parents=True)

        # Create file in subdirectory
        nested_file = subdir / 'test_day.nc'
        nested_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_output_files(tmp_path, output_type='et')

        assert len(result) == 1
        assert result[0] == nested_file

    def test_find_streamflow_files(self, tmp_path, mock_logger):
        """Test finding streamflow files."""
        streamflow_file = tmp_path / 'basin_streamflow.nc'
        streamflow_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_streamflow_files(tmp_path)

        assert len(result) == 1

    def test_find_model_specific_rhessys(self, tmp_path, mock_logger):
        """Test finding RHESSys-specific output files."""
        # Create RHESSys output file
        rhessys_file = tmp_path / 'test_basin.daily'
        rhessys_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_rhessys_output(tmp_path)

        assert len(result) == 1
        assert result[0] == rhessys_file

    def test_find_model_specific_hype(self, tmp_path, mock_logger):
        """Test finding HYPE-specific output files."""
        # Create HYPE output file
        hype_file = tmp_path / 'timeCOUT.txt'
        hype_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_hype_output(tmp_path)

        assert len(result) == 1
        assert result[0] == hype_file

    def test_find_mizuroute_output(self, tmp_path, mock_logger):
        """Test finding mizuRoute output files."""
        # Create mizuRoute directory and file
        mizuroute_dir = tmp_path / 'mizuRoute'
        mizuroute_dir.mkdir()
        mizuroute_file = mizuroute_dir / 'output.nc'
        mizuroute_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator._find_mizuroute_output(tmp_path)

        assert len(result) == 1


class TestOutputFileLocatorConvenienceMethods:
    """Test convenience methods for specific output types."""

    def test_find_et_files(self, tmp_path, mock_logger):
        """Test find_et_files convenience method."""
        et_file = tmp_path / 'et_daily.nc'
        et_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_et_files(tmp_path)

        assert len(result) == 1

    def test_find_snow_files(self, tmp_path, mock_logger):
        """Test find_snow_files convenience method."""
        snow_file = tmp_path / 'snow_day.nc'
        snow_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_snow_files(tmp_path)

        assert len(result) == 1

    def test_find_soil_moisture_files(self, tmp_path, mock_logger):
        """Test find_soil_moisture_files convenience method."""
        sm_file = tmp_path / 'sm_daily.nc'
        sm_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_soil_moisture_files(tmp_path)

        assert len(result) == 1

    def test_find_tws_files(self, tmp_path, mock_logger):
        """Test find_tws_files convenience method."""
        tws_file = tmp_path / 'tws_day.nc'
        tws_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_tws_files(tmp_path)

        assert len(result) == 1

    def test_find_groundwater_files(self, tmp_path, mock_logger):
        """Test find_groundwater_files convenience method."""
        gw_file = tmp_path / 'gw_daily.nc'
        gw_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.find_groundwater_files(tmp_path)

        assert len(result) == 1

    def test_get_most_recent(self, tmp_path, mock_logger):
        """Test get_most_recent returns most recently modified file."""
        import time

        # Create files with different modification times
        old_file = tmp_path / 'old_day.nc'
        old_file.touch()
        time.sleep(0.1)

        new_file = tmp_path / 'new_day.nc'
        new_file.touch()

        locator = OutputFileLocator(mock_logger)
        result = locator.get_most_recent(tmp_path, 'tws')

        assert result == new_file


class TestModuleLevelInstance:
    """Test module-level convenience functions."""

    def test_get_output_file_locator_returns_instance(self):
        """Test that get_output_file_locator returns an OutputFileLocator instance."""
        locator = get_output_file_locator()
        assert isinstance(locator, OutputFileLocator)

    def test_get_output_file_locator_caches_instance(self):
        """Test that subsequent calls return the same instance."""
        locator1 = get_output_file_locator()
        locator2 = get_output_file_locator()
        assert locator1 is locator2
