"""
Unit tests for WorkflowDiagnosticPlotter.

Tests diagnostic plotting methods for workflow step validation.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from symfluence.reporting.plotters.workflow_diagnostic_plotter import WorkflowDiagnosticPlotter


@pytest.fixture
def diagnostic_plotter(mock_config, mock_logger, mock_plot_config):
    """Create a WorkflowDiagnosticPlotter instance."""
    return WorkflowDiagnosticPlotter(mock_config, mock_logger, mock_plot_config)


@pytest.fixture
def sample_basin_gdf():
    """Create a mock GeoDataFrame for basin."""
    gdf = Mock()
    gdf.geometry = Mock()
    # Create a proper Series-like mock with sum() method
    area_series = Mock()
    area_series.sum = Mock(return_value=1e12)  # 1 million km² in m²
    gdf.geometry.area = area_series
    gdf.crs = Mock()
    gdf.crs.is_projected = True
    gdf.plot = Mock()
    gdf.to_crs = Mock(return_value=gdf)
    gdf.__len__ = Mock(return_value=1)
    return gdf


@pytest.fixture
def sample_hru_gdf():
    """Create a mock GeoDataFrame for HRUs."""
    gdf = Mock()
    gdf.geometry = Mock()
    gdf.geometry.area = pd.Series([1e8, 2e8, 1.5e8, 3e8])  # Various areas in m²
    gdf.crs = Mock()
    gdf.crs.is_projected = True
    gdf.plot = Mock()

    # to_crs should return a gdf with the same area series
    projected_gdf = Mock()
    projected_gdf.geometry = Mock()
    projected_gdf.geometry.area = pd.Series([1e8, 2e8, 1.5e8, 3e8])
    gdf.to_crs = Mock(return_value=projected_gdf)

    gdf.columns = ['HRU_ID', 'elevClass', 'geometry']
    gdf.__len__ = Mock(return_value=4)
    gdf.index = range(4)
    gdf.__getitem__ = Mock(return_value=pd.Series([1, 1, 2, 2]))
    return gdf


@pytest.fixture
def sample_obs_df():
    """Create sample observation DataFrame."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    np.random.seed(42)
    values = np.random.uniform(5, 50, len(dates))
    # Add some gaps
    values[50:55] = np.nan
    values[200:210] = np.nan
    df = pd.DataFrame({
        'datetime': dates,
        'discharge': values
    }).set_index('datetime')
    yield df
    # Cleanup
    del df, dates, values


@pytest.fixture
def sample_optimization_history():
    """Create sample optimization history."""
    return [
        {'iteration': 0, 'objective': 0.5},
        {'iteration': 1, 'objective': 0.55},
        {'iteration': 2, 'objective': 0.6},
        {'iteration': 3, 'objective': 0.65},
        {'iteration': 4, 'objective': 0.7},
        {'iteration': 5, 'objective': 0.72},
    ]


@pytest.fixture
def sample_best_params():
    """Create sample best parameters."""
    return {
        'k_macropore': 0.5,
        'theta_res': 0.05,
        'vGn_alpha': 0.02,
        'vGn_n': 1.5
    }


class TestWorkflowDiagnosticPlotter:
    """Test suite for WorkflowDiagnosticPlotter."""

    def test_initialization(self, diagnostic_plotter):
        """Test that WorkflowDiagnosticPlotter initializes correctly."""
        assert diagnostic_plotter.config is not None
        assert diagnostic_plotter.logger is not None
        assert diagnostic_plotter.plot_config is not None

    def test_ensure_diagnostic_dir(self, diagnostic_plotter):
        """Test diagnostic directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            diag_dir = diagnostic_plotter._ensure_diagnostic_dir('test_step')

            assert diag_dir.exists()
            assert 'workflow_diagnostics' in str(diag_dir)
            assert 'test_step' in str(diag_dir)

    def test_get_timestamp(self, diagnostic_plotter):
        """Test timestamp generation."""
        timestamp = diagnostic_plotter._get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # Format: YYYYMMDD_HHMMSS


class TestDomainDefinitionDiagnostics:
    """Test domain definition diagnostic methods."""

    def test_plot_domain_definition_diagnostic_success(
        self, diagnostic_plotter, sample_basin_gdf
    ):
        """Test successful domain definition diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.colorbar = Mock()
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_domain_definition_diagnostic(
                    basin_gdf=sample_basin_gdf,
                    dem_path=None
                )

                assert result is not None

    def test_plot_domain_definition_diagnostic_with_dem(
        self, diagnostic_plotter, sample_basin_gdf
    ):
        """Test domain definition diagnostic with DEM file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            dem_path = Path(tmpdir) / 'dem.tif'

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')), \
                 patch('rasterio.open') as mock_rasterio:

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.colorbar = Mock()
                mock_setup.return_value = (mock_plt, None)

                # Mock rasterio
                mock_src = Mock()
                mock_src.read.return_value = np.random.uniform(100, 3000, (100, 100))
                mock_src.nodata = -9999
                mock_rasterio.return_value.__enter__ = Mock(return_value=mock_src)
                mock_rasterio.return_value.__exit__ = Mock(return_value=False)

                # Create the DEM file (empty, since we mock rasterio)
                dem_path.touch()

                result = diagnostic_plotter.plot_domain_definition_diagnostic(
                    basin_gdf=sample_basin_gdf,
                    dem_path=dem_path
                )

                assert result is not None

    def test_plot_domain_definition_diagnostic_error_handling(
        self, diagnostic_plotter, mock_logger
    ):
        """Test error handling in domain definition diagnostic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib', side_effect=Exception("Test error")):
                result = diagnostic_plotter.plot_domain_definition_diagnostic(
                    basin_gdf=None,
                    dem_path=None
                )

                assert result is None
                mock_logger.error.assert_called()


class TestDiscretizationDiagnostics:
    """Test discretization diagnostic methods."""

    def test_plot_discretization_diagnostic_success(
        self, diagnostic_plotter, sample_hru_gdf
    ):
        """Test successful discretization diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_discretization_diagnostic(
                    hru_gdf=sample_hru_gdf,
                    method='elevation'
                )

                assert result is not None

    def test_plot_discretization_diagnostic_error_handling(
        self, diagnostic_plotter, mock_logger
    ):
        """Test error handling in discretization diagnostic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib', side_effect=Exception("Test error")):
                result = diagnostic_plotter.plot_discretization_diagnostic(
                    hru_gdf=None,
                    method='elevation'
                )

                assert result is None
                mock_logger.error.assert_called()


class TestObservationsDiagnostics:
    """Test observation diagnostic methods."""

    def test_plot_observations_diagnostic_success(
        self, diagnostic_plotter, sample_obs_df
    ):
        """Test successful observation diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.colorbar = Mock()
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_observations_diagnostic(
                    obs_df=sample_obs_df,
                    obs_type='streamflow'
                )

                assert result is not None

    def test_plot_observations_diagnostic_with_gaps(
        self, diagnostic_plotter, sample_obs_df
    ):
        """Test observation diagnostic with data gaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            # Add more gaps (on a copy to avoid modifying the fixture)
            test_df = sample_obs_df.copy()
            test_df.iloc[100:120] = np.nan

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.colorbar = Mock()
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_observations_diagnostic(
                    obs_df=test_df,
                    obs_type='streamflow'
                )

                assert result is not None

            # Explicit cleanup
            del test_df


class TestForcingDiagnostics:
    """Test forcing diagnostic methods."""

    def test_plot_forcing_raw_diagnostic_success(self, diagnostic_plotter):
        """Test successful raw forcing diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            forcing_nc = Path(tmpdir) / 'forcing.nc'
            forcing_nc.touch()

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')), \
                 patch('xarray.open_dataset') as mock_xr:

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                # Mock xarray dataset
                mock_ds = Mock()
                mock_ds.data_vars = ['precipitation_flux', 'air_temperature']
                mock_var = Mock()
                mock_var.dims = ['time', 'lat', 'lon']
                mock_var.values = np.random.random((10, 5, 5))
                mock_var.isel.return_value = mock_var
                mock_var.plot = Mock()
                mock_ds.__getitem__ = Mock(return_value=mock_var)
                mock_ds.dims = {'time': 10}
                mock_ds.__iter__ = Mock(return_value=iter(['precipitation_flux', 'air_temperature']))
                mock_ds.close = Mock()
                mock_xr.return_value = mock_ds

                result = diagnostic_plotter.plot_forcing_raw_diagnostic(
                    forcing_nc=forcing_nc,
                    domain_shp=None
                )

                assert result is not None

    def test_plot_forcing_remapped_diagnostic_success(self, diagnostic_plotter):
        """Test successful remapped forcing diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            raw_nc = Path(tmpdir) / 'raw.nc'
            remapped_nc = Path(tmpdir) / 'remapped.nc'
            raw_nc.touch()
            remapped_nc.touch()

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')), \
                 patch('xarray.open_dataset') as mock_xr:

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                # Create separate mock datasets for raw and remapped to avoid conflicts
                def create_mock_dataset():
                    mock_ds = Mock()
                    mock_ds.data_vars = ['precipitation_flux']

                    # Create a mock variable with all necessary attributes
                    mock_var = Mock()
                    mock_var.dims = ['time', 'hru']
                    mock_var.values = np.random.random((10, 5))

                    # Mock for isel result (selecting a time slice)
                    mock_isel_result = Mock()
                    mock_isel_result.dims = ['hru']
                    mock_isel_result.values = np.random.random(5)
                    mock_isel_result.plot = Mock(return_value=None)  # plot() should return None

                    # Create a separate mock for mean result to avoid circular refs
                    mock_mean_result = Mock()
                    mock_mean_result.dims = ['hru']
                    mock_mean_result.values = np.random.random(5)
                    mock_mean_result.plot = Mock(return_value=None)  # plot() should return None

                    # Mock for sum result
                    mock_sum_result = Mock()
                    mock_sum_result.values = 1000.0

                    mock_var.isel = Mock(return_value=mock_isel_result)
                    mock_var.sum = Mock(return_value=mock_sum_result)
                    mock_var.mean = Mock(return_value=mock_mean_result)
                    mock_var.plot = Mock(return_value=None)  # plot() should return None

                    # __getitem__ should return the appropriate mock based on check
                    mock_ds.__getitem__ = Mock(return_value=mock_var)
                    mock_ds.__contains__ = Mock(return_value=True)  # for 'var_name in ds' checks
                    mock_ds.close = Mock()
                    return mock_ds

                # Return a new mock dataset for each call
                mock_xr.side_effect = [create_mock_dataset(), create_mock_dataset()]

                result = diagnostic_plotter.plot_forcing_remapped_diagnostic(
                    raw_nc=raw_nc,
                    remapped_nc=remapped_nc,
                    hru_shp=None
                )

                assert result is not None


class TestModelPreprocessingDiagnostics:
    """Test model preprocessing diagnostic methods."""

    def test_plot_model_preprocessing_diagnostic_success(self, diagnostic_plotter):
        """Test successful model preprocessing diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            input_dir = Path(tmpdir) / 'SUMMA_input'
            input_dir.mkdir()

            # Create some test files
            (input_dir / 'forcing.nc').touch()
            (input_dir / 'attributes.nc').touch()
            (input_dir / 'config.txt').touch()

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_axes[0].pie = Mock()
                mock_axes[0].text = Mock()
                mock_axes[1].barh = Mock()
                mock_axes[1].set_yticks = Mock()
                mock_axes[1].set_yticklabels = Mock()
                mock_axes[2].axis = Mock()
                mock_axes[2].text = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.cm = Mock()
                mock_plt.cm.Set3 = Mock(return_value=[(0.5, 0.5, 0.5, 1.0)] * 10)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_model_preprocessing_diagnostic(
                    input_dir=input_dir,
                    model_name='SUMMA'
                )

                assert result is not None

    def test_plot_model_preprocessing_diagnostic_empty_dir(self, diagnostic_plotter):
        """Test preprocessing diagnostic with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            input_dir = Path(tmpdir) / 'empty_input'
            input_dir.mkdir()

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_model_preprocessing_diagnostic(
                    input_dir=input_dir,
                    model_name='SUMMA'
                )

                # Should still return a path (showing empty state)
                assert result is not None


class TestModelOutputDiagnostics:
    """Test model output diagnostic methods."""

    def test_plot_model_output_diagnostic_success(self, diagnostic_plotter):
        """Test successful model output diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            output_nc = Path(tmpdir) / 'output.nc'
            output_nc.touch()

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')), \
                 patch('xarray.open_dataset') as mock_xr:

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                # Mock xarray dataset
                mock_ds = Mock()
                mock_ds.data_vars = ['scalarTotalRunoff', 'scalarSWE']
                mock_var = MagicMock()
                mock_var.values = np.random.random((100,))
                mock_var.dims = ['time']
                mock_var.mean.return_value = mock_var
                mock_var.plot = Mock()
                mock_time_coord = MagicMock()
                mock_time_coord.values = np.arange(100)
                mock_var.__getitem__ = Mock(return_value=mock_time_coord)
                mock_ds.__getitem__ = Mock(return_value=mock_var)
                mock_ds.close = Mock()
                mock_xr.return_value = mock_ds

                result = diagnostic_plotter.plot_model_output_diagnostic(
                    output_nc=output_nc,
                    model_name='SUMMA'
                )

                assert result is not None


class TestAttributesDiagnostics:
    """Test attributes diagnostic methods."""

    def test_plot_attributes_diagnostic_success(self, diagnostic_plotter):
        """Test successful attributes diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)
            dem_path = Path(tmpdir) / 'dem.tif'
            dem_path.touch()

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')), \
                 patch('rasterio.open') as mock_rasterio:

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.colorbar = Mock()
                mock_plt.cm = Mock()
                mock_plt.cm.Set3 = Mock(return_value=[(0.5, 0.5, 0.5, 1.0)] * 15)
                mock_setup.return_value = (mock_plt, None)

                # Mock rasterio
                mock_src = Mock()
                mock_src.read.return_value = np.random.uniform(100, 3000, (100, 100))
                mock_src.nodata = -9999
                mock_rasterio.return_value.__enter__ = Mock(return_value=mock_src)
                mock_rasterio.return_value.__exit__ = Mock(return_value=False)

                result = diagnostic_plotter.plot_attributes_diagnostic(
                    dem_path=dem_path,
                    soil_path=None,
                    land_path=None
                )

                assert result is not None

    def test_plot_attributes_diagnostic_all_none(self, diagnostic_plotter):
        """Test attributes diagnostic with no paths provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_attributes_diagnostic(
                    dem_path=None,
                    soil_path=None,
                    land_path=None
                )

                # Should still return a path (showing unavailable state)
                assert result is not None


class TestCalibrationDiagnostics:
    """Test calibration diagnostic methods."""

    def test_plot_calibration_diagnostic_success(
        self, diagnostic_plotter, sample_optimization_history, sample_best_params
    ):
        """Test successful calibration diagnostic plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.cm = Mock()
                mock_plt.cm.viridis = Mock(return_value=[(0.5, 0.5, 0.5, 1.0)] * 10)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_calibration_diagnostic(
                    history=sample_optimization_history,
                    best_params=sample_best_params,
                    obs_vs_sim=None,
                    model_name='SUMMA'
                )

                assert result is not None

    def test_plot_calibration_diagnostic_with_obs_vs_sim(
        self, diagnostic_plotter, sample_optimization_history, sample_best_params
    ):
        """Test calibration diagnostic with obs vs sim data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            # Create obs_vs_sim data
            np.random.seed(42)
            observed = np.random.uniform(5, 50, 100)
            simulated = observed * 0.95 + np.random.normal(0, 2, 100)
            obs_vs_sim = {
                'observed': observed,
                'simulated': simulated
            }

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.cm = Mock()
                mock_plt.cm.viridis = Mock(return_value=[(0.5, 0.5, 0.5, 1.0)] * 10)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_calibration_diagnostic(
                    history=sample_optimization_history,
                    best_params=sample_best_params,
                    obs_vs_sim=obs_vs_sim,
                    model_name='SUMMA'
                )

                assert result is not None

    def test_plot_calibration_diagnostic_empty_history(
        self, diagnostic_plotter, sample_best_params
    ):
        """Test calibration diagnostic with empty history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(diagnostic_plotter, '_save_and_close', return_value=str(Path(tmpdir) / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_plt.cm = Mock()
                mock_plt.cm.viridis = Mock(return_value=[(0.5, 0.5, 0.5, 1.0)] * 10)
                mock_setup.return_value = (mock_plt, None)

                result = diagnostic_plotter.plot_calibration_diagnostic(
                    history=None,
                    best_params=sample_best_params,
                    obs_vs_sim=None,
                    model_name='SUMMA'
                )

                # Should still produce a plot (showing "no history" message)
                assert result is not None

    def test_plot_calibration_diagnostic_error_handling(
        self, diagnostic_plotter, mock_logger
    ):
        """Test error handling in calibration diagnostic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_plotter.project_dir = Path(tmpdir)

            with patch.object(diagnostic_plotter, '_setup_matplotlib', side_effect=Exception("Test error")):
                result = diagnostic_plotter.plot_calibration_diagnostic(
                    history=None,
                    best_params=None,
                    obs_vs_sim=None,
                    model_name='SUMMA'
                )

                assert result is None
                mock_logger.error.assert_called()


class TestPlotMethodRequired:
    """Test the required plot() method from BasePlotter."""

    def test_plot_method_exists(self, diagnostic_plotter):
        """Test that plot() method exists and is callable."""
        assert hasattr(diagnostic_plotter, 'plot')
        assert callable(diagnostic_plotter.plot)

    def test_plot_method_returns_none(self, diagnostic_plotter):
        """Test that plot() returns None (delegates to specific methods)."""
        result = diagnostic_plotter.plot()
        assert result is None
