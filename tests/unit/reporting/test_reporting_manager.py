from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.reporting.reporting_manager import ReportingManager


@pytest.fixture
def mock_data_processor():
    with patch('symfluence.reporting.processors.data_processor.DataProcessor') as mock:
        yield mock

@pytest.fixture
def mock_spatial_processor():
    with patch('symfluence.reporting.processors.spatial_processor.SpatialProcessor') as mock:
        yield mock

@pytest.fixture
def mock_config(tmp_path):
    """Create a minimal SymfluenceConfig for testing.

    Uses a per-test data dir: a fixed /tmp path can collide with a real
    directory or symlink on a developer machine (e.g. /tmp/SYMFLUENCE_data on
    a case-insensitive filesystem), which made test_init fail locally.
    """
    return SymfluenceConfig.from_minimal(
        domain_name='test_domain',
        model='summa',
        SYMFLUENCE_DATA_DIR=str(tmp_path / 'symfluence_data'),
        EXPERIMENT_TIME_START='2020-01-01 00:00',
        EXPERIMENT_TIME_END='2020-12-31 23:00'
    )

@pytest.fixture
def mock_domain_plotter():
    with patch('symfluence.reporting.plotters.domain_plotter.DomainPlotter') as mock:
        yield mock

@pytest.fixture
def mock_optimization_plotter():
    with patch('symfluence.reporting.plotters.optimization_plotter.OptimizationPlotter') as mock:
        yield mock

@pytest.fixture
def mock_analysis_plotter():
    with patch('symfluence.reporting.plotters.analysis_plotter.AnalysisPlotter') as mock:
        yield mock

@pytest.fixture
def mock_benchmark_plotter():
    with patch('symfluence.reporting.plotters.benchmark_plotter.BenchmarkPlotter') as mock:
        yield mock

@pytest.fixture
def mock_snow_plotter():
    with patch('symfluence.reporting.plotters.snow_plotter.SnowPlotter') as mock:
        yield mock

class TestReportingManager:

    def test_init(self, mock_config, mock_logger):
        """Test initialization of ReportingManager."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)

        assert manager.visualize is True
        # Use endswith to avoid platform-specific symlink issues (e.g., /tmp -> /private/tmp on macOS)
        assert manager.project_dir.as_posix().endswith('symfluence_data/domain_test_domain')

        # With cached_property, components are lazy-loaded and don't have _attributes
        # Verify that accessing the properties works and creates them lazily
        assert manager.plot_config is not None

    def test_visualization_disabled(self, mock_config, mock_logger, mock_domain_plotter):
        """Test that methods return None/Empty when visualization is disabled."""
        manager = ReportingManager(mock_config, mock_logger, visualize=False)

        # Test various methods
        assert manager.visualize_domain() is None
        assert manager.visualize_discretized_domain('elevation') is None
        assert manager.visualize_model_outputs([], []) is None
        assert manager.visualize_lumped_model_outputs([], []) is None
        assert manager.visualize_fuse_outputs([], []) is None
        assert manager.visualize_benchmarks({}) == []
        assert manager.visualize_snow_comparison([]) == {}

        # Ensure underlying plotter was NOT instantiated when viz is disabled
        mock_domain_plotter.assert_not_called()

    def test_visualization_enabled(self, mock_config, mock_logger, mock_domain_plotter):
        """Test that methods call underlying reporters when visualization is enabled."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        domain_instance = mock_domain_plotter.return_value

        # Setup return values
        domain_instance.plot_domain.return_value = "plot_path.png"

        # Test visualize_domain - this should trigger lazy loading
        result = manager.visualize_domain()
        assert result == "plot_path.png"
        domain_instance.plot_domain.assert_called_once()

        # Test visualize_discretized_domain
        manager.visualize_discretized_domain('landclass')
        domain_instance.plot_discretized_domain.assert_called_once_with('landclass')

    def test_model_output_visualization(self, mock_config, mock_logger, mock_analysis_plotter):
        """Test model output visualization delegation."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        analysis_instance = mock_analysis_plotter.return_value

        models = [('model', 'path')]
        obs = [('obs', 'path')]

        # Standard outputs
        manager.visualize_model_outputs(models, obs)
        analysis_instance.plot_streamflow_comparison.assert_called_once_with(models, obs)

        # Lumped outputs
        manager.visualize_lumped_model_outputs(models, obs)
        analysis_instance.plot_streamflow_comparison.assert_called_with(models, obs, lumped=True)

        # FUSE outputs
        manager.visualize_fuse_outputs(models, obs)
        analysis_instance.plot_fuse_streamflow.assert_called_once_with(models, obs)

    @patch.object(ReportingManager, 'data_processor', new_callable=PropertyMock)
    @patch.object(ReportingManager, 'analysis_plotter', new_callable=PropertyMock)
    def test_timeseries_results_visualization(self, mock_analysis_plotter_prop, mock_data_processor_prop, mock_config, mock_logger):
        """Test timeseries visualization delegation."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)

        # Configure the return value of the mocked properties
        mock_data_processor_prop.return_value = MagicMock()
        mock_analysis_plotter_prop.return_value = MagicMock()

        mock_df = MagicMock()
        mock_data_processor_prop.return_value.read_results_file.return_value = mock_df

        manager.visualize_timeseries_results()

        # Assert that the mocked methods were called
        mock_data_processor_prop.return_value.read_results_file.assert_called_once()
        mock_analysis_plotter_prop.return_value.plot_timeseries_results.assert_called_once()
        mock_analysis_plotter_prop.return_value.plot_diagnostics.assert_called_once()

    @patch.object(ReportingManager, 'spatial_processor', new_callable=PropertyMock)
    def test_update_sim_reach_id_always_runs(self, mock_spatial_processor_prop, mock_config, mock_logger):
        """Test that update_sim_reach_id runs regardless of visualization flag."""

        mock_spatial_processor_prop.return_value = MagicMock()

        # Case 1: visualize=False
        manager_false = ReportingManager(mock_config, mock_logger, visualize=False)
        manager_false.update_sim_reach_id("config.yaml")
        mock_spatial_processor_prop.return_value.update_sim_reach_id.assert_called_with("config.yaml")

        mock_spatial_processor_prop.return_value.reset_mock()

        # Case 2: visualize=True
        manager_true = ReportingManager(mock_config, mock_logger, visualize=True)
        manager_true.update_sim_reach_id("config.yaml")
        mock_spatial_processor_prop.return_value.update_sim_reach_id.assert_called_with("config.yaml")

    def test_visualize_benchmarks(self, mock_config, mock_logger, mock_benchmark_plotter):
        """Test benchmark visualization delegation."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        bench_instance = mock_benchmark_plotter.return_value

        bench_results = {'score': 100}
        bench_instance.plot_benchmarks.return_value = ['plot1.png']

        result = manager.visualize_benchmarks(bench_results)

        assert result == ['plot1.png']
        bench_instance.plot_benchmarks.assert_called_once_with(bench_results)

    def test_optimization_visualizations(self, mock_config, mock_logger, mock_optimization_plotter):
        """Test optimization-related visualization methods."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        opt_instance = mock_optimization_plotter.return_value
        output_dir = Path('/tmp/opt')
        history = [{'generation': 1, 'best_score': 0.8}]

        manager.visualize_optimization_progress(history, output_dir, 'flow', 'KGE')
        opt_instance.plot_optimization_progress.assert_called_once()

        manager.visualize_optimization_depth_parameters(history, output_dir)
        opt_instance.plot_depth_parameters.assert_called_once()

    @patch.object(ReportingManager, 'analysis_plotter', new_callable=PropertyMock)
    def test_analysis_visualizations(self, mock_analysis_plotter_prop, mock_config, mock_logger):
        """Test analysis-related visualization methods."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        mock_analysis_plotter_prop.return_value = MagicMock()
        analysis_instance = mock_analysis_plotter_prop.return_value

        manager.visualize_sensitivity_analysis(MagicMock(), Path('sens.png'))
        analysis_instance.plot_sensitivity_analysis.assert_called_once()

        manager.visualize_decision_impacts(Path('res.csv'), Path('out'))
        analysis_instance.plot_decision_impacts.assert_called_once()

        manager.visualize_drop_analysis([{'threshold': 10, 'mean_drop': 1}], 10, Path('proj'))
        analysis_instance.plot_drop_analysis.assert_called_once()

    @patch.object(ReportingManager, 'analysis_plotter', new_callable=PropertyMock)
    def test_model_specific_visualizations(self, mock_analysis_plotter_prop, mock_config, mock_logger):
        """Test model-specific visualization methods."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        mock_analysis_plotter_prop.return_value = MagicMock()
        analysis_instance = mock_analysis_plotter_prop.return_value

        # LSTM
        manager.visualize_lstm_results(MagicMock(), MagicMock(), MagicMock(), True, Path('out'), 'exp1')
        analysis_instance.plot_lstm_results.assert_called_once()

        # HYPE
        manager.visualize_hype_results(MagicMock(), MagicMock(), '1', 'dom', 'exp1', Path('proj'))
        analysis_instance.plot_hype_results.assert_called_once()

        # NGen
        manager.visualize_ngen_results(MagicMock(), None, 'exp1', Path('res'))
        analysis_instance.plot_ngen_results.assert_called_once()

    @patch.object(ReportingManager, 'analysis_plotter', new_callable=PropertyMock)
    def test_hydrographs_with_highlight(self, mock_analysis_plotter_prop, mock_config, mock_logger):
        """Test hydrograph visualization with highlight."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        mock_analysis_plotter_prop.return_value = MagicMock()
        analysis_instance = mock_analysis_plotter_prop.return_value

        manager.visualize_hydrographs_with_highlight(
            Path('res.csv'), {}, MagicMock(), {}, Path('out')
        )
        analysis_instance.plot_hydrographs_with_highlight.assert_called_once()
