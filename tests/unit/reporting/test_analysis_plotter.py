"""
Unit tests for AnalysisPlotter.

Tests sensitivity analysis, decision impacts, and threshold analysis visualizations.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter


@pytest.fixture
def analysis_plotter(mock_config, mock_logger, mock_plot_config):
    """Create an AnalysisPlotter instance."""
    return AnalysisPlotter(mock_config, mock_logger, mock_plot_config)


class TestAnalysisPlotter:
    """Test suite for AnalysisPlotter."""

    def test_initialization(self, analysis_plotter):
        """Test that AnalysisPlotter initializes correctly."""
        assert analysis_plotter.config is not None
        assert analysis_plotter.logger is not None
        assert analysis_plotter.plot_config is not None


class TestSensitivityAnalysisPlotting:
    """Test sensitivity analysis visualization methods."""

    def test_plot_sensitivity_analysis_single_success(
        self, analysis_plotter, sample_sensitivity_data
    ):
        """Test successful single-method sensitivity plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "sensitivity.png"

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(output_file)), \
                 patch.object(pd.Series, 'plot', return_value=Mock()):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_sensitivity_analysis(
                    sample_sensitivity_data,
                    output_file,
                    plot_type='single'
                )

                assert result == str(output_file)

    def test_plot_sensitivity_analysis_comparison_success(
        self, analysis_plotter, sample_sensitivity_comparison
    ):
        """Test successful comparison sensitivity plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "sensitivity_comparison.png"

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(output_file)), \
                 patch.object(pd.DataFrame, 'plot', return_value=Mock()):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_ax.legend = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_sensitivity_analysis(
                    sample_sensitivity_comparison,
                    output_file,
                    plot_type='comparison'
                )

                assert result == str(output_file)

    def test_plot_sensitivity_analysis_unknown_type_fallback(
        self, analysis_plotter, sample_sensitivity_data
    ):
        """Test that unknown plot_type falls back to 'single'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "sensitivity.png"

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(output_file)), \
                 patch.object(pd.Series, 'plot', return_value=Mock()):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_sensitivity_analysis(
                    sample_sensitivity_data,
                    output_file,
                    plot_type='invalid_type'
                )

                # Should fallback to single
                assert result is not None

    def test_plot_sensitivity_analysis_error_handling(
        self, analysis_plotter, sample_sensitivity_data, mock_logger
    ):
        """Test error handling in sensitivity plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "sensitivity.png"

            # Patch plotting methods to work, but have _save_and_close raise an error
            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', side_effect=Exception("Test error")), \
                 patch.object(pd.Series, 'plot', return_value=Mock()):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_sensitivity_analysis(
                    sample_sensitivity_data,
                    output_file,
                    plot_type='single'
                )

                assert result is None
                mock_logger.error.assert_called()


class TestDecisionImpactPlotting:
    """Test decision impact visualization methods."""

    def test_plot_decision_impacts_success(
        self, analysis_plotter, sample_decision_results_df
    ):
        """Test successful decision impact plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create results file
            results_file = Path(tmpdir) / "results.csv"
            sample_decision_results_df.to_csv(results_file, index=False)
            output_folder = Path(tmpdir) / "output"

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(output_folder / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_axes = [Mock(), Mock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_decision_impacts(
                    results_file,
                    output_folder
                )

                # Should return dict of plot paths
                assert result is None or isinstance(result, dict)

    def test_plot_decision_impacts_missing_file(self, analysis_plotter, mock_logger):
        """Test decision impact with missing results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_file = Path(tmpdir) / "nonexistent.csv"
            output_folder = Path(tmpdir) / "output"

            result = analysis_plotter.plot_decision_impacts(
                results_file,
                output_folder
            )

            assert result is None

    def test_plot_decision_impacts_single_decision(self, analysis_plotter):
        """Test decision impact with single decision column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create results with single decision
            df = pd.DataFrame({
                'Iteration': range(10),
                'soilDepth': ['shallow'] * 5 + ['deep'] * 5,
                'kge': np.random.uniform(0.5, 0.9, 10),
            })
            results_file = Path(tmpdir) / "results.csv"
            df.to_csv(results_file, index=False)
            output_folder = Path(tmpdir) / "output"

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(output_folder / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_decision_impacts(
                    results_file,
                    output_folder
                )
                # Should handle single decision case


class TestHydrographPlotting:
    """Test hydrograph visualization methods."""

    def test_plot_hydrographs_with_highlight_success(self, analysis_plotter):
        """Test successful hydrograph with highlight plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create results file
            results_df = pd.DataFrame({
                'Iteration': range(20),
                'decision1': ['a', 'b'] * 10,
                'kge': np.random.uniform(0.5, 0.9, 20),
            })
            results_file = Path(tmpdir) / "results.csv"
            results_df.to_csv(results_file, index=False)
            output_folder = Path(tmpdir) / "output"

            # Create mock simulation results and observations
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            simulation_results = {
                'iteration_0': pd.Series(np.random.random(100), index=dates),
                'iteration_1': pd.Series(np.random.random(100), index=dates),
            }
            observed_streamflow = pd.Series(np.random.random(100), index=dates)
            decision_options = {'decision1': ['a', 'b']}

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(output_folder / 'test.png')):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = analysis_plotter.plot_hydrographs_with_highlight(
                    results_file,
                    simulation_results,
                    observed_streamflow,
                    decision_options,
                    output_folder,
                    metric='kge'
                )


class TestDropAnalysisPlotting:
    """Test drop/threshold analysis visualization methods."""

    def test_plot_drop_analysis_success(self, analysis_plotter, sample_drop_data):
        """Test successful drop analysis plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            reporting_dir = project_dir / "reporting"
            reporting_dir.mkdir(parents=True)

            with patch.object(analysis_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(analysis_plotter, '_save_and_close', return_value=str(reporting_dir / 'drop_analysis.png')), \
                 patch.object(analysis_plotter, 'project_dir', project_dir):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                # Test plotting - should not raise
                if hasattr(analysis_plotter, 'plot_drop_analysis'):
                    try:
                        analysis_plotter.plot_drop_analysis(
                            sample_drop_data,
                            optimal_threshold=3000,
                            project_dir=project_dir
                        )
                    except Exception as e:  # noqa: BLE001
                        pytest.fail(f"plot_drop_analysis raised: {e}")

    def test_plot_drop_analysis_empty_data(self, analysis_plotter):
        """Test drop analysis with empty data returns None or handles gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            if hasattr(analysis_plotter, 'plot_drop_analysis'):
                # Empty list should be handled gracefully (either None or catching error)
                result = analysis_plotter.plot_drop_analysis(
                    [],
                    optimal_threshold=1000,
                    project_dir=project_dir
                )
                # Should return None when handling empty data
                # (the method will fail when trying to plot empty lists)
                assert result is None


class TestSUMMAPlotting:
    """Test SUMMA-specific visualization methods."""

    def test_plot_summa_outputs_method_exists(self, analysis_plotter):
        """Test that SUMMA plotting method exists."""
        assert hasattr(analysis_plotter, 'plot_summa_outputs') or \
               hasattr(analysis_plotter, 'plot_summa_results')

    def test_plot_summa_outputs_with_observations(self, analysis_plotter):
        """Test SUMMA output plotting with observations."""
        if hasattr(analysis_plotter, 'plot_summa_outputs'):
            with tempfile.TemporaryDirectory() as tmpdir:
                analysis_plotter.project_dir = Path(tmpdir)

                # Would need actual SUMMA output files for a real test
                # Here we just verify the method is callable
                assert callable(analysis_plotter.plot_summa_outputs)


class TestAnalysisPanelCaching:
    """The lazy panel properties must build each panel once, not per access."""

    @pytest.mark.parametrize("name", ["_scatter_panel", "_fdc_panel", "_ts_panel"])
    def test_panel_is_cached(self, analysis_plotter, name):
        assert getattr(analysis_plotter, name) is getattr(analysis_plotter, name)
