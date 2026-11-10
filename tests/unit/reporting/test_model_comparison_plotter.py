"""
Unit tests for ModelComparisonPlotter.

Tests the model comparison overview visualization functionality.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter


@pytest.fixture
def model_comparison_plotter(mock_config, mock_logger, mock_plot_config):
    """Create a ModelComparisonPlotter instance."""
    return ModelComparisonPlotter(mock_config, mock_logger, mock_plot_config)


class TestModelComparisonPlotter:
    """Test suite for ModelComparisonPlotter."""

    def test_initialization(self, model_comparison_plotter):
        """Test that ModelComparisonPlotter initializes correctly."""
        assert model_comparison_plotter.config is not None
        assert model_comparison_plotter.logger is not None
        assert model_comparison_plotter.plot_config is not None

    def test_model_colors_defined(self, model_comparison_plotter):
        """Test that model colors palette is defined."""
        assert hasattr(model_comparison_plotter, 'MODEL_COLORS')
        assert len(model_comparison_plotter.MODEL_COLORS) >= 8

    def test_plot_model_comparison_overview_success(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test successful model comparison overview plotting."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up project_dir
            model_comparison_plotter.project_dir = Path(tmpdir)
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir(parents=True)
            reporting_dir = Path(tmpdir) / "reporting" / "model_comparison"
            reporting_dir.mkdir(parents=True)

            # Save sample results
            sample_results_df.to_csv(results_dir / "test_exp_results.csv")

            # Only mock _collect_model_data - let matplotlib work normally
            with patch.object(model_comparison_plotter, '_collect_model_data',
                            return_value=(sample_results_df, sample_obs_series)):

                result = model_comparison_plotter.plot_model_comparison_overview(
                    experiment_id='test_exp',
                    context='run_model'
                )

                # Should return a path to the created plot
                assert result is not None
                assert Path(result).exists()

    def test_plot_model_comparison_overview_empty_results(
        self, model_comparison_plotter, empty_results_df
    ):
        """Test plotting with empty results DataFrame."""
        with patch.object(model_comparison_plotter, '_collect_model_data', return_value=(empty_results_df, None)):
            result = model_comparison_plotter.plot_model_comparison_overview(
                experiment_id='test_exp'
            )
            assert result is None

    def test_plot_model_comparison_overview_no_discharge_columns(
        self, model_comparison_plotter, results_df_no_discharge, sample_obs_series
    ):
        """Test plotting with no discharge columns in results."""
        with patch.object(model_comparison_plotter, '_collect_model_data', return_value=(results_df_no_discharge, sample_obs_series)):
            result = model_comparison_plotter.plot_model_comparison_overview(
                experiment_id='test_exp'
            )
            assert result is None

    def test_plot_model_comparison_overview_calibration_context(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test plotting in calibration context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_comparison_plotter.project_dir = Path(tmpdir)
            reporting_dir = Path(tmpdir) / "reporting" / "model_comparison"
            reporting_dir.mkdir(parents=True)

            with patch.object(model_comparison_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(model_comparison_plotter, '_save_and_close', return_value=str(reporting_dir / 'test.png')), \
                 patch.object(model_comparison_plotter, '_collect_model_data', return_value=(sample_results_df, sample_obs_series)):

                mock_plt = Mock()
                mock_fig = Mock()
                mock_setup.return_value = (mock_plt, None)
                mock_plt.figure.return_value = mock_fig

                result = model_comparison_plotter.plot_model_comparison_overview(
                    experiment_id='test_exp',
                    context='calibrate_model'
                )
                # Just verify it doesn't crash with calibration context

    def test_error_handling(self, model_comparison_plotter, mock_logger):
        """Test error handling in plotting."""
        with patch.object(model_comparison_plotter, '_collect_model_data', side_effect=Exception("Test error")):
            result = model_comparison_plotter.plot_model_comparison_overview(
                experiment_id='test_exp'
            )
            assert result is None
            mock_logger.error.assert_called()

    def test_calculate_all_metrics(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test metrics calculation for multiple models."""
        model_cols = ['SUMMA_discharge', 'FUSE_discharge']

        # This is an internal method, but we can still test it
        if hasattr(model_comparison_plotter, '_calculate_all_metrics'):
            metrics = model_comparison_plotter._calculate_all_metrics(
                sample_results_df, sample_obs_series, model_cols
            )
            # Should return metrics for each model
            assert isinstance(metrics, dict)


class TestModelComparisonPlotterPanels:
    """Test individual panel methods of ModelComparisonPlotter."""

    def test_plot_timeseries_panel(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test time series panel plotting."""
        if hasattr(model_comparison_plotter, '_plot_timeseries_panel'):
            mock_ax = Mock()
            model_cols = ['SUMMA_discharge', 'FUSE_discharge']

            # Should not raise
            try:
                model_comparison_plotter._plot_timeseries_panel(
                    mock_ax, sample_results_df, sample_obs_series, model_cols
                )
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"_plot_timeseries_panel raised: {e}")

    def test_plot_fdc_panel(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test flow duration curve panel plotting."""
        if hasattr(model_comparison_plotter, '_plot_fdc_panel'):
            mock_ax = Mock()
            model_cols = ['SUMMA_discharge', 'FUSE_discharge']

            try:
                model_comparison_plotter._plot_fdc_panel(
                    mock_ax, sample_results_df, sample_obs_series, model_cols
                )
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"_plot_fdc_panel raised: {e}")

    def test_plot_scatter_panels(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test scatter plot panels."""
        if hasattr(model_comparison_plotter, '_plot_scatter_panels'):
            mock_axes = [Mock(), Mock()]
            model_cols = ['SUMMA_discharge', 'FUSE_discharge']

            try:
                model_comparison_plotter._plot_scatter_panels(
                    mock_axes, sample_results_df, sample_obs_series, model_cols
                )
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"_plot_scatter_panels raised: {e}")

    def test_plot_metrics_table(self, model_comparison_plotter):
        """Test metrics table panel."""
        if hasattr(model_comparison_plotter, '_plot_metrics_table'):
            mock_ax = Mock()
            metrics_dict = {
                'SUMMA': {'KGE': 0.75, 'NSE': 0.70, 'RMSE': 2.5},
                'FUSE': {'KGE': 0.68, 'NSE': 0.65, 'RMSE': 3.1},
            }

            try:
                model_comparison_plotter._plot_metrics_table(mock_ax, metrics_dict)
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"_plot_metrics_table raised: {e}")

    def test_plot_monthly_boxplots(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test monthly boxplot panel."""
        if hasattr(model_comparison_plotter, '_plot_monthly_boxplots'):
            mock_ax = Mock()
            model_cols = ['SUMMA_discharge', 'FUSE_discharge']

            # Configure boxplot mock to return dict-like structure
            mock_boxes = [Mock()]
            boxplot_result = {'boxes': mock_boxes, 'whiskers': [], 'caps': [], 'medians': []}
            mock_ax.boxplot.return_value = boxplot_result

            try:
                model_comparison_plotter._plot_monthly_boxplots(
                    mock_ax, sample_results_df, sample_obs_series, model_cols
                )
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"_plot_monthly_boxplots raised: {e}")

    def test_plot_residual_analysis(
        self, model_comparison_plotter, sample_results_df, sample_obs_series
    ):
        """Test residual analysis panel."""
        if hasattr(model_comparison_plotter, '_plot_residual_analysis'):
            mock_ax = Mock()
            model_cols = ['SUMMA_discharge', 'FUSE_discharge']

            try:
                model_comparison_plotter._plot_residual_analysis(
                    mock_ax, sample_results_df, sample_obs_series, model_cols
                )
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"_plot_residual_analysis raised: {e}")


class TestModelComparisonDateAlignment:
    """Metrics pair sim and obs by timestamp, not by array position."""

    def test_calculated_metrics_align_on_index(self, model_comparison_plotter):
        idx = pd.date_range('2010-01-01', periods=400, freq='D')
        rng = np.random.default_rng(0)
        obs = pd.Series(rng.gamma(2, 5, len(idx)), index=idx)
        # Calibrated table covers a later, shorter window but the same dates
        # carry identical values -> perfect skill when paired by date.
        cal_df = pd.DataFrame({'SUMMA_discharge_cms': obs.iloc[100:].values}, index=idx[100:])
        metrics = model_comparison_plotter._calculate_all_metrics(
            cal_df, obs, ['SUMMA_discharge_cms']
        )
        assert metrics['SUMMA']['KGE'] == pytest.approx(1.0)

    def test_align_valid_pairs_with_series(self, model_comparison_plotter):
        a = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2010-01-01', periods=3))
        b = pd.Series([2.0, 3.0, 4.0], index=pd.date_range('2010-01-02', periods=3))
        obs, sim = model_comparison_plotter._align_valid_pairs(a, b)
        np.testing.assert_array_equal(obs, [2.0, 3.0])
        np.testing.assert_array_equal(sim, [2.0, 3.0])
