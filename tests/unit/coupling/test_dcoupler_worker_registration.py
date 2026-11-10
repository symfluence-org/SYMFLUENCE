"""Tests for DCouplerWorker registration and calibration pipeline integration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("torch", reason="requires the ml extra (pip install 'symfluence[ml]')")
import torch

from symfluence.core.registries import R


class TestDCouplerWorkerRegistration:
    """Test that DCouplerWorker is properly registered with OptimizerRegistry."""

    def test_registry_returns_dcoupler_worker(self):
        # Import triggers registration via decorator
        from symfluence.coupling.worker import DCouplerWorker

        worker_cls = R.workers.get('DCOUPLER')
        assert worker_cls is DCouplerWorker

    def test_dcoupler_in_worker_list(self):
        from symfluence.coupling.worker import DCouplerWorker  # noqa: F401

        workers = R.workers.keys()
        assert 'DCOUPLER' in workers


class TestDCouplerWorkerInit:
    """Test DCouplerWorker initialization and graph construction."""

    def test_init_stores_config(self):
        from symfluence.coupling.worker import DCouplerWorker

        config = {'HYDROLOGICAL_MODEL': 'SUMMA', 'ROUTING_MODEL': 'MIZUROUTE'}
        worker = DCouplerWorker(config)
        assert worker.config is config
        assert worker._graph is None  # Lazy init

    def test_graph_lazy_init(self):
        from symfluence.coupling.worker import DCouplerWorker

        config = {'HYDROLOGICAL_MODEL': 'SUMMA'}
        worker = DCouplerWorker(config)

        # Mock the builder to avoid importing dCoupler graph internals
        mock_graph = MagicMock()
        mock_graph.components = {'land': MagicMock()}
        worker._builder = MagicMock()
        worker._builder.build.return_value = mock_graph

        graph = worker.graph
        assert graph is mock_graph
        worker._builder.build.assert_called_once_with(config)

        # Second access should not rebuild
        graph2 = worker.graph
        assert graph2 is mock_graph
        assert worker._builder.build.call_count == 1


class TestDCouplerWorkerApplyParams:
    """Test parameter application logic."""

    def test_apply_process_params_logs_debug(self, caplog):
        import logging

        from symfluence.coupling.worker import DCouplerWorker

        # Ensure the symfluence logger hierarchy allows DEBUG capture even
        # when LoggingManager from a prior test has set propagate=False or
        # raised the effective level on the parent logger.
        for name in ("symfluence", "symfluence.coupling", "symfluence.coupling.worker"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.DEBUG)
            lg.propagate = True

        worker = DCouplerWorker({})
        mock_comp = MagicMock()
        mock_comp.name = "land"

        with caplog.at_level(logging.DEBUG, logger="symfluence.coupling.worker"):
            worker._apply_process_params(mock_comp, {'k1': 0.5}, '/tmp/settings')

        assert "Parameter application for process models" in caplog.text

    def test_apply_process_params_no_log_when_empty(self, caplog):
        import logging

        from symfluence.coupling.worker import DCouplerWorker

        for name in ("symfluence", "symfluence.coupling", "symfluence.coupling.worker"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.DEBUG)
            lg.propagate = True

        worker = DCouplerWorker({})
        mock_comp = MagicMock()
        mock_comp.name = "land"

        with caplog.at_level(logging.DEBUG, logger="symfluence.coupling.worker"):
            worker._apply_process_params(mock_comp, {}, '/tmp/settings')

        assert "Parameter application" not in caplog.text


class TestDCouplerWorkerRunModel:
    """Test run_model() with and without external inputs."""

    def test_run_model_without_external_inputs(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})

        mock_graph = MagicMock()
        mock_graph.forward.return_value = {
            'land': {'runoff': torch.tensor([1.0, 2.0])}
        }
        worker._graph = mock_graph

        success = worker.run_model({}, str(tmp_path), str(tmp_path / 'output'))
        assert success is True
        # Should have used empty dict for external_inputs
        call_kwargs = mock_graph.forward.call_args[1]
        assert call_kwargs['external_inputs'] == {}

    def test_run_model_with_external_inputs(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        worker.set_external_inputs(
            {'land': {'forcing': torch.randn(100, 7)}},
            n_timesteps=100,
            dt=3600.0,
        )

        mock_graph = MagicMock()
        mock_graph.forward.return_value = {
            'routing': {'discharge': torch.tensor([3.0, 4.0])}
        }
        worker._graph = mock_graph

        success = worker.run_model({}, str(tmp_path), str(tmp_path / 'output'))
        assert success is True
        call_kwargs = mock_graph.forward.call_args[1]
        assert call_kwargs['n_timesteps'] == 100
        assert call_kwargs['dt'] == 3600.0


class TestDCouplerWorkerMetrics:
    """Test calculate_metrics() with warmup handling."""

    def test_warmup_slices_data(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        # Set up cached outputs
        sim_data = torch.tensor([float(i) for i in range(100)])
        worker._last_outputs = {'land': {'runoff': sim_data}}

        # Mock observation loading
        obs_data = torch.tensor([float(i) * 0.9 for i in range(100)])
        with patch.object(worker, '_load_observations', return_value=obs_data):
            metrics = worker.calculate_metrics(
                str(tmp_path), {'WARMUP_DAYS': 10}
            )

        # Should have computed on 90 points, not 100
        assert 'KGE' in metrics
        assert metrics['KGE'] != -999.0

    def test_no_warmup_by_default(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        sim_data = torch.tensor([float(i) for i in range(50)])
        worker._last_outputs = {'routing': {'discharge': sim_data}}

        obs_data = torch.tensor([float(i) * 1.1 for i in range(50)])
        with patch.object(worker, '_load_observations', return_value=obs_data):
            metrics = worker.calculate_metrics(str(tmp_path), {})

        assert 'KGE' in metrics
        assert metrics['KGE'] != -999.0

    def test_supports_native_gradients_checks_components(self):
        from dcoupler.core.component import GradientMethod

        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})

        # Mock graph with all differentiable components
        mock_comp = MagicMock()
        mock_comp.gradient_method = GradientMethod.AUTOGRAD
        mock_graph = MagicMock()
        mock_graph.components = {'land': mock_comp}
        worker._graph = mock_graph

        assert worker.supports_native_gradients() is True

        # Now with a non-differentiable component
        mock_comp2 = MagicMock()
        mock_comp2.gradient_method = GradientMethod.NONE
        mock_graph.components = {'land': mock_comp, 'routing': mock_comp2}

        assert worker.supports_native_gradients() is False


class TestDCouplerWorkerDateAlignment:
    """Observations are paired with simulated steps by date, not position."""

    def _obs_csv(self, tmp_path, dates, values):
        import pandas as pd
        path = tmp_path / 'obs.csv'
        pd.DataFrame({'date': dates, 'q': values}).to_csv(path, index=False)
        return path

    def test_obs_starting_later_paired_by_date(self, tmp_path):
        import pandas as pd

        from symfluence.coupling.worker import DCouplerWorker

        rng = np.random.default_rng(0)
        sim_vals = rng.gamma(2, 5, 365)
        sim = torch.tensor(sim_vals, dtype=torch.float32)
        dates = pd.date_range('2010-03-01', periods=200, freq='D')
        # Obs equal the sim on matching dates (sim starts 2010-01-01)
        offset = (dates[0] - pd.Timestamp('2010-01-01')).days
        path = self._obs_csv(tmp_path, dates.strftime('%Y-%m-%d'), sim_vals[offset:offset + 200])
        config = {'OBSERVATIONS_FILE': str(path), 'EXPERIMENT_TIME_START': '2010-01-01'}
        worker = DCouplerWorker(config)
        worker._last_outputs = {'routing': {'discharge': sim}}
        metrics = worker.calculate_metrics(str(tmp_path), config)
        assert metrics['KGE'] == pytest.approx(1.0, abs=1e-5)

    def test_constant_obs_matches_core_semantics(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        sim = torch.linspace(1, 2, 50)
        obs = torch.full((50,), 3.0)
        assert worker._compute_metrics(sim, obs, {})['KGE'] == -999.0

    def test_metrics_match_core(self):
        from symfluence.core import metrics as core_metrics
        from symfluence.coupling.worker import DCouplerWorker

        rng = np.random.default_rng(1)
        obs = rng.gamma(2, 5, 100)
        sim = obs * 1.2 + rng.normal(0, 1, 100)
        out = DCouplerWorker({})._compute_metrics(
            torch.tensor(sim, dtype=torch.float64), torch.tensor(obs, dtype=torch.float64), {})
        assert out['KGE'] == pytest.approx(core_metrics.kge(obs, sim))
        assert out['NSE'] == pytest.approx(core_metrics.nse(obs, sim))
