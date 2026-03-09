# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Integration tests for SAC-SMA + Snow-17 model in SYMFLUENCE framework."""

import numpy as np
import pytest


class TestModelRegistration:
    """Test that SAC-SMA is registered correctly in the framework."""

    def test_runner_registered(self):
        from symfluence.models.registry import ModelRegistry
        runner_cls = ModelRegistry.get_runner('SACSMA')
        assert runner_cls is not None
        assert runner_cls.__name__ == 'SacSmaRunner'

    def test_preprocessor_registered(self):
        from symfluence.models.registry import ModelRegistry
        preprocessor_cls = ModelRegistry.get_preprocessor('SACSMA')
        assert preprocessor_cls is not None
        assert preprocessor_cls.__name__ == 'SacSmaPreProcessor'

    def test_config_adapter_registered(self):
        from symfluence.models.registry import ModelRegistry
        adapter = ModelRegistry.get_config_adapter('SACSMA')
        assert adapter is not None
        assert type(adapter).__name__ == 'SacSmaConfigAdapter'

    def test_result_extractor_registered(self):
        from symfluence.models.registry import ModelRegistry
        extractor = ModelRegistry.get_result_extractor('SACSMA')
        assert extractor is not None
        assert type(extractor).__name__ == 'SacSmaResultExtractor'


class TestCalibrationRegistration:
    """Test calibration infrastructure registration."""

    @pytest.fixture(autouse=True)
    def _import_calibration(self):
        """Ensure calibration modules are imported to trigger registration."""
        import jsacsma.calibration.optimizer  # noqa: F401
        import jsacsma.calibration.parameter_manager  # noqa: F401
        import jsacsma.calibration.worker  # noqa: F401

    def test_optimizer_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        optimizer_cls = OptimizerRegistry.get_optimizer('SACSMA')
        assert optimizer_cls is not None
        assert optimizer_cls.__name__ == 'SacSmaModelOptimizer'

    def test_worker_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        worker_cls = OptimizerRegistry.get_worker('SACSMA')
        assert worker_cls is not None
        assert worker_cls.__name__ == 'SacSmaWorker'

    def test_parameter_manager_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        pm_cls = OptimizerRegistry.get_parameter_manager('SACSMA')
        assert pm_cls is not None
        assert pm_cls.__name__ == 'SacSmaParameterManager'


class TestParameterBoundsRegistry:
    """Test parameter bounds in the central registry."""

    def test_get_sacsma_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_sacsma_bounds
        bounds = get_sacsma_bounds()
        assert len(bounds) == 26

    def test_all_params_present(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_sacsma_bounds
        bounds = get_sacsma_bounds()
        expected = {
            'SCF', 'PXTEMP', 'MFMAX', 'MFMIN', 'NMF', 'MBASE', 'TIPM', 'UADJ', 'PLWHC', 'DAYGM',
            'UZTWM', 'UZFWM', 'UZK', 'LZTWM', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK',
            'ZPERC', 'REXP', 'PFREE', 'PCTIM', 'ADIMP', 'RIVA', 'SIDE', 'RSERV',
        }
        assert set(bounds.keys()) == expected

    def test_log_transforms_preserved(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_sacsma_bounds
        bounds = get_sacsma_bounds()
        log_params = {'ZPERC', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK'}
        for name in log_params:
            assert bounds[name].get('transform') == 'log', f"{name} missing log transform"


class TestParameterManagerRoundTrip:
    """Test parameter manager normalize/denormalize round-trip."""

    def test_normalize_denormalize_round_trip(self):
        import logging

        from jsacsma.calibration.parameter_manager import SacSmaParameterManager
        from jsacsma.parameters import DEFAULT_PARAMS

        config = {
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp1',
        }
        pm = SacSmaParameterManager(config, logging.getLogger('test'), '/tmp')

        # Normalize defaults
        normalized = pm.normalize(DEFAULT_PARAMS)
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

        # Denormalize back
        recovered = pm.denormalize(normalized)

        for name in pm.calibration_params:
            assert abs(recovered[name] - DEFAULT_PARAMS[name]) < 1e-6, (
                f"{name}: expected {DEFAULT_PARAMS[name]}, got {recovered[name]}"
            )

    def test_log_transform_geometric_mean_initial(self):
        import logging

        from jsacsma.calibration.parameter_manager import SacSmaParameterManager
        from jsacsma.parameters import LOG_TRANSFORM_PARAMS, PARAM_BOUNDS

        config = {'DOMAIN_NAME': 'test', 'EXPERIMENT_ID': 'exp1'}
        pm = SacSmaParameterManager(config, logging.getLogger('test'), '/tmp')

        initial = pm.get_initial_parameters()
        for name in LOG_TRANSFORM_PARAMS:
            lo, hi = PARAM_BOUNDS[name]
            expected = np.sqrt(lo * hi)
            assert abs(initial[name] - expected) < 1e-6, f"{name}: expected {expected}, got {initial[name]}"


class TestWorkerSimulation:
    """Test worker-based simulation."""

    def test_run_simulation(self):
        from jsacsma.calibration.worker import SacSmaWorker
        from jsacsma.parameters import DEFAULT_PARAMS

        config = {
            'DOMAIN_NAME': 'test',
            'WARMUP_DAYS': 0,
            'SACSMA_SNOW_MODULE': 'none',
        }
        worker = SacSmaWorker(config=config)

        # Initialize
        assert worker._initialize_model()

        n = 365
        forcing = {
            'precip': np.full(n, 3.0),
            'temp': np.full(n, 10.0),
            'pet': np.full(n, 2.0),
        }

        runoff = worker._run_simulation(forcing, DEFAULT_PARAMS)
        assert len(runoff) == n
        assert np.all(runoff >= 0)
        assert runoff.sum() > 0


class TestEndToEndSmoke:
    """Smoke test: run full simulation and verify basic sanity."""

    def test_one_year_simulation(self):
        from jsacsma.model import simulate
        from jsacsma.parameters import DEFAULT_PARAMS

        n = 365
        # Synthetic annual forcing
        doy = np.arange(1, n + 1)
        temp = 10.0 * np.sin((doy - 81) * 2 * np.pi / 365)
        precip = np.maximum(0, 3.0 + 2.0 * np.random.RandomState(42).randn(n))
        pet = np.maximum(0, 2.0 + 1.5 * np.sin((doy - 81) * 2 * np.pi / 365))

        flow, state = simulate(
            precip, temp, pet,
            params=DEFAULT_PARAMS,
            day_of_year=doy,
            latitude=51.0,
        )

        # Basic sanity
        assert len(flow) == n
        assert np.all(flow >= 0)
        assert np.all(np.isfinite(flow))
        assert flow.sum() > 0

        # Total flow should be less than total precip
        assert flow.sum() < precip.sum()

        # Mean flow should be in reasonable range (0-20 mm/day)
        assert 0 < flow.mean() < 20

        # Final state should be valid
        assert state.snow17.w_i >= 0
        assert state.sacsma.uztwc >= 0
        assert state.sacsma.lztwc >= 0
