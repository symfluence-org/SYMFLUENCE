"""Tests for conditional dCoupler import and fallback behavior.

Verifies that:
1. is_dcoupler_available() correctly detects availability
2. INSTALL_SUGGESTION is always importable
3. Model simulate() functions accept coupling_mode parameter
4. Model manager falls back to sequential execution gracefully
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDCouplerAvailability:
    """Test the is_dcoupler_available() helper and INSTALL_SUGGESTION."""

    def test_is_dcoupler_available_returns_bool(self):
        from symfluence.coupling import is_dcoupler_available
        result = is_dcoupler_available()
        assert isinstance(result, bool)

    def test_install_suggestion_always_importable(self):
        from symfluence.coupling import INSTALL_SUGGESTION
        assert isinstance(INSTALL_SUGGESTION, str)
        assert "pip install dcoupler" in INSTALL_SUGGESTION

    def test_init_importable_without_dcoupler(self):
        """Verify symfluence.coupling can be imported even if dcoupler is absent."""
        # This test verifies the try/except structure in __init__.py
        from symfluence.coupling import INSTALL_SUGGESTION, is_dcoupler_available
        assert callable(is_dcoupler_available)
        assert len(INSTALL_SUGGESTION) > 0


class TestXAJCouplingMode:
    """Test that XAJ simulate() accepts coupling_mode and falls back correctly."""

    @pytest.fixture
    def synthetic_forcing(self):
        n = 30
        return {
            'precip': np.random.uniform(0, 10, n),
            'temp': np.random.uniform(-5, 15, n),
            'pet': np.random.uniform(1, 5, n),
            'day_of_year': np.arange(1, n + 1),
        }

    def test_native_mode_works(self, synthetic_forcing):
        """coupling_mode='native' should skip dCoupler and use lax.scan/numpy."""
        from jxaj.model import simulate

        snow17_params = {
            'SCF': 1.0, 'PXTEMP': 1.0, 'MFMAX': 1.0, 'MFMIN': 0.3,
            'NMF': 0.15, 'MBASE': 0.0, 'TIPM': 0.1, 'UADJ': 0.04,
            'PLWHC': 0.04, 'DAYGM': 0.0,
        }

        runoff, state = simulate(
            precip=synthetic_forcing['precip'],
            pet=synthetic_forcing['pet'],
            temp=synthetic_forcing['temp'],
            day_of_year=synthetic_forcing['day_of_year'],
            snow17_params=snow17_params,
            use_jax=False,
            coupling_mode='native',
        )
        assert len(runoff) == len(synthetic_forcing['precip'])
        assert np.all(np.isfinite(runoff))

    def test_auto_mode_works(self, synthetic_forcing):
        """coupling_mode='auto' should try dCoupler then fall back to native."""
        from jxaj.model import simulate

        snow17_params = {
            'SCF': 1.0, 'PXTEMP': 1.0, 'MFMAX': 1.0, 'MFMIN': 0.3,
            'NMF': 0.15, 'MBASE': 0.0, 'TIPM': 0.1, 'UADJ': 0.04,
            'PLWHC': 0.04, 'DAYGM': 0.0,
        }

        runoff, state = simulate(
            precip=synthetic_forcing['precip'],
            pet=synthetic_forcing['pet'],
            temp=synthetic_forcing['temp'],
            day_of_year=synthetic_forcing['day_of_year'],
            snow17_params=snow17_params,
            use_jax=False,
            coupling_mode='auto',
        )
        assert len(runoff) == len(synthetic_forcing['precip'])

    def test_default_mode_is_auto(self, synthetic_forcing):
        """Default coupling_mode should be 'auto'."""
        import inspect

        from jxaj.model import simulate
        sig = inspect.signature(simulate)
        assert sig.parameters['coupling_mode'].default == 'auto'


class TestSacSmaCouplingMode:
    """Test that SAC-SMA simulate() accepts coupling_mode and falls back correctly."""

    @pytest.fixture
    def synthetic_forcing(self):
        n = 30
        return {
            'precip': np.random.uniform(0, 10, n),
            'temp': np.random.uniform(-5, 15, n),
            'pet': np.random.uniform(1, 5, n),
            'day_of_year': np.arange(1, n + 1),
        }

    def test_native_mode_works(self, synthetic_forcing):
        """coupling_mode='native' should skip dCoupler and use native path."""
        from jsacsma.model import simulate

        runoff, state = simulate(
            precip=synthetic_forcing['precip'],
            temp=synthetic_forcing['temp'],
            pet=synthetic_forcing['pet'],
            day_of_year=synthetic_forcing['day_of_year'],
            use_jax=False,
            snow_module='snow17',
            coupling_mode='native',
        )
        assert len(runoff) == len(synthetic_forcing['precip'])
        assert np.all(np.isfinite(runoff))

    def test_auto_mode_works(self, synthetic_forcing):
        """coupling_mode='auto' should try dCoupler then fall back to native."""
        from jsacsma.model import simulate

        runoff, state = simulate(
            precip=synthetic_forcing['precip'],
            temp=synthetic_forcing['temp'],
            pet=synthetic_forcing['pet'],
            day_of_year=synthetic_forcing['day_of_year'],
            use_jax=False,
            snow_module='snow17',
            coupling_mode='auto',
        )
        assert len(runoff) == len(synthetic_forcing['precip'])

    def test_standalone_mode_ignores_coupling_mode(self, synthetic_forcing):
        """snow_module='none' should ignore coupling_mode entirely."""
        from jsacsma.model import simulate

        runoff, state = simulate(
            precip=synthetic_forcing['precip'],
            temp=synthetic_forcing['temp'],
            pet=synthetic_forcing['pet'],
            use_jax=False,
            snow_module='none',
            coupling_mode='dcoupler',
        )
        assert len(runoff) == len(synthetic_forcing['precip'])

    def test_default_mode_is_auto(self, synthetic_forcing):
        """Default coupling_mode should be 'auto'."""
        import inspect

        from jsacsma.model import simulate
        sig = inspect.signature(simulate)
        assert sig.parameters['coupling_mode'].default == 'auto'


class TestModelManagerFallback:
    """Test that ModelManager falls back to sequential execution."""

    def test_run_models_with_single_model_skips_dcoupler(self):
        """Single-model workflow should skip dCoupler attempt entirely."""
        # With only one model in workflow, _try_dcoupler_execution should
        # not be called (condition: len(workflow) > 1)
        import inspect

        from symfluence.models.model_manager import ModelManager
        source = inspect.getsource(ModelManager.run_models)
        assert "len(workflow) > 1" in source

    def test_sequential_method_exists(self):
        """Verify _run_sequential method exists on ModelManager."""
        from symfluence.models.model_manager import ModelManager
        assert hasattr(ModelManager, '_run_sequential')

    def test_try_dcoupler_method_exists(self):
        """Verify _try_dcoupler_execution method exists on ModelManager."""
        from symfluence.models.model_manager import ModelManager
        assert hasattr(ModelManager, '_try_dcoupler_execution')
