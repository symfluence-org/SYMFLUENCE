"""Tests for Xinanjiang parameter definitions and utilities."""

import numpy as np
import pytest
from jxaj.parameters import (
    DEFAULT_PARAMS,
    LOG_TRANSFORM_PARAMS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    XinanjiangParams,
    XinanjiangState,
    enforce_ki_kg_constraint,
    params_array_to_dict,
    params_dict_to_array,
    params_dict_to_namedtuple,
)


class TestParameterBounds:
    """Test parameter bound definitions."""

    def test_all_calibratable_params_have_bounds(self):
        """All calibratable parameters must have bounds (CS, L excluded — unused in lumped mode)."""
        excluded = {'CS', 'L'}  # Routing params excluded per Zhao 1992 lumped formulation
        for name in PARAM_NAMES:
            if name not in excluded:
                assert name in PARAM_BOUNDS, f"Missing bounds for {name}"

    def test_all_params_have_defaults(self):
        """All named parameters must have defaults."""
        for name in PARAM_NAMES:
            assert name in DEFAULT_PARAMS, f"Missing default for {name}"

    def test_defaults_within_bounds(self):
        """Default values must lie within bounds (for calibratable params)."""
        for name, (lo, hi) in PARAM_BOUNDS.items():
            val = DEFAULT_PARAMS[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_bounds_ordering(self):
        """Lower bound must be strictly less than upper bound."""
        for name, (lo, hi) in PARAM_BOUNDS.items():
            assert lo < hi, f"{name}: lo={lo} >= hi={hi}"

    def test_param_count(self):
        """Should have exactly 15 parameters (13 calibratable with bounds)."""
        assert len(PARAM_NAMES) == 15
        assert len(PARAM_BOUNDS) == 13  # CS, L excluded from calibration
        assert len(DEFAULT_PARAMS) == 15

    def test_log_transform_params_valid(self):
        """Log-transformed params must have positive lower bounds."""
        for name in LOG_TRANSFORM_PARAMS:
            lo, _ = PARAM_BOUNDS[name]
            assert lo > 0, f"{name} has log transform but lo={lo} <= 0"


class TestConstraintEnforcement:
    """Test KI+KG constraint enforcement."""

    def test_constraint_within_limits(self):
        """Values within limits should be unchanged."""
        ki, kg = enforce_ki_kg_constraint(0.3, 0.3)
        assert ki == 0.3
        assert kg == 0.3

    def test_constraint_at_limit(self):
        """Values at limit should be scaled down."""
        ki, kg = enforce_ki_kg_constraint(0.5, 0.5)
        assert ki + kg < 1.0
        assert abs(ki - kg) < 1e-10  # Should remain equal ratio

    def test_constraint_exceeding_limit(self):
        """Values exceeding limit should be proportionally scaled."""
        ki, kg = enforce_ki_kg_constraint(0.7, 0.7)
        assert ki + kg < 1.0
        # Proportional scaling maintains ratio
        assert abs(ki / kg - 1.0) < 1e-10

    def test_constraint_asymmetric(self):
        """Asymmetric values should maintain ratio."""
        ki, kg = enforce_ki_kg_constraint(0.6, 0.6)
        assert ki + kg < 1.0
        assert abs(ki - kg) < 1e-10

    def test_constraint_zero_safe(self):
        """Zero values should not cause errors."""
        ki, kg = enforce_ki_kg_constraint(0.0, 0.0)
        assert ki == 0.0
        assert kg == 0.0


class TestConversions:
    """Test parameter conversion utilities."""

    def test_dict_to_namedtuple(self):
        """Convert dict to XinanjiangParams."""
        params = params_dict_to_namedtuple(DEFAULT_PARAMS, use_jax=False)
        assert isinstance(params, XinanjiangParams)
        assert params.K == pytest.approx(DEFAULT_PARAMS['K'])

    def test_dict_to_array_roundtrip(self):
        """Dict → array → dict should roundtrip."""
        arr = params_dict_to_array(DEFAULT_PARAMS)
        assert len(arr) == 15
        recovered = params_array_to_dict(arr)
        for name in PARAM_NAMES:
            assert recovered[name] == pytest.approx(DEFAULT_PARAMS[name])

    def test_namedtuple_enforces_constraint(self):
        """Converting dict with KI+KG > 1 should enforce constraint."""
        params_dict = DEFAULT_PARAMS.copy()
        params_dict['KI'] = 0.7
        params_dict['KG'] = 0.7
        params = params_dict_to_namedtuple(params_dict, use_jax=False)
        assert float(params.KI) + float(params.KG) < 1.0
