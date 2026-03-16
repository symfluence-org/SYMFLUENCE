"""Tests for SAC-SMA + Snow-17 parameter definitions."""

import numpy as np
import pytest
from jsacsma.parameters import (
    DEFAULT_PARAMS,
    LOG_TRANSFORM_PARAMS,
    PARAM_BOUNDS,
    SACSMA_DEFAULTS,
    SACSMA_PARAM_BOUNDS,
    SACSMA_PARAM_NAMES,
    SNOW17_DEFAULTS,
    SNOW17_PARAM_BOUNDS,
    SacSmaParameters,
    Snow17Parameters,
    create_sacsma_params,
    create_snow17_params,
    get_param_transform,
    split_params,
)
from jsnow17.parameters import Snow17Params


class TestParameterBounds:
    """Test parameter bound definitions."""

    def test_snow17_has_10_params(self):
        assert len(SNOW17_PARAM_BOUNDS) == 10

    def test_sacsma_has_16_params(self):
        assert len(SACSMA_PARAM_BOUNDS) == 16

    def test_combined_has_26_params(self):
        assert len(PARAM_BOUNDS) == 26

    def test_all_bounds_min_lt_max(self):
        for name, (lo, hi) in PARAM_BOUNDS.items():
            assert lo < hi, f"{name}: min={lo} >= max={hi}"

    def test_log_transform_params_have_positive_min(self):
        """Log-transformed params must have min > 0."""
        for name in LOG_TRANSFORM_PARAMS:
            lo, _ = PARAM_BOUNDS[name]
            assert lo > 0, f"Log-transform param {name} has min={lo} <= 0"

    def test_log_transform_params_are_expected(self):
        assert LOG_TRANSFORM_PARAMS == {'ZPERC', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK'}

    def test_defaults_within_bounds(self):
        for name, default in DEFAULT_PARAMS.items():
            lo, hi = PARAM_BOUNDS[name]
            assert lo <= default <= hi, (
                f"{name}: default={default} not in [{lo}, {hi}]"
            )

    def test_snow17_defaults_complete(self):
        assert set(SNOW17_DEFAULTS.keys()) == set(SNOW17_PARAM_BOUNDS.keys())

    def test_sacsma_defaults_complete(self):
        assert set(SACSMA_DEFAULTS.keys()) == set(SACSMA_PARAM_BOUNDS.keys())

    def test_sacsma_param_names_count(self):
        assert len(SACSMA_PARAM_NAMES) == 16


class TestNamedTuples:
    """Test NamedTuple creation."""

    def test_snow17_params_from_defaults(self):
        params = create_snow17_params(SNOW17_DEFAULTS)
        assert isinstance(params, Snow17Params)
        assert params.SCF == SNOW17_DEFAULTS['SCF']

    def test_snow17_backward_compat_alias(self):
        """Snow17Parameters should be same as Snow17Params."""
        assert Snow17Parameters is Snow17Params

    def test_sacsma_params_from_defaults(self):
        params = create_sacsma_params(SACSMA_DEFAULTS)
        assert isinstance(params, SacSmaParameters)
        assert params.UZTWM == SACSMA_DEFAULTS['UZTWM']

    def test_split_params_returns_dicts(self):
        """split_params should return (dict, dict) not NamedTuples."""
        snow17_d, sacsma_d = split_params(DEFAULT_PARAMS)
        assert isinstance(snow17_d, dict)
        assert isinstance(sacsma_d, dict)

    def test_split_params_content(self):
        """Split dicts should contain correct keys."""
        snow17_d, sacsma_d = split_params(DEFAULT_PARAMS)
        assert 'SCF' in snow17_d
        assert 'UZTWM' in sacsma_d
        assert len(snow17_d) == 10
        assert len(sacsma_d) == 16

    def test_split_params_fills_defaults(self):
        """Missing keys should be filled from defaults."""
        snow17_d, sacsma_d = split_params({'SCF': 1.2, 'UZTWM': 80.0})
        assert snow17_d['SCF'] == 1.2
        assert snow17_d['MFMAX'] == SNOW17_DEFAULTS['MFMAX']
        assert sacsma_d['UZTWM'] == 80.0
        assert sacsma_d['UZK'] == SACSMA_DEFAULTS['UZK']

    def test_create_with_partial_dict(self):
        """Missing keys should be filled from defaults."""
        params = create_snow17_params({'SCF': 1.2})
        assert params.SCF == pytest.approx(1.2)
        assert params.MFMAX == pytest.approx(SNOW17_DEFAULTS['MFMAX'])

    def test_create_ignores_extra_keys(self):
        params = create_snow17_params({'SCF': 1.2, 'UZTWM': 100.0})
        assert params.SCF == pytest.approx(1.2)


class TestGetParamTransform:
    """Test transform type lookup."""

    def test_log_params(self):
        for name in LOG_TRANSFORM_PARAMS:
            assert get_param_transform(name) == 'log'

    def test_linear_params(self):
        for name in PARAM_BOUNDS:
            if name not in LOG_TRANSFORM_PARAMS:
                assert get_param_transform(name) == 'linear'
