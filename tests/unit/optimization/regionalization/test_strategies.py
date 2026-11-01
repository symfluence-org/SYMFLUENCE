"""
Unit Tests for model-agnostic parameter regionalization strategies.

Tests the shared regionalization framework at
symfluence.core.calibration.regionalization.strategies.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.optimization]


@pytest.fixture
def param_bounds():
    return {
        'MAXWATR_1': (25.0, 500.0),
        'BASERTE': (0.001, 0.1),
        'MBASE': (-5.0, 5.0),
    }


@pytest.fixture
def param_config():
    return {
        'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
        'BASERTE': {'attribute': 'aridity', 'calibrate_b': False},
        'MBASE': {'attribute': 'elev_m', 'calibrate_b': True},
    }


@pytest.fixture
def unit_attributes():
    return pd.DataFrame({
        'elev_m': [500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        'precip_mm_yr': [600.0, 800.0, 1000.0, 1200.0, 1400.0],
        'temp_C': [8.0, 5.0, 2.0, -1.0, -4.0],
        'aridity': [1.0, 0.8, 0.6, 0.4, 0.2],
        'snow_frac': [0.1, 0.3, 0.5, 0.7, 0.9],
    })


@pytest.fixture
def test_logger():
    logger = logging.getLogger('test_regionalization')
    logger.setLevel(logging.DEBUG)
    return logger


class TestParameterRegionalizationABC:
    def test_cannot_instantiate_directly(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            ParameterRegionalization,
        )
        with pytest.raises(TypeError):
            ParameterRegionalization(param_bounds, 5, test_logger)


class TestLumpedRegionalization:
    def test_name_property(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            LumpedRegionalization,
        )
        reg = LumpedRegionalization(param_bounds, 5, test_logger)
        assert reg.name == "lumped"

    def test_calibration_params_match_original_bounds(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            LumpedRegionalization,
        )
        reg = LumpedRegionalization(param_bounds, 5, test_logger)
        assert reg.get_calibration_parameters() == param_bounds

    def test_to_distributed_replicates_values(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            LumpedRegionalization,
        )
        n_units = 5
        reg = LumpedRegionalization(param_bounds, n_units, test_logger)
        params = {'MAXWATR_1': 200.0, 'BASERTE': 0.05, 'MBASE': 1.0}
        param_array, param_names = reg.to_distributed(params)
        assert param_array.shape == (n_units, 3)
        for col in range(3):
            np.testing.assert_array_equal(
                param_array[:, col], np.full(n_units, param_array[0, col]),
            )

    def test_returns_correct_names(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            LumpedRegionalization,
        )
        reg = LumpedRegionalization(param_bounds, 3, test_logger)
        params = {'MAXWATR_1': 200.0, 'BASERTE': 0.05, 'MBASE': 1.0}
        _, param_names = reg.to_distributed(params)
        assert set(param_names) == set(params.keys())


class TestTransferFunctionRegionalization:
    def test_name_property(self, param_bounds, param_config, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=param_config, logger=test_logger,
        )
        assert reg.name == "transfer_function"

    def test_normalizes_attributes(self, param_bounds, param_config, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=param_config, logger=test_logger,
        )
        assert 'elev_m_norm' in reg.attributes.columns
        assert reg.attributes['elev_m_norm'].min() >= 0.0
        assert reg.attributes['elev_m_norm'].max() <= 1.0

    def test_calibration_params_include_coefficients(
        self, param_bounds, unit_attributes, test_logger,
    ):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        pc = {
            'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
            'BASERTE': {'attribute': 'aridity', 'calibrate_b': False},
        }
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=pc, logger=test_logger,
        )
        cal_params = reg.get_calibration_parameters()
        assert 'MAXWATR_1_a' in cal_params
        assert 'MAXWATR_1_b' in cal_params
        assert 'BASERTE_a' in cal_params
        assert 'BASERTE_b' not in cal_params

    def test_produces_spatial_variation(self, param_bounds, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        pc = {'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True}}
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=pc, logger=test_logger,
        )
        coeffs = {'MAXWATR_1_a': 200.0, 'MAXWATR_1_b': 100.0}
        param_array, _ = reg.to_distributed(coeffs)
        assert param_array.shape[0] == 5
        assert param_array[:, 0].std() > 0

    def test_clips_to_bounds(self, param_bounds, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        pc = {'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True}}
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=pc, logger=test_logger,
        )
        coeffs = {'MAXWATR_1_a': 9999.0, 'MAXWATR_1_b': 9999.0}
        param_array, _ = reg.to_distributed(coeffs)
        p_min, p_max = param_bounds['MAXWATR_1']
        assert param_array[:, 0].max() <= p_max
        assert param_array[:, 0].min() >= p_min

    def test_b_zero_gives_uniform(self, param_bounds, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        pc = {'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True}}
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=pc, logger=test_logger,
        )
        coeffs = {'MAXWATR_1_a': 200.0, 'MAXWATR_1_b': 0.0}
        param_array, _ = reg.to_distributed(coeffs)
        np.testing.assert_array_almost_equal(param_array[:, 0], 200.0)

    def test_coefficient_transforms(self, param_bounds, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            TransferFunctionRegionalization,
        )
        pc = {
            'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True, 'transform': 'log'},
            'BASERTE': {'attribute': 'aridity', 'calibrate_b': False},
        }
        reg = TransferFunctionRegionalization(
            param_bounds, 5, unit_attributes, param_config=pc, logger=test_logger,
        )
        transforms = reg.get_coefficient_transforms()
        assert transforms['MAXWATR_1_a'] == 'log'
        assert transforms['MAXWATR_1_b'] == 'linear'
        assert transforms['BASERTE_a'] == 'linear'


class TestZoneRegionalization:
    def test_name_property(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            ZoneRegionalization,
        )
        zones = np.array([0, 0, 1, 1, 2])
        reg = ZoneRegionalization(param_bounds, 5, zones, test_logger)
        assert reg.name == "zones"

    def test_detects_number_of_zones(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            ZoneRegionalization,
        )
        zones = np.array([0, 0, 1, 1, 2])
        reg = ZoneRegionalization(param_bounds, 5, zones, test_logger)
        assert reg.n_zones == 3

    def test_calibration_params_per_zone(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            ZoneRegionalization,
        )
        zones = np.array([0, 0, 1])
        reg = ZoneRegionalization(param_bounds, 3, zones, test_logger)
        cal_params = reg.get_calibration_parameters()
        assert len(cal_params) == 6
        assert 'MAXWATR_1_z0' in cal_params
        assert 'MAXWATR_1_z1' in cal_params

    def test_maps_zone_values(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            ZoneRegionalization,
        )
        zones = np.array([0, 0, 1, 1, 1])
        reg = ZoneRegionalization(param_bounds, 5, zones, test_logger)
        cal_params = {
            'MAXWATR_1_z0': 100.0, 'MAXWATR_1_z1': 300.0,
            'BASERTE_z0': 0.01, 'BASERTE_z1': 0.05,
            'MBASE_z0': -1.0, 'MBASE_z1': 2.0,
        }
        param_array, param_names = reg.to_distributed(cal_params)
        assert param_array.shape == (5, 3)
        idx = param_names.index('MAXWATR_1')
        assert param_array[0, idx] == 100.0
        assert param_array[2, idx] == 300.0


class TestDistributedRegionalization:
    def test_name_property(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            DistributedRegionalization,
        )
        reg = DistributedRegionalization(param_bounds, 3, logger=test_logger)
        assert reg.name == "distributed"

    def test_calibration_params_per_unit(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            DistributedRegionalization,
        )
        reg = DistributedRegionalization(param_bounds, 3, logger=test_logger)
        cal_params = reg.get_calibration_parameters()
        assert len(cal_params) == 9
        assert 'MAXWATR_1_s0' in cal_params
        assert 'MAXWATR_1_s2' in cal_params

    def test_direct_mapping(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            DistributedRegionalization,
        )
        reg = DistributedRegionalization(param_bounds, 2, logger=test_logger)
        cal_params = {
            'MAXWATR_1_s0': 100.0, 'MAXWATR_1_s1': 200.0,
            'BASERTE_s0': 0.01, 'BASERTE_s1': 0.05,
            'MBASE_s0': -1.0, 'MBASE_s1': 2.0,
        }
        param_array, param_names = reg.to_distributed(cal_params)
        idx = param_names.index('MAXWATR_1')
        assert param_array[0, idx] == 100.0
        assert param_array[1, idx] == 200.0


class TestRegionalizationFactory:
    def test_creates_lumped(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            LumpedRegionalization,
            RegionalizationFactory,
        )
        reg = RegionalizationFactory.create('lumped', param_bounds, 5, logger=test_logger)
        assert isinstance(reg, LumpedRegionalization)

    def test_creates_transfer_function(self, param_bounds, param_config, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            RegionalizationFactory,
            TransferFunctionRegionalization,
        )
        config = {'TRANSFER_FUNCTION_PARAM_CONFIG': param_config}
        reg = RegionalizationFactory.create(
            'transfer_function', param_bounds, 5,
            config=config, attributes=unit_attributes, logger=test_logger,
        )
        assert isinstance(reg, TransferFunctionRegionalization)

    def test_transfer_function_requires_attributes(self, param_bounds, param_config, test_logger):
        from symfluence.core.calibration.regionalization.strategies import RegionalizationFactory
        config = {'TRANSFER_FUNCTION_PARAM_CONFIG': param_config}
        with pytest.raises(ValueError, match="requires 'attributes'"):
            RegionalizationFactory.create(
                'transfer_function', param_bounds, 5, config=config, logger=test_logger,
            )

    def test_transfer_function_requires_param_config(self, param_bounds, unit_attributes, test_logger):
        from symfluence.core.calibration.regionalization.strategies import RegionalizationFactory
        with pytest.raises(ValueError, match="TRANSFER_FUNCTION_PARAM_CONFIG"):
            RegionalizationFactory.create(
                'transfer_function', param_bounds, 5,
                attributes=unit_attributes, logger=test_logger,
            )

    def test_creates_zones(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            RegionalizationFactory,
            ZoneRegionalization,
        )
        config = {'zone_assignments': np.array([0, 0, 1, 1, 2])}
        reg = RegionalizationFactory.create('zones', param_bounds, 5, config=config, logger=test_logger)
        assert isinstance(reg, ZoneRegionalization)

    def test_creates_distributed(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            DistributedRegionalization,
            RegionalizationFactory,
        )
        reg = RegionalizationFactory.create('distributed', param_bounds, 5, logger=test_logger)
        assert isinstance(reg, DistributedRegionalization)

    def test_unknown_method_raises(self, param_bounds, test_logger):
        from symfluence.core.calibration.regionalization.strategies import RegionalizationFactory
        with pytest.raises(ValueError, match="Unknown regionalization method"):
            RegionalizationFactory.create('nonexistent', param_bounds, 5, logger=test_logger)

    def test_handles_hyphenated_name(self, param_bounds, param_config, test_logger):
        from symfluence.core.calibration.regionalization.strategies import (
            RegionalizationFactory,
            TransferFunctionRegionalization,
        )
        attrs = pd.DataFrame({
            'elev_m': [500.0, 1000.0], 'precip_mm_yr': [600.0, 800.0],
        })
        config = {'TRANSFER_FUNCTION_PARAM_CONFIG': param_config}
        reg = RegionalizationFactory.create(
            'transfer-function', param_bounds, 2,
            config=config, attributes=attrs, logger=test_logger,
        )
        assert isinstance(reg, TransferFunctionRegionalization)


class TestGetRegionalizationInfo:
    def test_returns_all_methods(self):
        from symfluence.core.calibration.regionalization.strategies import get_regionalization_info
        info = get_regionalization_info()
        assert 'lumped' in info
        assert 'transfer_function' in info
        assert 'zones' in info
        assert 'distributed' in info
        assert len(info) == 4
