from __future__ import annotations

import logging

import netCDF4 as nc
import numpy as np
import pytest

from symfluence.optimization.core.transformers import SoilDepthTransformer


def _make_full_config(tmp_path, **overrides):
    """Create a complete mock config with all required fields."""
    config = {
        # Required system paths
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        # Required domain settings
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        # Required model settings
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
    }
    config.update(overrides)
    return config


def test_soil_depth_transformation(tmp_path):
    """
    Test the SoilDepthTransformer with a real (mock) NetCDF file.
    """
    # 1. Create a dummy coldState.nc
    coldstate_path = tmp_path / "coldState.nc"
    with nc.Dataset(coldstate_path, 'w') as ds:
        ds.createDimension('hru', 1)
        ds.createDimension('midToto', 6)
        ds.createDimension('ifcToto', 7)

        depths = ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))
        heights = ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))

        original_depth_values = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 2.0])
        depths[:, 0] = original_depth_values

        h_vals = np.zeros(7)
        for i in range(6):
            h_vals[i+1] = h_vals[i] + original_depth_values[i]
        heights[:, 0] = h_vals

    # 2. Setup transformer
    config = _make_full_config(tmp_path, SETTINGS_SUMMA_COLDSTATE='coldState.nc')
    import logging
    logger = logging.getLogger('test')
    transformer = SoilDepthTransformer(config, logger)

    # 3. Apply transformation
    params = {'total_soil_depth_multiplier': 1.5, 'shape_factor': 1.0}
    success = transformer.apply(params, tmp_path)

    assert success is True

    # 4. Verify results
    with nc.Dataset(coldstate_path, 'r') as ds:
        new_depths = ds.variables['mLayerDepth'][:, 0]
        expected_depths = original_depth_values * 1.5
        assert np.allclose(new_depths, expected_depths)

        new_heights = ds.variables['iLayerHeight'][:, 0]
        assert new_heights[-1] == pytest.approx(np.sum(expected_depths))

def test_transformation_manager(tmp_path):

    """Test the TransformationManager orchestrator."""

    from symfluence.optimization.core.transformers import TransformationManager



    coldstate_path = tmp_path / "coldState.nc"

    with nc.Dataset(coldstate_path, 'w') as ds:

        ds.createDimension('hru', 1)

        ds.createDimension('midToto', 2)

        ds.createDimension('ifcToto', 3)

        ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))

        ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))

        ds.variables['mLayerDepth'][:, 0] = [1.0, 2.0]

        ds.variables['iLayerHeight'][:, 0] = [0, 1, 3]



    mgr = TransformationManager(_make_full_config(tmp_path), logging.getLogger('test'))

    success = mgr.transform({'total_soil_depth_multiplier': 2.0}, tmp_path)



    assert success is True

    with nc.Dataset(coldstate_path, 'r') as ds:

        assert ds.variables['mLayerDepth'][0, 0] == 2.0

        assert ds.variables['mLayerDepth'][1, 0] == 4.0


# ---------------------------------------------------------------------------
# One soil-depth routine: transformer, SUMMA parameter manager and SUMMA
# worker must rescale the layers identically (they used to be three copies).
# ---------------------------------------------------------------------------

_DEPTHS = np.array([0.025, 0.075, 0.15, 0.25, 0.5, 1.0])


def _summa_settings(root):
    settings = root / 'settings' / 'SUMMA'
    settings.mkdir(parents=True)
    with nc.Dataset(settings / 'attributes.nc', 'w') as ds:
        ds.createDimension('hru', 2)
        ds.createDimension('gru', 1)
        ds.createVariable('hruId', 'i4', ('hru',))[:] = [1, 2]
        ds.createVariable('gruId', 'i4', ('gru',))[:] = [1]
    with nc.Dataset(settings / 'coldState.nc', 'w') as ds:
        ds.createDimension('hru', 2)
        ds.createDimension('midToto', len(_DEPTHS))
        ds.createDimension('ifcToto', len(_DEPTHS) + 1)
        ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))[:] = np.tile(_DEPTHS[:, None], (1, 2))
        ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))[:] = np.tile(
            np.r_[0, _DEPTHS.cumsum()][:, None], (1, 2))
    (settings / 'localParamInfo.txt').write_text("k_soil | 1.0d-5 | 1.0d-7 | 1.0d-3\n")
    (settings / 'basinParamInfo.txt').write_text("routingGammaShape | 2.5 | 2.0 | 3.0\n")
    return settings


def _layers(settings):
    with nc.Dataset(settings / 'coldState.nc') as ds:
        return np.asarray(ds['mLayerDepth'][:]).copy(), np.asarray(ds['iLayerHeight'][:]).copy()


@pytest.mark.parametrize('depth_params', [
    {'total_mult': np.array([1.3]), 'shape_factor': np.array([1.7])},
    {'total_mult': 0.6, 'shape_factor': 0.4},
    {'total_soil_depth_multiplier': np.array([2.2])},
    {'total_soil_depth_multiplier': 1.5, 'total_mult': 9.0, 'shape_factor': 2.5},
    {'shape_factor': 3.0},
])
def test_every_soil_depth_path_gives_identical_layers(tmp_path, depth_params):
    from symfluence.models.summa.calibration.parameter_files import (
        calculate_soil_depths,
        depth_multipliers,
    )
    from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
    from symfluence.models.summa.calibration.worker_impl import _apply_parameters_worker

    log = logging.getLogger('test')
    flat = {'PARAMS_TO_CALIBRATE': 'k_soil,total_soil_depth_multiplier', 'BASIN_PARAMS_TO_CALIBRATE': '',
            'CALIBRATE_DEPTH': True}

    via_transformer = _summa_settings(tmp_path / 'transformer')
    transformer = SoilDepthTransformer(_make_full_config(tmp_path), log)
    assert transformer.apply(dict(depth_params), via_transformer) is True

    via_manager = _summa_settings(tmp_path / 'manager')
    assert SUMMAParameterManager(flat, log, via_manager).update_model_files(dict(depth_params))

    via_worker = _summa_settings(tmp_path / 'worker')
    debug = {'files_checked': [], 'errors': []}
    assert _apply_parameters_worker(
        dict(depth_params), {'config': flat, 'original_depths': _DEPTHS.tolist()}, via_worker, log, debug), debug

    expected = calculate_soil_depths(_DEPTHS, *depth_multipliers(depth_params))
    reference = _layers(via_transformer)
    np.testing.assert_array_equal(reference[0][:, 0], expected)
    np.testing.assert_array_equal(reference[0][:, 1], expected)
    np.testing.assert_allclose(reference[1][-1], expected.sum())
    for other in (via_manager, via_worker):
        depths, heights = _layers(other)
        np.testing.assert_array_equal(depths, reference[0])
        np.testing.assert_array_equal(heights, reference[1])
    np.testing.assert_array_equal(
        transformer._calculate_new_depths(_DEPTHS, *depth_multipliers(depth_params)), expected)


def test_transformer_module_does_not_import_models_at_import_time():
    """optimization.core must stay importable without the models layer."""
    import ast
    from pathlib import Path

    import symfluence.optimization.core.transformers as module

    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    top_level = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    ]
    assert not [name for name in top_level if name and name.startswith('symfluence.models')]
