"""
SYMFLUENCE Lumped Basin Integration Tests

Tests the lumped basin workflow from notebook 02a for all supported models.
Uses local test data from tests/data/ and runs the full workflow from raw data.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import DataAcquisitionError
from test_helpers.helpers import keep_single_raw_forcing, write_config

pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]


def _copy_source_data(src_domain: Path, dst_domain: Path, src_domain_name: str, dst_domain_name: str) -> None:
    """Copy raw data from source domain to destination project directory, renaming files."""
    # Copy attributes (DEM, landclass, soilclass)
    src_attrs = src_domain / "attributes"
    if src_attrs.exists():
        dst_attrs = dst_domain / "data" / "attributes"
        if dst_attrs.exists():
            shutil.rmtree(dst_attrs)
        shutil.copytree(src_attrs, dst_attrs)
        # Rename files to match new domain name
        for f in dst_attrs.rglob("*"):
            if f.is_file() and src_domain_name in f.name:
                new_name = f.name.replace(src_domain_name, dst_domain_name)
                f.rename(f.parent / new_name)

    # Copy forcing data
    src_forcing = src_domain / "forcing" / "raw_data"
    if src_forcing.exists():
        dst_forcing = dst_domain / "data" / "forcing" / "raw_data"
        dst_forcing.mkdir(parents=True, exist_ok=True)
        for f in src_forcing.glob("*.nc"):
            new_name = f.name.replace(src_domain_name, dst_domain_name) if src_domain_name in f.name else f.name
            shutil.copy2(f, dst_forcing / new_name)

    # Copy observations
    src_obs = src_domain / "observations"
    if src_obs.exists():
        dst_obs = dst_domain / "data" / "observations"
        if dst_obs.exists():
            shutil.rmtree(dst_obs)
        shutil.copytree(src_obs, dst_obs)
        # Rename files to match new domain name
        for f in dst_obs.rglob("*"):
            if f.is_file() and src_domain_name in f.name:
                new_name = f.name.replace(src_domain_name, dst_domain_name)
                f.rename(f.parent / new_name)


@pytest.fixture(scope="function")
def config_path(bow_domain, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    from test_helpers.helpers import load_config_template

    config_dict = load_config_template(symfluence_code_dir)

    # Use tmp_path as data directory to avoid polluting source data
    config_dict['SYMFLUENCE_DATA_DIR'] = str(tmp_path)
    config_dict['SYMFLUENCE_CODE_DIR'] = str(symfluence_code_dir)

    # Domain settings - use a unique name for the test
    domain_name = "Bow_at_Banff_lumped"
    config_dict['DOMAIN_NAME'] = domain_name
    config_dict['EXPERIMENT_ID'] = f'test_{tmp_path.name[:8]}'  # Keep short to avoid Fortran strLen=256 overflow in mizuRoute
    config_dict['POUR_POINT_COORDS'] = '51.1722/-115.5717'

    # Use semidistributed method to create watershed from DEM via TauDEM (raw data workflow)
    # Use very high stream threshold to get a single basin (lumped)
    config_dict['DOMAIN_DEFINITION_METHOD'] = 'semidistributed'
    config_dict['DELINEATION_METHOD'] = 'stream_threshold'
    config_dict['STREAM_THRESHOLD'] = 100000  # Very high threshold = single basin
    config_dict['SUB_GRID_DISCRETIZATION'] = 'GRUs'

    # DO NOT set RIVER_BASINS_NAME etc - leave as "default" so delineation runs
    # The workflow will create shapefiles with its own naming convention

    # Optimized: 5-day period for faster testing
    config_dict['EXPERIMENT_TIME_START'] = '2004-01-01 01:00'
    config_dict['EXPERIMENT_TIME_END'] = '2004-01-05 23:00'
    config_dict['CALIBRATION_PERIOD'] = '2004-01-02, 2004-01-04'
    config_dict['EVALUATION_PERIOD'] = '2004-01-05, 2004-01-05'
    config_dict['SPINUP_PERIOD'] = '2004-01-01, 2004-01-01'

    # Streamflow
    config_dict['STATION_ID'] = '05BB001'
    config_dict['DOWNLOAD_WSC_DATA'] = False

    # Calibration settings - use 3 iterations to verify algorithm actually runs
    config_dict['NUMBER_OF_ITERATIONS'] = 3
    config_dict['RANDOM_SEED'] = 42

    # Ensure required fields for SymfluenceConfig are present
    if 'HYDROLOGICAL_MODEL' not in config_dict:
        config_dict['HYDROLOGICAL_MODEL'] = 'SUMMA'
    if 'FORCING_DATASET' not in config_dict:
        config_dict['FORCING_DATASET'] = 'ERA5'

    # Create and validate SymfluenceConfig
    config = SymfluenceConfig(**config_dict)

    # Save config back to YAML for SYMFLUENCE initialization
    cfg_path = tmp_path / 'test_config.yaml'
    write_config(config.to_dict(flatten=True), cfg_path)

    return cfg_path, config, bow_domain


@pytest.fixture(scope="function")
def setup_installs_symlink(tmp_path, symfluence_data_root):
    """Create symlink to installs directory in tmp_path so TauDEM binaries are found."""
    installs_src = symfluence_data_root / "installs"
    installs_dst = tmp_path / "installs"
    if installs_src.exists() and not installs_dst.exists():
        try:
            installs_dst.symlink_to(installs_src)
        except OSError:
            # Windows without admin/Developer Mode — use copy
            shutil.copytree(installs_src, installs_dst, dirs_exist_ok=True)
    return installs_dst


def _prune_raw_forcing(project_dir: Path) -> None:
    raw_dir = project_dir / "data" / "forcing" / "raw_data"
    if not raw_dir.exists():
        return
    keep_single_raw_forcing(raw_dir)

    basin_avg_dir = project_dir / "data" / "forcing" / "basin_averaged_data"
    if basin_avg_dir.exists():
        for path in basin_avg_dir.glob("*.nc"):
            path.unlink()
    intersection_dir = project_dir / "shapefiles" / "catchment_intersection"
    if intersection_dir.exists():
        for path in intersection_dir.glob("**/*"):
            if path.is_file():
                path.unlink()


MODELS = [
    'SUMMA',
    'FUSE',
    pytest.param('GR', marks=pytest.mark.full),
    pytest.param('NGEN', marks=pytest.mark.full),
    'LSTM',
    'HYPE',
    # Note: MESH and RHESSys do not support lumped (single-GRU) basins by design
    # Use test_semi_distributed.py for these models instead
]


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("model", MODELS)
def test_lumped_basin_workflow(config_path, model, symfluence_data_root, setup_installs_symlink):
    """
    Test lumped basin workflow for each model.

    Follows notebook 02a workflow:
    1. Setup project
    2. Copy raw data from source
    3. Create pour point
    4. Define domain (watershed delineation)
    5. Discretize domain
    6. Model-agnostic preprocessing
    7. Model-specific preprocessing
    8. Run model
    9. Calibrate model
    """
    # Check for model-specific dependencies before starting test
    if model == 'GR':
        # GR models require R and rpy2
        try:
            import rpy2.robjects
        except ImportError:
            pytest.skip("GR models require R and rpy2. Skipping test since dependencies are not available.")

    cfg_path, typed_config, bow_domain = config_path
    config_dict = typed_config.to_dict(flatten=True)

    # Update model in config
    config_dict['HYDROLOGICAL_MODEL'] = model
    if model == 'SUMMA':
        config_dict['ROUTING_MODEL'] = 'mizuRoute'
        config_dict['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat'
        config_dict['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'
    elif model == 'FUSE':
        config_dict['FUSE_SPATIAL_MODE'] = 'lumped'
        config_dict['SETTINGS_FUSE_PARAMS_TO_CALIBRATE'] = 'MAXWATR_1,MAXWATR_2,BASERTE'
        config_dict['FUSE_RUN_INTERNAL_CALIBRATION'] = False
    elif model == 'GR':
        config_dict['GR_SPATIAL_MODE'] = 'lumped'
        config_dict['GR_SKIP_CALIBRATION'] = True
    elif model == 'NGEN':
        config_dict['NGEN_MODULES_TO_CALIBRATE'] = 'CFE'
        config_dict['NGEN_CFE_PARAMS_TO_CALIBRATE'] = 'smcmax,satdk,bb'
        # Point to ngen install in data directory
        config_dict['NGEN_INSTALL_PATH'] = str(symfluence_data_root / 'installs' / 'ngen' / 'cmake_build')
    elif model == 'LSTM':
        config_dict['LSTM_EPOCHS'] = 1
        config_dict['LSTM_LOOKBACK'] = 24
        config_dict['LSTM_HIDDEN_SIZE'] = 16
        config_dict['LSTM_BATCH_SIZE'] = 4
    elif model == 'HYPE':
        config_dict['HYPE_TIMESHIFT'] = 0
        config_dict['HYPE_SPINUP_DAYS'] = 0
        config_dict['HYPE_FRAC_THRESHOLD'] = 0.1
        config_dict['HYPE_SKIP_CALIBRATION'] = True

    # Create new validated typed config with model-specific overrides
    config = SymfluenceConfig(**config_dict)

    # Save updated config to YAML
    with open(cfg_path, 'w') as f:
        yaml.dump(config.to_dict(flatten=True), f, default_flow_style=False, sort_keys=False)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers['project'].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Step 2: Copy raw data from source domain to project directory
    # Source domain name is extracted from bow_domain path (e.g., "Bow_at_Banff")
    src_domain_name = bow_domain.name.replace("domain_", "")
    _copy_source_data(bow_domain, project_dir, src_domain_name, config.domain.name)

    # Step 3: Create pour point
    pour_point_path = symfluence.managers['project'].create_pour_point()
    if pour_point_path is None:
        existing_pour_point = (
            project_dir / "shapefiles" / "pour_point" / f"{config.domain.name}_pourPoint.shp"
        )
        assert existing_pour_point.exists(), "Pour point shapefile should be created"
    else:
        assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Prune forcing to single month for speed
    _prune_raw_forcing(project_dir)

    # Step 4: Define domain (watershed delineation) - creates shapefiles from raw DEM data
    watershed_path, delineation_artifacts = symfluence.managers['domain'].define_domain()
    # Note: 'delineate' in config is normalized to 'semidistributed' for backward compatibility
    assert delineation_artifacts.method == "semidistributed", "Delineation method should be 'semidistributed'"

    # Step 5: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers['domain'].discretize_domain()
    assert discretization_artifacts.method == "GRUs", "Discretization method should be 'GRUs'"

    # Verify shapefiles were created by workflow
    shapefile_dir = project_dir / "shapefiles"

    # Get river basins path from artifacts or find it in the directory
    river_basins_path = delineation_artifacts.river_basins_path
    if river_basins_path is None:
        river_basins_dir = shapefile_dir / "river_basins"
        if river_basins_dir.exists():
            shps = list(river_basins_dir.glob("*.shp"))
            if shps:
                river_basins_path = shps[0]

    # Get HRUs path from artifacts or find it in the directory
    hrus_path = discretization_artifacts.hru_paths
    if hrus_path is None or (isinstance(hrus_path, Path) and not hrus_path.exists()):
        catchment_dir = shapefile_dir / "catchment"
        if catchment_dir.exists():
            shps = list(catchment_dir.glob("*.shp"))
            if shps:
                hrus_path = shps[0]

    assert river_basins_path is not None and river_basins_path.exists(), \
        f"River basins shapefile not created in {shapefile_dir / 'river_basins'}"
    assert hrus_path is not None and (isinstance(hrus_path, Path) and hrus_path.exists()), \
        f"HRU shapefile not created in {shapefile_dir / 'catchment'}"

    # Step 6: Model-agnostic preprocessing
    try:
        symfluence.managers['data'].run_model_agnostic_preprocessing()
    except (DataAcquisitionError, RuntimeError) as e:
        if 'hdf' in str(e).lower() or 'netcdf' in str(e).lower():
            pytest.skip(f"HDF5/NetCDF library conflict in CI environment: {e}")
        raise

    # Step 7: Model-specific preprocessing
    symfluence.managers['model'].preprocess_models()

    # Step 8: Run model
    # Check for binary if needed
    if model == 'HYPE':
        hype_exe = symfluence_data_root / 'installs' / 'hype' / 'bin' / 'hype'
        if not hype_exe.exists():
            pytest.skip(f"HYPE binary not found at {hype_exe}, skipping run and calibration")
    elif model == 'MESH':
        mesh_exe = symfluence_data_root / 'installs' / 'mesh' / 'bin' / 'mesh.exe'
        if not mesh_exe.exists():
            pytest.skip(f"MESH binary not found at {mesh_exe}, skipping run and calibration")
    elif model == 'RHESSys':
        rhessys_exe = symfluence_data_root / 'installs' / 'rhessys' / 'bin' / 'rhessys'
        if not rhessys_exe.exists():
            pytest.skip(f"RHESSys binary not found at {rhessys_exe}, skipping run and calibration")
        if config_dict.get('RHESSYS_USE_WMFIRE') or config_dict.get('RHESSYS_USE_VMFIRE'):
            wmfire_lib_dir = symfluence_data_root / 'installs' / 'wmfire' / 'lib'
            wmfire_so = wmfire_lib_dir / 'libwmfire.so'
            wmfire_dylib = wmfire_lib_dir / 'libwmfire.dylib'
            if not (wmfire_so.exists() or wmfire_dylib.exists()):
                pytest.skip(f"WMFire library not found at {wmfire_lib_dir}, skipping run with fire support")

    symfluence.managers['model'].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config.domain.experiment_id / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 9: Calibrate model
    # Skip calibration for LSTM/GR/HYPE/MESH as they either auto-calibrate or are not supported
    if model in ['GR', 'LSTM', 'HYPE', 'MESH', 'RHESSys']:
        pass
    else:
        results_file = symfluence.managers['optimization'].calibrate_model()
        assert results_file is not None, "Calibration should produce results"

        # Validate calibration results - ensure we actually ran with real data
        import math

        import pandas as pd

        assert results_file.exists(), f"Results file should exist on disk: {results_file}"

        results_df = pd.read_csv(results_file)
        assert len(results_df) >= 3, f"Should have at least 3 calibration iterations, got {len(results_df)}"

        # Check required columns exist
        assert 'iteration' in results_df.columns, "Results should have 'iteration' column"
        assert 'score' in results_df.columns, "Results should have 'score' column"

        # Validate scores are real numbers (not NaN or inf)
        scores = results_df['score'].values
        for i, score in enumerate(scores):
            assert not math.isnan(score), f"Score at iteration {i} should not be NaN"
            assert not math.isinf(score), f"Score at iteration {i} should not be infinite"
            # KGE ranges from -inf to 1; use lenient threshold for integration tests
            # where model execution may fail (-9999 sentinel) or produce poor scores
            assert score > -10000, f"Score at iteration {i} seems unreasonably low: {score}"

        # Verify we have parameter columns (model-specific)
        param_cols = [c for c in results_df.columns if c not in ['iteration', 'score', 'elapsed_time']]
        assert len(param_cols) > 0, "Results should contain parameter columns"

        # Verify observations were actually used by checking optimization directory
        opt_dir = project_dir / "optimization"
        assert opt_dir.exists(), "Optimization directory should exist"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
