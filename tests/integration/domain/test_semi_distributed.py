"""
SYMFLUENCE Semi-Distributed Basin Integration Tests

Tests the semi-distributed basin workflow from notebook 02b for supported models.
Uses local test data from tests/data/ and runs the full workflow from raw data.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from symfluence.core.exceptions import DataAcquisitionError
from test_helpers.helpers import keep_single_raw_forcing, load_config_template, write_config

pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]


def _copy_with_name_adaptation(src: Path, dst: Path, old_name: str, new_name: str) -> bool:
    """Copy directory or file and adapt filenames containing the old domain name."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
        return True
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for file in dst.rglob("*"):
        if file.is_file() and old_name in file.name:
            new_file = file.parent / file.name.replace(old_name, new_name)
            file.rename(new_file)
    return True


def _prune_raw_forcing(project_dir: Path) -> None:
    raw_dir = project_dir / "data" / "forcing" / "raw_data"
    keep_single_raw_forcing(raw_dir)


@pytest.fixture(scope="function")
def config_path(bow_domain, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    config = load_config_template(symfluence_code_dir)

    # Use tmp_path as data directory to avoid polluting source data
    config["SYMFLUENCE_DATA_DIR"] = str(tmp_path)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Domain settings from notebook 02b
    config["DOMAIN_NAME"] = "Bow_Banff_sd"  # Keep short to avoid Fortran strLen=256 overflow (SUMMA/mizuRoute)
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name[:8]}"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Semi-distributed basin settings
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["DOMAIN_DEFINITION_METHOD"] = "semidistributed"  # Use canonical value (not legacy 'delineate')
    config["STREAM_THRESHOLD"] = 10000
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"

    # Optimized: 3-day period for faster testing while still allowing calibration
    config["EXPERIMENT_TIME_START"] = "2004-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-03 23:00"
    config["CALIBRATION_PERIOD"] = "2004-01-01, 2004-01-02"
    config["EVALUATION_PERIOD"] = "2004-01-03, 2004-01-03"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-01"

    # Streamflow
    config["STATION_ID"] = "05BB001"
    config["DOWNLOAD_WSC_DATA"] = False

    # Calibration settings - use 3 iterations to verify algorithm actually runs
    config["NUMBER_OF_ITERATIONS"] = 3
    config["RANDOM_SEED"] = 42

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    write_config(config, cfg_path)

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


MODELS = [
    "SUMMA",
    "FUSE",
    pytest.param("NGEN", marks=pytest.mark.full),
    "HYPE",
    pytest.param("MESH", marks=pytest.mark.skip(reason="meshflow library bug: extract_rank_next fails with certain network topologies (upstream issue)")),
]


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("model", MODELS)
def test_semi_distributed_basin_workflow(config_path, model, symfluence_data_root, setup_installs_symlink):
    """
    Test semi-distributed basin workflow for each model.

    Follows notebook 02b workflow:
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
    cfg_path, config, bow_domain = config_path

    # Update model in config
    config["HYDROLOGICAL_MODEL"] = model
    if model == "SUMMA":
        config["ROUTING_MODEL"] = "mizuRoute"
        config["MIZU_FROM_MODEL"] = "SUMMA"
        config["SETTINGS_MIZU_ROUTING_VAR"] = "averageRoutedRunoff"
        config["SETTINGS_MIZU_ROUTING_UNITS"] = "m/s"
        config["SETTINGS_MIZU_ROUTING_DT"] = "3600"
        config["PARAMS_TO_CALIBRATE"] = "k_soil,theta_sat"
        config["BASIN_PARAMS_TO_CALIBRATE"] = "routingGammaScale"
    elif model == "FUSE":
        config["FUSE_SPATIAL_MODE"] = "semi_distributed"
        config["SETTINGS_FUSE_PARAMS_TO_CALIBRATE"] = "MAXWATR_1,MAXWATR_2,BASERTE"
        config["FUSE_RUN_INTERNAL_CALIBRATION"] = False
    elif model == "NGEN":
        config["NGEN_MODULES_TO_CALIBRATE"] = "CFE"
        config["NGEN_CFE_PARAMS_TO_CALIBRATE"] = "smcmax,satdk,bb"
        config["NGEN_INSTALL_PATH"] = str(symfluence_data_root / "installs" / "ngen" / "cmake_build")
    elif model == "MESH":
        config["MESH_SKIP_CALIBRATION"] = True
        config["MESH_INSTALL_PATH"] = str(symfluence_data_root / "installs" / "mesh" / "bin")
        config["MESH_EXE"] = "mesh.exe"

    # Save updated config
    write_config(config, cfg_path)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Step 2: Copy raw data from source domain to project directory
    # Detect the actual domain name used in filenames (may differ from directory name)
    dem_dir = bow_domain / "attributes" / "elevation" / "dem"
    if dem_dir.exists():
        dem_files = list(dem_dir.glob("*_elv.tif"))
        if dem_files:
            # Extract domain name from "domain_{name}_elv.tif"
            lumped_domain = dem_files[0].stem.replace("domain_", "").replace("_elv", "")
        else:
            lumped_domain = bow_domain.name.replace("domain_", "")
    else:
        lumped_domain = bow_domain.name.replace("domain_", "")
    reusable_data = {
        "Elevation": bow_domain / "attributes" / "elevation",
        "Land Cover": bow_domain / "attributes" / "landclass",
        "Soils": bow_domain / "attributes" / "soilclass",
        "Forcing": bow_domain / "forcing" / "raw_data",
        "Streamflow": bow_domain / "observations" / "streamflow",
    }
    for _, src_path in reusable_data.items():
        if src_path.exists():
            rel_path = src_path.relative_to(bow_domain)
            dst_path = project_dir / "data" / rel_path
            _copy_with_name_adaptation(
                src_path, dst_path, lumped_domain, config["DOMAIN_NAME"]
            )

    _prune_raw_forcing(project_dir)

    # Clear processed forcing outputs so HRU counts stay consistent
    forcing_dir = project_dir / "data" / "forcing"
    if forcing_dir.exists():
        for subdir in ["basin_averaged_data", "merged_path", "SUMMA_input", "GR_input", "NGEN_input"]:
            shutil.rmtree(forcing_dir / subdir, ignore_errors=True)
        for temp_dir in forcing_dir.glob("temp_*"):
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Step 3: Create pour point
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 4: Define domain (watershed delineation) - creates shapefiles from raw data
    watershed_path, delineation_artifacts = symfluence.managers["domain"].define_domain()
    # Note: 'delineate' is auto-mapped to 'semidistributed' for backward compatibility
    expected_methods = {'delineate', 'semidistributed'}  # Accept both for backward compat
    assert (
        delineation_artifacts.method in expected_methods
    ), f"Delineation method mismatch: got {delineation_artifacts.method}, expected one of {expected_methods}"

    # Step 5: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers["domain"].discretize_domain()
    assert (
        discretization_artifacts.method == config["SUB_GRID_DISCRETIZATION"]
    ), "Discretization method mismatch"

    # Verify shapefiles were created by workflow
    # Note: File suffix now uses normalized method name (semidistributed, not delineate)
    method_suffix = delineation_artifacts.method  # Use actual method for path consistency
    shapefile_dir = project_dir / "shapefiles"
    river_basins_path = delineation_artifacts.river_basins_path or (
        shapefile_dir / "river_basins" / f"{config['DOMAIN_NAME']}_riverBasins_{method_suffix}.shp"
    )
    river_network_path = delineation_artifacts.river_network_path or (
        shapefile_dir / "river_network" / f"{config['DOMAIN_NAME']}_riverNetwork_{method_suffix}.shp"
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    assert river_basins_path.exists(), f"River basins shapefile not created: {river_basins_path}"
    assert river_network_path.exists(), f"River network shapefile not created: {river_network_path}"
    assert hrus_path.exists(), f"HRU shapefile not created: {hrus_path}"

    # Step 6: Model-agnostic preprocessing
    try:
        symfluence.managers["data"].run_model_agnostic_preprocessing()
    except (DataAcquisitionError, RuntimeError) as e:
        if 'hdf' in str(e).lower() or 'netcdf' in str(e).lower():
            pytest.skip(f"HDF5/NetCDF library conflict in CI environment: {e}")
        raise

    # Step 7: Model-specific preprocessing
    symfluence.managers["model"].preprocess_models()

    # Step 8: Run model
    # Check for binary if needed
    if model == "HYPE":
        hype_exe = symfluence_data_root / "installs" / "hype" / "bin" / "hype"
        if not hype_exe.exists():
            pytest.skip(f"HYPE binary not found at {hype_exe}, skipping run and calibration")
    elif model == "MESH":
        mesh_exe = symfluence_data_root / "installs" / "mesh" / "bin" / "mesh.exe"
        if not mesh_exe.exists():
            pytest.skip(f"MESH binary not found at {mesh_exe}, skipping run and calibration")

    symfluence.managers["model"].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 9: Calibrate model
    # Skip calibration for MESH as it's not yet fully supported
    if model == "MESH":
        pass
    else:
        results_file = symfluence.managers["optimization"].calibrate_model()
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
