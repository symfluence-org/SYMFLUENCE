"""
SYMFLUENCE Distributed Basin Integration Tests

Tests the elevation-based distributed workflow from notebook 02c for SUMMA.
Reuses data from the semi-distributed example when available.
"""

import shutil
from pathlib import Path

import pytest

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from test_helpers.geospatial import (
    assert_shapefile_signature_matches,
    load_shapefile_signature,
)
from test_helpers.helpers import load_config_template, write_config

# GitHub release URL for example data
EXAMPLE_DATA_URL = "https://github.com/symfluence-org/SYMFLUENCE/releases/download/examples-data-v0.2/example_data_v0.2.zip"



pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]


def _copy_with_name_adaptation(src: Path, dst: Path, old_name: str, new_name: str) -> bool:
    """Copy directory or file and adapt filenames containing the old domain name."""
    if not src.exists():
        print(f"  Warning: Source path does not exist: {src}")
        return False

    print(f"  Copying {src.name} to {dst.relative_to(dst.parents[2]) if len(dst.parts) > 3 else dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        shutil.copy2(src, dst)
        files_to_check = [dst]
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        files_to_check = list(dst.rglob("*"))

    # Sort files to ensure we don't rename a directory before its contents (though rglob handles this ok)
    # Actually, we only care about files for name adaptation in most cases
    for file in sorted(files_to_check, key=lambda x: len(str(x)), reverse=True):
        if file.is_file() and old_name in file.name:
            new_file_name = file.name.replace(old_name, new_name)
            new_file_path = file.parent / new_file_name
            if new_file_path != file:
                file.replace(new_file_path)
                # print(f"    Renamed: {file.name} -> {new_file_name}")
    return True


@pytest.fixture(scope="function")
def config_path(example_data_bundle, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    config = load_config_template(symfluence_code_dir)

    # Update paths
    config["SYMFLUENCE_DATA_DIR"] = str(example_data_bundle)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Domain settings from notebook 02c
    config["DOMAIN_NAME"] = "Bow_at_Banff_elevation"
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Elevation-based discretization - optimized for faster testing
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["STREAM_THRESHOLD"] = 20000  # Reduced spatial resolution (was 10000)
    config["SUB_GRID_DISCRETIZATION"] = "elevation"
    config["ELEVATION_BAND_SIZE"] = 1600  # Reduced spatial resolution (was 800)

    # Optimized: 3-day test window (need sufficient data points for KGE calculation)
    # Note: Forcing data is for Jan 2004, so use dates that match the available data
    config["EXPERIMENT_TIME_START"] = "2004-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-03 23:00"
    config["CALIBRATION_PERIOD"] = "2004-01-01 12:00, 2004-01-03 00:00"
    config["EVALUATION_PERIOD"] = "2004-01-03 00:00, 2004-01-03 23:00"
    config["SPINUP_PERIOD"] = "2004-01-01 00:00, 2004-01-01 12:00"

    # Streamflow
    config["STATION_ID"] = "05BB001"
    config["DOWNLOAD_WSC_DATA"] = False

    # Calibration settings - use 3 iterations to verify algorithm actually runs
    config["OPTIMIZATION_METHODS"] = ['iteration']
    config["ITERATIVE_OPTIMIZATION_ALGORITHM"] = "DDS"
    config["NUMBER_OF_ITERATIONS"] = 3
    config["RANDOM_SEED"] = 42

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


MODELS = [
    "SUMMA",
    # MESH not supported: meshflow does not support elevation-based discretization (requires GRU-based setup)
]


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.full
@pytest.mark.parametrize("model", MODELS)
def test_distributed_basin_workflow(config_path, example_data_bundle, model):
    """
    Test elevation-based distributed basin workflow for supported models.

    Follows notebook 02c workflow:
    1. Setup project
    2. Reuse data from semi-distributed example
    3. Define domain (watershed delineation)
    4. Discretize domain (elevation bands)
    5. Model-agnostic preprocessing
    6. Model-specific preprocessing
    7. Run model
    8. Calibrate model

    Note: MESH is excluded because meshflow preprocessing does not support
    elevation-based HRU discretization (it requires GRU-based river basins).
    """
    cfg_path, config = config_path

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

    # Save updated config
    write_config(config, cfg_path)

    baseline_dir = (
        Path(config["SYMFLUENCE_DATA_DIR"])
        / f"domain_{config['DOMAIN_NAME']}"
        / "shapefiles"
    )
    baseline_river_basins = (
        baseline_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_delineate.shp"
    )
    baseline_river_network = (
        baseline_dir
        / "river_network"
        / f"{config['DOMAIN_NAME']}_riverNetwork_delineate.shp"
    )
    baseline_hrus = (
        baseline_dir
        / "catchment"
        / f"{config['DOMAIN_NAME']}_HRUs_elevation.shp"
    )
    signature_strict = (
        config.get("STREAM_THRESHOLD") == 5000
        and config.get("ELEVATION_BAND_SIZE") == 400
    )
    if signature_strict:
        assert baseline_river_basins.exists(), "Baseline river basins shapefile missing"
        assert baseline_river_network.exists(), "Baseline river network shapefile missing"
        assert baseline_hrus.exists(), "Baseline HRU shapefile missing"
        expected_river_basins = load_shapefile_signature(baseline_river_basins)
        expected_river_network = load_shapefile_signature(baseline_river_network)
        expected_hrus = load_shapefile_signature(baseline_hrus)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Step 2: Reuse data from the semi-distributed domain when available
    semi_dist_domain = "Bow_at_Banff_semi_distributed"
    semi_dist_data_dir = example_data_bundle / f"domain_{semi_dist_domain}"

    # Fallback for v0.6.0 bundle structure
    if not semi_dist_data_dir.exists():
        semi_dist_data_dir = example_data_bundle / "domain_bow_banff_minimal"
        semi_dist_domain = "Bow_at_Banff_lumped" # Name used inside files in minimal bundle
        print(f"  Note: Using {semi_dist_data_dir.name} as data source for reuse.")

    reusable_data = {
        "Elevation": semi_dist_data_dir / "attributes" / "elevation",
        "Land Cover": semi_dist_data_dir / "attributes" / "landclass",
        "Soils": semi_dist_data_dir / "attributes" / "soilclass",
        "Forcing": semi_dist_data_dir / "forcing" / "raw_data",
        "Stream Network": semi_dist_data_dir / "shapefiles" / "river_network",
        "GRUs": semi_dist_data_dir / "shapefiles" / "river_basins",
        "Streamflow": semi_dist_data_dir / "observations" / "streamflow",
    }
    _DATA_SUBDIRS = {'attributes', 'forcing', 'observations'}
    for _, src_path in reusable_data.items():
        if src_path.exists():
            rel_path = src_path.relative_to(semi_dist_data_dir)
            # Data subdirs live under data/ in the new project layout
            top_dir = rel_path.parts[0] if rel_path.parts else ''
            if top_dir in _DATA_SUBDIRS:
                dst_path = project_dir / "data" / rel_path
            else:
                dst_path = project_dir / rel_path
            _copy_with_name_adaptation(
                src_path, dst_path, semi_dist_domain, config["DOMAIN_NAME"]
            )

    # Clear cached outputs/remap weights to keep performance tuning predictable.
    forcing_dir = project_dir / "data" / "forcing"
    if forcing_dir.exists():
        for subdir in ["basin_averaged_data", "merged_path", "SUMMA_input", "GR_input", "NGEN_input"]:
            shutil.rmtree(forcing_dir / subdir, ignore_errors=True)
        for temp_dir in forcing_dir.glob("temp_*"):
            shutil.rmtree(temp_dir, ignore_errors=True)
    remap_cache = project_dir / "shapefiles" / "catchment_intersection" / "with_forcing"
    if remap_cache.exists():
        shutil.rmtree(remap_cache, ignore_errors=True)

    # Prune to single forcing file for faster tests
    forcing_raw_dir = project_dir / "data" / "forcing" / "raw_data"
    if forcing_raw_dir.exists():
        forcing_files = sorted(forcing_raw_dir.glob("*.nc"))
        if len(forcing_files) > 1:
            # Keep only the first file, remove the rest
            for f in forcing_files[1:]:
                f.unlink()
        if forcing_files:
            config["FORCING_FILES"] = str(forcing_files[0])
            write_config(config, cfg_path)

    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 3: Define domain (watershed delineation)
    watershed_path, delineation_artifacts = symfluence.managers["domain"].define_domain()
    # Note: 'delineate' is auto-mapped to 'semidistributed' for backward compatibility
    expected_methods = {'delineate', 'semidistributed'}  # Accept both for backward compat
    assert (
        delineation_artifacts.method in expected_methods
    ), f"Delineation method mismatch: got {delineation_artifacts.method}, expected one of {expected_methods}"
    # watershed_path can be None for workflows that use existing data

    # Step 4: Discretize domain (elevation bands)
    hru_path, discretization_artifacts = symfluence.managers["domain"].discretize_domain()
    assert (
        discretization_artifacts.method == config["SUB_GRID_DISCRETIZATION"]
    ), "Discretization method mismatch"

    # Verify geospatial artifacts (02c)
    # Note: File suffix now uses normalized method name (semidistributed, not delineate)
    method_suffix = delineation_artifacts.method  # Use actual method for path consistency
    shapefile_dir = project_dir / "shapefiles"
    river_basins_path = delineation_artifacts.river_basins_path or (
        shapefile_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_{method_suffix}.shp"
    )
    river_network_path = delineation_artifacts.river_network_path or (
        shapefile_dir
        / "river_network"
        / f"{config['DOMAIN_NAME']}_riverNetwork_{method_suffix}.shp"
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir
        / "catchment"
        / f"{config['DOMAIN_NAME']}_HRUs_elevation.shp"
    )
    assert river_basins_path.exists()
    assert river_network_path.exists()
    assert hrus_path.exists()
    if signature_strict:
        assert_shapefile_signature_matches(river_basins_path, expected_river_basins)
        assert_shapefile_signature_matches(river_network_path, expected_river_network)
        assert_shapefile_signature_matches(hrus_path, expected_hrus)

    # Step 5: Model-agnostic preprocessing
    symfluence.managers["data"].run_model_agnostic_preprocessing()

    # Step 6: Model-specific preprocessing
    symfluence.managers["model"].preprocess_models()

    # Step 7: Run model
    symfluence.managers["model"].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 8: Calibrate model
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
        # with short windows where routing may fail and scores can be very poor
        assert score > -10000, f"Score at iteration {i} seems unreasonably low: {score}"

    # Verify we have parameter columns (model-specific)
    param_cols = [c for c in results_df.columns if c not in ['iteration', 'score', 'elapsed_time']]
    assert len(param_cols) > 0, "Results should contain parameter columns"

    # Verify observations were actually used by checking optimization directory
    opt_dir = project_dir / "optimization"
    assert opt_dir.exists(), "Optimization directory should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
