"""
SYMFLUENCE Point-Scale Integration Tests

Tests the point-scale workflow from notebook 01a (Paradise SNOTEL example).
Runs a short SUMMA simulation for a point domain.
"""

import shutil
import zipfile
from pathlib import Path

import pytest
import requests

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from symfluence.core.exceptions import DataAcquisitionError
from test_helpers.geospatial import (
    assert_shapefile_signature_matches,
    load_shapefile_signature,
)
from test_helpers.helpers import load_config_template, write_config

# GitHub release URL for example data
EXAMPLE_DATA_URL = "https://github.com/symfluence-org/SYMFLUENCE/releases/download/examples-data-v0.5.5/example_data_v0.5.5.zip"



pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]

@pytest.fixture(scope="module")
def test_data_dir(symfluence_data_root):
    """Ensure example data exists in a writable SYMFLUENCE_data directory."""
    data_root = symfluence_data_root

    # Check if example domain already exists
    example_domain = "domain_paradise"
    example_domain_path = data_root / example_domain

    # Download if it doesn't exist
    if not example_domain_path.exists():
        print(f"\nDownloading example data to {data_root}...")
        zip_path = data_root / "example_data_v0.5.5.zip"

        # Download
        response = requests.get(EXAMPLE_DATA_URL, stream=True, timeout=600)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting example data...")
        # Extract to a temp location
        extract_dir = data_root / "temp_extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Move the domain to data root
        example_data_dir = extract_dir / "example_data_v0.5.5"
        src_domain = example_data_dir / example_domain

        if src_domain.exists():
            src_domain.rename(example_domain_path)
            print(f"Created domain: {example_domain}")
        else:
            raise FileNotFoundError(f"{example_domain} not found in downloaded data")

        # Cleanup
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

        print(f"Test data ready at {example_domain_path}")
    else:
        print(f"Using existing test data at {example_domain_path}")

    return data_root


@pytest.fixture(scope="function")
def config_path(test_data_dir, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    config = load_config_template(symfluence_code_dir)

    # Update paths
    config["SYMFLUENCE_DATA_DIR"] = str(test_data_dir)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Point-scale settings from notebook 01a
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"
    config["BOUNDING_BOX_COORDS"] = "46.781/-121.751/46.779/-121.749"
    config["POUR_POINT_COORDS"] = "46.78/-121.75"

    # Data sources
    config["DOWNLOAD_SNOTEL"] = False
    config["SNOTEL_STATION"] = "679"

    # Model and forcing
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["FORCING_DATASET"] = "ERA5"

    # Optimized: 1-day test period for faster testing
    config["EXPERIMENT_TIME_START"] = "2000-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2000-01-01 23:00"
    config["CALIBRATION_PERIOD"] = "2000-01-01 06:00, 2000-01-01 18:00"
    config["EVALUATION_PERIOD"] = "2000-01-01 06:00, 2000-01-01 18:00"
    config["SPINUP_PERIOD"] = "2000-01-01 00:00, 2000-01-01 06:00"

    # Domain and experiment ids
    config["DOMAIN_NAME"] = "paradise"
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


@pytest.mark.slow
@pytest.mark.requires_data
def test_point_scale_workflow(config_path):
    """
    Test point-scale workflow for SUMMA.

    Follows notebook 01a workflow:
    1. Setup project
    2. Create pour point
    3. Define domain (point)
    4. Discretize domain
    5. Model-agnostic preprocessing
    6. Model-specific preprocessing
    7. Run model
    """
    cfg_path, config = config_path

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Prune to single forcing file for faster tests (before preprocessing)
    forcing_raw_dir = project_dir / "forcing" / "raw_data"
    if forcing_raw_dir.exists():
        forcing_files = sorted(forcing_raw_dir.glob("*.nc"))
        if len(forcing_files) > 1:
            # Keep only the first file, remove the rest
            for f in forcing_files[1:]:
                f.unlink()
            print(f"Pruned forcing files: kept 1 out of {len(forcing_files)} files")

    # Step 2: Create pour point
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 3: Define domain (point)
    domain_path, delineation_artifacts = symfluence.managers["domain"].define_domain()
    assert (
        delineation_artifacts.method == config["DOMAIN_DEFINITION_METHOD"]
    ), "Delineation method mismatch"

    # Step 4: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers["domain"].discretize_domain()
    assert (
        discretization_artifacts.method == config["SUB_GRID_DISCRETIZATION"]
    ), "Discretization method mismatch"

    # Verify geospatial artifacts (01a)
    shapefile_dir = project_dir / "shapefiles"
    river_basins_path = delineation_artifacts.river_basins_path or (
        shapefile_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_point.shp"
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    assert river_basins_path.exists()
    assert hrus_path.exists()

    # Only check signature if baseline exists
    baseline_dir = (
        Path(config["SYMFLUENCE_DATA_DIR"])
        / f"domain_{config['DOMAIN_NAME']}"
        / "shapefiles"
    )
    baseline_river_basins = (
        baseline_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_point.shp"
    )
    baseline_hrus = (
        baseline_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    if baseline_river_basins.exists() and baseline_hrus.exists():
        expected_river_basins = load_shapefile_signature(baseline_river_basins)
        expected_hrus = load_shapefile_signature(baseline_hrus)
        assert_shapefile_signature_matches(river_basins_path, expected_river_basins)
        assert_shapefile_signature_matches(hrus_path, expected_hrus)

    # Step 5: Model-agnostic preprocessing
    try:
        symfluence.managers["data"].run_model_agnostic_preprocessing()
    except (DataAcquisitionError, RuntimeError) as e:
        if 'hdf' in str(e).lower() or 'netcdf' in str(e).lower():
            pytest.skip(f"HDF5/NetCDF library conflict in CI environment: {e}")
        raise

    # Step 6: Model-specific preprocessing
    symfluence.managers["model"].preprocess_models()

    # Step 7: Run model
    symfluence.managers["model"].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / "SUMMA"
    assert sim_dir.exists(), "SUMMA simulation output directory should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
