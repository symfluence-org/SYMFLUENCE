"""
SYMFLUENCE Cloud Data Acquisition Integration Tests

Uses the Paradise point-scale setup to validate cloud attribute acquisition
and multiple cloud forcing datasets with short time windows.
"""

import multiprocessing
import os
import shutil
import traceback
from pathlib import Path

import pytest
import yaml

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from test_helpers.helpers import (
    has_cds_credentials,
    is_cds_data_available,
    is_em_earth_s3_available,
    is_rdrs_s3_available,
    load_config_template,
    write_config,
)

pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.requires_cloud, pytest.mark.slow]

def _ensure_summa_binary(data_root: Path, symfluence_code_dir: Path) -> None:
    """Symlink a local SUMMA binary into the test data root."""
    candidates = []
    env_exe = os.environ.get("SUMMA_EXE_PATH")
    if env_exe:
        candidates.append(Path(env_exe))

    env_install = os.environ.get("SUMMA_INSTALL_PATH")
    env_exe_name = os.environ.get("SUMMA_EXE", "summa_sundials.exe")
    if env_install:
        install_path = Path(env_install)
        candidates.extend([
            install_path / env_exe_name,
            install_path / "bin" / env_exe_name,
        ])

    repo_install = symfluence_code_dir / "installs" / "summa" / "bin"
    candidates.extend([
        repo_install / "summa_sundials.exe",
        repo_install / "summa.exe",
    ])
    data_install = data_root / "installs" / "summa" / "bin"
    candidates.extend([
        data_install / "summa_sundials.exe",
        data_install / "summa.exe",
    ])
    extra_data_roots = []
    env_data_root = os.environ.get("SYMFLUENCE_DATA_DIR")
    if env_data_root:
        extra_data_roots.append(Path(env_data_root))
    extra_data_roots.extend([
        symfluence_code_dir.parent / "SYMFLUENCE_data",
        symfluence_code_dir.parent / "data" / "SYMFLUENCE_data",
    ])
    for root in extra_data_roots:
        data_install = root / "installs" / "summa" / "bin"
        candidates.extend([
            data_install / "summa_sundials.exe",
            data_install / "summa.exe",
        ])

    for exe_name in ("summa_sundials.exe", "summa.exe"):
        which_path = shutil.which(exe_name)
        if which_path:
            candidates.append(Path(which_path))

    source = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            continue
        source = resolved
        break
    if source is None:
        pytest.skip("Skipping cloud forcing tests: SUMMA binary not found for symlink")

    dest_dir = data_root / "installs" / "summa" / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "summa_sundials.exe"
    if source == dest:
        return  # Binary already present at destination
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source)
    except OSError:
        # Windows requires admin/Developer Mode for symlinks — fall back to copy
        shutil.copy2(source, dest)

@pytest.fixture(scope="module")
def base_config(tmp_path_factory, symfluence_code_dir):
    """Create a base Paradise config for cloud acquisition tests."""
    tmp_path = tmp_path_factory.mktemp("cloud_acq")
    cfg_path = tmp_path / "test_config.yaml"

    # Load template
    config = load_config_template(symfluence_code_dir)

    # Use persistent data directory for caching attribute data
    data_root = Path(symfluence_code_dir).parent / "SYMFLUENCE_data_test_cache"
    data_root.mkdir(exist_ok=True)

    _ensure_summa_binary(data_root, Path(symfluence_code_dir))
    _ensure_summa_binary(data_root, Path(symfluence_code_dir))

    # Base paths
    config["SYMFLUENCE_DATA_DIR"] = str(data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Paradise point-scale setup (small bbox for high-resolution datasets like AORC/HRRR)
    config["DOMAIN_NAME"] = "paradise_cloud"
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"
    config["BOUNDING_BOX_COORDS"] = "46.79/-121.76/46.77/-121.74"  # Small 0.02° x 0.02° box (~2km x 2km)
    config["POUR_POINT_COORDS"] = "46.78/-121.75"

    # Cloud access for attributes/forcings
    config["DATA_ACCESS"] = "cloud"
    config["DEM_SOURCE"] = "copernicus"

    # Avoid live SNOTEL downloads in this test
    config["DOWNLOAD_SNOTEL"] = False
    config["SNOTEL_STATION"] = "679"

    # Model setup (point-scale SUMMA)
    config["HYDROLOGICAL_MODEL"] = "SUMMA"  # Ensure it's a string, not a list

    # Placeholder experiment window; per-dataset windows set in tests
    config["EXPERIMENT_ID"] = "cloud_acq"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-02 00:00"
    config["CALIBRATION_PERIOD"] = None
    config["EVALUATION_PERIOD"] = None
    config["SPINUP_PERIOD"] = None

    write_config(config, cfg_path)

    return cfg_path


@pytest.fixture(scope="module")
def prepared_project(base_config):
    """Acquire cloud attributes once and set up the point-scale domain."""
    symfluence = SYMFLUENCE(base_config)

    project_dir = symfluence.managers["project"].setup_project()
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Check if attribute data already exists to avoid re-downloading
    domain_name = symfluence.config['DOMAIN_NAME']
    data_dir = Path(symfluence.config["SYMFLUENCE_DATA_DIR"]) / f"domain_{domain_name}"

    # Check for existing attribute files
    dem_file = data_dir / "attributes" / "elevation" / f"domain_{domain_name}_elevation.tif"
    soil_file = data_dir / "attributes" / "soilclass" / f"domain_{domain_name}_soil_classes.tif"
    land_file = data_dir / "attributes" / "landclass" / f"domain_{domain_name}_land_classes.tif"

    # Only acquire attributes if they don't exist
    if not (dem_file.exists() and soil_file.exists() and land_file.exists()):
        print("Acquiring cloud attributes (DEM, soil, land cover)...")
        # Acquire cloud attributes: Copernicus DEM, MODIS land cover, soil classes
        symfluence.managers["data"].acquire_attributes()
    else:
        print("Using cached attribute data...")

    # Skip if SoilGrids download produced an unreadable raster.
    soil_raster = soil_file
    try:
        import rasterio
        with rasterio.open(soil_raster):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Skipping cloud forcing tests: SoilGrids raster unreadable ({exc})")

    # Define and discretize the point domain
    symfluence.managers["domain"].define_domain()
    symfluence.managers["domain"].discretize_domain()

    return base_config, project_dir


FORCING_CASES = [
    {
        "dataset": "ERA5",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": ["ERA5_*.nc", "*ERA5_merged_*.nc", "*ERA5_CDS_*.nc"],  # CDS pathway also produces ERA5_CDS files
    },
    # ERA5_CDS removed - not a valid ForcingDatasetType. Use ERA5 with ERA5_USE_CDS=True instead.
    {
        "dataset": "AORC",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": "*AORC_*.nc",
    },
    # Skip NEX-GDDP-CMIP6 due to pandas compatibility issue with EASYMORE
    # {
    #     "dataset": "NEX-GDDP-CMIP6",
    #     "start": "2010-01-01 00:00",
    #     "end": "2010-01-02 00:00",  # 1 day (NEX is daily data)
    #     "expect_glob": "NEXGDDP_all_*.nc",
    #     "extras": {
    #         "NEX_MODELS": ["ACCESS-CM2"],
    #         "NEX_SCENARIOS": ["historical"],
    #         "NEX_ENSEMBLES": ["r1i1p1f1"],
    #         "NEX_VARIABLES": ["pr", "tas", "huss", "rlds", "rsds", "sfcWind"],
    #     },
    # },
    {
        "dataset": "CONUS404",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": "*CONUS404_*.nc",
    },
    {
        "dataset": "HRRR",
        "start": "2020-01-01 00:00",
        "end": "2020-01-01 01:00",  # Just 1 hour
        "expect_glob": ["paradise_cloud_HRRR_hourly_*.nc", "HRRR_*.nc"],
        "extras": {
            "HRRR_BOUNDING_BOX_COORDS": "46.79/-121.76/46.77/-121.74",
            "HRRR_VARS": ["TMP", "SPFH", "PRES", "UGRD", "VGRD", "DSWRF", "DLWRF"],  # All SUMMA-required variables
            "HRRR_BUFFER_CELLS": 0,
        },
    },
    {
        "dataset": "CARRA",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 06:00",  # 6 hours (2 timesteps for 3-hourly data)
        "expect_glob": "*CARRA*.nc",
        "domain_override": {
            "DOMAIN_NAME": "ellioaar_iceland",
            "BOUNDING_BOX_COORDS": "64.13/-21.96/64.11/-21.94",  # Elliðaár, Reykjavik (very small ~2km x 2km)
            "POUR_POINT_COORDS": "64.12/-21.95",
        },
        "extras": {
            # Remove CARRA_DOMAIN to use bounding box instead of full domain
        },
    },
    {
        "dataset": "CERRA",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 03:00",  # 3 hours (1 timestep for 3-hourly data)
        "expect_glob": "*CERRA*.nc",
        "domain_override": {
            "DOMAIN_NAME": "fyris_uppsala",
            "BOUNDING_BOX_COORDS": "59.87/17.64/59.85/17.66",  # Fyrisån, Uppsala (very small ~2km x 2km)
            "POUR_POINT_COORDS": "59.86/17.65",
        },
    },
    {
        "dataset": "RDRS",
        "start": "2015-01-01 00:00",
        "end": "2015-01-01 01:00",
        "expect_glob": "*RDRS*.nc",
        "domain_override": {
            "DOMAIN_NAME": "bow_banff",
            "BOUNDING_BOX_COORDS": "51.20/-115.60/51.15/-115.55",
            "POUR_POINT_COORDS": "51.17/-115.57",
        },
    },
    {
        "dataset": "EM-EARTH",
        "start": "2010-01-01 00:00",
        "end": "2010-01-03 00:00",  # 2 days (EM-Earth is daily data)
        "expect_glob": "*EM-Earth_*.nc",
    },
]

def _selected_cases():
    """Optionally filter forcing cases with CLOUD_DATASET env var."""
    selected = os.environ.get("CLOUD_DATASET")
    if not selected:
        return FORCING_CASES
    selected_norm = selected.strip().lower()
    return [
        case
        for case in FORCING_CASES
        if case["dataset"].strip().lower() == selected_norm
    ]

def _run_case_logic(cfg_path: Path, project_dir: Path, case: dict) -> None:
    # Load base config and update for this dataset
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Use larger bounding box for coarse resolution datasets (ERA5, CONUS404)
    if case.get("dataset") in ["ERA5", "ERA5_CDS", "CONUS404"]:
        # ERA5 has 0.25° resolution, need at least 0.5° x 0.5° box to get data
        # CONUS404 also has coarse resolution
        config["BOUNDING_BOX_COORDS"] = "46.85/-121.85/46.70/-121.65"  # ~0.15° x 0.20° box (~15km x 20km)

    # Handle domain override for datasets requiring different geographic locations
    # (e.g., CARRA needs Arctic, CERRA needs Europe)
    if "domain_override" in case:
        for key, value in case["domain_override"].items():
            config[key] = value
        # Update project_dir to match new domain name
        data_root = Path(config["SYMFLUENCE_DATA_DIR"])
        project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"

        # Save updated config first (before initializing SYMFLUENCE)
        write_config(config, cfg_path)

        # Setup new domain if needed
        symfluence_temp = SYMFLUENCE(cfg_path)
        # Check if domain is fully set up (HRUs file exists)
        domain_name = config["DOMAIN_NAME"]
        hrus_file = project_dir / "shapefiles" / "catchment" / f"{domain_name}_HRUs_GRUs.shp"
        if not hrus_file.exists():
            symfluence_temp.managers["project"].setup_project()
            pour_point_path = symfluence_temp.managers["project"].create_pour_point()
            assert Path(pour_point_path).exists(), "Pour point shapefile should be created"
            symfluence_temp.managers["data"].acquire_attributes()
            symfluence_temp.managers["domain"].define_domain()
            symfluence_temp.managers["domain"].discretize_domain()

    config["FORCING_DATASET"] = case["dataset"]
    config["EXPERIMENT_TIME_START"] = case["start"]
    config["EXPERIMENT_TIME_END"] = case["end"]
    config["EXPERIMENT_ID"] = f"cloud_{case['dataset'].lower().replace('-', '_')}"

    # Copy catchment shapefiles from whatever prior experiment dir exists to the new one
    domain_def = config.get("DOMAIN_DEFINITION_METHOD", "point")
    catchment_parent = project_dir / "shapefiles" / "catchment" / domain_def
    new_catchment = catchment_parent / config["EXPERIMENT_ID"]
    if catchment_parent.exists() and not new_catchment.exists():
        # Find any existing experiment subdirectory with shapefiles
        existing = [d for d in catchment_parent.iterdir() if d.is_dir() and list(d.glob("*.shp"))]
        if existing:
            new_catchment.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(existing[0], new_catchment)

    # Ensure HYDROLOGICAL_MODEL stays as string (not list)
    if isinstance(config.get("HYDROLOGICAL_MODEL"), list):
        config["HYDROLOGICAL_MODEL"] = config["HYDROLOGICAL_MODEL"][0]
    elif "HYDROLOGICAL_MODEL" not in config:
        config["HYDROLOGICAL_MODEL"] = "SUMMA"

    for key, value in case.get("extras", {}).items():
        config[key] = value

    write_config(config, cfg_path)

    # Clean forcing outputs from prior dataset runs
    for subdir in ["raw_data", "basin_averaged_data", "merged_path"]:
        shutil.rmtree(project_dir / "data" / "forcing" / subdir, ignore_errors=True)

    symfluence = SYMFLUENCE(cfg_path)
    symfluence.managers["data"].acquire_forcings()

    raw_data_dir = project_dir / "data" / "forcing" / "raw_data"
    expect_glob = case["expect_glob"]
    if isinstance(expect_glob, (list, tuple)):
        matches = []
        for pattern in expect_glob:
            matches.extend(raw_data_dir.glob(pattern))
    else:
        matches = list(raw_data_dir.glob(expect_glob))
    assert matches, f"No forcing output found for {case['dataset']} in {raw_data_dir}"

    # Run full preprocessing and model
    symfluence.managers["data"].run_model_agnostic_preprocessing()
    symfluence.managers["model"].preprocess_models()
    symfluence.managers["model"].run_models()

    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / "SUMMA"
    assert sim_dir.exists(), f"SUMMA simulation output directory missing for {case['dataset']}"

def _run_case_worker(cfg_path_str: str, project_dir_str: str, case: dict, result_queue) -> None:
    try:
        _run_case_logic(Path(cfg_path_str), Path(project_dir_str), case)
        result_queue.put({"ok": True})
    except Exception:  # noqa: BLE001
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})

def _execute_case_in_subprocess(cfg_path: Path, project_dir: Path, case: dict) -> None:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_case_worker,
        args=(str(cfg_path), str(project_dir), case, result_queue),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        error = "subprocess crashed"
        if not result_queue.empty():
            payload = result_queue.get()
            if not payload.get("ok"):
                error = payload.get("traceback", error)
        raise AssertionError(
            f"Case {case['dataset']} failed in subprocess (exitcode {proc.exitcode}):\n{error}"
        )

    if not result_queue.empty():
        payload = result_queue.get()
        if not payload.get("ok"):
            raise AssertionError(
                f"Case {case['dataset']} failed in subprocess:\n{payload.get('traceback')}"
            )


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.timeout(600)  # CDS-based cases (CARRA, CERRA) involve server-side queuing
@pytest.mark.parametrize("case", _selected_cases())
def test_cloud_forcing_acquisition(prepared_project, case):
    """
    Download a short forcing window for each cloud-supported dataset, then
    run the full preprocessing + model pipeline.
    """
    # Skip CDS-based tests if CDS credentials are not available
    if case["dataset"] in ["CARRA", "CERRA", "ERA5_CDS"] and not has_cds_credentials():
        pytest.skip(f"Skipping {case['dataset']} test: CDS API credentials not found in ~/.cdsapirc")

    # Skip CARRA if CDS API access is restricted (external service issue)
    if case["dataset"] == "CARRA" and not is_cds_data_available("reanalysis-carra-single-levels"):
        pytest.skip("Skipping CARRA test: CDS API reanalysis data access currently restricted")

    # Skip RDRS if S3 access is restricted (external service issue)
    if case["dataset"] == "RDRS" and not is_rdrs_s3_available():
        pytest.skip("Skipping RDRS test: S3 Zarr store access restricted (external data source unavailable)")

    # Skip EM-EARTH if S3 access is restricted
    if case["dataset"] == "EM-EARTH" and not is_em_earth_s3_available():
        pytest.skip("Skipping EM-EARTH test: S3 bucket access restricted (anonymous access not available)")

    cfg_path, project_dir = prepared_project

    _execute_case_in_subprocess(cfg_path, project_dir, case)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
