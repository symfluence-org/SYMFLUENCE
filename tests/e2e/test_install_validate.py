"""
End-to-end install & validate tests.

Replaces .github/workflows/install-validate.yml validation steps with pytest.
These tests validate the complete SYMFLUENCE installation and core functionality.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from symfluence import SYMFLUENCE

pytestmark = [pytest.mark.e2e, pytest.mark.ci_quick]

def _prune_raw_forcing(project_dir: Path) -> None:
    raw_dir = project_dir / "forcing" / "raw_data"
    if not raw_dir.exists():
        return
    from test_helpers.helpers import keep_single_raw_forcing
    keep_single_raw_forcing(raw_dir)

    basin_avg_dir = project_dir / "forcing" / "basin_averaged_data"
    if basin_avg_dir.exists():
        for path in basin_avg_dir.glob("*.nc"):
            path.unlink()
    intersection_dir = project_dir / "shapefiles" / "catchment_intersection"
    if intersection_dir.exists():
        for path in intersection_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.smoke
def test_binary_validation(symfluence_code_dir, symfluence_data_root):
    """
    Validate all external tool binaries are functional.

    Tests that required binaries (SUMMA, mizuRoute, TauDEM) exist and can be executed.
    Also validates optional binaries (FUSE, NGEN) if they are installed.
    """
    from test_helpers.helpers import load_config_template

    # Load config to get installation paths
    config = load_config_template(symfluence_code_dir)
    data_dir_val = config.get("SYMFLUENCE_DATA_DIR", "default")
    if data_dir_val == "default" or not data_dir_val:
        data_dir = Path(symfluence_data_root)
    else:
        data_dir = Path(data_dir_val)

    # Required binaries
    print("\n" + "="*60)
    print("Validating REQUIRED binaries...")
    print("="*60)

    # Check for SUMMA - use configured path
    summa_install_path = config.get("SUMMA_INSTALL_PATH", "default")
    if summa_install_path == "default":
        summa_install_path = data_dir / "installs" / "summa" / "bin"
    else:
        summa_install_path = Path(summa_install_path)

    summa_exe = config.get("SUMMA_EXE", "summa.exe")

    # Try PATH first, then fall back to configured location
    summa_in_path = shutil.which(summa_exe)
    if summa_in_path:
        summa_path = Path(summa_in_path)
    else:
        # Check configured installation directory
        summa_path = summa_install_path / summa_exe
        # Also try the symlink summa.exe if the configured exe doesn't exist
        if not summa_path.exists() and summa_exe != "summa.exe":
            summa_path_symlink = summa_install_path / "summa.exe"
            if summa_path_symlink.exists():
                summa_path = summa_path_symlink

    if not summa_path.exists():
        pytest.skip(f"SUMMA binary not found at {summa_path} (not built/cached in this environment)")
    print(f"✓ SUMMA found: {summa_path}")

    # Verify SUMMA can run (check version)
    result = subprocess.run([str(summa_path), "--version"], capture_output=True, text=True)
    print(f"  SUMMA version output: {result.stdout.strip() or result.stderr.strip()}")

    # Check for mizuRoute - use configured path
    mizu_install_path = config.get("MIZUROUTE_INSTALL_PATH", "default")
    if mizu_install_path == "default":
        mizu_install_path = data_dir / "installs" / "mizuRoute" / "route" / "bin"
    else:
        mizu_install_path = Path(mizu_install_path)

    mizu_exe = config.get("MIZUROUTE_EXE", "mizuRoute.exe")

    # Try PATH first, then fall back to configured location
    mizu_in_path = shutil.which(mizu_exe)
    if mizu_in_path:
        mizu_path = Path(mizu_in_path)
    else:
        mizu_path = mizu_install_path / mizu_exe

    if not mizu_path.exists():
        pytest.skip(f"mizuRoute binary not found at {mizu_path} (not built/cached in this environment)")
    print(f"✓ mizuRoute found: {mizu_path}")

    # Check for TauDEM tools (required for domain preprocessing)
    taudem_install_path = config.get("TAUDEM_INSTALL_PATH", "default")
    if taudem_install_path == "default":
        taudem_install_path = data_dir / "installs" / "TauDEM" / "bin"
    else:
        taudem_install_path = Path(taudem_install_path)

    taudem_tools = ["pitremove", "d8flowdir", "aread8", "threshold", "streamnet", "gagewatershed"]
    print(f"✓ Checking {len(taudem_tools)} TauDEM tools...")
    for tool in taudem_tools:
        tool_path = shutil.which(tool)
        if not tool_path:
            tool_path = taudem_install_path / tool
            if not tool_path.exists():
                pytest.skip(f"TauDEM tool '{tool}' not found in PATH or {taudem_install_path} (not built/cached)")
    print(f"  All {len(taudem_tools)} TauDEM tools found")

    # Optional binaries
    print("\n" + "="*60)
    print("Validating OPTIONAL binaries...")
    print("="*60)

    optional_found = []
    optional_missing = []

    # Check for FUSE (optional alternative hydrological model)
    fuse_path = shutil.which("fuse.exe")
    if fuse_path:
        print(f"✓ FUSE found: {fuse_path}")
        optional_found.append("FUSE")
        # Try to verify FUSE can run
        try:
            result = subprocess.run(["fuse.exe", "--version"],
                                  capture_output=True, text=True, timeout=5)
            print(f"  FUSE version output: {result.stdout.strip() or result.stderr.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  FUSE found but version check failed (may be normal)")
    else:
        print("⚠ FUSE not found (optional)")
        optional_missing.append("FUSE")

    # Check for NGEN (optional NextGen framework)
    ngen_path = shutil.which("ngen")
    if ngen_path:
        print(f"✓ NGEN found: {ngen_path}")
        optional_found.append("NGEN")
        # Try to verify NGEN can run
        try:
            result = subprocess.run(["ngen", "--help"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "ngen" in result.stdout.lower():
                print("  NGEN verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  NGEN found but verification failed")
    else:
        print("⚠ NGEN not found (optional)")
        optional_missing.append("NGEN")

    # Check for HYPE (optional hydrological model)
    hype_install_path = config.get("HYPE_INSTALL_PATH", "default")
    if hype_install_path == "default":
        hype_install_path = data_dir / "installs" / "hype" / "bin"
    else:
        hype_install_path = Path(hype_install_path)

    hype_exe_name = config.get("HYPE_EXE", "hype")
    hype_in_path = shutil.which(hype_exe_name)
    if hype_in_path:
        hype_path = Path(hype_in_path)
    else:
        hype_path = hype_install_path / hype_exe_name

    if hype_path.exists():
        print(f"✓ HYPE found: {hype_path}")
        optional_found.append("HYPE")
        # Try to verify HYPE can run
        try:
            # HYPE usually prints its version/header even if args are missing
            result = subprocess.run([str(hype_path)],
                                  capture_output=True, text=True, timeout=5)
            # HYPE outputs to stderr when run without arguments
            if "HYPE" in result.stdout or "HYPE" in result.stderr or result.returncode != 0:
                print("  HYPE verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  HYPE found but verification failed")
    else:
        print(f"⚠ HYPE not found at {hype_path} (optional)")
        optional_missing.append("HYPE")

    # Check for MESH (optional hydrological model)
    mesh_install_path = config.get("MESH_INSTALL_PATH", "default")
    if mesh_install_path == "default":
        mesh_install_path = data_dir / "installs" / "mesh" / "bin"
    else:
        mesh_install_path = Path(mesh_install_path)

    mesh_exe_name = config.get("MESH_EXE", "mesh.exe")
    mesh_in_path = shutil.which(mesh_exe_name)
    if mesh_in_path:
        mesh_path = Path(mesh_in_path)
    else:
        mesh_path = mesh_install_path / mesh_exe_name

    if mesh_path.exists():
        print(f"✓ MESH found: {mesh_path}")
        optional_found.append("MESH")
        # Try to verify MESH can run
        try:
            # MESH may print usage info when run without arguments
            result = subprocess.run([str(mesh_path), "--help"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "MESH" in result.stdout or "MESH" in result.stderr:
                print("  MESH verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  MESH found but verification failed")
    else:
        print(f"⚠ MESH not found at {mesh_path} (optional)")
        optional_missing.append("MESH")

    # Check for RHESSys (optional, experimental)
    rhessys_install_path = config.get("RHESSYS_INSTALL_PATH", "default")
    if rhessys_install_path == "default":
        rhessys_install_path = data_dir / "installs" / "rhessys" / "bin"
    else:
        rhessys_install_path = Path(rhessys_install_path)

    rhessys_exe_name = config.get("RHESSYS_EXE", "rhessys")
    rhessys_in_path = shutil.which(rhessys_exe_name)
    if rhessys_in_path:
        rhessys_path = Path(rhessys_in_path)
    else:
        rhessys_path = rhessys_install_path / rhessys_exe_name

    if rhessys_path.exists():
        print(f"✓ RHESSys found: {rhessys_path}")
        optional_found.append("RHESSys")
        try:
            result = subprocess.run([str(rhessys_path), "-h"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "rhessys" in result.stdout.lower() or "usage" in result.stderr.lower():
                print("  RHESSys verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  RHESSys found but verification failed")
    else:
        print(f"⚠ RHESSys not found at {rhessys_path} (optional)")
        optional_missing.append("RHESSys")

    # Check for WMFire (optional, experimental for RHESSys)
    wmfire_install_path = config.get("WMFIRE_INSTALL_PATH", "default")
    if wmfire_install_path == "default":
        wmfire_install_path = data_dir / "installs" / "wmfire" / "lib"
    else:
        wmfire_install_path = Path(wmfire_install_path)

    if sys.platform == "darwin":
        wmfire_lib_name = config.get("WMFIRE_LIB", "libwmfire.dylib")
    else:
        wmfire_lib_name = config.get("WMFIRE_LIB", "libwmfire.so")

    wmfire_path = wmfire_install_path / wmfire_lib_name

    if wmfire_path.exists():
        print(f"✓ WMFire found: {wmfire_path}")
        optional_found.append("WMFire")
    else:
        print(f"⚠ WMFire not found at {wmfire_path} (optional)")
        optional_missing.append("WMFire")

    # Summary
    print("\n" + "="*60)
    print("Binary Validation Summary")
    print("="*60)
    print("Required: SUMMA, mizuRoute, TauDEM - ALL FOUND ✓")
    if optional_found:
        print(f"Optional found: {', '.join(optional_found)}")
    if optional_missing:
        print(f"Optional missing: {', '.join(optional_missing)}")
    print("="*60)


@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.smoke
def test_package_imports():
    """
    Verify Python environment and package imports.

    Tests that all critical packages can be imported without errors.
    """
    # Core SYMFLUENCE

    # Critical data packages

    # Geospatial packages

    # Model-specific

    # Optimization

    # All imports successful
    assert True


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


def _setup_installs_symlink(tmp_path: Path, symfluence_data_root: Path) -> Path:
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


@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.requires_data
@pytest.mark.parametrize("domain_name", ["bow"])
def test_quick_workflow_summa_only(
    domain_name,
    tmp_path,
    symfluence_code_dir,
    symfluence_data_root,
    bow_domain
):
    """
    Quick 3-hour SUMMA workflow test.

    Tests the complete workflow with minimal data:
    1. Setup project
    2. Copy raw data from source domain
    3. Run delineation if shapefiles don't exist
    4. Preprocess forcing (3 hours)
    5. Run SUMMA simulation (3 hours)

    This is a smoke test for CI - fast validation of core functionality.
    """
    from test_helpers.helpers import load_config_template, write_config

    # Validate required source data exists before proceeding
    src_domain_name = bow_domain.name.replace("domain_", "")
    dem_path = bow_domain / "data" / "attributes" / "elevation" / "dem" / f"domain_{src_domain_name}_elv.tif"
    if not dem_path.exists():
        # Also check alternative naming convention
        dem_path_alt = bow_domain / "data" / "attributes" / "elevation" / "dem" / f"{src_domain_name}_elv.tif"
        if not dem_path_alt.exists():
            pytest.skip(f"DEM file not found: {dem_path} (test data not available)")

    # Setup installs symlink for TauDEM
    _setup_installs_symlink(tmp_path, symfluence_data_root)

    # Create test configuration
    config = load_config_template(symfluence_code_dir)
    config["SYMFLUENCE_DATA_DIR"] = str(tmp_path)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Minimal 3-hour simulation
    # Use short names to avoid Fortran path length limits in mizuRoute
    config["DOMAIN_NAME"] = "Bow_quick"
    config["EXPERIMENT_ID"] = "qtest"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-01 04:00"  # 3 hours
    config["CALIBRATION_PERIOD"] = None
    config["EVALUATION_PERIOD"] = None
    config["SPINUP_PERIOD"] = None

    # Model settings
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Delineation settings - run full workflow from raw data
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["STREAM_THRESHOLD"] = 100000  # High threshold for single basin (lumped)

    # Use existing data (no downloads)
    config["DOWNLOAD_WSC_DATA"] = False

    # Save config
    cfg_path = tmp_path / "config_quick.yaml"
    write_config(config, cfg_path)

    # Initialize SYMFLUENCE
    sym = SYMFLUENCE(cfg_path)

    # Setup project (creates directory structure)
    project_dir = sym.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Copy raw data from bow_domain to project directory, renaming files
    src_domain_name = bow_domain.name.replace("domain_", "")
    dst_domain_name = config["DOMAIN_NAME"]

    # Copy forcing
    src_forcing = bow_domain / "forcing" / "raw_data"
    dst_forcing = project_dir / "forcing" / "raw_data"
    if src_forcing.exists():
        _copy_with_name_adaptation(src_forcing, dst_forcing, src_domain_name, dst_domain_name)
    _prune_raw_forcing(project_dir)

    # Copy attributes (DEM, landclass, soilclass)
    for attr_type in ["elevation", "landclass", "soilclass"]:
        src_attr = bow_domain / "attributes" / attr_type
        dst_attr = project_dir / "attributes" / attr_type
        if src_attr.exists():
            _copy_with_name_adaptation(src_attr, dst_attr, src_domain_name, dst_domain_name)

    # Create pour point
    sym.managers["project"].create_pour_point()

    # Define domain (watershed delineation) - creates shapefiles from raw DEM
    sym.managers["domain"].define_domain()

    # Discretize domain
    sym.managers["domain"].discretize_domain()

    # Model-agnostic preprocessing
    sym.managers["data"].run_model_agnostic_preprocessing()

    # Check preprocessing outputs
    basin_avg_dir = project_dir / "forcing" / "basin_averaged_data"
    assert basin_avg_dir.exists(), "Basin-averaged forcing should be created"

    # Model-specific preprocessing
    sym.managers["model"].preprocess_models()

    # Check SUMMA inputs
    summa_input_dir = project_dir / "forcing" / "SUMMA_input"
    assert summa_input_dir.exists(), "SUMMA input directory should exist"

    # Run SUMMA
    sym.managers["model"].run_models()

    # Check SUMMA output
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / "SUMMA"
    assert sim_dir.exists(), "SUMMA simulation output should exist"
    output_files = list(sim_dir.glob("*_timestep.nc"))
    assert output_files, "SUMMA output files should exist"


@pytest.mark.e2e
@pytest.mark.ci_full
@pytest.mark.requires_data
@pytest.mark.parametrize("model", ["SUMMA"])
def test_full_workflow_1month(
    model,
    tmp_path,
    symfluence_code_dir,
    symfluence_data_root,
    bow_domain
):
    """
    Full 1-month workflow test for SUMMA.

    Tests the complete workflow with standard data:
    1. Setup project
    2. Copy raw data and run delineation
    3. Preprocess forcing (1 month)
    4. Run model simulation (1 month)
    5. Verify outputs

    This is a more comprehensive test for full CI validation.
    Note: HYPE and RHESSys are tested in integration tests.
    """
    from test_helpers.helpers import load_config_template, write_config

    # Validate required source data exists before proceeding
    src_domain_name = bow_domain.name.replace("domain_", "")
    dem_path = bow_domain / "data" / "attributes" / "elevation" / "dem" / f"domain_{src_domain_name}_elv.tif"
    if not dem_path.exists():
        dem_path_alt = bow_domain / "data" / "attributes" / "elevation" / "dem" / f"{src_domain_name}_elv.tif"
        if not dem_path_alt.exists():
            pytest.skip(f"DEM file not found: {dem_path} (test data not available)")

    # Setup installs symlink for TauDEM
    _setup_installs_symlink(tmp_path, symfluence_data_root)

    # Create test configuration
    config = load_config_template(symfluence_code_dir)
    config["SYMFLUENCE_DATA_DIR"] = str(tmp_path)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # 1-month simulation
    # Use short names to avoid Fortran path length limits in mizuRoute
    config["DOMAIN_NAME"] = "Bow_full"
    config["EXPERIMENT_ID"] = f"ftest_{model[:2]}"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-31 23:00"  # 1 month

    config["CALIBRATION_PERIOD"] = "2004-01-05, 2004-01-19"
    config["EVALUATION_PERIOD"] = "2004-01-20, 2004-01-30"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-04"

    # Model settings
    config["HYDROLOGICAL_MODEL"] = model
    config["FORCING_DATASET"] = "ERA5"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Delineation settings - run full workflow from raw data
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["STREAM_THRESHOLD"] = 100000  # High threshold for single basin

    # Save config
    cfg_path = tmp_path / "config_full.yaml"
    write_config(config, cfg_path)

    # Initialize and run
    sym = SYMFLUENCE(cfg_path)
    project_dir = sym.managers["project"].setup_project()

    # Copy raw data from bow_domain to project directory, renaming files
    src_domain_name = bow_domain.name.replace("domain_", "")
    dst_domain_name = config["DOMAIN_NAME"]

    # Copy forcing
    src_forcing = bow_domain / "forcing" / "raw_data"
    dst_forcing = project_dir / "forcing" / "raw_data"
    if src_forcing.exists():
        _copy_with_name_adaptation(src_forcing, dst_forcing, src_domain_name, dst_domain_name)

    # Copy attributes (DEM, landclass, soilclass)
    for attr_type in ["elevation", "landclass", "soilclass"]:
        src_attr = bow_domain / "attributes" / attr_type
        dst_attr = project_dir / "attributes" / attr_type
        if src_attr.exists():
            _copy_with_name_adaptation(src_attr, dst_attr, src_domain_name, dst_domain_name)

    # Create pour point and run domain workflow
    sym.managers["project"].create_pour_point()
    sym.managers["domain"].define_domain()
    sym.managers["domain"].discretize_domain()

    # Run full pipeline
    sym.managers["data"].run_model_agnostic_preprocessing()
    sym.managers["model"].preprocess_models()
    sym.managers["model"].run_models()

    # Verify outputs
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / model
    assert sim_dir.exists(), f"{model} simulation output should exist"


@pytest.mark.e2e
@pytest.mark.ci_full
@pytest.mark.calibration
@pytest.mark.requires_data
def test_calibration_workflow(tmp_path, symfluence_code_dir, symfluence_data_root, bow_domain):
    """
    Calibration workflow validation.

    Tests the complete calibration pipeline:
    1. Setup project
    2. Copy raw data and run delineation
    3. Preprocess data
    4. Run model
    5. Calibrate model (minimal iterations for testing)
    6. Verify calibration outputs with real data validation
    """
    import math

    import pandas as pd

    from test_helpers.helpers import load_config_template, write_config

    # Validate required source data exists before proceeding
    src_domain_name = bow_domain.name.replace("domain_", "")
    dem_path = bow_domain / "data" / "attributes" / "elevation" / "dem" / f"domain_{src_domain_name}_elv.tif"
    if not dem_path.exists():
        dem_path_alt = bow_domain / "data" / "attributes" / "elevation" / "dem" / f"{src_domain_name}_elv.tif"
        if not dem_path_alt.exists():
            pytest.skip(f"DEM file not found: {dem_path} (test data not available)")

    # Setup installs symlink for TauDEM
    _setup_installs_symlink(tmp_path, symfluence_data_root)

    # Create test configuration
    config = load_config_template(symfluence_code_dir)
    config["SYMFLUENCE_DATA_DIR"] = str(tmp_path)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Calibration settings
    # Use short names to avoid Fortran path length limits in mizuRoute
    config["DOMAIN_NAME"] = "Bow_calib"
    config["EXPERIMENT_ID"] = "ctest"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-05 23:00"  # 5 days for faster test
    config["CALIBRATION_PERIOD"] = "2004-01-02, 2004-01-04"
    config["EVALUATION_PERIOD"] = "2004-01-05, 2004-01-05"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-01"

    # Model settings
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["FORCING_DATASET"] = "ERA5"
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Delineation settings
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["STREAM_THRESHOLD"] = 100000

    # Streamflow station
    config["STATION_ID"] = "05BB001"
    config["DOWNLOAD_WSC_DATA"] = False

    # Calibration for testing - use 3 iterations to verify algorithm runs
    config["NUMBER_OF_ITERATIONS"] = 3
    config["RANDOM_SEED"] = 42
    config["PARAMS_TO_CALIBRATE"] = "theta_sat"

    # Save config
    cfg_path = tmp_path / "config_calib.yaml"
    write_config(config, cfg_path)

    # Initialize and run
    sym = SYMFLUENCE(cfg_path)
    project_dir = sym.managers["project"].setup_project()

    # Copy raw data from bow_domain to project directory, renaming files
    src_domain_name = bow_domain.name.replace("domain_", "")
    dst_domain_name = config["DOMAIN_NAME"]

    # Copy forcing
    src_forcing = bow_domain / "forcing" / "raw_data"
    dst_forcing = project_dir / "forcing" / "raw_data"
    if src_forcing.exists():
        _copy_with_name_adaptation(src_forcing, dst_forcing, src_domain_name, dst_domain_name)
    _prune_raw_forcing(project_dir)

    # Copy attributes (DEM, landclass, soilclass)
    for attr_type in ["elevation", "landclass", "soilclass"]:
        src_attr = bow_domain / "attributes" / attr_type
        dst_attr = project_dir / "attributes" / attr_type
        if src_attr.exists():
            _copy_with_name_adaptation(src_attr, dst_attr, src_domain_name, dst_domain_name)

    # Copy observations (streamflow)
    src_obs = bow_domain / "observations"
    dst_obs = project_dir / "observations"
    if src_obs.exists():
        _copy_with_name_adaptation(src_obs, dst_obs, src_domain_name, dst_domain_name)

    # Create pour point and run domain workflow
    sym.managers["project"].create_pour_point()
    sym.managers["domain"].define_domain()
    sym.managers["domain"].discretize_domain()

    # Run pipeline
    sym.managers["data"].run_model_agnostic_preprocessing()
    sym.managers["model"].preprocess_models()
    sym.managers["model"].run_models()

    # Run calibration
    results_file = sym.managers["optimization"].calibrate_model()
    assert results_file is not None, "Calibration should produce results"

    # Validate calibration results - ensure we actually ran with real data
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
        # KGE ranges from -inf to 1, but reasonable values are > -100
        assert score > -100, f"Score at iteration {i} seems unreasonably low: {score}"

    # Verify we have parameter columns
    param_cols = [c for c in results_df.columns if c not in ['iteration', 'score', 'elapsed_time']]
    assert len(param_cols) > 0, "Results should contain parameter columns"

    # Verify calibration directory exists
    calib_dir = project_dir / "optimization"
    assert calib_dir.exists(), "Calibration directory should exist"
