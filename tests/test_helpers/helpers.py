"""
Helper functions for SYMFLUENCE tests.

Configuration management utilities for loading and writing test configs.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml


def load_config_template(symfluence_code_dir):
    """
    Load the configuration template from the config files directory.

    Args:
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        dict: Configuration dictionary loaded from template (flat format)

    Raises:
        FileNotFoundError: If config template doesn't exist

    Note:
        symfluence_code_dir parameter is deprecated, templates are now loaded from package data
    """
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.resources import get_config_template

    template_path = get_config_template()
    # Use SymfluenceConfig to load WITHOUT validation, then convert to dict for test manipulation
    # Tests will set model-specific configurations before validation
    config_obj = SymfluenceConfig.from_file(template_path, validate=False)
    return config_obj.to_dict(flatten=True)


def write_config(config, output_path):
    """
    Write a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path where config should be written

    Creates parent directories if they don't exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def has_cds_credentials():
    """Return True when CDS credentials are configured via env or ~/.cdsapirc."""
    env_url = os.environ.get("CDSAPI_URL")
    env_key = os.environ.get("CDSAPI_KEY")
    if env_url and env_key:
        return True

    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        return False

    try:
        content = cdsapirc.read_text(encoding="utf-8")
    except OSError:
        return False

    has_url = False
    has_key = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("url:"):
            has_url = bool(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("key:"):
            has_key = bool(stripped.split(":", 1)[1].strip())

    return has_url and has_key


# Cache for cloud service availability checks (avoid repeated network calls)
_cloud_availability_cache = {}


def is_rdrs_s3_available():
    """Check if RDRS S3 Zarr store is accessible."""
    if "rdrs" in _cloud_availability_cache:
        return _cloud_availability_cache["rdrs"]

    try:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        # Check if the RDRS Zarr store exists
        result = fs.exists("msc-open-data/reanalysis/rdrs/v3.1/zarr/.zmetadata")
        _cloud_availability_cache["rdrs"] = result
        return result
    except Exception:  # noqa: BLE001
        _cloud_availability_cache["rdrs"] = False
        return False


def is_em_earth_s3_available():
    """Check if EM-Earth S3 bucket is accessible and files can be read.

    Note: The bucket may allow directory listing but deny file access,
    so we need to actually try to read a file to verify access.
    """
    if "em_earth" in _cloud_availability_cache:
        return _cloud_availability_cache["em_earth"]

    try:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        # Check if the EM-Earth bucket directory is accessible
        if not fs.exists("emearth/nc/deterministic_raw_daily/prcp"):
            _cloud_availability_cache["em_earth"] = False
            return False

        # List files and try to actually read one (directory listing may work
        # even when file access is denied)
        files = fs.ls("emearth/nc/deterministic_raw_daily/prcp/")
        if not files:
            _cloud_availability_cache["em_earth"] = False
            return False

        # Try to open the first file to verify read access
        with fs.open(files[0], 'rb') as f:
            # Read just the first few bytes to verify access
            f.read(100)

        _cloud_availability_cache["em_earth"] = True
        return True
    except Exception:  # noqa: BLE001
        _cloud_availability_cache["em_earth"] = False
        return False


def is_cds_data_available(dataset="reanalysis-carra-single-levels"):
    """Check if CDS API can access a specific dataset (beyond just credentials)."""
    if not has_cds_credentials():
        return False

    cache_key = f"cds_{dataset}"
    if cache_key in _cloud_availability_cache:
        return _cloud_availability_cache[cache_key]

    try:
        import cdsapi
        c = cdsapi.Client(quiet=True)
        # Do a minimal info request to check dataset availability
        # This is a lightweight check that doesn't download data
        c.retrieve(
            dataset,
            {"product_type": "reanalysis"},
            str(Path(tempfile.gettempdir()) / f"cds_test_{dataset}.nc")
        )
        # If we get here without error, access is available
        # (we don't actually want to download, so this will likely fail
        # but a permissions error is different from an access denied error)
        _cloud_availability_cache[cache_key] = True
        return True
    except Exception as e:  # noqa: BLE001
        # Check if the error is an access/permissions issue vs a request format issue
        error_str = str(e).lower()
        if "access denied" in error_str or "forbidden" in error_str or "restricted" in error_str:
            _cloud_availability_cache[cache_key] = False
            return False
        # Other errors (like missing required fields) suggest the API is accessible
        _cloud_availability_cache[cache_key] = True
        return True


# The Bow-at-Banff raw forcing for January 2004, newest bundle name first. The
# bundle also ships a 3-step sample for the same month; both map to one
# remapped output name, so tests must keep exactly one of them.
BOW_JAN_2004_FORCING_GLOBS = ("*ERA5*200401*1month*.nc", "domain_*_ERA5_merged_200401.nc")


def keep_single_raw_forcing(raw_dir, globs=BOW_JAN_2004_FORCING_GLOBS):
    """Delete every raw forcing file in *raw_dir* except the first match of *globs*.

    Fails when the folder holds several files and none matches, so a renamed
    test bundle cannot silently leave overlapping forcing behind.
    """
    from pathlib import Path

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    files = sorted(raw_dir.glob("*.nc"))
    keep = next((m for g in globs for m in sorted(raw_dir.glob(g))), None)
    if keep is None:
        assert len(files) <= 1, f"No raw forcing in {raw_dir} matches {globs}: {[f.name for f in files]}"
        return files[0] if files else None
    for path in files:
        if path != keep:
            path.unlink()
    return keep
