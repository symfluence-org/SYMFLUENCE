"""
Data Fixtures for SYMFLUENCE Tests

Provides fixtures for downloading and managing test data bundles from GitHub releases.
"""

import shutil
import zipfile

import pytest
import requests

from symfluence.data.cache import RawForcingCache

# Test data bundle configuration
BUNDLE_VERSION = "v0.7.0"
BUNDLE_NAME = f"example_data_{BUNDLE_VERSION}"
BUNDLE_URL = f"https://github.com/symfluence-org/SYMFLUENCE/releases/download/examples-data-{BUNDLE_VERSION}/{BUNDLE_NAME}.zip"

# Fallback to v0.6.0 (same data, different tag)
FALLBACK_VERSION = "v0.6.0"
FALLBACK_NAME = f"example_data_{FALLBACK_VERSION}"
FALLBACK_URL = f"https://github.com/symfluence-org/SYMFLUENCE/releases/download/examples-data-{FALLBACK_VERSION}/{FALLBACK_NAME}.zip"


@pytest.fixture(scope="session")
def example_data_bundle(symfluence_data_root, symfluence_code_dir):
    """
    Download and extract example data bundle from GitHub release.

    This is a session-scoped fixture that downloads the test data once per test session
    and reuses it across all tests.

    Args:
        symfluence_data_root: Path to SYMFLUENCE_data directory

    Returns:
        Path: Path to the data root containing all domains

    Yields:
        Path: Data root path during test session
    """
    data_root = symfluence_data_root
    marker_file = data_root / f".{BUNDLE_NAME}_installed"
    read_only_root = symfluence_code_dir.parent / "SYMFLUENCE_data"
    read_only_marker = read_only_root / f".{BUNDLE_NAME}_installed"

    if marker_file.exists():
        return data_root

    if read_only_root != data_root and read_only_marker.exists():
        return read_only_root

    # Check if already downloaded
    if not marker_file.exists():
        print(f"\nDownloading example data {BUNDLE_VERSION}...")
        zip_path = data_root / f"{BUNDLE_NAME}.zip"

        try:
            # Try primary URL
            response = requests.get(BUNDLE_URL, stream=True, timeout=600)
            response.raise_for_status()
        except (requests.RequestException, requests.HTTPError):
            # Fall back to v0.5.5
            print(f"Warning: {BUNDLE_VERSION} not available, falling back to {FALLBACK_VERSION}")
            response = requests.get(FALLBACK_URL, stream=True, timeout=600)
            response.raise_for_status()
            zip_path = data_root / f"{FALLBACK_NAME}.zip"

        # Download
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting example data...")
        # Extract to temp location
        extract_dir = data_root / "temp_extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Move domains to data root
        extracted_bundle = extract_dir / BUNDLE_NAME
        if not extracted_bundle.exists():
            # Try fallback name
            extracted_bundle = extract_dir / FALLBACK_NAME

        if extracted_bundle.exists():
            # Move each domain to data root
            for domain_dir in extracted_bundle.iterdir():
                if domain_dir.is_dir() and domain_dir.name.startswith("domain_"):
                    dest = data_root / domain_dir.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    domain_dir.rename(dest)
                    print(f"Installed: {domain_dir.name}")

        # Cleanup
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

        # Create marker file
        marker_file.touch()
        print(f"Test data bundle ready at {data_root}")

    return data_root


@pytest.fixture(scope="session")
def ellioaar_domain(example_data_bundle):
    """
    Elliðaár Iceland test domain (CARRA forcing).

    Small 2km x 2km basin in Reykjavik, Iceland.
    Uses CARRA (Arctic Regional Reanalysis) forcing data.

    Returns:
        Path: Path to domain_ellioaar_iceland directory
    """
    domain_path = example_data_bundle / "domain_ellioaar_iceland"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def fyris_domain(example_data_bundle):
    """
    Fyrisån Uppsala test domain (CERRA forcing).

    Small 2km x 2km basin in Uppsala, Sweden.
    Uses CERRA (European Regional Reanalysis) forcing data.

    Returns:
        Path: Path to domain_fyris_uppsala directory
    """
    domain_path = example_data_bundle / "domain_fyris_uppsala"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def bow_domain(example_data_bundle, symfluence_code_dir):
    """
    Bow at Banff test domain (ERA5 forcing).

    Small basin in Banff, Alberta, Canada.
    Uses ERA5 (global reanalysis) forcing data.
    Includes streamflow observations.

    Prefers local tests/data/ over downloaded bundle for faster tests.
    Integration tests will run the full workflow to create shapefiles from raw data.

    Returns:
        Path: Path to domain directory
    """
    # Prefer local test data (faster, no network required)
    tests_dir = symfluence_code_dir / "tests"
    local_data = tests_dir / "data" / "domain_Bow_at_Banff"
    if local_data.exists():
        return local_data

    # Fallback to downloaded bundle for backwards compatibility
    domain_path = example_data_bundle / "domain_bow_banff_minimal"
    if not domain_path.exists():
        pytest.skip(f"Bow domain not found in {example_data_bundle}")
    return domain_path


@pytest.fixture(scope="session")
def iceland_domain(example_data_bundle):
    """
    Iceland regional domain (ERA5 forcing).

    Regional domain covering part of Iceland.
    Uses ERA5 (global reanalysis) forcing data.

    Returns:
        Path: Path to domain_Iceland directory
    """
    domain_path = example_data_bundle / "domain_Iceland"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def paradise_domain(example_data_bundle):
    """
    Paradise SNOTEL point-scale domain (ERA5 forcing).

    Point-scale domain at Paradise, Mt. Rainier, Washington.
    Uses ERA5 (global reanalysis) forcing data.
    Includes SNOTEL observations.

    Returns:
        Path: Path to domain_paradise directory
    """
    domain_path = example_data_bundle / "domain_paradise"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def raw_forcing_cache(example_data_bundle, symfluence_data_root):
    """
    Session-scoped fixture providing pre-downloaded raw forcing data.

    Checks for raw forcing data in the example data bundle first,
    then falls back to the global cache if not found.

    This fixture enables fast test execution by avoiding redundant
    API calls to CDS, ERA5, AORC, etc.

    Returns:
        callable: Function to retrieve forcing data by dataset/domain/time
                  Returns Path if cached, None if needs download
    """
    # Check if raw_forcing_data exists in bundle
    bundle_forcing_dir = example_data_bundle / "raw_forcing_data"

    # Global cache location
    cache_root = symfluence_data_root / "cache" / "raw_forcing"

    def _get_forcing(dataset: str, domain: str, start_time: str, end_time: str):
        """
        Get cached raw forcing data if available.

        Parameters
        ----------
        dataset : str
            Forcing dataset name (e.g., "ERA5", "CARRA", "CERRA")
        domain : str
            Domain name (e.g., "paradise", "iceland", "sweden")
        start_time : str
            Start time in ISO format
        end_time : str
            End time in ISO format

        Returns
        -------
        Path or None
            Path to cached forcing file if available, None otherwise
        """
        # Try bundle first (pre-packaged test data)
        if bundle_forcing_dir.exists():
            # Look for file matching pattern: {domain}_{start}_{end}.nc
            pattern = f"{domain}_{start_time}_{end_time}.nc"
            dataset_dir = bundle_forcing_dir / dataset.upper()
            if dataset_dir.exists():
                cache_file = dataset_dir / pattern
                if cache_file.exists():
                    return cache_file

                # Try without exact time match (fuzzy match)
                for f in dataset_dir.glob(f"{domain}*.nc"):
                    return f  # Return first match

        # Try global cache second
        if cache_root.exists():
            cache = RawForcingCache(cache_root=cache_root)
            # Note: This requires knowing the cache key generation logic
            # For now, return None and let the download proceed
            # Future enhancement: integrate with cache.get() using proper key

        # No cached data available
        return None

    return _get_forcing
