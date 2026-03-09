"""Shared fixtures for jFUSE model tests."""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import jfuse  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]

from symfluence.core.config.models import SymfluenceConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def jfuse_config(temp_dir):
    """Create a jFUSE-specific configuration."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(temp_dir / "data"),
        "SYMFLUENCE_CODE_DIR": str(temp_dir / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "jfuse_test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-12-31 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "JFUSE",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 3600,
        "JFUSE_MODEL_CONFIG_NAME": "prms_gradient",
        "JFUSE_SPATIAL_MODE": "lumped",
        "JFUSE_ENABLE_SNOW": True,
        "JFUSE_WARMUP_DAYS": 365,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def distributed_jfuse_config(temp_dir):
    """Create a distributed jFUSE configuration."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(temp_dir / "data"),
        "SYMFLUENCE_CODE_DIR": str(temp_dir / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "jfuse_dist_test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-12-31 23:00",
        "DOMAIN_DEFINITION_METHOD": "delineate",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "JFUSE",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 3600,
        "JFUSE_SPATIAL_MODE": "distributed",
        "JFUSE_N_HRUS": 10,
        "JFUSE_ENABLE_ROUTING": True,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def setup_jfuse_directories(temp_dir, jfuse_config):
    """Set up directory structure for jFUSE testing."""
    data_dir = jfuse_config.system.data_dir
    domain_dir = data_dir / f"domain_{jfuse_config.domain.name}"

    settings_dir = domain_dir / "settings" / "JFUSE"
    forcing_dir = domain_dir / "data" / "forcing" / "merged_data" / "JFUSE_input"
    simulations_dir = domain_dir / "simulations" / "jfuse_test" / "JFUSE"

    for d in [settings_dir, forcing_dir, simulations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "domain_dir": domain_dir,
        "settings_dir": settings_dir,
        "forcing_dir": forcing_dir,
        "simulations_dir": simulations_dir,
    }


@pytest.fixture
def mock_jfuse_modules():
    """Mock jfuse and JAX modules for testing without these dependencies."""
    mock_jfuse = MagicMock()
    mock_jfuse.PARAM_BOUNDS = {"S1_max": (1, 1000), "S2_max": (1, 2000)}
    mock_jfuse.PRMS_CONFIG = MagicMock()
    mock_jfuse.SACRAMENTO_CONFIG = MagicMock()
    mock_jfuse.TOPMODEL_CONFIG = MagicMock()
    mock_jfuse.VIC_CONFIG = MagicMock()

    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_eqx = MagicMock()

    modules = {
        "jfuse": mock_jfuse,
        "jfuse.fuse": MagicMock(),
        "jfuse.fuse.config": MagicMock(),
        "jax": mock_jax,
        "jax.numpy": mock_jnp,
        "equinox": mock_eqx,
    }

    with patch.dict("sys.modules", modules):
        yield {
            "jfuse": mock_jfuse,
            "jax": mock_jax,
            "jnp": mock_jnp,
            "eqx": mock_eqx,
        }
