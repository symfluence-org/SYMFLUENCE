# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Model Runner."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


def _find_pcraster_site_packages() -> Optional[str]:
    """Locate PCRaster's site-packages, searching multiple strategies.

    Search order:
    1. Current environment (pcraster already importable — no path needed).
    2. Conda env matching active Python version (e.g. ``pcraster312``).
    3. Conda env with generic name ``pcraster``.
    4. Any conda env whose name starts with ``pcraster``.
    5. LISFLOOD_PCRASTER_PATH config / environment variable.
    """
    # 1. Already importable?
    try:
        import pcraster  # noqa: F401

        return None
    except ImportError:
        pass

    # 2-4. Search conda environments
    pyver = f"{sys.version_info.major}{sys.version_info.minor}"
    candidates = [f"pcraster{pyver}", "pcraster"]
    for env_name in candidates:
        path = _conda_env_site_packages(env_name)
        if path:
            return path

    # Try any env whose name starts with "pcraster"
    try:
        result = run_subprocess(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json

            envs = json.loads(result.stdout).get("envs", [])
            for env_path in envs:
                name = os.path.basename(env_path)
                if name.startswith("pcraster") and name not in candidates:
                    path = _conda_env_site_packages(name)
                    if path:
                        return path
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass

    # 5. Explicit path from environment variable
    explicit = os.environ.get("LISFLOOD_PCRASTER_PATH", "")
    if explicit and explicit != "default" and os.path.isdir(explicit):
        return explicit

    logger.warning(
        "PCRaster not found. Install via: conda create -n pcraster%s -c conda-forge python=%s.%s pcraster",
        pyver,
        sys.version_info.major,
        sys.version_info.minor,
    )
    return None


def _conda_env_site_packages(env_name: str) -> Optional[str]:
    """Return site-packages path for a named conda environment, or None."""
    try:
        result = run_subprocess(
            ["conda", "run", "-n", env_name, "python", "-c", "import site; print(site.getsitepackages()[0])"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if os.path.isdir(path):
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@R.runners.add("LISFLOOD")
class LisfloodRunner(BaseModelRunner):
    """Runner for the LISFLOOD model (Python/PCRaster-based).

    LISFLOOD is a pure-Python model that requires PCRaster. If PCRaster
    is not in the current environment, the runner searches for a conda
    environment with PCRaster and injects its site-packages via PYTHONPATH.
    """

    MODEL_NAME = "LISFLOOD"

    def _setup_model_specific_paths(self) -> None:
        self.settings_dir = self.project_dir / "settings" / "LISFLOOD"

    def run(self, **kwargs) -> Optional[Path]:
        settings_file = self._get_config_value(
            lambda: self.config.model.lisflood.settings_file,
            dict_key='LISFLOOD_SETTINGS_FILE',
        )
        settings_path = self.settings_dir / settings_file
        if not settings_path.exists():
            raise FileNotFoundError(f"LISFLOOD settings not found: {settings_path}")

        env = dict(os.environ)
        pcraster_site = _find_pcraster_site_packages()
        if pcraster_site:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{pcraster_site}:{existing}" if existing else pcraster_site
            pcraster_prefix = str(Path(pcraster_site).parent.parent.parent)
            env["CONDA_PREFIX"] = pcraster_prefix

        num_threads = self._get_config_value(
            lambda: self.config.model.lisflood.num_threads,
            dict_key='LISFLOOD_NUM_THREADS',
        )
        if num_threads > 1:
            env["PCRASTER_NR_WORKER_THREADS"] = str(num_threads)
        else:
            env.pop("PCRASTER_NR_WORKER_THREADS", None)

        cmd = [
            sys.executable,
            "-c",
            f'from lisflood.main import main; main("{settings_path}")',
        ]
        timeout = self._get_config_value(lambda: self.config.model.lisflood.timeout, dict_key='LISFLOOD_TIMEOUT')

        log_dir = self.get_simulation_dir('LISFLOOD', self._get_experiment_id()) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running LISFLOOD for domain: {self.domain_name}")
        result = self.execute_subprocess(
            command=cmd,
            log_file=log_dir / f"lisflood_run_{self._timestamp()}.log",
            cwd=self.settings_dir,
            env=env,
            timeout=timeout,
        )
        if not result.success:
            self.logger.error(f"LISFLOOD simulation failed (rc={result.returncode})")
        output_dir = self.get_simulation_dir('LISFLOOD', self._get_experiment_id())
        return output_dir if result.success else None

    def _get_experiment_id(self) -> str:
        return self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default="run_1",
        )

    @staticmethod
    def _timestamp() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")
