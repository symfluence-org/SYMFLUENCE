# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODFLOW 6 Model Runner

Executes MODFLOW 6 (mf6) from a prepared simulation directory.
MODFLOW 6 reads mfsim.nam from the current working directory to
discover all model packages and input files.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base.base_runner import BaseModelRunner
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.runners.add("MODFLOW")
class MODFLOWRunner(BaseModelRunner):
    """
    Runs MODFLOW 6 via direct mf6 invocation.

    Handles:
    - Executable path resolution (download or source build)
    - Input file copying to simulation directory
    - Model execution (mf6 reads mfsim.nam from cwd)
    - Output verification (*.hds and *.bud files)
    """


    MODEL_NAME = "MODFLOW"
    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.settings_dir = self.project_dir / "settings" / "MODFLOW"

    def _get_mf6_executable(self) -> Path:
        """Get the MODFLOW 6 (mf6) executable path."""
        return self.get_model_executable(
            install_path_key='MODFLOW_INSTALL_PATH',
            default_install_subpath='installs/modflow',
            default_exe_name='mf6',
            typed_exe_accessor=lambda: (
                self.config.model.modflow.exe
                if self.config.model and self.config.model.modflow
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.modflow.timeout,
            default=3600,
        )

    def run_modflow(self, sim_dir: Optional[Path] = None, coupling_source_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute MODFLOW 6.

        Args:
            sim_dir: Optional override for simulation directory. If None,
                     uses standard output path.
            coupling_source_dir: Optional override for the land-surface model
                output directory used by recharge extraction.  When called from
                the CoupledGWWorker during calibration, this points to the
                iteration-specific SUMMA output so recharge reflects the
                current parameter set (not the project-level default).

        Returns:
            Path to output directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        logger.debug(f"Running MODFLOW 6 for domain: {self.config.domain.name}")

        # Prepare coupled recharge from upstream land surface model
        self._prepare_coupled_recharge(source_dir_override=coupling_source_dir)

        with symfluence_error_handler(
            "MODFLOW 6 model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            # Setup output directory
            if sim_dir is None:
                self.output_dir = (
                    self.project_dir / "simulations"
                    / self.config.domain.experiment_id / "MODFLOW"
                )
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            mf6_exe = self._get_mf6_executable()
            logger.debug(f"Using MODFLOW 6 executable: {mf6_exe}")

            # Copy input files to simulation directory
            self._setup_sim_directory(self.output_dir)

            # Execute: MODFLOW 6 reads mfsim.nam from cwd
            cmd = [str(mf6_exe)]
            logger.debug(f"Executing MODFLOW 6 from: {self.output_dir}")

            env = os.environ.copy()
            timeout = self._get_timeout()

            result = run_subprocess(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.stdout:
                logger.debug(f"MODFLOW stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"MODFLOW stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"MODFLOW execution returned code {result.returncode}")
                logger.error(
                    f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
                )
                # Also check mfsim.lst for error details
                lst_file = self.output_dir / "mfsim.lst"
                if lst_file.exists():
                    lst_content = lst_file.read_text()
                    # Find error lines
                    error_lines = [
                        l for l in lst_content.splitlines()
                        if 'error' in l.lower() or 'failed' in l.lower()
                    ]
                    if error_lines:
                        logger.error(f"MODFLOW listing errors: {error_lines[-5:]}")

                raise ModelExecutionError(
                    f"MODFLOW 6 execution failed with return code {result.returncode}"
                )

            logger.debug("MODFLOW 6 execution completed successfully")
            self._verify_output()

            return self.output_dir

    def _prepare_coupled_recharge(self, source_dir_override: Optional[Path] = None) -> None:
        """Extract recharge from upstream land surface model output.

        Checks ``coupling_source`` in MODFLOW config. If a source model
        is configured (e.g. SUMMA), uses SUMMAToMODFLOWCoupler to read
        its output, convert soil drainage to MODFLOW recharge, and write
        the gwf.rch file into the settings directory.

        Args:
            source_dir_override: When provided, read land-surface output
                from this directory instead of the project-level default.
                Used during calibration so each iteration gets recharge
                from the current parameter set.
        """
        coupling_source = self._get_config_value(
            lambda: self.config.model.modflow.coupling_source if self.config.model.modflow else None,
            default=None,
        )

        # When source_dir_override is provided (calibration), always prepare
        # recharge from the override directory regardless of coupling_source
        # config (which may be unreadable when config is a flat dict).
        if source_dir_override is not None:
            source_output_dir = Path(source_dir_override)
            coupling_source = coupling_source or 'SUMMA'
            coupling_source = str(coupling_source).upper()
            logger.debug(f"Preparing coupled recharge from {coupling_source}")
        elif coupling_source and str(coupling_source).lower() not in ('none', 'default', ''):
            coupling_source = str(coupling_source).upper()
            logger.debug(f"Preparing coupled recharge from {coupling_source}")
            experiment_id = self.config.domain.experiment_id
            source_output_dir = (
                self.project_dir / "simulations" / experiment_id / coupling_source
            )
        else:
            logger.debug("No coupling source configured — skipping recharge preparation")
            return

        if self._grid_mode() == 'basin':
            self._prepare_basin_recharge(coupling_source, source_output_dir)
            return

        if not source_output_dir.exists():
            logger.warning(
                f"Coupling source output directory not found: {source_output_dir}. "
                "Skipping recharge extraction — MODFLOW will use existing RCH file."
            )
            return

        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler

        recharge_var = self._get_config_value(
            lambda: self.config.model.modflow.recharge_variable,
            default='scalarSoilDrainage',
        )

        nc_files = sorted(source_output_dir.glob("*.nc"))
        logger.debug(
            f"Coupling source dir: {source_output_dir}, "
            f"NC files: {[f.name for f in nc_files]}"
        )

        coupler = SUMMAToMODFLOWCoupler(self.config_dict, logger_instance=logger)
        recharge_series = coupler.extract_recharge_from_summa(
            source_output_dir, variable=recharge_var,
        )

        rch_path = self.settings_dir / "gwf.rch"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        coupler.write_modflow_recharge_rch(recharge_series, rch_path)

        logger.debug(f"Wrote coupled recharge ({len(recharge_series)} periods) to {rch_path}")

    def _grid_mode(self) -> str:
        return str(self._get_config_value(
            lambda: self.config.model.modflow.grid_mode, default='lumped', dict_key='MODFLOW_GRID_MODE',
        )).lower()

    def _prepare_basin_recharge(self, land: str, source_output_dir: Path) -> None:
        """Basin mode: zone recharge from the land model's output -> RCH time series + TDIS."""
        import pandas as pd

        from .basin_model import WEIGHTS_FILE, set_simulation_length, write_coupling_record, write_recharge_timeseries
        from .land_adapters import adapter_options, get_land_adapter

        if not source_output_dir.exists():
            raise ModelExecutionError(f"{land} output for MODFLOW recharge not found: {source_output_dir}")
        land_settings = self.settings_dir.parent / land
        if not land_settings.exists():
            land_settings = self.project_dir / 'settings' / land
        adapter = get_land_adapter(land, adapter_options(self.config_dict or self.config))
        recharge = adapter.read_recharge(source_output_dir, land_settings)
        weights = pd.read_csv(self.settings_dir / WEIGHTS_FILE)
        dates = write_recharge_timeseries(self.settings_dir, recharge, sorted(weights.zone_id.unique()))
        set_simulation_length(self.settings_dir, len(dates))
        write_coupling_record(self.settings_dir, land, land_settings, source_output_dir, dates[0])
        logger.debug(f"MODFLOW basin recharge from {land}: {len(dates)} days from {dates[0].date()}, "
                     f"{len(recharge.columns)} zones")

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Copy all MODFLOW input files to simulation directory."""
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"MODFLOW settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )
        if self._grid_mode() == 'basin':
            if Path(sim_dir).resolve() != self.settings_dir.resolve():
                for src in self.settings_dir.iterdir():
                    if src.is_file():
                        shutil.copy2(src, Path(sim_dir) / src.name)
            return

        # Copy all MODFLOW input files
        modflow_files = [
            'mfsim.nam', 'gwf.nam', 'gwf.tdis', 'gwf.dis',
            'gwf.ic', 'gwf.npf', 'gwf.sto', 'gwf.rch',
            'gwf.drn', 'gwf.oc', 'gwf.ims', 'recharge.ts',
        ]

        for name in modflow_files:
            src = self.settings_dir / name
            if src.exists():
                shutil.copy2(src, sim_dir / name)
                logger.debug(f"Copied {name} to simulation directory")
            else:
                logger.warning(f"MODFLOW input file not found: {src}")

    def _verify_output(self) -> None:
        """Verify MODFLOW produced valid output files."""
        if self._grid_mode() == 'basin':
            from .basin_model import BUDGET_CSV, check_run
            check_run(self.output_dir)
            if not (self.output_dir / BUDGET_CSV).exists():
                raise ModelExecutionError(f"MODFLOW did not write {BUDGET_CSV} in {self.output_dir}")
            return
        # Check for head and budget files
        hds_files = list(self.output_dir.glob("*.hds"))
        bud_files = list(self.output_dir.glob("*.bud"))

        if not hds_files:
            raise RuntimeError(
                f"MODFLOW did not produce expected *.hds output in {self.output_dir}"
            )

        if not bud_files:
            logger.warning("MODFLOW did not produce *.bud budget files")

        for f in hds_files:
            if f.stat().st_size == 0:
                raise RuntimeError(f"MODFLOW head output file is empty: {f}")

        logger.debug(
            f"Verified MODFLOW output: {len(hds_files)} head file(s), "
            f"{len(bud_files)} budget file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for MODFLOW execution."""
        return self.run_modflow(**kwargs)
