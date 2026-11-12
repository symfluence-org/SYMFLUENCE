# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MizuRoute Model Runner.

Manages the execution of the mizuRoute routing model.
Refactored to use the Unified Model Execution Framework.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.utilities.runoff_loader import (
    MODEL_CONFIGS,
    normalize_cf_time_units,
    resolve_runoff_file,
    round_time_stamps,
)
from symfluence.core.modeling.utilities.runoff_loader import (
    fix_time_precision as shared_fix_time_precision,
)
from symfluence.core.registries import R

#: Slack for filesystems that store whole-second modification times.
_MTIME_TOLERANCE_S = 1.0


@R.runners.add('MIZUROUTE', runner_method='run_mizuroute')
class MizuRouteRunner(BaseModelRunner):  # type: ignore[misc]
    """
    A class to run the mizuRoute model.

    This class handles the execution of the mizuRoute model, including setting up paths,
    running the model, and managing log files.

    Uses the Unified Model Execution Framework for subprocess execution.

    Attributes:

        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """

    MODEL_NAME = "MizuRoute"
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # MizuRoute uses 'root_path' alias for backwards compatibility
        self.setup_path_aliases({'root_path': 'data_dir'})

    def _should_create_output_dir(self) -> bool:
        """MizuRoute creates directories on-demand."""
        return False

    @staticmethod
    def _normalize_time_units(units: str) -> str:
        """Keep the same CF epoch while spelling it for the Fortran parser."""
        return normalize_cf_time_units(units)

    @staticmethod
    def _validate_routing_output(log_path: Path, output_path: Path, run_started: float) -> None:
        """Require output written by *this* run.

        Fortran STOP may return zero after fatal initialization errors, and
        the output directory usually still holds the previous run's files, so
        neither the exit status nor "some .nc exists" is evidence of a routed
        result. ``run_started`` is ``time.time()`` taken before launch.
        """
        if 'FATAL ERROR' in log_path.read_text(errors='replace').upper():
            raise ModelExecutionError(f'mizuRoute reported a fatal error; see {log_path}')
        outputs = list(output_path.glob('*.nc'))
        if not outputs:
            raise ModelExecutionError(f'mizuRoute produced no NetCDF output in {output_path}')
        # Whole-second mtimes on some filesystems: allow the truncation.
        fresh = [f for f in outputs if f.stat().st_mtime >= run_started - _MTIME_TOLERANCE_S]
        if not fresh:
            newest = max(outputs, key=lambda f: f.stat().st_mtime)
            raise ModelExecutionError(
                f'mizuRoute wrote no NetCDF output during this run: the newest '
                f'file in {output_path} ({newest.name}) predates the run start. '
                f'Refusing to treat stale output as the result; see {log_path}'
            )

    def _get_time_rounding_freq(self) -> Optional[str]:
        """Get time rounding frequency from config.

        Returns:
            Rounding frequency string (e.g., 'h', 'min', 's') or None if disabled.
        """
        freq = self._get_config_value(
            lambda: self.config.model.mizuroute.time_rounding_freq, default='h'
        )
        if freq and freq.lower() == 'none':
            return None
        return freq

    def _times_misaligned(self, stamps) -> bool:
        """Whether ANY timestamp is off the rounding grid.

        SUMMA writes times computed from Julian days, so later steps can carry
        microsecond jitter even when the first is exact. mizuRoute derives the
        runoff step from the first two times and then needs the last step to
        end exactly at <sim_end>; a 3599.99998 s step fails there with
        "index of idxFront lower than idxEnd".
        """
        freq = self._get_time_rounding_freq()
        if not freq:
            return False
        stamps = pd.DatetimeIndex(stamps)
        return bool((stamps != self._round_stamps(stamps, freq)).any())

    def _round_stamps(self, stamps, freq):
        """Round *stamps* to *freq*, or to whole seconds if that would merge steps.

        Thin wrapper: the rule lives in runoff_loader.round_time_stamps, the
        one implementation every routing model uses.
        """
        return round_time_stamps(pd.DatetimeIndex(stamps), freq)

    def fix_time_precision(self) -> Optional[Path]:
        """
        Fix model output time precision by rounding to configurable frequency.
        This fixes compatibility issues with mizuRoute time matching.
        Now supports both SUMMA and FUSE outputs with proper time format detection.
        Rounding frequency can be configured via MIZUROUTE_TIME_ROUNDING_FREQ
        (default: 'h' for hour, use 'none' to disable rounding).
        Returns the path to the runoff file if resolved, None otherwise.
        """
        # Determine which model's output to process
        models_raw = self.hydrological_model or ''
        mizu_from = self._get_config_value(lambda: self.config.model.mizuroute.from_model, dict_key='MIZU_FROM_MODEL')

        # Combine models and filter out 'DEFAULT' or empty strings
        all_models = f"{models_raw},{mizu_from}".split(',')
        active_models = sorted(list(set([
            m.strip().upper() for m in all_models
            if m.strip() and m.strip().upper() != 'DEFAULT'
        ])))

        self.logger.debug(f"Detected active models for time precision fix: {active_models}")

        # Which source model's output to route: the routable active models, in
        # the alphabetical order ``active_models`` is already sorted into, first
        # match wins. The routable set comes from the registered runoff
        # declarations rather than a literal tuple, so an external model package
        # becomes selectable by registering a schema -- and a model that cannot
        # feed a routing model (HYPE: its 'cout' is already routed discharge) is
        # excluded by not declaring one, instead of by a name in a list here.
        #
        # Alphabetical reproduces the precedence this method has always applied
        # (FUSE > GR > NGEN > SUMMA), but only incidentally: a future routable
        # model sorting before FUSE would take precedence over it. Precedence
        # only bites when two source models are active at once, which is not a
        # supported configuration -- pin an explicit order here if it becomes
        # one.
        routable = {name.upper() for name in MODEL_CONFIGS}
        candidates = [m for m in active_models if m in routable]

        # KNOWN GAP, behaviour preserved: with no routable active model this
        # still falls back to SUMMA, re-introducing the silent default that
        # get_model_config() deliberately dropped (it now raises rather than
        # resolving SUMMA paths for an unroutable model). A HYPE-only or
        # MESH-only run therefore resolves SUMMA paths, and touches a SUMMA
        # file of the same experiment id if one exists. Made loud rather than
        # changed, because the fallback is pinned by
        # tests/unit/models/test_mizuroute_runner_source_resolution.py.
        source_model = candidates[0] if candidates else 'SUMMA'
        if not candidates:
            self.logger.warning(
                f"No active model declares a routable runoff artifact "
                f"(active models: {active_models}; routable: {sorted(routable)}). "
                f"Falling back to SUMMA — any file found is a guess, not a "
                f"resolved output of the model that was run."
            )

        self.logger.info(
            f"Fixing {source_model} time precision for mizuRoute compatibility "
            f"(active models: {active_models})"
        )

        # Directory, filename pattern and the FUSE 6-char file-id truncation all
        # come from the model's registered runoff declaration; this method used
        # to carry its own if/elif copy, which is how it came to expect a HYPE
        # file nothing produces.
        runoff_filepath = resolve_runoff_file(
            source_model=source_model,
            project_dir=self.project_dir,
            experiment_id=self.experiment_id,
            domain_name=self.domain_name,
            config=self.config,
        )

        if runoff_filepath is None:
            self.logger.error(
                f"No routable {source_model} output found for experiment "
                f"'{self.experiment_id}'. mizuRoute needs a NetCDF runoff file "
                f"from the source model before it can route."
            )
            return None

        self.logger.info(f"Resolved runoff filepath: {runoff_filepath}")

        # One implementation for every routing model (runoff_loader): decodes
        # the CF units, rounds every stamp, refuses to create duplicates and
        # leaves an already-clean file untouched. normalize_epoch respells an
        # ISO reference ('T', timezone suffix) that mizuRoute's Fortran parser
        # rejects, even when rounding is disabled.
        return shared_fix_time_precision(
            runoff_filepath,
            rounding_freq=self._get_time_rounding_freq(),
            normalize_epoch=True,
        )

    def sync_control_file_dimensions(self, control_path: Path, netcdf_path: Path):
        """
        Ensure mizuRoute control file dimension/variable names match the NetCDF input.
        This prevents hangs/crashes when preprocessor assumes 'gru' but SUMMA outputs 'hru'.
        """
        try:
            import xarray as xr
            self.logger.debug(f"Syncing control file dimensions for {netcdf_path}")

            # Which variable mizuRoute will read. Its dimensions are the only
            # authoritative answer: a SUMMA file carries BOTH 'gru' and 'hru',
            # so picking from the file's dimension list (as this did, gru
            # first) is a guess that happens to be right for SUMMA and wrong
            # for any model whose runoff sits on the other one.
            qsim_var = None
            try:
                for line in control_path.read_text(encoding='utf-8').splitlines():
                    if '<vname_qsim>' in line:
                        parts = line.split('!')[0].split()
                        if len(parts) > 1:
                            qsim_var = parts[-1]
                        break
            except OSError:
                pass

            with xr.open_dataset(netcdf_path, decode_times=False) as ds:
                dname = None
                if qsim_var and qsim_var in ds.variables:
                    spatial = [d for d in ds[qsim_var].dims if d != 'time']
                    if len(spatial) == 1:
                        dname = spatial[0]
                if dname is None:
                    # Fall back to the file's dimensions when the runoff
                    # variable cannot be identified.
                    if 'gru' in ds.dims:
                        dname = 'gru'
                    elif 'hru' in ds.dims:
                        dname = 'hru'
                    else:
                        self.logger.warning(f"Could not find 'gru' or 'hru' dimension in {netcdf_path}. Available: {list(ds.dims)}")

                # Detect ID variable
                vname = None
                # Prefer the ID variable matching the dimension actually used.
                if dname and f'{dname}Id' in ds.variables:
                    vname = f'{dname}Id'
                elif 'gruId' in ds.variables:
                    vname = 'gruId'
                elif 'hruId' in ds.variables:
                    vname = 'hruId'
                # fallback checks
                elif 'gru_id' in ds.variables:
                    vname = 'gru_id'
                elif 'hru_id' in ds.variables:
                    vname = 'hru_id'

                if dname and not vname:
                     self.logger.warning(f"Could not find ID variable in {netcdf_path}")
                     # Try to find integer variable with same name as dim?
                     if dname in ds.variables:
                         vname = dname

            if dname and vname:
                self.logger.debug(f"Detected in NetCDF: dimension='{dname}', variable='{vname}'")

                # Read control file
                with open(control_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                new_lines = []
                modified = False
                for line in lines:
                    if '<dname_hruid>' in line:
                        # Compare the VALUE, not the whole line. `dname not in
                        # line` was a substring test against a line whose own
                        # tag is '<dname_hruid>' — which contains 'hru'. So for
                        # dname='hru' the test was always False and the sync
                        # could never rewrite gru -> hru, while the vname test
                        # below could (its tag is lowercase 'hruid' vs the value
                        # 'hruId'). The result was a control file left
                        # internally inconsistent — dimension 'gru', variable
                        # 'hruId' — while logging that it had applied 'hru'.
                        current = line.split('!')[0].split()[-1] if line.split('!')[0].split()[1:] else ''
                        if current != dname:
                            parts = line.split('!')
                            comment = '!' + parts[1] if len(parts) > 1 else ''
                            new_lines.append(f"<dname_hruid>           {dname}    {comment}")
                            modified = True
                        else:
                            new_lines.append(line)
                    elif '<vname_hruid>' in line:
                        # Value comparison, for the same reason as above.
                        current = line.split('!')[0].split()[-1] if line.split('!')[0].split()[1:] else ''
                        if current != vname:
                            parts = line.split('!')
                            comment = '!' + parts[1] if len(parts) > 1 else ''
                            new_lines.append(f"<vname_hruid>           {vname}    {comment}")
                            modified = True
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                # Write back if modified
                if modified:
                    self.logger.info(f"Updating control file to use dimension '{dname}' and variable '{vname}'")
                    with open(control_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                else:
                    self.logger.debug("Control file already matches NetCDF dimensions.")
            else:
                self.logger.warning("Could not determine dimensions to sync.")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error syncing control file dimensions: {e}", exc_info=True)

    def run_mizuroute(self):
        """
        Run the mizuRoute model.

        This method sets up the necessary paths, executes the mizuRoute model,
        and handles any errors that occur during the run.
        """
        self.logger.debug("Starting mizuRoute run")

        with symfluence_error_handler(
            "mizuRoute model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            runoff_path = self.fix_time_precision()

            # Set up paths and filenames
            # Legacy keys (INSTALL_PATH_MIZUROUTE, EXE_NAME_MIZUROUTE) are handled
            # by DEPRECATED_KEYS in transformers.py and mapped to the standard keys.
            self.mizu_exe = self.get_model_executable(
                install_path_key='MIZUROUTE_INSTALL_PATH',
                default_install_subpath='installs/mizuRoute/route/bin',
                exe_name_key='MIZUROUTE_EXE',
                default_exe_name='mizuRoute.exe',
                must_exist=True
            )
            settings_path = self.get_config_path('SETTINGS_MIZU_PATH', 'settings/mizuRoute/')
            control_file = self._get_config_value(
                lambda: self.config.model.mizuroute.control_file, default='mizuroute.control'
            )

            # Sane defaults for control file if not specified
            if not control_file or control_file == 'default':
                mizu_from = self._get_config_value(
                    lambda: self.config.model.mizuroute.from_model,
                    dict_key='MIZU_FROM_MODEL',
                ).upper()
                if mizu_from == 'GR':
                    control_file = 'mizuRoute_control_GR.txt'
                elif mizu_from == 'FUSE':
                    control_file = 'mizuRoute_control_FUSE.txt'
                elif mizu_from == 'NGEN':
                    control_file = 'mizuroute.control'
                else:
                    control_file = 'mizuroute.control'
                self.logger.debug(f"Using default mizuRoute control file: {control_file}")

            # Sync control file dimensions with actual runoff file
            if runoff_path and runoff_path.exists():
                control_path = settings_path / control_file
                if control_path.exists():
                    self.sync_control_file_dimensions(control_path, runoff_path)
                else:
                    self.logger.warning(f"Control file not found at {control_path}, skipping dimension sync")

            mizu_log_path = self.get_config_path('EXPERIMENT_LOG_MIZUROUTE', f"simulations/{self.experiment_id}/mizuRoute/mizuRoute_logs/")
            mizu_log_name = "mizuRoute_log.txt"

            mizu_out_path = self.get_config_path('EXPERIMENT_OUTPUT_MIZUROUTE', f"simulations/{self.experiment_id}/mizuRoute/")

            # Backup settings if required
            backup = self._get_config_value(
                lambda: self.config.model.summa.backup_settings, default='no',
                dict_key='EXPERIMENT_BACKUP_SETTINGS'
            )
            if backup == 'yes':
                self.backup_settings(settings_path, backup_subdir="run_settings")

            # Run mizuRoute (with OpenMP thread count if configured)
            mizu_log_path.mkdir(parents=True, exist_ok=True)
            mizu_command = [str(self.mizu_exe), str(settings_path / control_file)]

            mizu_num_threads = self._get_config_value(
                lambda: self.config.model.mizuroute.num_threads,
                default=1, dict_key='MIZUROUTE_NUM_THREADS',
            )
            mizu_env = {**__import__('os').environ, 'OMP_NUM_THREADS': str(mizu_num_threads)}
            self.logger.debug(f'Running mizuRoute with command: {" ".join(mizu_command)} (OMP_NUM_THREADS={mizu_num_threads})')

            run_started = time.time()
            self.execute_subprocess(
                mizu_command,
                mizu_log_path / mizu_log_name,
                success_message="mizuRoute process exited",
                env=mizu_env,
            )

            # Fortran STOP can return zero even after a fatal initialization
            # error. Exit status alone is not evidence of a routed result.
            self._validate_routing_output(mizu_log_path / mizu_log_name, mizu_out_path, run_started)
            self.logger.info('mizuRoute run completed successfully')

            return mizu_out_path
