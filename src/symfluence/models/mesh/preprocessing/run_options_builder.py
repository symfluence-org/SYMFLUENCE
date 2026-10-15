# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Run Options Configuration Builder

Handles all modifications to MESH_input_run_options.ini.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from symfluence.core.mixins import ConfigMixin


class RunOptionsConfigBuilder(ConfigMixin):
    """Builds and fixes MESH run options configuration.

    Manages variable name replacements, snow parameters, control flags,
    output directories, and forcing file references in run_options.ini.

    Args:
        run_options_path: Path to MESH_input_run_options.ini
        config: Configuration dictionary
        logger: Logger instance
    """

    def __init__(
        self,
        run_options_path: Path,
        config: dict,
        logger: logging.Logger,
    ):
        self._path = run_options_path
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Variable name fixes
    # ------------------------------------------------------------------

    def fix_var_names(self) -> None:
        """Fix variable names in run options to match forcing file."""
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()

            var_replacements = {
                'name_var=SWRadAtm': 'name_var=FSIN',
                'name_var=spechum': 'name_var=QA',
                'name_var=airtemp': 'name_var=TA',
                'name_var=windspd': 'name_var=UV',
                'name_var=pptrate': 'name_var=PRE',
                'name_var=airpres': 'name_var=PRES',
                'name_var=LWRadAtm': 'name_var=FLIN',
            }

            modified = False
            for old_name, new_name in var_replacements.items():
                if old_name in content:
                    content = content.replace(old_name, new_name)
                    modified = True

            if modified:
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.update_control_flag_count()
                self.logger.info("Fixed run options variable names")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix run options variable names: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Snow / routing parameters
    # ------------------------------------------------------------------

    def fix_snow_params(self, get_num_cells_fn: Callable[[], int]) -> None:
        """Fix run options snow/ice parameters for stable multi-year simulations.

        Args:
            get_num_cells_fn: Callable returning number of subbasins from DDB
        """
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Run mode. Unset (None) => auto-pick from domain size below; an
            # explicit 'runrte'/'noroute' is always honoured. (The previous
            # call passed the key string as the typed accessor and omitted
            # dict_key, so MESH_RUNMODE was silently ignored and every domain
            # defaulted to 'runrte' before the single-cell override.)
            runmode = self._get_config_value(
                lambda: self.config.model.mesh.run_mode,
                default=None,
                dict_key='MESH_RUNMODE',
            )
            explicit_runmode = runmode is not None

            num_cells = get_num_cells_fn()
            if not explicit_runmode:
                # A single cell has no channel network for WATROUTE to route
                # through, so the wf_lzs lower-zone baseflow store cannot be
                # exercised — use 'noroute' (extractor handles RFF+DRAINSOL).
                # Multi-cell domains route and get the baseflow store.
                runmode = 'noroute' if num_cells == 1 else 'runrte'
                if num_cells == 1:
                    self.logger.info(
                        "Single-cell domain detected (lumped mode). "
                        "Using RUNMODE 'noroute' (extractor handles RFF+DRAINSOL)."
                    )
            elif num_cells == 1 and runmode != 'noroute':
                self.logger.info(
                    "Single-cell domain with explicit MESH_RUNMODE='%s': "
                    "honouring the routing request so the wf_lzs baseflow store "
                    "(FLZ/PWR) stays active.", runmode
                )

            if runmode == 'noroute':
                streamflow_flag = 'none'
                outfiles_flag = 'daily'
                basinavgwb_flag = 'daily'
            else:
                streamflow_flag = 'csv'
                outfiles_flag = 'daily'
                basinavgwb_flag = 'daily'

            enable_frozen = self._get_config_value(
                lambda: self.config.model.mesh.enable_frozen_soil,
                default=False, dict_key="MESH_ENABLE_FROZEN_SOIL"
            )
            frozen_flag = "1" if enable_frozen else "0"
            if enable_frozen:
                self.logger.info("FROZENSOILINFILFLAG enabled (1); explicit counted frozen-soil inputs will be validated")

            modified = False
            replacements = [
                (r'FREZTH\s+[-\d.]+', 'FREZTH                0.0'),
                (r'SWELIM\s+[-\d.]+', f'SWELIM                {self._get_config_value(lambda: self.config.model.mesh.swelim, default=800.0, dict_key="MESH_SWELIM")}'),
                (r'SNDENLIM\s+[-\d.]+', 'SNDENLIM              600.0'),
                (r'PBSMFLAG\s+\w+', 'PBSMFLAG              off'),
                (r'FROZENSOILINFILFLAG\s+\d+', f'FROZENSOILINFILFLAG   {frozen_flag}'),
                (r'RUNMODE\s+\w+', f'RUNMODE               {runmode}'),
                (r'METRICSSPINUP\s+\d+', f'METRICSSPINUP         {int(self._get_config_value(lambda: self.config.model.mesh.spinup_days, default=730, dict_key="MESH_SPINUP_DAYS"))}'),
                (r'DIAGNOSEMODE\s+\w+', 'DIAGNOSEMODE          off'),
                (r'SHDFILEFLAG\s+\w+', 'SHDFILEFLAG           nc_subbasin pad_outlets'),
                (r'BASINFORCINGFLAG\s+\w+', 'BASINFORCINGFLAG      nc_subbasin'),
                (r'OUTFILESFLAG\s+\w+', f'OUTFILESFLAG         {outfiles_flag}'),
                (r'OUTFIELDSFLAG\s+\w+', 'OUTFIELDSFLAG        none'),
                (r'STREAMFLOWOUTFLAG\s+\w+', f'STREAMFLOWOUTFLAG     {streamflow_flag}'),
                (r'BASINAVGWBFILEFLAG\s+\w+', f'BASINAVGWBFILEFLAG    {basinavgwb_flag}'),
                (r'PRINTSIMSTATUS\s+\w+', 'PRINTSIMSTATUS        date_monthly'),
            ]

            for pattern, replacement in replacements:
                if re.search(pattern, content):
                    content_new = re.sub(pattern, replacement, content)
                    if content_new != content:
                        content = content_new
                        modified = True

            self.logger.info(
                f"MESH RUNMODE set to '{runmode}' with streamflow output '{streamflow_flag}', "
                f"basin WB output '{basinavgwb_flag}'"
            )

            if modified:
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.update_control_flag_count()
                self.logger.info("Fixed run options snow/ice parameters")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix run options snow parameters: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Control flag count
    # ------------------------------------------------------------------

    def update_control_flag_count(self) -> None:
        """Update the number of control flags in MESH_input_run_options.ini."""
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            flag_start_idx = -1
            count_line_idx = -1
            for i, line in enumerate(lines):
                if 'Number of control flags' in line:
                    count_line_idx = i
                if line.startswith('----#'):
                    flag_start_idx = i + 1
                    break

            if count_line_idx == -1 or flag_start_idx == -1:
                return

            flag_count = 0
            for i in range(flag_start_idx, len(lines)):
                if lines[i].startswith('#####'):
                    break
                if lines[i].strip() and not lines[i].strip().startswith('#'):
                    flag_count += 1

            old_line = lines[count_line_idx]
            match = re.search(r'(\s*)(\d+)(\s*#.*)', old_line)
            if match:
                new_line = f"{match.group(1)}{flag_count:2d}{match.group(3)}\n"
                if new_line != old_line:
                    lines[count_line_idx] = new_line
                    with open(self._path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    self.logger.info(f"Updated control flag count to {flag_count}")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to update control flag count: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------

    def fix_output_dirs(self) -> None:
        """Fix output directory paths in run options file."""
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()

            if 'CLASSOUT' in content:
                content = content.replace('CLASSOUT', './' + ' ' * 6)
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info("Fixed output directory paths in run options")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix run options output dirs: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Safe forcing updates
    # ------------------------------------------------------------------

    def update_for_safe_forcing(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        actual_spinup_days: Optional[int] = None,
    ) -> None:
        """Update run options for safe forcing file.

        Args:
            start_time: Simulation start time (including spinup)
            end_time: Simulation end time
            actual_spinup_days: Actual spinup days used (may differ from configured)
        """
        if not self._path.exists():
            return

        with open(self._path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        modified = False
        new_lines = []

        for line in lines:
            if 'fname=MESH_forcing' in line and 'fname=MESH_forcing_safe' not in line:
                line = line.replace('fname=MESH_forcing', 'fname=MESH_forcing_safe')
                modified = True

            if 'start_date=' in line:
                new_start_date = start_time.strftime('%Y%m%d')
                line = re.sub(r'start_date=\d+', f'start_date={new_start_date}', line)
                modified = True

            if 'METRICSSPINUP' in line and actual_spinup_days:
                line = re.sub(
                    r'METRICSSPINUP\s+\d+',
                    f'METRICSSPINUP         {actual_spinup_days}',
                    line
                )
                modified = True

            new_lines.append(line)

        date_line_indices = self._find_date_lines(new_lines)
        if len(date_line_indices) >= 2:
            start_idx = date_line_indices[-2]
            end_idx = date_line_indices[-1]
            # Config bounds identify forcing records. MESH stops before its
            # stop timestamp, so include the complete final forcing interval.
            dt = int(self._get_config_value(
                lambda: self.config.forcing.time_step_size,
                default=3600, dict_key='FORCING_TIME_STEP_SIZE'))
            stop = pd.Timestamp(end_time) + pd.Timedelta(seconds=dt)
            new_lines[start_idx] = f"{start_time.year:04d} {start_time.dayofyear:03d} {start_time.hour:3d} {start_time.minute:3d}"
            new_lines[end_idx] = f"{stop.year:04d} {stop.dayofyear:03d} {stop.hour:3d} {stop.minute:3d}"
            modified = True
            self.logger.info(
                f"Updated simulation dates: {start_time.year:04d}/{start_time.dayofyear:03d} "
                f"to {end_time.year:04d}/{end_time.dayofyear:03d}"
            )

        if modified:
            with open(self._path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))

    def _find_date_lines(self, lines: list) -> list:
        """Find lines that look like date specifications."""
        date_line_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('-'):
                parts = stripped.split()
                if len(parts) >= 4 and parts[0].isdigit() and len(parts[0]) == 4:
                    try:
                        int(parts[0])
                        int(parts[1])
                        int(parts[2])
                        int(parts[3])
                        date_line_indices.append(i)
                    except ValueError as e:
                        self.logger.debug(f"Line does not match date format '{stripped}': {e}")
        return date_line_indices
