# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH Parameter Fixer

Facade that coordinates MESH parameter file fixes via composed helpers:
- RunOptionsConfigBuilder: run_options.ini modifications
- DDBFileManager: drainage database GRU operations
- CLASSFileManager: CLASS parameter file operations
- GRUCountManager: GRU alignment between DDB and CLASS

Also retains hydrology, reservoir, forcing, and lumped-output methods
directly (they don't belong to any single helper).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.mixins import ConfigMixin

from .class_file_manager import CLASSFileManager
from .config_defaults import MESHConfigDefaults
from .ddb_file_manager import DDBFileManager
from .gru_count_manager import GRUCountManager
from .run_options_builder import RunOptionsConfigBuilder


class MESHParameterFixer(ConfigMixin):
    """
    Fixes MESH parameter files for compatibility and stability.

    Handles:
    - Run options variable name fixes
    - Snow/ice parameter fixes for multi-year stability
    - GRU count mismatches between CLASS and DDB
    - CLASS initial conditions for snow simulation
    - Hydrology WF_R2 parameter
    - Safe forcing file creation
    """

    def __init__(
        self,
        forcing_dir: Path,
        setup_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None,
        time_window_func=None
    ):
        self.forcing_dir = forcing_dir
        self.setup_dir = setup_dir
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)
        self.get_simulation_time_window = time_window_func
        self._actual_spinup_days = None

        # Composed helpers
        self._run_options = RunOptionsConfigBuilder(
            self.run_options_path, config, self.logger
        )
        self._ddb = DDBFileManager(self.ddb_path, self.logger)
        self._class_mgr = CLASSFileManager(self.class_file_path, self.logger)
        self._gru = GRUCountManager(
            self._ddb, self._class_mgr, config, self.logger
        )

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------

    @property
    def run_options_path(self) -> Path:
        return self.forcing_dir / "MESH_input_run_options.ini"

    @property
    def class_file_path(self) -> Path:
        return self.forcing_dir / "MESH_parameters_CLASS.ini"

    @property
    def hydro_path(self) -> Path:
        return self.forcing_dir / "MESH_parameters_hydrology.ini"

    @property
    def ddb_path(self) -> Path:
        return self.forcing_dir / "MESH_drainage_database.nc"

    # ==================================================================
    # Run options delegations
    # ==================================================================

    def fix_run_options_var_names(self) -> None:
        """Fix variable names in run options to match forcing file."""
        self._run_options.fix_var_names()

    def fix_run_options_snow_params(self) -> None:
        """Fix run options snow/ice parameters for stable multi-year simulations."""
        self._run_options.fix_snow_params(self._get_num_cells)

    def _update_control_flag_count(self) -> None:
        """Update the number of control flags in MESH_input_run_options.ini."""
        self._run_options.update_control_flag_count()

    def fix_run_options_output_dirs(self) -> None:
        """Fix output directory paths in run options file."""
        self._run_options.fix_output_dirs()

    # ==================================================================
    # GRU count management delegations
    # ==================================================================

    def fix_gru_count_mismatch(self) -> None:
        """Ensure CLASS NM matches GRU count and renormalize GRU fractions."""
        self._gru.fix_gru_count_mismatch()

    # ==================================================================
    # DDB delegations (private — used by tests and internal methods)
    # ==================================================================

    def _get_num_cells(self) -> int:
        """Get number of cells from drainage database."""
        return self._ddb.get_num_cells()

    def _get_ddb_gru_count(self) -> Optional[int]:
        """Get the number of GRU columns in the DDB."""
        return self._ddb.get_gru_count()

    def _trim_ddb_to_active_grus(self, target_count: int) -> None:
        """Trim DDB GRU columns to exactly target_count."""
        self._ddb.trim_to_active_grus(target_count)

    def _ensure_gru_normalization(self) -> None:
        """Ensure GRU fractions in DDB sum to 1.0 for every subbasin."""
        self._ddb.ensure_gru_normalization()

    def _renormalize_mesh_active_grus(self, active_count: int) -> None:
        """Renormalize the first N GRU fractions to sum to 1.0."""
        self._ddb.renormalize_active_grus(active_count)

    def _get_mesh_active_gru_count(self) -> Optional[int]:
        """Return NGRU-1 (what MESH actually reads)."""
        return self._ddb.get_mesh_active_gru_count()

    def _get_spatial_dim(self, ds: xr.Dataset) -> Optional[str]:
        """Get the spatial dimension name from dataset."""
        return DDBFileManager.get_spatial_dim(ds)

    def _get_domain_latitude(self) -> Optional[float]:
        """Get representative latitude from drainage database."""
        return self._ddb.get_domain_latitude()

    def _remove_small_grus(self) -> None:
        """Remove GRUs below the minimum fraction threshold."""
        min_fraction = float(self._get_config_value(
            lambda: self.config.model.mesh.gru_min_total,
            default=0.05,
            dict_key='MESH_GRU_MIN_TOTAL'
        ))
        keep_mask = self._ddb.remove_small_grus(min_fraction)
        if keep_mask is not None:
            self._class_mgr.remove_blocks_by_mask(keep_mask)

    # ==================================================================
    # CLASS delegations (private — used by tests and internal methods)
    # ==================================================================

    def _get_class_block_count(self) -> Optional[int]:
        """Get the number of CLASS parameter blocks."""
        return self._class_mgr.get_block_count()

    def _read_nm_from_lines(self, lines: list) -> Optional[int]:
        """Read NM value from CLASS file lines."""
        return self._class_mgr.read_nm_from_lines(lines)

    def _update_class_nm(self, new_nm: int) -> None:
        """Update NM in CLASS parameters file."""
        self._class_mgr.update_nm(new_nm)

    def _trim_class_to_count(self, target_count: int) -> None:
        """Trim CLASS parameter blocks to a specific count."""
        self._class_mgr.trim_to_count(target_count)

    def _trim_empty_gru_columns(self) -> Optional[list]:
        """Trim empty GRU columns from drainage database."""
        min_total = float(self._get_config_value(
            lambda: self.config.model.mesh.gru_min_total,
            default=0.02, dict_key='MESH_GRU_MIN_TOTAL'
        ))
        return self._ddb.trim_empty_gru_columns(min_total)

    def _remove_class_blocks_by_mask(self, keep_mask: np.ndarray) -> None:
        """Remove CLASS parameter blocks by mask."""
        self._class_mgr.remove_blocks_by_mask(keep_mask)

    def _fix_class_nm(self, keep_mask: Optional[list]) -> None:
        """Fix CLASS NM parameter to match block count."""
        self._class_mgr.fix_nm(keep_mask)
        self._ddb.ensure_gru_normalization()

    def _trim_class_blocks(self, lines: list, keep_mask: list) -> bool:
        """Trim CLASS parameter blocks to match DDB GRU columns."""
        return self._class_mgr.trim_blocks_by_mask(lines, keep_mask)

    def fix_class_vegetation_parameters(self) -> None:
        """Fix CLASS vegetation parameters for different GRU types."""
        self._class_mgr.fix_vegetation_parameters()

    def fix_class_initial_conditions(self) -> None:
        """Fix CLASS initial conditions for proper snow simulation."""
        self._class_mgr.fix_initial_conditions(
            time_window_fn=self.get_simulation_time_window,
            latitude=self._ddb.get_domain_latitude(),
        )

    def create_elevation_band_class_blocks(self, elevation_info: list) -> bool:
        """Create CLASS parameter blocks for elevation bands."""
        return self._class_mgr.create_elevation_band_blocks(
            elevation_info, self._ddb.get_num_cells
        )

    def _get_climate_adjusted_snow_params(
        self, start_month: int, latitude: Optional[float]
    ) -> dict:
        """Get snow initial conditions adjusted for climate zone and season."""
        return CLASSFileManager._get_climate_adjusted_snow_params(
            start_month, latitude
        )

    def apply_class_field_overrides(self) -> None:
        """Override meshflow-derived CLASS fields with configured values.

        meshflow hard-derives regime-determining CLASS fields (soil texture,
        drainage density, initial states, vegetation params) from the input
        data. When the corresponding ``MESH_*`` config keys are set they
        override those fields across all GRU blocks, letting the surface-runoff
        regime be controlled from config alone.  Left unset => keep meshflow's
        values (no-op).
        """
        overrides = {
            'sand': self._get_config_value(
                lambda: self.config.model.mesh.soil_sand, default=None, dict_key='MESH_SOIL_SAND'),
            'clay': self._get_config_value(
                lambda: self.config.model.mesh.soil_clay, default=None, dict_key='MESH_SOIL_CLAY'),
            'orgm': self._get_config_value(
                lambda: self.config.model.mesh.soil_orgm, default=None, dict_key='MESH_SOIL_ORGM'),
            'dd': self._get_config_value(
                lambda: self.config.model.mesh.drainage_density, default=None, dict_key='MESH_DD'),
            'mid': self._get_config_value(
                lambda: self.config.model.mesh.mid, default=None, dict_key='MESH_MID'),
            'tbar': self._get_config_value(
                lambda: self.config.model.mesh.init_tbar, default=None, dict_key='MESH_INIT_TBAR'),
            'thlq': self._get_config_value(
                lambda: self.config.model.mesh.init_thlq, default=None, dict_key='MESH_INIT_THLQ'),
            'cmas': self._get_config_value(
                lambda: self.config.model.mesh.veg_cmas, default=None, dict_key='MESH_VEG_CMAS'),
            'qa50': self._get_config_value(
                lambda: self.config.model.mesh.veg_qa50, default=None, dict_key='MESH_VEG_QA50'),
            'vpda': self._get_config_value(
                lambda: self.config.model.mesh.veg_vpda, default=None, dict_key='MESH_VEG_VPDA'),
            'vpdb': self._get_config_value(
                lambda: self.config.model.mesh.veg_vpdb, default=None, dict_key='MESH_VEG_VPDB'),
        }
        self._class_mgr.apply_field_overrides(overrides)

    def apply_hydrology_field_overrides(self) -> None:
        """Override meshflow-derived hydrology.ini fields with configured values.

        Currently supports the IWF interflow flag: when ``MESH_IWF`` is set,
        the ``IWF`` line's value(s) are replaced (0=off, 1=on). Left unset =>
        keep meshflow's value (no-op).
        """
        iwf = self._get_config_value(
            lambda: self.config.model.mesh.iwf, default=None, dict_key='MESH_IWF'
        )
        if iwf is None or not self.hydro_path.exists():
            return

        try:
            with open(self.hydro_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            for i, line in enumerate(lines):
                if re.match(r'^\s*IWF\b', line):
                    # Preserve any trailing comment; replace all numeric tokens
                    # with the configured value (one per GRU column).
                    comment = ''
                    body = line.rstrip('\n')
                    for marker in ('#', '!'):
                        if marker in body:
                            idx = body.index(marker)
                            comment = ' ' + body[idx:]
                            body = body[:idx]
                            break
                    tokens = body.split()
                    n_values = max(1, len(tokens) - 1)  # tokens[0] == 'IWF'
                    values = '    '.join([str(int(iwf))] * n_values)
                    lines[i] = f"IWF   {values}{comment}\n"
                    modified = True
                    break

            if modified:
                with open(self.hydro_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                self.logger.info(f"Set hydrology IWF={int(iwf)} from config")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to apply hydrology IWF override: {e}", exc_info=True)

    # ==================================================================
    # Hydrology methods (kept directly — no suitable helper group)
    # ==================================================================

    def fix_hydrology_wf_r2(self) -> None:
        """Ensure WF_R2 is in the hydrology file.

        Note: WF_R2 (WATFLOOD channel roughness) is DIFFERENT from R2N (overland Manning's n).
        """
        settings_hydro = self.setup_dir / "MESH_parameters_hydrology.ini"

        if not self.hydro_path.exists() or self.hydro_path.stat().st_size == 0:
            if settings_hydro.exists() and settings_hydro.stat().st_size > 0:
                import shutil
                shutil.copy2(settings_hydro, self.hydro_path)
                self.logger.info("Copied hydrology file from settings")
            else:
                self.logger.warning("No valid hydrology file found")
                return

        try:
            with open(self.hydro_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            if not content.strip() and settings_hydro.exists():
                with open(settings_hydro, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                if content.strip():
                    with open(self.hydro_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.logger.info("Restored hydrology file from settings")

            configured_wf_r2 = self._get_config_value('MESH_WF_R2', None)

            if configured_wf_r2 is None:
                try:
                    hydro_params = self._get_config_value(lambda: None, default={}, dict_key='hydrology_params')
                    if isinstance(hydro_params, dict):
                        configured_wf_r2 = hydro_params.get('routing', [{}])[0].get('wf_r2')
                except Exception:  # noqa: BLE001 — model execution resilience
                    pass

            default_wf_r2 = 0.30
            target_wf_r2 = float(configured_wf_r2) if configured_wf_r2 is not None else default_wf_r2

            if 'WF_R2' in content:
                if configured_wf_r2 is not None:
                     self.logger.debug(f"WF_R2 already present, but updating to configured value {target_wf_r2}")
                     pattern = r'(WF_R2\s+)([\d\.\s]+)(.*)'

                     def replace_wf_r2(match):
                         prefix = match.group(1)
                         suffix = match.group(3)
                         existing_values = match.group(2).split()
                         n_vals = len(existing_values)
                         new_vals = [f"{target_wf_r2:.4f}"] * n_vals
                         return f"{prefix}{'    '.join(new_vals)}{suffix}"

                     content = re.sub(pattern, replace_wf_r2, content)
                     with open(self.hydro_path, 'w', encoding='utf-8') as f:
                         f.write(content)
                     return

                self.logger.debug("WF_R2 already present")
                return

            new_lines = []
            r2n_found = False
            for line in lines:
                if line.startswith('R2N') and not r2n_found:
                    parts = line.split()
                    if len(parts) >= 2:
                        n_values = len(parts) - 1

                        wf_r2_values = [f"{target_wf_r2:.4f}"] * n_values
                        wf_r2_line = "WF_R2  " + "    ".join(wf_r2_values) + "  # channel roughness (calibratable)"
                        new_lines.append(wf_r2_line)
                        r2n_found = True
                        self.logger.info(f"Added WF_R2={target_wf_r2} for {n_values} routing class(es)")

                        for j in range(len(new_lines) - 1, -1, -1):
                            if "Number of channel routing parameters" in new_lines[j]:
                                match = re.match(r'\s*(\d+)', new_lines[j])
                                if match:
                                    old_count = int(match.group(1))
                                    new_count = old_count + 1
                                    new_lines[j] = new_lines[j].replace(
                                        str(old_count), str(new_count), 1
                                    )
                                break

                new_lines.append(line)

            if r2n_found:
                with open(self.hydro_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to add WF_R2: {e}", exc_info=True)

    def fix_missing_hydrology_params(self) -> None:
        """Verify and pre-populate hydrology parameters for MESH."""
        if not self.hydro_path.exists():
            self.logger.warning("Hydrology file not found, skipping parameter verification")
            return

        is_noroute = False
        if self.run_options_path.exists():
            with open(self.run_options_path, 'r', encoding='utf-8') as f:
                run_options_content = f.read()
            if re.search(r'RUNMODE\s+noroute', run_options_content):
                is_noroute = True

        try:
            with open(self.hydro_path, 'r', encoding='utf-8') as f:
                content = f.read()

            modified = False

            if not is_noroute:
                required_params = ['R2N', 'R1N', 'PWR', 'FLZ']
                missing = [p for p in required_params if p not in content]
                if missing:
                    self.logger.warning(f"Missing routing parameters in hydrology file: {missing}")
                else:
                    self.logger.debug("All standard routing parameters present (R2N, R1N, PWR, FLZ)")

            # Do not append RCHARG/FRZTH: neither is recognized by the
            # supported native reader. Frozen inputs are written explicitly
            # after GRU/glacier geometry is finalized.

            if modified:
                with open(self.hydro_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to verify hydrology parameters: {e}", exc_info=True)

    # GRU-dependent hydrology parameters (one value per GRU column).
    # meshflow emits these under the "GRU class dependent hydrologic parameters"
    # section; frozen fields are written later by the explicit frozen-soil writer.
    _GRU_DEPENDENT_HYDRO_PARAMS = ('ZSNL', 'ZPLS', 'ZPLG', 'FRZC', 'DDEN', 'MANN')

    def fix_gru_dependent_hydrology_params(self) -> None:
        """Expand GRU-dependent hydrology parameters to one value per GRU.

        MESH reads GRU-dependent hydrology parameters (ZSNL, ZPLS, ZPLG, …) as an
        array with exactly ``NTYPE`` values — one per GRU. meshflow, tuned for the
        single-GRU lumped case, writes a single value per line; on any multi-GRU
        domain (elevation bands, semi-distributed, distributed) MESH then aborts
        during initialization with::

            Error converting GRU dependent parameter ZSNL GRU  2: ''
            A value is required for all active GRUs.

        Only the parameters MESH actually reads — the first ``count`` lines of the
        section, as declared by that section's own count — are expanded. Trailing
        legacy scalar lines *after* the counted
        block are left untouched: MESH stops reading at ``count``
        and never sees them, exactly as in the working lumped file. The section
        count is preserved. Single-GRU (lumped) files are already correct and
        pass through unchanged.
        """
        if not self.hydro_path.exists():
            return

        # NTYPE = number of CLASS.ini GRU blocks, exactly what MESH reads.
        num_grus = self._count_class_gru_blocks()
        if num_grus <= 1:
            return  # lumped: single value per line is already correct

        try:
            with open(self.hydro_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Locate the GRU-dependent section header and its count line.
            header_idx = next(
                (i for i, ln in enumerate(lines)
                 if 'GRU class dependent hydrologic parameters' in ln), None
            )
            if header_idx is None:
                return
            count_idx = next(
                (i for i in range(header_idx + 1, len(lines))
                 if re.match(r'\s*\d+\s', lines[i]) and 'Number' in lines[i]), None
            )
            if count_idx is None:
                # Fall back to the first bare-integer line after the header.
                count_idx = next(
                    (i for i in range(header_idx + 1, len(lines))
                     if re.match(r'\s*\d+\s*$', lines[i].split('#')[0])), None
                )
            if count_idx is None:
                return

            # Only the first `declared_count` parameter lines are read by MESH;
            # expand exactly those and leave any trailing scalar lines untouched.
            count_match = re.match(r'\s*(\d+)', lines[count_idx])
            declared_count = int(count_match.group(1)) if count_match else 0
            if declared_count <= 0:
                return

            n_expanded = 0
            seen = 0
            changed = False
            for i in range(count_idx + 1, len(lines)):
                if seen >= declared_count:
                    break
                raw = lines[i]
                stripped = raw.strip()
                if not stripped or stripped.startswith(('!', '#')):
                    continue
                m = re.match(r'^(\s*)([A-Za-z_]\w*)\s+(.*)$', raw)
                if not m:
                    continue
                seen += 1  # this is one of the counted GRU-dependent parameters
                indent = m.group(1)
                rest = m.group(3)
                comment = ''
                for marker in ('#', '!'):
                    if marker in rest:
                        rest, comment = rest.split(marker, 1)
                        comment = marker + comment
                        break
                values = rest.split()
                if not values:
                    continue
                if len(values) >= num_grus:
                    continue  # already expanded (idempotent re-runs)
                expanded = '    '.join([values[0]] * num_grus)
                new_line = f"{indent}{m.group(2)}  {expanded}"
                if comment:
                    new_line += f"  {comment.rstrip()}"
                new_line += '\n'
                lines[i] = new_line
                n_expanded += 1
                changed = True

            if changed:
                with open(self.hydro_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                self.logger.info(
                    f"Expanded {n_expanded} GRU-dependent hydrology parameter(s) to "
                    f"{num_grus} values each (NTYPE={num_grus}) in {self.hydro_path.name}"
                )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(
                f"Failed to expand GRU-dependent hydrology parameters: {e}", exc_info=True
            )

    def _count_class_gru_blocks(self) -> int:
        """Number of GRU parameter blocks in CLASS.ini (== MESH NTYPE).

        Counts the ``DRN/SDEP`` marker that opens each GRU block. Falls back to
        the drainage-database GRU count, then 1 (lumped) if neither is readable.
        """
        try:
            if self.class_file_path.exists():
                with open(self.class_file_path, 'r', encoding='utf-8') as f:
                    n = sum(1 for ln in f if 'DRN/SDEP' in ln)
                if n > 0:
                    return n
        except OSError:
            pass
        ddb_count = self._ddb.get_gru_count()
        return ddb_count if ddb_count and ddb_count > 0 else 1

    # ==================================================================
    # Reservoir
    # ==================================================================

    def fix_reservoir_file(self) -> None:
        """Fix reservoir input file to match IREACH in drainage database."""
        reservoir_file = self.forcing_dir / "MESH_input_reservoir.txt"

        max_ireach = 0
        if self.ddb_path.exists():
            try:
                with xr.open_dataset(self.ddb_path) as ds:
                    if 'IREACH' in ds:
                        ireach_vals = ds['IREACH'].values
                        valid_vals = ireach_vals[ireach_vals >= 0]
                        if len(valid_vals) > 0:
                            max_ireach = int(np.max(valid_vals))
                        self.logger.debug(f"Max IREACH from DDB: {max_ireach}")
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.debug(f"Could not read IREACH from DDB: {e}", exc_info=True)

        try:
            if max_ireach == 0:
                with open(reservoir_file, 'w', encoding='utf-8') as f:
                    f.write("0\n")
                self.logger.info("Fixed reservoir file: 0 reservoirs")
            else:
                if reservoir_file.exists():
                    with open(reservoir_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    first_line = content.split('\n')[0] if content else ""
                    parts = first_line.split()
                    file_count = int(parts[0]) if parts else 0
                    if file_count != max_ireach:
                        self.logger.warning(
                            f"Reservoir file count ({file_count}) != IREACH max ({max_ireach}). "
                            f"Reservoir routing may fail."
                        )
                else:
                    self.logger.warning(
                        f"Reservoir file not found but IREACH max = {max_ireach}. "
                        f"Creating placeholder with 0 reservoirs."
                    )
                    with open(reservoir_file, 'w', encoding='utf-8') as f:
                        f.write("0\n")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix reservoir file: {e}", exc_info=True)

    # ==================================================================
    # Lumped outputs
    # ==================================================================

    def configure_lumped_outputs(self) -> None:
        """Configure outputs_balance.txt for lumped mode calibration."""
        outputs_balance = self.forcing_dir / "outputs_balance.txt"
        if not outputs_balance.exists():
            return

        num_cells = self._get_num_cells()
        if num_cells > 1:
            return

        try:
            with open(outputs_balance, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith('!RFF') and ('D' in stripped or 'H' in stripped) and 'csv' in stripped.lower():
                    new_line = stripped[1:].replace(' H ', ' D ')
                    new_lines.append(new_line + '\n')
                    modified = True
                    continue
                new_lines.append(line)

            if not any('RFF' in l and 'csv' in l.lower() and not l.strip().startswith('!')
                      for l in new_lines):
                new_lines.append('RFF     D  csv\n')
                modified = True

            if modified:
                with open(outputs_balance, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                self.logger.info("Configured outputs_balance.txt for lumped mode (daily RFF csv)")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to configure lumped outputs: {e}", exc_info=True)

    # ==================================================================
    # Safe forcing
    # ==================================================================

    def create_safe_forcing(self) -> None:
        """Create a trimmed forcing file for the simulation period."""
        forcing_nc = self.forcing_dir / "MESH_forcing.nc"
        safe_forcing_nc = self.forcing_dir / "MESH_forcing_safe.nc"

        if not forcing_nc.exists():
            self.logger.warning("No MESH_forcing.nc found")
            return

        try:
            time_window = self._get_time_window()
            if not time_window:
                self.logger.warning("No simulation time window configured")
                return

            analysis_start, end_time = time_window

            with xr.open_dataset(forcing_nc) as ds_check:
                forcing_times = pd.to_datetime(ds_check['time'].values)
                forcing_start = forcing_times[0]
                forcing_end = forcing_times[-1]

            configured_spinup = self._get_config_value(
                lambda: self.config.model.mesh.spinup_days,
                default=None, dict_key='MESH_SPINUP_DAYS'
            )

            if configured_spinup is not None:
                spinup_days = int(configured_spinup)
            else:
                latitude = self._get_domain_latitude()
                spinup_days = MESHConfigDefaults.get_recommended_spinup_days(latitude=latitude)
                self.logger.info(f"Using recommended spinup of {spinup_days} days (based on lat={latitude})")

            from datetime import timedelta
            requested_start = pd.Timestamp(analysis_start - timedelta(days=spinup_days))

            if requested_start < forcing_start:
                actual_spinup_days = (analysis_start - forcing_start).days
                start_time = pd.Timestamp(forcing_start)
                self.logger.warning(f"Limiting spinup to {actual_spinup_days} days")
                self._actual_spinup_days = actual_spinup_days
            else:
                start_time = requested_start
                self._actual_spinup_days = spinup_days

            end_time = pd.Timestamp(end_time)
            if end_time > forcing_end:
                end_time = forcing_end

            end_time_padded = min(end_time + timedelta(days=2), forcing_end)

            with xr.open_dataset(forcing_nc) as ds:
                if 'time' not in ds.dims:
                    return

                times = pd.to_datetime(ds['time'].values)
                start_idx, end_idx = 0, len(times)

                for i, t in enumerate(times):
                    if t >= start_time:
                        start_idx = max(0, i - 1)
                        break

                for i, t in enumerate(times):
                    if t > end_time_padded:
                        end_idx = i
                        break

                ds_safe = ds.isel(time=slice(start_idx, end_idx))
                n_timesteps = end_idx - start_idx

                self.logger.info(f"Creating MESH_forcing_safe.nc with {n_timesteps} timesteps")

                from netCDF4 import Dataset as NC4Dataset
                n_spatial = ds_safe.sizes.get('subbasin', 1)

                with NC4Dataset(safe_forcing_nc, 'w', format='NETCDF4') as ncfile:
                    ncfile.createDimension('time', None)
                    ncfile.createDimension('subbasin', n_spatial)

                    var_time = ncfile.createVariable('time', 'f8', ('time',))
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.axis = 'T'
                    var_time.units = 'hours since 1900-01-01 00:00:00'
                    var_time.calendar = 'gregorian'

                    reference = pd.Timestamp('1900-01-01')
                    time_hours = np.array([
                        (pd.Timestamp(t) - reference).total_seconds() / 3600.0
                        for t in ds_safe['time'].values
                    ])
                    var_time[:] = time_hours

                    var_n = ncfile.createVariable('subbasin', 'i4', ('subbasin',))
                    var_n[:] = np.arange(1, n_spatial + 1)

                    for coord_var in ['lat', 'lon']:
                        if coord_var in ds_safe:
                            var = ncfile.createVariable(coord_var, 'f8', ('subbasin',))
                            for attr in ds_safe[coord_var].attrs:
                                var.setncattr(attr, ds_safe[coord_var].attrs[attr])
                            var[:] = ds_safe[coord_var].values

                    if 'crs' in ds_safe:
                        var_crs = ncfile.createVariable('crs', 'i4')
                        for attr in ds_safe['crs'].attrs:
                            var_crs.setncattr(attr, ds_safe['crs'].attrs[attr])

                    forcing_vars = ['PRES', 'QA', 'TA', 'UV', 'PRE', 'FSIN', 'FLIN']
                    for var_name in forcing_vars:
                        if var_name in ds_safe:
                            var = ncfile.createVariable(
                                var_name, 'f4', ('time', 'subbasin'),
                                fill_value=-9999.0
                            )
                            for attr in ds_safe[var_name].attrs:
                                if attr != '_FillValue':
                                    var.setncattr(attr, ds_safe[var_name].attrs[attr])
                            var.missing_value = -9999.0
                            var[:] = ds_safe[var_name].values

                    ncfile.author = "University of Calgary"
                    ncfile.license = "GNU General Public License v3 (or any later version)"
                    ncfile.purpose = "Create forcing .nc file for MESH"
                    ncfile.Conventions = "CF-1.6"
                    if 'history' in ds.attrs:
                        ncfile.history = ds.attrs['history']

            self._run_options.update_for_safe_forcing(
                start_time, end_time, self._actual_spinup_days
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.warning(f"Failed to create safe forcing: {e}", exc_info=True)
            self.logger.debug(traceback.format_exc())

    def _get_time_window(self) -> Optional[Tuple]:
        """Get simulation time window from config or callback."""
        if self.get_simulation_time_window:
            time_window = self.get_simulation_time_window()
            if time_window:
                return time_window

        cal_period = self._get_config_value(
            lambda: self.config.domain.calibration_period,
            dict_key='CALIBRATION_PERIOD'
        )
        eval_period = self._get_config_value(
            lambda: self.config.domain.evaluation_period,
            dict_key='EVALUATION_PERIOD'
        )

        if cal_period:
            cal_parts = [p.strip() for p in str(cal_period).split(',')]
            if len(cal_parts) >= 2:
                analysis_start = pd.Timestamp(cal_parts[0])
                if eval_period:
                    eval_parts = [p.strip() for p in str(eval_period).split(',')]
                    end_time = pd.Timestamp(eval_parts[1] if len(eval_parts) >= 2 else eval_parts[0])
                else:
                    end_time = pd.Timestamp(cal_parts[1])
                return (analysis_start, end_time)

        return None
