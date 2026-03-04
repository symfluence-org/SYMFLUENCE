#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
NextGen (ngen) Parameter Manager

Handles ngen parameter bounds, normalization, denormalization, and
configuration file updates for model calibration.

Author: SYMFLUENCE Development Team
Date: 2025
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_ngen_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('NGEN')
class NgenParameterManager(BaseParameterManager):
    """Manages ngen calibration parameters across CFE, NOAH-OWP, and PET modules"""

    def __init__(self, config: Dict, logger: logging.Logger, ngen_settings_dir: Path):
        """
        Initialize ngen parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger object
            ngen_settings_dir: Path to ngen settings directory
        """
        # Initialize base class
        super().__init__(config, logger, ngen_settings_dir)

        # Ngen-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse which modules to calibrate
        self.modules_to_calibrate = self._parse_modules_to_calibrate()

        # Parse parameters to calibrate for each module
        self.params_to_calibrate = self._parse_parameters_to_calibrate()

        # Path to ngen configuration files
        self.ngen_setup_dir = Path(ngen_settings_dir)

        # Configuration file paths
        self.realization_config = self.ngen_setup_dir / 'realization_config.json'
        self.cfe_txt_dir = self.ngen_setup_dir / 'CFE'
        self.noah_dir = self.ngen_setup_dir / 'NOAH'
        self.pet_dir  = self.ngen_setup_dir / 'PET'

        # expected JSONs (may not exist; that's fine)
        self.noah_config = self.noah_dir / 'noah_config.json'
        self.pet_config  = self.pet_dir  / 'pet_config.json'
        self.cfe_config = self.cfe_txt_dir / 'cfe_config.json'

        # BMI text dirs
        self.pet_txt_dir  = self.pet_dir

        # Determine hydro_id for configuration file matching
        # For lumped catchments, we can find the ID from the files themselves
        self.hydro_id = self._resolve_hydro_id()

        # Default TBL mappings for NOAH (used if JSON or namelist overrides aren't available)
        # Format: de_param -> (tbl_file, variable_name, column_index or None for single value)
        # Column indices are 0-indexed into parts[] after line.split(), where parts[0] is
        # the row index. SOILPARM.TBL columns: BB=1, DRYSMC=2, F11=3, MAXSMC=4,
        # REFSMC=5, SATPSI=6, SATDK=7, SATDW=8, WLTSMC=9, QTZ=10
        self.noah_tbl_map = {
            "refkdt": ("GENPARM.TBL", "REFKDT_DATA", None),
            "slope":  ("GENPARM.TBL", "SLOPE_DATA", 1), # Default to first slope category
            "smcmax": ("SOILPARM.TBL", "MAXSMC", 4),
            "dksat":  ("SOILPARM.TBL", "SATDK", 7),
            "bb":     ("SOILPARM.TBL", "BB", 1),
            "bexp":   ("SOILPARM.TBL", "BB", 1),  # bexp is alias for BB (pore size distribution)
        }

        self.logger.debug("NgenParameterManager initialized")
        self.logger.debug(f"Calibrating modules: {self.modules_to_calibrate}")
        self.logger.debug(f"Total parameters to calibrate: {len(self.all_param_names)}")

    def _resolve_hydro_id(self) -> Optional[str]:
        """Resolve the active catchment ID (hydro_id) from available configuration files."""
        # Try to find a cat-*.txt file in CFE directory
        if self.cfe_txt_dir.exists():
            candidates = list(self.cfe_txt_dir.glob("cat-*_bmi_config_cfe_*.txt"))
            if candidates:
                # Extract '1' from 'cat-1_bmi_config_cfe_pass.txt'
                filename = candidates[0].name
                match = re.search(r'cat-([a-zA-Z0-9_-]+)', filename)
                if match:
                    res = match.group(1)
                    # Strip any trailing suffixes if needed, e.g. _bmi_config...
                    if '_' in res:
                        res = res.split('_')[0]
                    return res

        # Fallback to NOAH directory
        if self.noah_dir.exists():
            candidates = list(self.noah_dir.glob("cat-*.input"))
            if candidates:
                filename = candidates[0].name
                match = re.search(r'cat-([a-zA-Z0-9_-]+)', filename)
                if match:
                    res = match.group(1)
                    if '.' in res:
                        res = res.split('.')[0]
                    return res

        return None

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS FROM BASE CLASS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return ngen parameter names in module.param format."""
        all_params = []
        for module, params in self.params_to_calibrate.items():
            # Prefix parameters with module name to avoid conflicts
            all_params.extend([f"{module}.{p}" for p in params])
        return all_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return ngen parameter bounds, merging config YAML with registry defaults.

        Checks NGEN_CFE_PARAM_BOUNDS, NGEN_NOAH_PARAM_BOUNDS, and NGEN_PET_PARAM_BOUNDS
        from the config YAML first. Config bounds override min/max values but preserve
        the 'transform' key from the registry (e.g., 'log' for parameters spanning
        multiple orders of magnitude like satdk, dksat, Cgw).
        """
        base_bounds = get_ngen_bounds()
        bounds = {}

        # Load config-specified bounds per module and merge into base_bounds
        # (preserves transform metadata via _apply_config_bounds_override)
        config_bounds_keys = {
            'CFE': ('NGEN_CFE_PARAM_BOUNDS', lambda: self.config.model.ngen.cfe_param_bounds),
            'NOAH': ('NGEN_NOAH_PARAM_BOUNDS', lambda: self.config.model.ngen.noah_param_bounds),
            'PET': ('NGEN_PET_PARAM_BOUNDS', lambda: self.config.model.ngen.pet_param_bounds),
        }
        for module, (config_key, typed_accessor) in config_bounds_keys.items():
            module_bounds = self._get_config_value(
                typed_accessor,
                default=None,
                dict_key=config_key
            )
            if isinstance(module_bounds, dict):
                # Snapshot registry bounds before override for comparison
                registry_snapshot = {
                    k: dict(v) for k, v in base_bounds.items()
                    if k in module_bounds
                }
                self._apply_config_bounds_override(base_bounds, module_bounds)
                # Warn when config widens bounds beyond registry defaults
                for param, orig in registry_snapshot.items():
                    if param in base_bounds:
                        new = base_bounds[param]
                        if new['min'] < orig['min'] or new['max'] > orig['max']:
                            self.logger.warning(
                                f"Config {config_key} widens {param} bounds: "
                                f"registry [{orig['min']:.2g}, {orig['max']:.2g}] -> "
                                f"config [{new['min']:.2g}, {new['max']:.2g}]. "
                                f"Wide bounds may increase crash rate."
                            )

        for module, params in self.params_to_calibrate.items():
            for param in params:
                full_param_name = f"{module}.{param}"
                if param in base_bounds:
                    bounds[full_param_name] = base_bounds[param]
                else:
                    self.logger.warning(
                        f"No bounds defined for parameter {param}, using default [0.1, 10.0]"
                    )
                    bounds[full_param_name] = {'min': 0.1, 'max': 10.0}

        return bounds

    def _get_default_ngen_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return default ngen bounds without module prefixes."""
        return get_ngen_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update ngen config files (JSON or BMI text)."""
        return self.update_config_files(params)

    def get_initial_parameters(self) -> Dict[str, float]:
        """Get initial ngen parameters (midpoint of bounds)."""
        return self.get_default_parameters()

    def _parse_modules_to_calibrate(self) -> List[str]:
        """Parse which ngen modules to calibrate from config"""
        modules_str = self._get_config_value(lambda: self.config.model.ngen.modules_to_calibrate, default='CFE', dict_key='NGEN_MODULES_TO_CALIBRATE')
        if modules_str is None:
            modules_str = 'CFE'
        modules = [m.strip().upper() for m in modules_str.split(',') if m.strip()]

        # Validate modules (filter invalid ones without mutating during iteration)
        valid_modules = ['CFE', 'NOAH', 'PET']
        validated = []
        for module in modules:
            if module in valid_modules:
                validated.append(module)
            else:
                self.logger.warning(f"Unknown module '{module}', skipping")

        return validated if validated else ['CFE']  # Default to CFE

    def _parse_parameters_to_calibrate(self) -> Dict[str, List[str]]:
        """Parse parameters to calibrate for each module"""
        params = {}

        # CFE parameters
        if 'CFE' in self.modules_to_calibrate:
            cfe_params_str = self._get_config_value(lambda: self.config.model.ngen.cfe_params_to_calibrate, default='maxsmc,satdk,bb,slop,Cgw,max_gw_storage,K_nash,K_lf,soil_depth', dict_key='NGEN_CFE_PARAMS_TO_CALIBRATE')
            if cfe_params_str is None:
                cfe_params_str = 'maxsmc,satdk,bb,slop,Cgw,max_gw_storage,K_nash,K_lf,soil_depth'
            params['CFE'] = [p.strip() for p in cfe_params_str.split(',') if p.strip()]

        # NOAH-OWP parameters
        if 'NOAH' in self.modules_to_calibrate:
            noah_params_str = self._get_config_value(lambda: self.config.model.ngen.noah_params_to_calibrate, default='refkdt,slope,smcmax,dksat,bexp', dict_key='NGEN_NOAH_PARAMS_TO_CALIBRATE')
            if noah_params_str is None:
                noah_params_str = 'refkdt,slope,smcmax,dksat,bexp'
            params['NOAH'] = [p.strip() for p in noah_params_str.split(',') if p.strip()]

        # PET parameters
        if 'PET' in self.modules_to_calibrate:
            pet_params_str = self._get_config_value(lambda: self.config.model.ngen.pet_params_to_calibrate, default='wind_speed_measurement_height_m', dict_key='NGEN_PET_PARAMS_TO_CALIBRATE')
            if pet_params_str is None:
                pet_params_str = 'wind_speed_measurement_height_m'
            params['PET'] = [p.strip() for p in pet_params_str.split(',') if p.strip()]

        return params

    # Note: Parameter bounds are now provided by the central ParameterBoundsRegistry
    # Note: all_param_names property and get_parameter_bounds() are inherited from BaseParameterManager
    def get_default_parameters(self) -> Dict[str, float]:
        """Get default parameter values (middle of bounds).

        Uses geometric midpoint for log-transformed parameters to avoid
        bias toward the upper bound in log-space.
        """
        import math
        bounds = self.param_bounds
        params = {}

        for param_name, param_bounds in bounds.items():
            transform = param_bounds.get('transform', 'linear')
            if transform == 'log' and param_bounds['min'] > 0:
                # Geometric midpoint for log-space parameters
                params[param_name] = math.sqrt(param_bounds['min'] * param_bounds['max'])
            else:
                params[param_name] = (param_bounds['min'] + param_bounds['max']) / 2.0

        return params

    # ========================================================================
    # NOTE: The following methods are now inherited from BaseParameterManager:
    # - normalize_parameters()
    # - denormalize_parameters()
    # - validate_parameters()
    # These shared implementations eliminate ~80 lines of duplicated code!
    # ========================================================================

    # Validation function to help debug parameter updates
    def validate_parameter_updates(self, param_dict: Dict[str, float], config_file_path: Path) -> bool:
        """
        Validate that parameters were actually written to config file.
        Use this for debugging calibration issues.

        Args:
            param_dict: Dictionary of parameters that should have been updated
            config_file_path: Path to the CFE BMI config file

        Returns:
            True if all parameters found in file, False otherwise
        """
        if not config_file_path.exists():
            self.logger.error(f"Config file not found: {config_file_path}")
            return False

        content = config_file_path.read_text(encoding='utf-8')
        all_found = True

        for param_name, expected_value in param_dict.items():
            # Check if parameter appears in file
            if param_name in content or param_name.replace('_', '.') in content:
                self.logger.info(f"✓ Parameter {param_name} found in config")
            else:
                self.logger.error(f"✗ Parameter {param_name} NOT found in config")
                all_found = False

        return all_found

    # Note: validate_parameters() is now inherited from BaseParameterManager
    def update_config_files(self, params: Dict[str, float]) -> bool:
        """
        Update ngen configuration files with new parameter values.

        Args:
            params: Dictionary of parameters (with module.param naming)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Group parameters by module
            module_params: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                if '.' in param_name:
                    module, param = param_name.split('.', 1)
                    if module not in module_params:
                        module_params[module] = {}
                    module_params[module][param] = value

            # Update each module's config file
            success = True
            if 'CFE' in module_params:
                success = success and self._update_cfe_config(module_params['CFE'])

            if 'NOAH' in module_params:
                success = success and self._update_noah_config(module_params['NOAH'])

            if 'PET' in module_params:
                success = success and self._update_pet_config(module_params['PET'])

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating ngen config files: {e}")
            return False


    def _update_cfe_config(self, params: Dict[str, float]) -> bool:
        """
        Update CFE configuration: prefer JSON, fallback to BMI .txt.
        Preserves units in [brackets] for BMI text files.

        """
        try:
            # --- Preferred path: JSON file ---
            if self.cfe_config.exists():
                with open(self.cfe_config, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                updated = 0
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
                        updated += 1
                    else:
                        self.logger.warning(f"CFE parameter {k} not found in JSON config")
                with open(self.cfe_config, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2)
                self.logger.debug(f"Updated CFE JSON with {updated} parameters")
                return True

            # --- Fallback: BMI text file ---
            candidates = []
            if getattr(self, "hydro_id", None):
                pattern = f"cat-{self.hydro_id}_bmi_config_cfe_*.txt"
                candidates = list(self.cfe_txt_dir.glob(pattern))

            if not candidates:
                candidates = list(self.cfe_txt_dir.glob("*.txt"))

            if len(candidates) == 0:
                self.logger.error(f"CFE config not found (no JSON, no BMI .txt in {self.cfe_txt_dir})")
                return False
            if len(candidates) > 1:
                self.logger.error(f"Multiple BMI .txt files in {self.cfe_txt_dir}; please set NGEN_ACTIVE_CATCHMENT_ID or prune files")
                return False

            path = candidates[0]
            lines = path.read_text(encoding='utf-8').splitlines()

            # FIXED: Complete parameter mapping including groundwater and routing params
            keymap = {
                # Soil parameters
                "bb": "soil_params.b",
                "satdk": "soil_params.satdk",
                "slop": "soil_params.slop",
                "maxsmc": "soil_params.smcmax",
                "smcmax": "soil_params.smcmax",
                "wltsmc": "soil_params.wltsmc",
                "satpsi": "soil_params.satpsi",
                "expon": "soil_params.expon",

                # Groundwater parameters (CRITICAL - these were missing!)
                "Cgw": "Cgw",
                "max_gw_storage": "max_gw_storage",

                # Routing parameters (CRITICAL - these were missing!)
                "K_nash": "K_nash",
                "K_lf": "K_lf",
                "Kn": "K_nash",      # Alias
                "Klf": "K_lf",       # Alias

                # Other CFE parameters
                "alpha_fc": "alpha_fc",
                "refkdt": "refkdt",
                "soil_depth": "soil_params.depth",
            }

            # Helper: write numeric value preserving any trailing [units]
            num_units_re = re.compile(r"""
                ^\s*             # leading space
                (?P<num>[+-]?(?:\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)  # number (incl. sci)
                (?P<tail>\s*(\[[^\]]*\])?.*)$   # optional units and remainder
            """, re.VERBOSE)

            def render_value(original_rhs: str, new_val: float) -> str:
                m = num_units_re.match(original_rhs.strip())
                if m:
                    tail = m.group('tail') or ''
                    return f"{new_val:.8g}{tail}"
                return f"{new_val:.8g}"

            # Determine num_timesteps from config using FORCING_TIME_STEP_SIZE
            start_time = self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')
            end_time = self._get_config_value(lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END')
            forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
            try:
                forcing_timestep = int(forcing_timestep)
            except (ValueError, TypeError):
                forcing_timestep = 3600

            if start_time and end_time:
                try:
                    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
                    # Use configured timestep, add 1 for inclusive end bound
                    num_steps = int(duration.total_seconds() / forcing_timestep) + 1
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Could not parse time range, defaulting to 1 timestep: {e}")
                    num_steps = 1
            else:
                num_steps = 1

            updated = set()
            for i, line in enumerate(lines):
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, rhs = line.split("=", 1)
                k = k.strip()
                rhs_keep = rhs.rstrip("\n")

                # Update num_timesteps
                if k == "num_timesteps":
                    lines[i] = f"num_timesteps={num_steps}"
                    continue

                # Surface runoff scheme configuration - respect user config if set
                # Default to Schaake partitioning and GIUH routing if not specified
                cfe_partition_scheme = self._get_config_value(
                    lambda: None, default='Schaake', dict_key='NGEN_CFE_PARTITION_SCHEME')
                cfe_runoff_scheme = self._get_config_value(
                    lambda: None, default='GIUH', dict_key='NGEN_CFE_RUNOFF_SCHEME')
                if k == "surface_water_partitioning_scheme":
                    lines[i] = f"surface_water_partitioning_scheme={cfe_partition_scheme}"
                    continue
                if k == "surface_runoff_scheme":
                    lines[i] = f"surface_runoff_scheme={cfe_runoff_scheme}"
                    continue

                # Match parameters by mapped BMI key
                for p, bmi_k in keymap.items():
                    if p in params and k == bmi_k:
                        new_rhs = render_value(rhs_keep, params[p])
                        lines[i] = f"{k}={new_rhs}"
                        updated.add(p)

            # Warn about any requested params we couldn't find in the BMI file
            for p in params:
                if p not in updated and p in keymap:
                    self.logger.warning(f"CFE parameter {p} not found in BMI config {path.name}")

            path.write_text("\n".join(lines) + "\n", encoding='utf-8')
            self.logger.debug(f"Updated CFE BMI text ({path.name}) with {len(updated)} parameters")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating CFE config: {e}")
            return False




    def _update_noah_config(self, params: Dict[str, float]) -> bool:
        """
        Update NOAH configuration for calibration:
        1) Prefer JSON if present.
        2) Fallback to NOAH BMI input file ({{id}}.input).
        3) Optionally update TBL parameters in NOAH/parameters (if mappings supplied).
        """
        try:
            if self.noah_config.exists():
                return self._update_noah_json(params)

            if not self.noah_dir.exists():
                self.logger.error(f"NOAH directory missing: {self.noah_dir}")
                return False

            input_file = self._select_noah_input_file()
            if input_file is None:
                return False

            if self._update_noah_input_namelist(input_file, params):
                return True

            return self._update_noah_tbl_parameters(params, input_file)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating NOAH config: {e}")
            return False

    def _update_noah_json(self, params: Dict[str, float]) -> bool:
        with open(self.noah_config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        updated = 0
        for key, value in params.items():
            if key in cfg:
                cfg[key] = value
                updated += 1
            else:
                self.logger.warning(f"NOAH parameter {key} not in JSON config")

        with open(self.noah_config, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)

        self.logger.debug(f"Updated NOAH JSON with {updated} parameters")
        return True

    def _select_noah_input_file(self) -> Optional[Path]:
        input_candidates: List[Path] = []
        if getattr(self, "hydro_id", None):
            input_candidates = list(self.noah_dir.glob(f"cat-{self.hydro_id}.input"))
        if not input_candidates:
            input_candidates = list(self.noah_dir.glob("*.input"))

        if len(input_candidates) == 0:
            self.logger.error(f"NOAH config not found: no JSON and no *.input under {self.noah_dir}")
            return None
        if len(input_candidates) > 1:
            self.logger.error("Multiple NOAH *.input files; set NGEN_ACTIVE_CATCHMENT_ID to disambiguate")
            return None
        return input_candidates[0]

    def _replace_noah_namelist_value(
        self,
        text: str,
        section: str,
        key: str,
        new_val: float,
    ) -> Tuple[str, bool]:
        section_pattern = re.compile(rf"(?s)&\s*{re.escape(section)}\b(.*?)/")
        section_match = section_pattern.search(text)
        if not section_match:
            return text, False

        section_body = section_match.group(1)
        key_pattern = re.compile(rf"(^|\n)(\s*{re.escape(key)}\s*=\s*)([^,\n/]+)", re.MULTILINE)

        def _render(match):
            prefix = match.group(2)
            rhs = match.group(3).strip()
            if rhs.startswith('"') and rhs.endswith('"'):
                return f'{match.group(1)}{prefix}"{new_val:.8g}"'
            return f"{match.group(1)}{prefix}{new_val:.8g}"

        updated_body, replacements = key_pattern.subn(_render, section_body, count=1)
        if replacements == 0:
            return text, False

        updated_text = text[:section_match.start(1)] + updated_body + text[section_match.end(1):]
        return updated_text, True

    def _update_noah_input_namelist(self, input_file: Path, params: Dict[str, float]) -> bool:
        text = input_file.read_text(encoding='utf-8')
        keymap = {
            "rain_snow_thresh": ("forcing", "rain_snow_thresh"),
            "ZREF": ("forcing", "ZREF"),
            "dt": ("timing", "dt"),
        }

        updated_inputs = 0
        for param_name, (section, key) in keymap.items():
            if param_name not in params:
                continue

            text, updated = self._replace_noah_namelist_value(
                text, section, key, params[param_name]
            )
            if updated:
                updated_inputs += 1
            else:
                self.logger.warning(
                    f"NOAH param {param_name} ({section}.{key}) not found in {input_file.name}"
                )

        if updated_inputs > 0:
            input_file.write_text(text, encoding='utf-8')
            self.logger.debug(
                f"Updated NOAH input ({input_file.name}) with {updated_inputs} parameter(s)"
            )
            return True
        return False

    def _detect_noah_isltyp(self, input_file: Path) -> int:
        isltyp = 1
        try:
            input_text = input_file.read_text(encoding='utf-8')
            match = re.search(r"isltyp\s*=\s*(\d+)", input_text)
            if match:
                isltyp = int(match.group(1))
        except (OSError, IOError, ValueError) as e:
            self.logger.debug(f"Could not read isltyp from input file, using default: {e}")
        return isltyp

    def _update_soil_table_line(
        self,
        lines: List[str],
        line_idx: int,
        parts: List[str],
        col: Optional[int],
        new_val: float,
        isltyp: int,
    ) -> bool:
        try:
            idx_str = parts[0].rstrip(',')
            if int(idx_str) != isltyp or col is None or col >= len(parts):
                return False

            fmt_val = f"{new_val:.4E}" if new_val < 0.001 else f"{new_val:.6g}"
            parts[col] = fmt_val

            if parts[0].endswith(','):
                clean_parts = [p.rstrip(',') for p in parts]
                lines[line_idx] = f"{clean_parts[0] + ',':<4} {', '.join(clean_parts[1:])}"
            else:
                lines[line_idx] = " ".join(parts)
            return True
        except (ValueError, IndexError):
            return False

    def _update_genparm_table_line(
        self,
        lines: List[str],
        line_idx: int,
        parts: List[str],
        variable: str,
        col: Optional[int],
        new_val: float,
    ) -> bool:
        if not parts[0].startswith(variable):
            return False

        if col is None:
            target_line = line_idx + 1
            if target_line < len(lines):
                lines[target_line] = f"{new_val:.8g}"
                return True
            return False

        if variable == "SLOPE_DATA":
            target_line = line_idx + 1 + col
            if target_line < len(lines):
                lines[target_line] = f"{new_val:.8g}"
                return True
            return False

        for candidate in range(line_idx + 1, min(line_idx + 10, len(lines))):
            stripped = lines[candidate].strip()
            if stripped and not stripped.startswith("'"):
                lines[candidate] = f"{new_val:.8g}"
                return True
        return False

    def _edit_noah_tbl_value(
        self,
        table_path: Path,
        variable: str,
        col: Optional[int],
        new_val: float,
        isltyp: int,
    ) -> bool:
        if not table_path.exists():
            return False

        lines = table_path.read_text(encoding='utf-8').splitlines()
        is_soil_tbl = "SOILPARM" in table_path.name

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("'"):
                continue

            parts = line.split()
            if not parts:
                continue

            changed = False
            if is_soil_tbl:
                changed = self._update_soil_table_line(
                    lines, idx, parts, col, new_val, isltyp
                )
            else:
                changed = self._update_genparm_table_line(
                    lines, idx, parts, variable, col, new_val
                )

            if changed:
                table_path.write_text("\n".join(lines) + "\n", encoding='utf-8')
                return True

        return False

    def _update_noah_tbl_parameters(self, params: Dict[str, float], input_file: Path) -> bool:
        tbl_map: Dict[str, Tuple[str, str, Optional[int]]] = getattr(self, "noah_tbl_map", {})
        if not tbl_map:
            return True

        params_dir = self.noah_dir / "parameters"
        if not params_dir.exists():
            self.logger.error(f"NOAH parameters directory missing: {params_dir}")
            return False

        isltyp = self._detect_noah_isltyp(input_file)
        updated_tbls = 0

        for param_name, (fname, variable, col) in tbl_map.items():
            if param_name not in params:
                continue

            table_path = params_dir / fname
            if self._edit_noah_tbl_value(table_path, variable, col, params[param_name], isltyp):
                updated_tbls += 1
            else:
                self.logger.warning(
                    f"NOAH TBL param {param_name} ({fname}:{variable}[{col}]) not found/updated"
                )

        if updated_tbls > 0:
            self.logger.debug(f"Updated NOAH TBLs with {updated_tbls} parameter(s)")
        return True


    def _update_pet_config(self, params: Dict[str, float]) -> bool:
        """
        Update PET configuration:
        1) Prefer JSON if present.
        2) Fallback to PET BMI text file: PET/{{id}}_pet_config.txt (or the only *.txt).
        """
        try:
            # ---------- 1) JSON ----------
            if self.pet_config.exists():
                with open(self.pet_config, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                up = 0
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
                        up += 1
                    else:
                        self.logger.warning(f"PET parameter {k} not in JSON config")
                with open(self.pet_config, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2)
                self.logger.debug(f"Updated PET JSON with {up} parameter(s)")
                return True

            # ---------- 2) BMI text ----------
            # pick file by hydro_id if present, else a single *.txt under PET/
            if not self.pet_txt_dir.exists():
                self.logger.error(f"PET directory missing: {self.pet_txt_dir}")
                return False

            candidates = []
            if getattr(self, "hydro_id", None):
                candidates = list(self.pet_txt_dir.glob(f"cat-{self.hydro_id}_pet_config.txt"))
            if not candidates:
                candidates = list(self.pet_txt_dir.glob("*.txt"))

            if len(candidates) == 0:
                self.logger.error(f"PET config not found: no JSON and no *.txt in {self.pet_txt_dir}")
                return False
            if len(candidates) > 1:
                self.logger.error("Multiple PET *.txt configs; set NGEN_ACTIVE_CATCHMENT_ID to disambiguate")
                return False

            path = candidates[0]
            lines = path.read_text(encoding='utf-8').splitlines()

            # Determine num_timesteps from config using FORCING_TIME_STEP_SIZE
            start_time = self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')
            end_time = self._get_config_value(lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END')
            forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
            try:
                forcing_timestep = int(forcing_timestep)
            except (ValueError, TypeError):
                forcing_timestep = 3600

            if start_time and end_time:
                try:
                    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
                    # Use configured timestep, add 1 for inclusive end bound
                    num_steps = int(duration.total_seconds() / forcing_timestep) + 1
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Could not parse time range, defaulting to 1 timestep: {e}")
                    num_steps = 1
            else:
                num_steps = 1

            import re
            num_units_re = re.compile(r"""
                ^\s*
                (?P<num>[+-]?(?:\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)
                (?P<tail>\s*(\[[^\]]*\])?.*)$
            """, re.VERBOSE)

            def render_value(rhs: str, new_val: float) -> str:
                m = num_units_re.match(rhs.strip())
                if m:
                    tail = m.group('tail') or ''
                    return f"{new_val:.8g}{tail}"
                return f"{new_val:.8g}"

            # Map calibration param names -> keys in PET text config file
            # Parameter names match the actual keys in cat-X_pet_config.txt
            keymap = {
                "vegetation_height_m": "vegetation_height_m",
                "zero_plane_displacement_height_m": "zero_plane_displacement_height_m",
                "momentum_transfer_roughness_length": "momentum_transfer_roughness_length",
                "heat_transfer_roughness_length_m": "heat_transfer_roughness_length_m",
                "surface_shortwave_albedo": "surface_shortwave_albedo",
                "surface_longwave_emissivity": "surface_longwave_emissivity",
                "wind_speed_measurement_height_m": "wind_speed_measurement_height_m",
                "humidity_measurement_height_m": "humidity_measurement_height_m",
            }

            updated = set()
            for i, line in enumerate(lines):
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, rhs = line.split("=", 1)
                key = k.strip()
                if not key:
                    continue

                # Update num_timesteps
                if key == "num_timesteps":
                    lines[i] = f"num_timesteps={num_steps}"
                    updated.add("num_timesteps")
                    continue

                for p, txt_key in keymap.items():
                    if p in params and key == txt_key:
                        lines[i] = f"{key}={render_value(rhs, params[p])}"
                        updated.add(p)

            for p in params:
                if p in keymap and p not in updated:
                    self.logger.warning(f"PET parameter {p} not found in {path.name}")

            if updated:
                path.write_text("\n".join(lines) + "\n", encoding='utf-8')
                self.logger.debug(f"Updated PET BMI text ({path.name}) with {len(updated)} parameter(s)")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating PET config: {e}")
            return False
