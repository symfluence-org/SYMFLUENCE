# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Parameter Manager

Handles PIHM parameter bounds, normalization, and input file updates.
Parameters are written into PIHM .soil, .calib, and .lc files.

Snow-17 parameters (SCF, MFMAX, PXTEMP) are also supported for
forcing regeneration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R

PIHM_DEFAULT_BOUNDS = {
    'K_SAT': {
        'min': 1e-8, 'max': 1e-3,
        'transform': 'log',
        'description': 'Saturated hydraulic conductivity (m/s)',
    },
    'POROSITY': {
        'min': 0.05, 'max': 0.6,
        'transform': 'linear',
        'description': 'Total porosity (-)',
    },
    'VG_ALPHA': {
        'min': 0.01, 'max': 10.0,
        'transform': 'log',
        'description': 'van Genuchten alpha (1/m)',
    },
    'VG_N': {
        'min': 1.1, 'max': 5.0,
        'transform': 'linear',
        'description': 'van Genuchten n shape parameter (-)',
    },
    'MACROPORE_K': {
        'min': 1e-7, 'max': 1e-2,
        'transform': 'log',
        'description': 'Macropore hydraulic conductivity (m/s)',
    },
    'MANNINGS_N': {
        'min': 0.005, 'max': 0.3,
        'transform': 'log',
        'description': "Manning's roughness coefficient (s/m^1/3)",
    },
    'SOIL_DEPTH': {
        'min': 0.5, 'max': 10.0,
        'transform': 'linear',
        'description': 'Active soil depth (m)',
    },
    'GEOL_K_RATIO': {
        'min': 0.0001, 'max': 0.5,
        'transform': 'log',
        'description': 'Ratio of geological K to soil K_SAT (-)',
    },
    'INIT_GW_DEPTH': {
        'min': 0.1, 'max': 5.0,
        'transform': 'linear',
        'description': 'Initial groundwater depth (m)',
    },
    'SNOW17_SCF': {
        'min': 0.7, 'max': 1.4,
        'transform': 'linear',
        'description': 'Snowfall correction factor',
    },
    'SNOW17_MFMAX': {
        'min': 0.5, 'max': 4.0,
        'transform': 'linear',
        'description': 'Max melt factor Jun 21 (mm/C/6hr)',
    },
    'SNOW17_PXTEMP': {
        'min': -4.0, 'max': 3.0,
        'transform': 'linear',
        'description': 'Rain/snow threshold temperature (C)',
    },
    'SFCTMP': {
        'min': -3.0, 'max': 3.0,
        'transform': 'linear',
        'description': 'Surface temperature offset (K) — rain/snow partition',
    },
    'PRCP': {
        'min': 0.7, 'max': 1.5,
        'transform': 'linear',
        'description': 'Precipitation multiplier — ERA5 correction',
    },
    'CZIL': {
        'min': 0.1, 'max': 10.0,
        'transform': 'log',
        'description': 'Zilitinkevich coefficient multiplier — surface exchange',
    },
    'RS': {
        'min': 0.5, 'max': 5.0,
        'transform': 'log',
        'description': 'Min stomatal resistance multiplier — transpiration control',
    },
}

SNOW17_PARAM_NAMES = {'SNOW17_SCF', 'SNOW17_MFMAX', 'SNOW17_PXTEMP'}
LSM_CALIB_PARAM_NAMES = {'CZIL', 'RS'}
SCENARIO_PARAM_NAMES = {'PRCP', 'SFCTMP'}
CALIB_ONLY_PARAM_NAMES = SNOW17_PARAM_NAMES | LSM_CALIB_PARAM_NAMES | SCENARIO_PARAM_NAMES


@R.parameter_managers.add('PIHM')
class PIHMParameterManager(BaseParameterManager):
    """Handles PIHM parameter bounds, normalization, and input file updates."""

    def __init__(self, config: Dict, logger: logging.Logger, pihm_settings_dir: Path):
        super().__init__(config, logger, pihm_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        pihm_params_str = self._get_config_value(
            lambda: self.config.model.pihm.params_to_calibrate,
            default='K_SAT,POROSITY,VG_ALPHA,VG_N,MACROPORE_K,MANNINGS_N,SOIL_DEPTH',
            dict_key='PIHM_PARAMS_TO_CALIBRATE'
        )
        self.pihm_params = [p.strip() for p in str(pihm_params_str).split(',') if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        """Return PIHM parameter names from config."""
        return self.pihm_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Return PIHM parameter bounds with transform info preserved."""
        bounds: Dict[str, Dict[str, Any]] = {
            k: {
                'min': float(v['min']),
                'max': float(v['max']),
                'transform': v.get('transform', 'linear'),
            }
            for k, v in PIHM_DEFAULT_BOUNDS.items()
        }

        # Support both dict and Pydantic config objects
        config_bounds = self._get_config_value(lambda: None, default=None, dict_key='PIHM_PARAM_BOUNDS')

        if config_bounds and isinstance(config_bounds, dict):
            self.logger.info("Using config-specified PIHM parameter bounds")
            self._apply_config_bounds_override(bounds, config_bounds)

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update PIHM input files with new parameter values.

        Updates all files that contain calibrated parameters:
        - .soil (K_SAT, POROSITY, VG_ALPHA, VG_N, MACROPORE_K)
        - .geol (K_SAT-derived deep K)
        - .riv  (MANNINGS_N roughness, K_SAT riverbed K)
        - .mesh (SOIL_DEPTH via ZMIN adjustment)
        - .ic   (SOIL_DEPTH, POROSITY via initial conditions)
        - .calib (multiplier form of all params)
        - .lc   (MANNINGS_N surface roughness)
        """
        try:
            if SNOW17_PARAM_NAMES.intersection(params):
                raise ValueError("PIHM Snow17 forcing regeneration is not implemented")
            subsurface_params = {k: v for k, v in params.items() if k not in CALIB_ONLY_PARAM_NAMES}
            lsm_params = {k: v for k, v in params.items() if k in LSM_CALIB_PARAM_NAMES}
            scenario_params = {k: v for k, v in params.items() if k in SCENARIO_PARAM_NAMES}

            # Update .soil file
            self._update_soil_file(subsurface_params)

            # Update .geol file (deep subsurface K from K_SAT * GEOL_K_RATIO)
            if 'K_SAT' in subsurface_params or 'GEOL_K_RATIO' in subsurface_params:
                self._update_geol_file(subsurface_params)

            # Update .riv file (Manning's N and riverbed K)
            if 'MANNINGS_N' in subsurface_params or 'K_SAT' in subsurface_params:
                self._update_riv_file(subsurface_params)

            # Update .mesh file (SOIL_DEPTH changes ZMIN)
            if 'SOIL_DEPTH' in subsurface_params:
                self._update_mesh_file(subsurface_params['SOIL_DEPTH'])

            # Update .ic file (SOIL_DEPTH, POROSITY, INIT_GW_DEPTH affect initial conditions)
            if any(k in subsurface_params for k in ('SOIL_DEPTH', 'POROSITY', 'INIT_GW_DEPTH')):
                self._update_ic_file(subsurface_params)

            # Update .calib file (subsurface multipliers reset + LSM/SCENARIO values)
            self._update_calib_file(subsurface_params, lsm_params, scenario_params)

            # Update .lc file for Manning's n
            if 'MANNINGS_N' in subsurface_params:
                self._update_lc_file(subsurface_params['MANNINGS_N'])

            self.logger.debug(f"Updated PIHM files with {len(params)} parameters")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to update PIHM files: {e}", exc_info=True)
            return False

    def _update_soil_file(self, params: Dict[str, float]) -> None:
        """Update .soil file with new parameter values.

        MM-PIHM .soil format:
            NUMSOIL  <n>
            INDEX  SILT  CLAY  OM  BD  KINF  KSATV  KSATH  MAXSMC  MINSMC  ALPHA  BETA  MACHF  MACVF  DMAC  QTZ
            1      ...   ...   ...
            DINF   <depth>
            KMACV_RO  <ratio>
            KMACH_RO  <ratio>

        Parameter mapping:
            K_SAT      -> KINF, KSATV, KSATH columns
            POROSITY   -> MAXSMC column
            VG_ALPHA   -> ALPHA column
            VG_N       -> BETA column
            MACROPORE_K -> MACHF, MACVF columns
            SOIL_DEPTH changes mesh ZMIN, not the surface infiltration depth DINF.
        """
        soil_files = list(self.settings_dir.glob('*.soil'))
        if not soil_files:
            return

        soil_path = soil_files[0]
        lines = soil_path.read_text().strip().split('\n')

        if len(lines) < 3:
            return

        # Line 0: NUMSOIL header, Line 1: column header, Line 2+: data/keywords
        new_lines = [lines[0], lines[1]]  # Keep NUMSOIL and column header

        # Parse column header to find column indices
        col_names = lines[1].split()
        col_idx = {name: i for i, name in enumerate(col_names)}

        # Map calibration params to .soil column names
        param_to_col = {
            'K_SAT': ['KINF', 'KSATV', 'KSATH'],
            'POROSITY': ['MAXSMC'],
            'VG_ALPHA': ['ALPHA'],
            'VG_N': ['BETA'],
            'MACROPORE_K': ['MACHF', 'MACVF'],
        }

        for line in lines[2:]:
            parts = line.split()
            if not parts:
                new_lines.append(line)
                continue

            # Keyword lines: DINF, KMACV_RO, KMACH_RO
            if parts[0] in ('DINF', 'KMACV_RO', 'KMACH_RO'):
                new_lines.append(line)
                continue

            # Data row — try to parse first field as integer (soil index)
            try:
                int(parts[0])
            except ValueError:
                new_lines.append(line)
                continue

            # Update data columns
            values = list(parts)
            for param_name, col_list in param_to_col.items():
                if param_name not in params:
                    continue
                val = params[param_name]
                for col_name in col_list:
                    if col_name in col_idx:
                        idx = col_idx[col_name]
                        if idx < len(values):
                            values[idx] = f"{val:.6e}"

            new_lines.append('\t'.join(values))

        soil_path.write_text('\n'.join(new_lines) + '\n')

    def _update_calib_file(
        self,
        subsurface_params: Dict[str, float],
        lsm_params: Optional[Dict[str, float]] = None,
        scenario_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update .calib file with section-aware parameter handling.

        The .calib file has three sections:
        - SUBSURFACE (implicit, before any section header): multipliers reset
          to 1.0 because absolute values are written to .soil/.geol/.riv/.mesh.
        - LSM_CALIBRATION: multipliers applied to base values from vegprmt.tbl
          lookup table (not from any file we write), so calibrated values are
          written directly.
        - SCENARIO: PRCP as multiplier, SFCTMP as additive offset (K).
        """
        calib_files = list(self.settings_dir.glob('*.calib'))
        if not calib_files:
            return

        lsm_params = lsm_params or {}
        scenario_params = scenario_params or {}

        # Map each subsurface .calib keyword to the calibration parameter
        calib_key_to_param = {
            'KSATH': 'K_SAT',
            'KSATV': 'K_SAT',
            'KINF': 'K_SAT',
            'KMACSATH': 'MACROPORE_K',
            'KMACSATV': 'MACROPORE_K',
            'POROSITY': 'POROSITY',
            'ALPHA': 'VG_ALPHA',
            'BETA': 'VG_N',
            'ROUGH': 'MANNINGS_N',
            'DROOT': 'SOIL_DEPTH',
        }

        calib_path = calib_files[0]
        lines = calib_path.read_text().strip().split('\n')
        new_lines = []
        current_section = 'SUBSURFACE'

        for line in lines:
            stripped = line.strip()

            # Detect section transitions
            if stripped == 'LSM_CALIBRATION':
                current_section = 'LSM_CALIBRATION'
                new_lines.append(line)
                continue
            elif stripped == 'SCENARIO':
                current_section = 'SCENARIO'
                new_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) < 2:
                new_lines.append(line)
                continue

            key = parts[0]

            if current_section == 'SUBSURFACE':
                param_name = calib_key_to_param.get(key)
                if param_name and param_name in subsurface_params:
                    # Reset to 1.0 — absolute values are already in the
                    # direct input files (.soil, .riv, .mesh, .geol).
                    line = f"{key}\t\t1.000000"

            elif current_section == 'LSM_CALIBRATION':
                if key in lsm_params:
                    line = f"{key}\t\t{lsm_params[key]:.6f}"

            elif current_section == 'SCENARIO':
                if key in scenario_params:
                    if key == 'SFCTMP':
                        # Additive offset (K)
                        line = f"{key}\t\t{scenario_params[key]:.6f}"
                    else:
                        # Multiplier (e.g. PRCP)
                        line = f"{key}\t\t{scenario_params[key]:.6f}"

            new_lines.append(line)

        calib_path.write_text('\n'.join(new_lines) + '\n')

    def _update_lc_file(self, mannings_n: float) -> None:
        """Update .lc file with new Manning's n."""
        lc_files = list(self.settings_dir.glob('*.lc'))
        if not lc_files:
            return

        lc_path = lc_files[0]
        lines = lc_path.read_text().strip().split('\n')
        new_lines = [lines[0]]  # header

        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                new_lines.append(f"{parts[0]} {parts[1]} {mannings_n:.4f} {parts[3] if len(parts) > 3 else '0.0'}")
            else:
                new_lines.append(line)

        lc_path.write_text('\n'.join(new_lines) + '\n')

    def _update_geol_file(self, params: Dict[str, float]) -> None:
        """Update .geol file with deep-subsurface K.

        Uses GEOL_K_RATIO * K_SAT if available, otherwise K_SAT / 100.
        """
        geol_files = list(self.settings_dir.glob('*.geol'))
        if not geol_files:
            return

        k_sat = params.get('K_SAT')
        if k_sat is None:
            return

        geol_k_ratio = params.get('GEOL_K_RATIO', 0.01)
        geol_k = k_sat * geol_k_ratio
        geol_path = geol_files[0]
        lines = geol_path.read_text().strip().split('\n')
        new_lines = []

        for line in lines:
            parts = line.split()
            if not parts:
                new_lines.append(line)
                continue
            # Data rows start with an integer index
            try:
                int(parts[0])
            except ValueError:
                new_lines.append(line)
                continue
            # Replace KSATV (col 1) and KSATH (col 2) in data row
            if len(parts) >= 3:
                parts[1] = f"{geol_k:.6e}"
                parts[2] = f"{geol_k:.6e}"
            new_lines.append('\t'.join(parts))

        geol_path.write_text('\n'.join(new_lines) + '\n')

    def _update_riv_file(self, params: Dict[str, float]) -> None:
        """Update .riv file with Manning's roughness and riverbed K.

        MM-PIHM .riv MATERIAL section format:
            INDEX  ROUGH  CWR  KH
            #-     s/m1/3  -   m/s
            1      <n>    0.6  <k>
        """
        riv_files = list(self.settings_dir.glob('*.riv'))
        if not riv_files:
            return

        riv_path = riv_files[0]
        lines = riv_path.read_text().strip().split('\n')
        new_lines = []
        in_material = False

        for line in lines:
            parts = line.split()
            if not parts:
                new_lines.append(line)
                continue

            # Track which section we're in
            if parts[0] == 'MATERIAL':
                in_material = True
                new_lines.append(line)
                continue
            elif parts[0] in ('BC', 'RES', 'SHAPE', 'NUMRIV'):
                in_material = False
                new_lines.append(line)
                continue

            # In MATERIAL section, update data rows (start with integer index)
            if in_material:
                try:
                    int(parts[0])
                except ValueError:
                    new_lines.append(line)
                    continue
                # MATERIAL data row: INDEX ROUGH CWR KH
                if len(parts) >= 4:
                    if 'MANNINGS_N' in params:
                        parts[1] = f"{params['MANNINGS_N']:.4f}"
                    if 'K_SAT' in params:
                        parts[3] = f"{params['K_SAT']:.3E}"
                    new_lines.append('\t'.join(parts))
                    continue

            new_lines.append(line)

        riv_path.write_text('\n'.join(new_lines) + '\n')

    def _update_mesh_file(self, soil_depth: float) -> None:
        """Update .mesh file ZMIN values when SOIL_DEPTH changes.

        ZMIN = ZMAX - soil_depth for each node.
        """
        mesh_files = list(self.settings_dir.glob('*.mesh'))
        if not mesh_files:
            return

        mesh_path = mesh_files[0]
        lines = mesh_path.read_text().strip().split('\n')
        new_lines = []
        in_nodes = False

        for line in lines:
            parts = line.split()
            if not parts:
                new_lines.append(line)
                continue

            if parts[0] == 'NUMNODE':
                in_nodes = True
                new_lines.append(line)
                continue
            elif parts[0] in ('NUMELE', 'INDEX') and not in_nodes:
                new_lines.append(line)
                continue

            if in_nodes and parts[0] == 'INDEX':
                new_lines.append(line)
                continue

            # Node data rows: INDEX X Y ZMIN ZMAX
            if in_nodes and len(parts) >= 5:
                try:
                    int(parts[0])
                    zmax = float(parts[4])
                    parts[3] = f"{zmax - soil_depth:.1f}"
                    new_lines.append('\t'.join(parts))
                    continue
                except (ValueError, IndexError):
                    pass

            new_lines.append(line)

        mesh_path.write_text('\n'.join(new_lines) + '\n')

    def _update_ic_file(self, params: Dict[str, float]) -> None:
        """Update binary .ic file when SOIL_DEPTH or POROSITY change.

        Flux-PIHM IC format:
            Per element: cmc, sneqv, surf, unsat, gw, t1, snowh,
                        stc[11], smc[11], swc[11]  = 40 doubles
            Per river segment: stage = 1 double
            Total: NUMELE*40 + NUMRIV doubles

        The element/river counts are read from the .mesh/.riv files so this works
        for both the lumped (2 elem + 1 river) and distributed (TIN) meshes,
        instead of assuming a fixed 2-element layout.
        """
        import struct

        ic_files = list(self.settings_dir.glob('*.ic'))
        if not ic_files:
            return

        soil_depth = params.get('SOIL_DEPTH', 2.0)
        porosity = params.get('POROSITY', 0.4)
        init_gw_depth = params.get('INIT_GW_DEPTH', soil_depth * 0.5)

        # Ensure GW depth doesn't exceed soil depth
        gw = min(init_gw_depth, soil_depth * 0.95)
        init_satn = 0.5
        deficit = soil_depth - gw
        unsat = init_satn * deficit

        MAXLYR = 11
        min_smc = 0.05
        init_smc = min_smc + init_satn * (porosity - min_smc)
        init_temp = 277.15

        def pack_elem():
            d_ = struct.pack('ddddd', 0.0, 0.0, 0.0, unsat, gw)
            d_ += struct.pack('dd', init_temp, 0.0)
            d_ += struct.pack(f'{MAXLYR}d', *([init_temp] * MAXLYR))
            d_ += struct.pack(f'{MAXLYR}d', *([init_smc] * MAXLYR))
            d_ += struct.pack(f'{MAXLYR}d', *([init_smc] * MAXLYR))
            return d_

        # Size the .ic to the actual mesh (NUMELE elements + NUMRIV river stages),
        # not a hardcoded 2-element lumped layout, so distributed meshes work.
        def _count(glob_pat: str, keyword: str, default: int) -> int:
            files = list(self.settings_dir.glob(glob_pat))
            if not files:
                return default
            with open(files[0]) as fh:
                first = fh.readline().split()
            if len(first) >= 2 and first[0].upper().startswith(keyword):
                try:
                    return int(first[1])
                except ValueError:
                    return default
            return default

        n_ele = _count('*.mesh', 'NUMELE', 2)
        n_riv = _count('*.riv', 'NUMRIV', 1)

        data = pack_elem() * n_ele
        data += struct.pack('d', 0.1) * n_riv  # one stage per river segment
        ic_files[0].write_bytes(data)

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from defaults."""
        defaults = {
            'K_SAT': 1e-5,
            'POROSITY': 0.4,
            'VG_ALPHA': 1.0,
            'VG_N': 2.0,
            'MACROPORE_K': 1e-4,
            'MANNINGS_N': 0.03,
            'SOIL_DEPTH': 2.0,
            'SNOW17_SCF': 1.0,
            'SNOW17_MFMAX': 1.0,
            'SNOW17_PXTEMP': 0.0,
            'SFCTMP': 0.0,
            'PRCP': 1.0,
            'CZIL': 1.0,
            'RS': 1.0,
        }
        return {k: v for k, v in defaults.items() if k in self.pihm_params}
