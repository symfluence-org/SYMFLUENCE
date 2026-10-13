#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
MESH Parameter Manager

Handles MESH parameter bounds, normalization, and .ini file updates.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.parameters.parameter_bounds_registry import get_mesh_bounds
from symfluence.core.logging_utils import log_once
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.registries import R

from .glacier_parameters import ICE_CLASS, ICE_HYDRO, LAND_HYDRO, ice_mask, update_gru_arrays

logger = logging.getLogger(__name__)

@R.parameter_managers.add('MESH')
class MESHParameterManager(BaseParameterManager):
    """Handles MESH parameter bounds, normalization, and file updates"""

    # Marker comments used by meshflow to identify parameter lines in CLASS.ini
    # Maps: marker substring -> (expected param names at each position, total_value_count)
    # Only params listed in param_file_map are detected (others are skipped).
    # NOTE: vegetation params are NOT positioned here — see VEG_PARAM_LAYOUT.
    CLASS_LINE_MARKERS = {
        'DRN/SDEP': (['DRN', 'SDEP', None, 'DD'], 4),
        'XSLP/XDRAINH/MANN/KSAT': (['XSLP', 'XDRAINH', 'MANN_CLASS', 'KSAT'], 5),
        'FCAN': ([None] * 9, 9),
        '5xLNZ0/4xLAMN': ([None] * 9, 9),
        '5xALVC/4xCMAS': ([None] * 9, 9),
        'ALIC': ([None] * 9, 9),
        'RSMN': ([None] * 8, 8),
        '4xVPDA/4xVPDB': ([None] * 8, 8),
        '4xPSGA/4xPSGB': ([None] * 8, 8),
    }

    # CLASS carries one value per canopy category (needleleaf, broadleaf, crops,
    # grass) in each vegetation parameter group. Only the category whose FCAN is
    # non-zero is active for a GRU; a value written to any other column is inert
    # (it silently does nothing) and, when it activates an otherwise-absent
    # category, can drive CLASS into a snow-energy-balance crash.
    #
    # The column is therefore NOT fixed: it is ``group_start + active_canopy_index``,
    # where the active index comes from the block's FCAN line. This mirrors how
    # CLASSFileManager.apply_field_overrides targets the active column.
    #
    # Map: param -> (line marker, group start position, values on line)
    VEG_PARAM_LAYOUT = {
        'LAMX':  ('5xFCAN/4xLAMX', 5, 9),
        'LAMN':  ('5xLNZ0/4xLAMN', 5, 9),
        'CMAS':  ('5xALVC/4xCMAS', 5, 9),
        'ROOT':  ('5xALIC/4xROOT', 5, 9),
        'RSMIN': ('4xRSMN/4xQA50', 0, 8),
        'QA50':  ('4xRSMN/4xQA50', 4, 8),
        'VPDA':  ('4xVPDA/4xVPDB', 0, 8),
        'VPDB':  ('4xVPDA/4xVPDB', 4, 8),
        'PSGA':  ('4xPSGA/4xPSGB', 0, 8),
    }

    # Line offsets of each vegetation parameter relative to the block's
    # DRN/SDEP line (CLASS line 12), used to locate the line in every GRU block.
    VEG_LINE_OFFSET_FROM_DRN = {
        'LAMX': -7, 'LAMN': -6, 'CMAS': -5, 'ROOT': -4,
        'RSMIN': -3, 'QA50': -3, 'VPDA': -2, 'VPDB': -2, 'PSGA': -1,
    }

    # Number of canopy categories in a CLASS 4x vegetation group.
    N_CANOPY_CATEGORIES = 4

    # Default multipliers for landcover-specific parameter scaling
    # These allow differentiated soil/surface parameters by landcover type
    # Keys are NALCMS/IGBP class IDs, values are multiplier dictionaries
    DEFAULT_LANDCOVER_MULTIPLIERS = {
        1: {'KSAT': 0.8, 'SDEP': 1.2, 'MANN_CLASS': 1.5},   # Needleleaf forest: slower infiltration, deeper soil, rougher
        8: {'KSAT': 1.0, 'SDEP': 1.0, 'MANN_CLASS': 1.2},   # Shrubland: moderate
        9: {'KSAT': 0.9, 'SDEP': 1.1, 'MANN_CLASS': 1.3},   # Broadleaf deciduous: similar to needleleaf
        10: {'KSAT': 1.2, 'SDEP': 0.9, 'MANN_CLASS': 0.8},  # Grassland: faster infiltration, shallower, smoother
        15: {'KSAT': 0.5, 'SDEP': 0.3, 'MANN_CLASS': 0.5},  # Snow/Ice: very low infiltration, minimal soil
        16: {'KSAT': 1.5, 'SDEP': 0.5, 'MANN_CLASS': 0.6},  # Barren: fast infiltration (fractured rock), shallow soil
    }

    def __init__(self, config: Dict, logger: logging.Logger, mesh_settings_dir: Path):
        super().__init__(config, logger, mesh_settings_dir)

        # MESH-specific setup
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')
        self.project_dir = self._resolve_project_dir()

        # Parse MESH parameters to calibrate from config
        mesh_params_str = self._get_config_value(lambda: self.config.model.mesh.params_to_calibrate, default=None, dict_key='MESH_PARAMS_TO_CALIBRATE')
        if mesh_params_str is None:
            # Default: 10 high-impact parameters for DDS convergence.
            # CLASS soil/surface: KSAT (infiltration), DRN (drainage),
            #   SDEP (soil depth), XSLP (slope), XDRAINH (horiz. drainage),
            #   MANN_CLASS (roughness)
            # CLASS vegetation (ET partitioning): RSMIN (stomatal resistance),
            #   LAMX (max LAI — canopy interception & transpiration)
            # Hydrology: FLZ/PWR (baseflow recession)
            #
            # Excluded from default:
            #   ZSNL — snow depth for full cover; opt in for snow observations
            #   ZPLS — ponding depth over snow
            #   RCHARG — groundwater recharge; can cause water balance issues
            #             without paired DRAINFRAC
            #   WF_R2/R2N — routing params, inert in noroute mode (auto-removed
            #               below even if user adds them explicitly)
            #   FRZTH — inert when FROZENSOILINFILFLAG=0 (the default)
            mesh_params_str = 'KSAT,DRN,SDEP,XSLP,XDRAINH,MANN_CLASS,FLZ,PWR,RSMIN,LAMX'

        self.mesh_params = [p.strip() for p in str(mesh_params_str).split(',') if p.strip()]

        # Auto-remove routing parameters that are inert in noroute mode
        routing_params_in_set = [p for p in self.mesh_params if p in ('WF_R2', 'R2N')]
        if routing_params_in_set:
            routing_mode = self._get_config_value(lambda: None, default='noroute', dict_key='ROUTING_MODE')
            if routing_mode == 'noroute':
                self.mesh_params = [p for p in self.mesh_params if p not in ('WF_R2', 'R2N')]
                self.logger.info(
                    f"Removed routing parameters {routing_params_in_set} from "
                    f"calibration set — ROUTING_MODE='noroute' makes them "
                    f"inert (zero gradient). Calibrating {len(self.mesh_params)} "
                    f"parameters: {self.mesh_params}"
                )

        # Paths to parameter files
        # mesh_settings_dir is the base directory for model files in this run context
        self.mesh_settings_dir = mesh_settings_dir

        # MESH parameter files (usually in forcing directory, but mirrored in settings for workers)
        self.class_params_file = self.mesh_settings_dir / 'MESH_parameters_CLASS.ini'
        self.hydro_params_file = self.mesh_settings_dir / 'MESH_parameters_hydrology.ini'
        self.routing_params_file = self.mesh_settings_dir / 'MESH_parameters.txt'

        # Map parameters to files
        # NOTE: meshflow creates MESH_parameters_hydrology.ini with ZSNL, ZPLS, ZPLG, WF_R2
        # CLASS.ini has fixed-format parameters that control runoff generation
        self.param_file_map = {
            **{key: 'CLASS' for key in ICE_CLASS},
            **{key: 'hydrology' for key in ICE_HYDRO},
            # CLASS.ini parameters (fixed-format, control runoff)
            'KSAT': 'CLASS',      # Saturated hydraulic conductivity (mm/hr) - Line 13, pos 4
            'DRN': 'CLASS',       # Drainage parameter - Line 12, pos 1
            'SDEP': 'CLASS',      # Soil depth (m) - Line 12, pos 2
            'DD': 'CLASS',        # Drainage density - Line 12, pos 4
            'XSLP': 'CLASS',      # Slope - Line 13, pos 1
            'XDRAINH': 'CLASS',   # Horizontal drainage - Line 13, pos 2
            'MANN_CLASS': 'CLASS', # Manning for overland flow - Line 13, pos 3
            # CLASS.ini vegetation parameters (control ET partitioning)
            'LAMX': 'CLASS',      # Max LAI for first veg class - Line 05, pos 5
            'LAMN': 'CLASS',      # Min LAI for first veg class - Line 06, pos 5
            'ROOT': 'CLASS',      # Root depth (m) for first veg class - Line 08, pos 5
            'CMAS': 'CLASS',      # Canopy mass (kg/m²) - Line 07, pos 5
            'RSMIN': 'CLASS',     # Minimum stomatal resistance (s/m) - Line 09, pos 0
            'QA50': 'CLASS',      # VPD half-response (Pa) - Line 09, pos 4
            'VPDA': 'CLASS',      # VPD slope param - Line 10, pos 0
            'PSGA': 'CLASS',      # Soil moisture stress param A - Line 11, pos 0
            # Meshflow-generated parameters (MESH_parameters_hydrology.ini)
            'ZSNL': 'hydrology', 'ZPLG': 'hydrology', 'ZPLS': 'hydrology',
            'WF_R2': 'hydrology',  # Channel roughness coefficient
            'R2N': 'hydrology',    # Overland routing roughness (Manning's n)
            'R1N': 'hydrology',    # River routing parameter
            'FLZ': 'hydrology',    # Baseflow recession coefficient (critical for winter flow)
            'PWR': 'hydrology',    # Power coefficient for LZS drainage
            # Legacy parameters (may not exist in meshflow-generated files)
            # FRZTH is a soil parameter written to hydrology.ini via key-value format.
            # FREZTH is a distinct glacier pond-refreezing temperature (degrees C);
            # FRZTH here is the frozen soil infiltration *threshold* depth (m).
            'FRZTH': 'hydrology',
            'MANN': 'hydrology',
            'RCHARG': 'hydrology', 'DRAINFRAC': 'hydrology', 'BASEFLW': 'hydrology',
            'DTMINUSR': 'routing',  # In main MESH_parameters.txt
        }

        # CLASS.ini line mappings (0-indexed line number, 0-indexed position in line)
        # Format: param_name -> (line_index, value_position, num_values_in_line)
        # These are detected dynamically via _detect_class_param_positions()
        # and fall back to standard meshflow layout if detection fails.
        self.class_param_positions = self._detect_class_param_positions()

        # Multi-GRU parameter distribution settings
        # MESH_APPLY_PARAMS_ALL_GRUS: Apply calibrated params to all landcover classes (default: True)
        # MESH_USE_LANDCOVER_MULTIPLIERS: Apply landcover-specific multipliers (default: True)
        self.apply_to_all_grus = self._get_config_value(lambda: None, default=True, dict_key='MESH_APPLY_PARAMS_ALL_GRUS')
        self.use_landcover_multipliers = self._get_config_value(lambda: None, default=True, dict_key='MESH_USE_LANDCOVER_MULTIPLIERS')

        # Load custom multipliers from config or use defaults
        self.landcover_multipliers = self._get_config_value(
            lambda: None, default=self.DEFAULT_LANDCOVER_MULTIPLIERS, dict_key='MESH_LANDCOVER_MULTIPLIERS'
        )

        # Detect all GRU blocks in CLASS.ini for multi-GRU parameter distribution
        self.all_gru_blocks = self._detect_all_gru_blocks()

        # Build mapping from block index to NALCMS class ID for multipliers
        # This reads the landcover stats CSV to determine which NALCMS classes
        # correspond to which CLASS.ini blocks (blocks are ordered by frac_* column order)
        self.block_to_nalcms = (
            self._build_block_to_nalcms_mapping() if self.use_landcover_multipliers else {}
        )

    def _get_parameter_names(self) -> List[str]:
        """Return MESH parameter names from config."""
        return self.mesh_params

    def _detect_class_param_positions(self) -> Dict[str, tuple]:
        """Dynamically detect parameter positions in CLASS.ini by scanning for
        marker comments like ``DRN/SDEP`` and ``XSLP/XDRAINH/MANN/KSAT``.

        Falls back to the standard meshflow layout (lines 12-13, 0-indexed)
        when the file is missing or markers are not found.

        Returns:
            Dict mapping parameter name to (line_index, col_position, n_values).
        """
        # Standard meshflow defaults (0-indexed lines). Vegetation columns default
        # to the grass slot (canopy index 3), which is what meshflow emits for a
        # single-GRU lumped domain; the real index is detected from FCAN below.
        grass = self.N_CANOPY_CATEGORIES - 1
        defaults = {
            'LAMX': (5, 5 + grass, 9),
            'LAMN': (6, 5 + grass, 9),
            'CMAS': (7, 5 + grass, 9),
            'ROOT': (8, 5 + grass, 9),
            'RSMIN': (9, 0 + grass, 8),
            'QA50': (9, 4 + grass, 8),
            'VPDA': (10, 0 + grass, 8),
            'VPDB': (10, 4 + grass, 8),
            'PSGA': (11, 0 + grass, 8),
            'DRN': (12, 0, 4),
            'SDEP': (12, 1, 4),
            'DD': (12, 3, 4),
            'XSLP': (13, 0, 5),
            'XDRAINH': (13, 1, 5),
            'MANN_CLASS': (13, 2, 5),
            'KSAT': (13, 3, 5),
        }

        class_file = self._resolve_ini_file('CLASS')
        if class_file is None:
            return defaults

        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except OSError:
            return defaults

        detected: Dict[str, tuple] = {}

        # --- Non-vegetation (fixed-column) params -------------------------
        for line_idx, raw_line in enumerate(lines):
            for marker, (param_names, n_values) in self.CLASS_LINE_MARKERS.items():
                if marker not in raw_line:
                    continue
                # meshflow puts the comment on the same line as the values, e.g.
                #   "   1.000   2.500   1.000  50.000  12 DRN/SDEP/FARE/DD"
                # but some formats place the comment on a header line above.
                data_line_idx = line_idx
                numeric_count = 0
                for tok in raw_line.split():
                    try:
                        float(tok)
                        numeric_count += 1
                    except ValueError:
                        break
                if numeric_count < n_values and line_idx + 1 < len(lines):
                    data_line_idx = line_idx + 1

                for pos, pname in enumerate(param_names):  # type: ignore[var-annotated]
                    if pname in self.param_file_map:
                        detected[pname] = (data_line_idx, pos, n_values)

        # --- Vegetation params: column depends on the active canopy -------
        fcan_line = self._find_marker_line(lines, self.VEG_PARAM_LAYOUT['LAMX'][0])
        active = self._active_veg_index(lines, fcan_line)
        for pname, (marker, group_start, n_values) in self.VEG_PARAM_LAYOUT.items():
            if pname not in self.param_file_map:
                continue
            line_idx = self._find_marker_line(lines, marker)
            if line_idx is None:
                continue
            detected[pname] = (line_idx, group_start + active, n_values)

        if detected:
            self.logger.debug(
                f"Detected CLASS param positions (active canopy index {active}): "
                f"{', '.join(f'{k}=line{v[0]+1}:pos{v[1]}' for k, v in detected.items())}"
            )
            # Merge with defaults so that any params NOT matched still have
            # fallback positions (useful when a marker is missing)
            merged = dict(defaults)
            merged.update(detected)
            return merged

        self.logger.debug("No CLASS markers found; using default line positions")
        return defaults

    @staticmethod
    def _find_marker_line(lines: List[str], marker: str) -> Optional[int]:
        """Index of the first line carrying ``marker``, or None."""
        for i, line in enumerate(lines):
            if marker in line:
                return i
        return None

    @classmethod
    def _active_veg_index(
        cls, lines: List[str], fcan_line_idx: Optional[int]
    ) -> int:
        """Return the active canopy-category index (0-3) for a GRU block.

        CLASS vegetation groups carry one value per canopy category
        (needleleaf, broadleaf, crops, grass). The active category is the one
        with a non-zero FCAN on CLASS line 05; writing a calibrated value to
        any other column is inert. Defaults to grass when FCAN is unreadable
        or all-zero, matching meshflow's single-GRU lumped output.
        """
        grass = cls.N_CANOPY_CATEGORIES - 1
        if fcan_line_idx is None or not (0 <= fcan_line_idx < len(lines)):
            return grass

        parts = lines[fcan_line_idx].split()
        fcan = []
        for i in range(cls.N_CANOPY_CATEGORIES):
            try:
                fcan.append(float(parts[i]))
            except (IndexError, ValueError):
                fcan.append(0.0)

        if max(fcan) > 0.0:
            return max(range(cls.N_CANOPY_CATEGORIES), key=lambda i: fcan[i])
        return grass

    def _detect_all_gru_blocks(self) -> List[Dict[str, Any]]:
        """Detect all GRU/landcover parameter blocks in CLASS.ini.

        Each landcover class has its own parameter block (lines 05-19 repeated).
        This method finds all blocks and extracts their class IDs.

        Returns:
            List of dicts with 'class_id', 'drn_line', 'xslp_line' for each block.
        """
        class_file = self._resolve_ini_file('CLASS')
        if class_file is None:
            return []

        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except OSError:
            return []

        blocks: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for DRN/SDEP marker (line 12 of each block)
            if 'DRN/SDEP' in line:
                drn_line = i
                # XSLP/XDRAINH line should be next (line 13)
                xslp_line = i + 1 if i + 1 < len(lines) else None

                # Try to extract class ID from the XSLP line comment (e.g., "100 Temp_sub-_need_fore")
                class_id = None
                if xslp_line and xslp_line < len(lines):
                    xslp_content = lines[xslp_line]
                    # Look for 3-digit class ID (e.g., 100, 101, 102 for NALCMS mapping)
                    import re
                    match = re.search(r'\s(\d{3})\s+\w', xslp_content)
                    if match:
                        mesh_class_id = int(match.group(1))
                        # MESH class IDs are typically 100 + NALCMS class
                        class_id = mesh_class_id - 100 if mesh_class_id >= 100 else mesh_class_id

                blocks.append({
                    'class_id': class_id,
                    'drn_line': drn_line,
                    'xslp_line': xslp_line,
                    'block_index': len(blocks)
                })
            i += 1

        if blocks:
            self.logger.debug(
                f"Detected {len(blocks)} GRU blocks in CLASS.ini: "
                f"{[b['class_id'] for b in blocks]}"
            )
        return blocks

    def _detect_all_gru_blocks_from_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect all GRU blocks from a list of lines (for use during file update).

        Similar to _detect_all_gru_blocks but operates on already-loaded lines.
        """
        blocks: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'DRN/SDEP' in line:
                drn_line = i
                xslp_line = i + 1 if i + 1 < len(lines) else None

                class_id = None
                if xslp_line and xslp_line < len(lines):
                    xslp_content = lines[xslp_line]
                    match = re.search(r'\s(\d{3})\s+\w', xslp_content)
                    if match:
                        mesh_class_id = int(match.group(1))
                        class_id = mesh_class_id - 100 if mesh_class_id >= 100 else mesh_class_id

                blocks.append({
                    'class_id': class_id,
                    'drn_line': drn_line,
                    'xslp_line': xslp_line,
                    'block_index': len(blocks)
                })
            i += 1
        return blocks

    def _resolve_project_dir(self) -> Optional[Path]:
        """Resolve the real project directory from config.

        In parallel runs, mesh_settings_dir may point into process_N/forcing,
        which does not contain attributes/gistool-outputs. Use the config's
        SYMFLUENCE_DATA_DIR + DOMAIN_NAME to locate the true project dir.
        """
        domain_name = self.domain_name
        if domain_name is None:
            domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')

        data_dir = self._get_config_value(lambda: self.config.system.data_dir, default=None, dict_key='SYMFLUENCE_DATA_DIR')

        if domain_name and data_dir:
            project_dir = Path(data_dir) / f"domain_{domain_name}"
            if project_dir.exists():
                return project_dir

        return None

    def _build_block_to_nalcms_mapping(self) -> Dict[int, int]:
        """Build mapping from CLASS.ini block index to NALCMS class ID.

        Reads the landcover stats CSV to determine which NALCMS classes are present
        and their order (which corresponds to CLASS.ini block order).

        Returns:
            Dict mapping block index (0, 1, 2, ...) to NALCMS class ID (1, 8, 9, ...)
        """
        import pandas as pd

        # Look for landcover stats in various locations
        project_dir = self.project_dir or self.mesh_settings_dir.parent.parent
        possible_paths = [
            resolve_data_subdir(project_dir, 'forcing') / 'MESH_input' / 'temp_modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv',
            resolve_data_subdir(project_dir, 'attributes') / 'gistool-outputs' / 'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv',
            resolve_data_subdir(project_dir, 'attributes') / 'gistool-outputs' / 'landcover_stats_multi_gru.csv',
        ]

        for lc_path in possible_paths:
            if lc_path.exists():
                try:
                    df = pd.read_csv(lc_path)
                    # Extract class IDs from frac_* columns (preserving order)
                    frac_cols = [col for col in df.columns if col.startswith('frac_')]
                    class_ids = []
                    for col in frac_cols:
                        match = re.match(r'frac_(\d+)', col)
                        if match:
                            class_ids.append(int(match.group(1)))

                    if class_ids:
                        mapping = {i: class_id for i, class_id in enumerate(class_ids)}
                        self.logger.debug(f"Block to NALCMS mapping: {mapping}")
                        return mapping
                except Exception as e:  # noqa: BLE001 — calibration resilience
                    self.logger.warning(f"Failed to read landcover mapping from {lc_path}: {e}", exc_info=True)

        # Fallback: assume sequential mapping
        self.logger.warning("Could not determine NALCMS class mapping; multipliers may not work correctly")
        return {}

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return MESH parameter bounds, preferring config-specified bounds.

        Config-specified bounds override min/max only, preserving
        transform metadata (e.g. ``'log'``) from the registry so that
        DDS normalization works correctly in log-space for parameters
        spanning orders of magnitude.
        """
        # Start with registry defaults
        bounds = get_mesh_bounds()

        # Override with config-specified bounds if available
        config_bounds = self._get_config_value(lambda: self.config.model.mesh.param_bounds, default={}, dict_key='MESH_PARAM_BOUNDS')
        if config_bounds:
            self.logger.debug(f"Using config-specified MESH bounds for: {list(config_bounds.keys())}")
            self._apply_config_bounds_override(bounds, config_bounds)

        for name in getattr(self, 'mesh_params', []):
            if (name in ICE_CLASS or name in ICE_HYDRO) and name not in bounds:
                raise ValueError(f'{name} requires explicit MESH_PARAM_BOUNDS; no glacier bounds are assumed')

        return bounds

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        """
        Denormalize parameters and enforce MESH-specific feasibility constraints.

        Overrides base implementation to apply physical constraints that prevent
        CLASS snow energy balance crashes.
        """
        params = super().denormalize_parameters(normalized_array)
        params = self._validate_parameter_feasibility(params)
        return params

    def _validate_parameter_feasibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce physical feasibility constraints between MESH parameters.

        These constraints prevent CLASS energy balance crashes caused by
        physically impossible or numerically unstable parameter combinations.

        Constraints enforced:
        1. KSAT * DRN <= 400 mm/hr (cap unrealistic drainage rate)
        2. KSAT * DRN * (1/SDEP) <= 250 (prevent instant soil emptying)
        3. XSLP >= 0.005 (prevent CLASS numerical issues)
        4. SDEP >= 0.5m always (shallow soils + high KSAT cause crashes)
        5. XDRAINH <= 0.5 when KSAT > 100 (cap combined drainage)
        6. FRZTH <= SDEP (frozen threshold can't exceed soil depth)
        7. MANN_CLASS >= 0.01 (prevent zero-friction overland flow instability)
        8. RCHARG + DRAINFRAC <= 1.0 (water balance closure)
        9. ROOT <= SDEP (root depth can't exceed soil depth)
        10. LAMX >= 0.1 (CLASS needs minimum positive LAI)
        """
        validated = params.copy()

        ksat = validated.get('KSAT')
        drn = validated.get('DRN')
        sdep = validated.get('SDEP')
        xslp = validated.get('XSLP')
        xdrainh = validated.get('XDRAINH')
        frzth = validated.get('FRZTH')
        mann_class = validated.get('MANN_CLASS')
        rcharg = validated.get('RCHARG')
        drainfrac = validated.get('DRAINFRAC')

        # Constraint 4: SDEP >= 0.5m (applied early so downstream constraints use it)
        # Note: 1.5m was too restrictive for alpine basins with thin soils.
        # CLASS is stable with SDEP >= 0.5m when KSAT*DRN constraints are enforced.
        # Allowing shallower soil is critical for mountain basins where deep soil
        # causes excessive ET (too much root-zone water available for evaporation).
        if sdep is not None and float(sdep) < 0.5:
            self.logger.debug(
                f"Feasibility: raised SDEP {float(sdep):.2f} -> 0.5m "
                f"(minimum for CLASS stability)"
            )
            validated['SDEP'] = 0.5
            sdep = 0.5

        # Constraint 1: KSAT * DRN <= 400
        # Moderately tightened from 500 to reduce crashes without
        # over-constraining DRN (which controls runoff generation)
        if ksat is not None and drn is not None:
            product = float(ksat) * float(drn)
            if product > 400.0:
                scale = 400.0 / product
                validated['DRN'] = float(drn) * scale
                self.logger.debug(
                    f"Feasibility: scaled DRN {float(drn):.3f} -> {validated['DRN']:.3f} "
                    f"(KSAT*DRN={product:.1f} > 400)"
                )
                drn = validated['DRN']

        # Constraint 2: KSAT * DRN * (1/SDEP) <= 250
        # Moderately tightened from 300 to prevent timestep-scale soil
        # depletion while preserving sufficient drainage search space
        if ksat is not None and drn is not None and sdep is not None:
            drain_intensity = float(ksat) * float(drn) / float(sdep)
            if drain_intensity > 250.0:
                scale = 250.0 / drain_intensity
                validated['DRN'] = float(drn) * scale
                self.logger.debug(
                    f"Feasibility: scaled DRN {float(drn):.3f} -> {validated['DRN']:.3f} "
                    f"(KSAT*DRN/SDEP={drain_intensity:.1f} > 250)"
                )
                drn = validated['DRN']

        # Constraint 2b: DRN * XDRAINH / SDEP <= 1.5 (lateral-drainage intensity).
        # The KSAT*DRN constraints above only bound the *vertical* drainage path,
        # so they miss the interflow instability: a large DRN with high horizontal
        # drainage (XDRAINH) draining a shallow soil (small SDEP) lets the column
        # saturate and dump an entire high-elevation snowmelt pulse as lateral flow
        # in a single day (observed: DRN~10, XDRAINH~0.7, SDEP~0.8 -> LATFLW
        # 60 mm/day -> ~10x the observed peak, blowing up on melt-heavy years while
        # scoring fine on milder calibration years). Scale DRN down to keep the
        # interflow path in the physically stable regime.
        if drn is not None and xdrainh is not None and sdep is not None:
            lateral_intensity = float(drn) * float(xdrainh) / float(validated['SDEP'])
            if lateral_intensity > 1.5:
                scale = 1.5 / lateral_intensity
                validated['DRN'] = float(drn) * scale
                self.logger.debug(
                    f"Feasibility: scaled DRN {float(drn):.3f} -> {validated['DRN']:.3f} "
                    f"(DRN*XDRAINH/SDEP={lateral_intensity:.2f} > 1.5; "
                    f"prevents snowmelt interflow blow-up)"
                )
                drn = validated['DRN']

        # Constraint 3: XSLP >= 0.005
        if xslp is not None and float(xslp) < 0.005:
            validated['XSLP'] = 0.005
            self.logger.debug(
                f"Feasibility: raised XSLP {float(xslp):.4f} -> 0.005 "
                f"(minimum for CLASS stability)"
            )

        # Constraint 5: XDRAINH <= 0.5 when KSAT > 100
        if xdrainh is not None and ksat is not None:
            if float(ksat) > 100.0 and float(xdrainh) > 0.5:
                validated['XDRAINH'] = 0.5
                self.logger.debug(
                    f"Feasibility: capped XDRAINH {float(xdrainh):.3f} -> 0.5 "
                    f"(KSAT={float(ksat):.1f} > 100)"
                )

        # Constraint 6: FRZTH <= SDEP
        if frzth is not None and sdep is not None:
            if float(frzth) > float(validated['SDEP']):
                validated['FRZTH'] = float(validated['SDEP'])
                self.logger.debug(
                    f"Feasibility: capped FRZTH {float(frzth):.2f} -> {validated['FRZTH']:.2f} "
                    f"(must be <= SDEP)"
                )

        # Constraint 7: MANN_CLASS >= 0.01
        # Very low Manning's n causes overland flow velocity to blow up,
        # leading to numerical instability in the surface water balance
        if mann_class is not None and float(mann_class) < 0.01:
            validated['MANN_CLASS'] = 0.01
            self.logger.debug(
                f"Feasibility: raised MANN_CLASS {float(mann_class):.4f} -> 0.01 "
                f"(minimum for numerical stability)"
            )

        # Constraint 8: RCHARG + DRAINFRAC <= 1.0
        # These fractions partition soil drainage; their sum cannot exceed 1.0
        if rcharg is not None and drainfrac is not None:
            total = float(rcharg) + float(drainfrac)
            if total > 1.0:
                scale = 1.0 / total
                validated['RCHARG'] = float(rcharg) * scale
                validated['DRAINFRAC'] = float(drainfrac) * scale
                self.logger.debug(
                    f"Feasibility: scaled RCHARG+DRAINFRAC {total:.3f} -> 1.0"
                )

        # Constraint 9: ROOT <= SDEP (root depth can't exceed soil depth)
        root = validated.get('ROOT')
        if root is not None and sdep is not None:
            if float(root) > float(validated['SDEP']):
                validated['ROOT'] = float(validated['SDEP'])
                self.logger.debug(
                    f"Feasibility: capped ROOT {float(root):.2f} -> {validated['ROOT']:.2f} "
                    f"(must be <= SDEP)"
                )

        # Constraint 10: LAMX >= 0.1 (CLASS needs minimum positive LAI)
        lamx = validated.get('LAMX')
        if lamx is not None and float(lamx) < 0.1:
            validated['LAMX'] = 0.1
            self.logger.debug(
                f"Feasibility: raised LAMX {float(lamx):.3f} -> 0.1 "
                f"(minimum for CLASS stability)"
            )
            lamx = 0.1

        # Constraint 11: LAMN <= LAMX (minimum LAI can't exceed maximum)
        lamn = validated.get('LAMN')
        if lamn is not None and lamx is not None and float(lamn) > float(lamx):
            validated['LAMN'] = float(lamx) * 0.9
            self.logger.debug(
                f"Feasibility: capped LAMN {float(lamn):.3f} -> {validated['LAMN']:.3f} "
                f"(must be <= LAMX={float(lamx):.3f})"
            )

        return validated

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update MESH parameter .ini files."""
        self.logger.debug(f"Updating MESH files with params: {params}")
        return self.update_mesh_params(params)

    def _resolve_ini_file(self, file_type: str) -> Optional[Path]:
        """Resolve .ini file path, falling back to forcing/MESH_input/.

        The optimizer creates this parameter manager with settings/MESH/
        as the base directory, but the actual .ini files created by
        meshflow live in forcing/MESH_input/.  When reading initial
        values we need to find the real files.
        """
        if file_type == 'CLASS':
            filename = 'MESH_parameters_CLASS.ini'
        elif file_type == 'hydrology':
            filename = 'MESH_parameters_hydrology.ini'
        elif file_type == 'routing':
            filename = 'MESH_parameters.txt'
        else:
            return None

        # Primary: settings dir (mesh_settings_dir)
        primary = self.mesh_settings_dir / filename
        if primary.exists():
            return primary

        # Fallback: forcing/MESH_input/ (where meshflow writes the files)
        # mesh_settings_dir is typically {project_dir}/settings/MESH/
        # so project_dir is two levels up
        project_dir = self.mesh_settings_dir.parent.parent
        fallback = resolve_data_subdir(project_dir, 'forcing') / 'MESH_input' / filename
        if fallback.exists():
            self.logger.debug(
                f"File {filename} not in {self.mesh_settings_dir}, "
                f"using fallback: {fallback}"
            )
            return fallback

        return None

    def get_initial_parameters(self, skip_boundary_check: bool = False) -> Optional[Dict[str, float]]:
        """Get initial parameter values from .ini files or defaults.

        Values read from files are checked against parameter bounds.
        If a value falls in the bottom or top 5% of the feasible range,
        it is replaced with the bounds midpoint to give DDS a more
        central starting position in the search space.

        For log-transformed parameters, the boundary check and midpoint
        use log-space so the starting point is the geometric mean rather
        than the arithmetic mean.

        Args:
            skip_boundary_check: If True, return raw file values without
                replacing near-boundary values with midpoints.  Used by
                warm-start to preserve the values the model was actually
                running with when it achieved its best score.
        """
        try:
            params: Dict[str, float] = {}

            for param_name in self.mesh_params:
                bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                transform = bounds.get('transform', 'linear')
                is_log = (transform == 'log' and bounds['min'] > 0 and bounds['max'] > 0)

                # Midpoint: geometric mean for log params, arithmetic for linear
                if is_log:
                    midpoint = np.exp((np.log(bounds['min']) + np.log(bounds['max'])) / 2)
                else:
                    midpoint = (bounds['min'] + bounds['max']) / 2

                value = self._read_param_from_file(param_name)

                if value is not None:
                    if not skip_boundary_check:
                        # Check if value is near the boundary (bottom/top 5%)
                        if is_log:
                            log_range = np.log(bounds['max']) - np.log(bounds['min'])
                            if log_range > 0 and value > 0:
                                normalized = (np.log(max(value, bounds['min'])) - np.log(bounds['min'])) / log_range
                            else:
                                normalized = 0.5
                        else:
                            param_range = bounds['max'] - bounds['min']
                            normalized = (value - bounds['min']) / param_range if param_range > 0 else 0.5

                        if normalized < 0.05 or normalized > 0.95:
                            self.logger.info(
                                f"Initial {param_name}={value:.4g} is near boundary "
                                f"[{bounds['min']:.4g}, {bounds['max']:.4g}] "
                                f"(normalized={normalized:.3f}); "
                                f"using midpoint {midpoint:.4g} for DDS start"
                            )
                            value = midpoint
                    params[param_name] = value
                else:
                    params[param_name] = midpoint

            self.logger.info(
                "Initial parameters: "
                + ", ".join(f"{k}={v:.4g}" for k, v in params.items())
            )

            return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {e}", exc_info=True)
            return self._get_default_initial_values()

    def update_mesh_params(self, params: Dict[str, float]) -> bool:
        """
        Update MESH parameter files with new values.

        MESH uses .ini format: KEY value
        """
        try:
            # Group parameters by file
            class_params = {}
            hydro_params = {}
            routing_params = {}

            for param_name, value in params.items():
                file_type = self.param_file_map.get(param_name, 'unknown')
                if file_type == 'CLASS':
                    class_params[param_name] = value
                elif file_type == 'hydrology':
                    hydro_params[param_name] = value
                elif file_type == 'routing':
                    routing_params[param_name] = value

            success = True

            # Update CLASS parameters (fixed-format file)
            if class_params:
                success = success and self._update_class_file(
                    self.class_params_file, class_params
                )

            # Update hydrology parameters
            if hydro_params:
                success = success and self._update_ini_file(
                    self.hydro_params_file, hydro_params
                )

            # Update routing parameters (in main MESH_parameters.txt)
            if routing_params:
                success = success and self._update_ini_file(
                    self.routing_params_file, routing_params
                )

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating MESH parameters: {e}", exc_info=True)
            return False

    # Parameters that have multiple values per line (one per GRU/river class)
    ARRAY_PARAMS = {'WF_R2'}

    def _update_class_file(self, file_path: Path, params: Dict[str, float]) -> bool:
        """Update CLASS.ini fixed-format parameter file for ALL GRU blocks.

        CLASS.ini has repeated parameter blocks for each landcover class.
        This method updates parameters in ALL blocks, optionally applying
        landcover-specific multipliers for differentiation.

        When MESH_APPLY_PARAMS_ALL_GRUS=True (default), calibrated parameters
        are applied to all landcover class blocks.

        When MESH_USE_LANDCOVER_MULTIPLIERS=True (default), landcover-specific
        multipliers are applied (e.g., forest gets deeper soil, barren gets shallower).
        """
        try:
            self.logger.debug(f"Updating CLASS file: {file_path}")
            if not file_path.exists():
                self.logger.error(f"CLASS parameter file not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Re-detect GRU blocks in case file changed
            blocks = self._detect_all_gru_blocks_from_lines(lines)
            surfaces = ice_mask(lines)
            if any(name in ICE_CLASS for name in params):
                targets = surfaces if self.apply_to_all_grus else surfaces[:1]
                if not any(targets):
                    raise ValueError('Requested glacier CLASS parameters have no targeted ice GRU')
            if not blocks:
                # Fall back to single-block behavior
                blocks = [{'class_id': None, 'drn_line': 12, 'xslp_line': 13, 'block_index': 0}]

            total_updated = 0
            n_blocks = len(blocks) if self.apply_to_all_grus else 1

            for block_idx, block in enumerate(blocks[:n_blocks]):
                drn_line = block['drn_line']
                xslp_line = block['xslp_line']

                # Get NALCMS class ID for this block using the mapping
                # block_to_nalcms maps block index (0, 1, 2, ...) to NALCMS class ID (1, 8, 9, ...)
                nalcms_class_id = self.block_to_nalcms.get(block_idx)

                # Get multipliers for this landcover class
                multipliers = {}
                if self.use_landcover_multipliers and nalcms_class_id is not None:
                    multipliers = self.landcover_multipliers.get(nalcms_class_id, {})

                # First pass: compute multiplied values and validate as a set
                multiplied_params = {}
                for param_name, base_value in params.items():
                    multiplier = multipliers.get(param_name, 1.0)
                    multiplied_params[param_name] = base_value * multiplier

                # Enforce physical bounds on multiplied values (prevents crashes
                # when landcover multipliers push parameters outside feasible range,
                # e.g. Snow/Ice SDEP*0.3 or Barren KSAT*1.5)
                m_sdep = multiplied_params.get('SDEP')
                if m_sdep is not None and m_sdep < 0.5:
                    multiplied_params['SDEP'] = 0.5
                    m_sdep = 0.5
                m_mann = multiplied_params.get('MANN_CLASS')
                if m_mann is not None and m_mann < 0.01:
                    multiplied_params['MANN_CLASS'] = 0.01
                m_ksat = multiplied_params.get('KSAT')
                m_drn = multiplied_params.get('DRN')
                if m_ksat is not None and m_drn is not None:
                    if m_ksat * m_drn > 400.0:
                        multiplied_params['DRN'] = 400.0 / m_ksat
                    if m_sdep is not None and m_ksat * multiplied_params['DRN'] / m_sdep > 250.0:
                        multiplied_params['DRN'] = 250.0 * m_sdep / m_ksat
                m_frzth = multiplied_params.get('FRZTH')
                if m_frzth is not None and m_sdep is not None and m_frzth > m_sdep:
                    multiplied_params['FRZTH'] = m_sdep

                for param_name, base_value in params.items():
                    multiplier = multipliers.get(param_name, 1.0)
                    value = multiplied_params[param_name]

                    is_ice = surfaces[block_idx] if block_idx < len(surfaces) else False
                    if is_ice != (param_name in ICE_CLASS):
                        continue

                    # Determine which line this parameter is on
                    if param_name in ICE_CLASS:
                        marker, pos_idx = ICE_CLASS[param_name]
                        start = drn_line - 7
                        line_idx = next(i for i in range(max(0, start), drn_line) if marker in lines[i])
                        num_values = 9
                    elif param_name in ['DRN', 'SDEP']:
                        line_idx = drn_line
                        pos_idx = 0 if param_name == 'DRN' else 1
                        num_values = 4
                    elif param_name == 'DD':
                        line_idx = drn_line
                        pos_idx = 3
                        num_values = 4
                    elif param_name in ['XSLP', 'XDRAINH', 'MANN_CLASS', 'KSAT']:
                        line_idx = xslp_line
                        pos_map = {'XSLP': 0, 'XDRAINH': 1, 'MANN_CLASS': 2, 'KSAT': 3}
                        pos_idx = pos_map[param_name]
                        num_values = 5
                    elif param_name in self.VEG_PARAM_LAYOUT:
                        # Vegetation params: the column is the block's ACTIVE canopy
                        # category, not a fixed slot. Writing to an inactive
                        # (FCAN=0) column is a silent no-op and can destabilise
                        # CLASS, so resolve it per block from that block's FCAN line.
                        _marker, group_start, num_values = self.VEG_PARAM_LAYOUT[param_name]
                        line_idx = drn_line + self.VEG_LINE_OFFSET_FROM_DRN[param_name]
                        fcan_line = drn_line + self.VEG_LINE_OFFSET_FROM_DRN['LAMX']
                        pos_idx = group_start + self._active_veg_index(lines, fcan_line)
                    elif param_name in self.class_param_positions:
                        line_idx_base, pos_idx, num_values = self.class_param_positions[param_name]
                        default_drn = self.class_param_positions.get('DRN', (12, 0, 4))[0]
                        line_idx = drn_line + (line_idx_base - default_drn)
                    else:
                        continue  # Not a CLASS.ini parameter

                    if line_idx is not None and line_idx < 0:
                        continue

                    if line_idx is None or line_idx >= len(lines):
                        continue

                    line = lines[line_idx]
                    parts = line.split()

                    if pos_idx >= len(parts):
                        continue

                    # Format the new value
                    if abs(value) < 0.01:
                        new_val_str = f"{value:.4f}"
                    elif abs(value) < 1:
                        new_val_str = f"{value:.3f}"
                    elif abs(value) < 100:
                        new_val_str = f"{value:.2f}"
                    else:
                        new_val_str = f"{value:.1f}"

                    old_val = parts[pos_idx]
                    parts[pos_idx] = new_val_str

                    # Reconstruct line with proper spacing. A token that fills
                    # its 8-char field (e.g. 1.000E-02) would otherwise glue to
                    # the previous field, shifting every later column and
                    # consuming the trailing MID integer — MESH then aborts
                    # ('Unable to read the file') on every subsequent read.
                    new_line = ""
                    for i, part in enumerate(parts):
                        if i < num_values:
                            field = f"{part:>8}"
                            if new_line and not field.startswith(" "):
                                field = " " + field
                            new_line += field
                        else:
                            new_line += " " + part
                    new_line += "\n"

                    lines[line_idx] = new_line
                    total_updated += 1

                    if block_idx == 0 or multiplier != 1.0:
                        mult_info = f" (x{multiplier:.2f})" if multiplier != 1.0 else ""
                        self.logger.debug(
                            f"Block {block_idx} (NALCMS {nalcms_class_id}): "
                            f"{param_name}: {old_val} -> {new_val_str}{mult_info}"
                        )

            if total_updated == 0:
                self.logger.error(
                    f"No CLASS parameters were updated in {file_path.name} "
                    f"(requested: {list(params.keys())}). "
                    f"File has {len(lines)} lines; check that line positions are correct."
                )
                return False

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            n_blocks_updated = len(blocks[:n_blocks])
            self.logger.debug(
                f"Updated CLASS params across {n_blocks_updated} GRU blocks "
                f"({total_updated} total updates) in {file_path.name}"
            )
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating CLASS file {file_path.name}: {e}", exc_info=True)
            return False

    def _update_ini_file(self, file_path: Path, params: Dict[str, float]) -> bool:
        """Update a .ini format parameter file.

        If a scalar parameter is not found in the file, it is appended so that
        subsequent iterations (and MESH itself) can pick it up.
        """
        try:
            self.logger.debug(f"Updating file: {file_path}")
            if not file_path.exists():
                self.logger.debug(f"File not found: {file_path}")
                if file_path.parent.exists():
                    self.logger.debug(f"Directory contents of {file_path.parent}: {os.listdir(file_path.parent)}")
                else:
                    self.logger.debug(f"Parent directory does not exist: {file_path.parent}")
                self.logger.error(f"Parameter file not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            gru_params = {key: value for key, value in params.items() if key in LAND_HYDRO or key in ICE_HYDRO}
            if 'GRU class dependent hydrologic parameters' not in content and not any(key in ICE_HYDRO for key in params):
                gru_params = {}  # Legacy flat key/value format has no counted arrays.
            if gru_params:
                class_file = self._resolve_ini_file('CLASS')
                land_mask = None
                spatial_manifest = class_file.parent / 'spatial_units.json'
                if spatial_manifest.exists():
                    units = json.loads(spatial_manifest.read_text())['units']
                    land_mask = [u['surface'] == 'land' and u.get('modis') != 17 for u in units]
                content = update_gru_arrays(content, gru_params, ice_mask(class_file.read_text().splitlines()), land_mask)

            updated = len(gru_params)
            injected = 0
            for param_name, value in params.items():
                if param_name in gru_params:
                    continue
                if param_name in self.ARRAY_PARAMS:
                    # Handle array parameters (e.g., WF_R2 has one value per river class)
                    # Format: WF_R2  0.30    0.30    0.30  # comment
                    # Replace all numeric values on the line with the calibrated value
                    pattern = rf'^(?P<prefix>\s*{param_name}\s*(?:=|\s)\s*)(?P<vals>[\d\.\s\-\+eE]+)(?P<comment>\s*[!#].*)?$'

                    def replace_array_values(m, _value=value):
                        prefix = m.group('prefix')
                        values_str = m.group('vals')
                        comment = m.group('comment') or ''
                        # Count how many values there were
                        num_values = len(re.findall(r'[\d\.\-\+eE]+', values_str))
                        # Create new values string with same count
                        new_values = '    '.join([f"{_value:.2f}"] * num_values)
                        return prefix + new_values + comment

                    content, n = re.subn(
                        pattern,
                        replace_array_values,
                        content,
                        count=1,
                        flags=re.MULTILINE | re.IGNORECASE
                    )
                else:
                    # Match: KEY value (ignore comments starting with !)
                    # Use word boundary \b to avoid matching partial names
                    # and handle KEY=value or KEY value
                    pattern = rf'\b({param_name})\b\s*[\s=]+\s*([\d\.\-\+eE]+)'
                    content, n = re.subn(pattern, lambda m, _value=value: m.group(1) + " " + f"{_value:.6f}", content, count=1, flags=re.IGNORECASE)  # type: ignore[misc]

                if n > 0:
                    updated += 1
                else:
                    # Parameter not found — inject it at the end of the file
                    # so MESH can read it and future regex updates will match.
                    if param_name not in self.ARRAY_PARAMS:
                        inject_line = f"{param_name}  {value:.6f}  # injected by SYMFLUENCE calibration\n"
                        if not content.endswith('\n'):
                            content += '\n'
                        content += inject_line
                        injected += 1
                        log_once(
                            self.logger, logging.INFO,
                            key=f'mesh-injected-{file_path.name}-{param_name}',
                            message=(
                                f"Injected missing parameter {param_name}={value:.6f} "
                                f"into {file_path.name}"
                            ),
                        )
                    else:
                        log_once(
                            self.logger, logging.WARNING,
                            key=f'mesh-array-param-missing-{file_path.name}-{param_name}',
                            message=(
                                f"Array parameter {param_name} not found in "
                                f"{file_path.name}; cannot inject automatically"
                            ),
                        )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            total = updated + injected
            if total == 0:
                self.logger.error(
                    f"No hydrology parameters were updated or injected in "
                    f"{file_path.name} (requested: {list(params.keys())})"
                )
                return False

            self.logger.debug(
                f"Hydrology file {file_path.name}: "
                f"{updated} updated, {injected} injected, "
                f"{len(params) - total} failed"
            )
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating {file_path.name}: {e}", exc_info=True)
            return False

    def _read_param_from_file(self, param_name: str) -> Optional[float]:
        """Read a parameter value from the appropriate file.

        CLASS parameters use fixed-format positional lines, so they are
        read using ``self.class_param_positions`` instead of a key-value
        regex.  Hydrology and routing parameters use a standard
        ``KEY value`` .ini format.

        Falls back to forcing/MESH_input/ if the file is not found in
        the primary settings directory.
        """
        try:
            file_type = self.param_file_map.get(param_name)
            if file_type is None:
                return None

            file_path = self._resolve_ini_file(file_type)
            if file_path is None:
                return None

            if param_name in ICE_CLASS:
                lines = file_path.read_text().splitlines()
                blocks = self._detect_all_gru_blocks_from_lines(lines)
                for block, is_ice in zip(blocks, ice_mask(lines)):
                    if is_ice:
                        marker, position = ICE_CLASS[param_name]
                        row = next(line for line in lines[max(0, block['drn_line']-7):block['drn_line']]
                                   if marker in line)
                        return float(row.split()[position])
                return None
            if param_name in ICE_HYDRO:
                flags = ice_mask(self._resolve_ini_file('CLASS').read_text().splitlines())
                if not any(flags):
                    return None
                row = next((line.split('#')[0].split()[1:] for line in file_path.read_text().splitlines()
                            if line.strip().startswith(param_name+' ')), None)
                return float(row[flags.index(True)] if len(row) > 1 else row[0]) if row else ICE_HYDRO[param_name]

            # CLASS parameters are positional (fixed-format lines)
            if file_type == 'CLASS':
                if param_name not in self.class_param_positions:
                    log_once(
                        self.logger, logging.WARNING,
                        key=f'mesh-class-param-unmapped-{param_name}',
                        message=f"No positional mapping for CLASS param {param_name}",
                    )
                    return None

                line_idx, pos_idx, _num_values = self.class_param_positions[param_name]

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_idx >= len(lines):
                    return None

                parts = lines[line_idx].split()
                if pos_idx >= len(parts):
                    return None

                return float(parts[pos_idx])

            # Hydrology / routing: key-value .ini format
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            pattern = rf'^{param_name}\s+([\d\.\-\+eE]+)'
            match = re.search(pattern, content, re.MULTILINE)

            if match:
                return float(match.group(1))

            return None

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error reading {param_name}: {e}", exc_info=True)
            return None

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds).

        Uses geometric mean for log-transformed parameters.
        """
        params = {}
        for param_name in self.mesh_params:
            bounds = self.param_bounds[param_name]
            transform = bounds.get('transform', 'linear')
            if transform == 'log' and bounds['min'] > 0 and bounds['max'] > 0:
                params[param_name] = np.exp(
                    (np.log(bounds['min']) + np.log(bounds['max'])) / 2
                )
            else:
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params
