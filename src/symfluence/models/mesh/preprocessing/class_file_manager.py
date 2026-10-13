# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLASS File Manager

Handles all operations on the MESH CLASS parameter file (CLASS.ini).
Manages GRU blocks, NM parameter, vegetation corrections, initial
conditions, and elevation band block creation.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


class CLASSFileManager:
    """Manages the MESH CLASS parameter file (MESH_parameters_CLASS.ini).

    Provides methods for block counting, trimming, NM updates, vegetation
    parameter fixes, initial condition fixes, and elevation band creation.

    Args:
        class_file_path: Path to MESH_parameters_CLASS.ini
        logger: Logger instance
    """

    def __init__(self, class_file_path: Path, logger: logging.Logger):
        self._path = class_file_path
        self.logger = logger

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _ice_line_indices(lines: list) -> set:
        """Protect explicitly configured ice blocks from terrestrial defaults."""
        starts = [i for i, line in enumerate(lines) if '05 5xFCAN' in line]
        protected = set()
        for start in starts:
            end = next((i+1 for i in range(start, len(lines))
                        if '19 RCAN/SCAN' in lines[i]), len(lines))
            sand = next((line for line in lines[start:end] if '3xSAND' in line), '')
            if sand and float(sand.split()[0]) == -4:
                protected.update(range(start, end))
        return protected

    # ------------------------------------------------------------------
    # Block counting
    # ------------------------------------------------------------------

    def get_block_count(self) -> Optional[int]:
        """Get the number of CLASS parameter blocks."""
        if not self._path.exists():
            return None

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            block_count = sum(
                1 for line in lines
                if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_')
            )
            return block_count if block_count > 0 else None
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    # ------------------------------------------------------------------
    # NM management
    # ------------------------------------------------------------------

    def read_nm_from_lines(self, lines: list) -> Optional[int]:
        """Read NM value from CLASS file lines."""
        for line in lines:
            if '04 DEGLAT' in line or 'NL/NM' in line or line.startswith('NM '):
                parts = line.split()
                if line.startswith('NM '):
                    try:
                        return int(parts[1])
                    except (ValueError, IndexError) as e:
                        self.logger.debug(
                            f"Could not parse NM from '{line.strip()}': {e}"
                        )
                else:
                    if len(parts) >= 9:
                        try:
                            return int(parts[8])
                        except (ValueError, IndexError) as e:
                            self.logger.debug(
                                f"Could not parse NM from column 9 of "
                                f"'{line.strip()}': {e}"
                            )
                break
        return None

    def update_nm(self, new_nm: int) -> None:
        """Update NM in CLASS parameters file."""
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            for i, line in enumerate(lines):
                if line.startswith('NM '):
                    parts = line.split()
                    old_nm = parts[1]
                    lines[i] = f"NM {new_nm}    ! number of landcover classes (GRUs)\n"
                    modified = True
                    self.logger.info(f"Updated CLASS NM from {old_nm} to {new_nm}")
                    break

                if '04 DEGLAT' in line or 'NL/NM' in line:
                    parts = line.split()
                    if len(parts) >= 9:
                        old_nm = parts[8]
                        tokens = re.split(r'(\s+)', line)
                        value_count = 0
                        for j, tok in enumerate(tokens):
                            if tok.strip():
                                value_count += 1
                                if value_count == 9:
                                    tokens[j] = str(new_nm)
                                    break
                        lines[i] = ''.join(tokens)
                        modified = True
                        self.logger.info(f"Updated CLASS NM from {old_nm} to {new_nm}")
                    break

            if modified:
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to update CLASS NM: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Block trimming
    # ------------------------------------------------------------------

    def trim_to_count(self, target_count: int) -> None:
        """Trim CLASS parameter blocks to a specific count, preserving footer."""
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            ini_blocks = [i for i, line in enumerate(lines) if line.startswith('[GRU_')]
            legacy_blocks = [
                i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line
            ]

            footer_start = None
            for i, line in enumerate(lines):
                if '20 ' in line or '20\t' in line or line.strip().endswith('20'):
                    footer_start = i
                    break

            footer = []
            if footer_start is not None:
                footer = lines[footer_start:]
                lines = lines[:footer_start]

            if ini_blocks:
                header = lines[:ini_blocks[0]]
                block_starts = ini_blocks + [len(lines)]
                blocks = [
                    lines[block_starts[i]:block_starts[i + 1]]
                    for i in range(len(block_starts) - 1)
                ]
            elif legacy_blocks:
                legacy_blocks = [i for i in legacy_blocks if i < len(lines)]
                if not legacy_blocks:
                    return
                header = lines[:legacy_blocks[0]]
                block_starts = legacy_blocks + [len(lines)]
                blocks = [
                    lines[block_starts[i]:block_starts[i + 1]]
                    for i in range(len(block_starts) - 1)
                ]
            else:
                return

            kept_blocks = blocks[:target_count]

            if len(kept_blocks) != len(blocks):
                new_lines = (
                    header
                    + [line for block in kept_blocks for line in block]
                    + footer
                )
                content = '\n'.join(new_lines)
                if not content.endswith('\n'):
                    content += '\n'
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(
                    f"Trimmed CLASS parameters to {len(kept_blocks)} GRU block(s)"
                )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to trim CLASS to count {target_count}: {e}", exc_info=True)

    def trim_blocks_by_mask(self, lines: list, keep_mask: list) -> bool:
        """Trim CLASS parameter blocks to match DDB GRU columns.

        Args:
            lines: CLASS file lines
            keep_mask: Boolean list — True = keep, False = remove

        Returns:
            True if blocks were trimmed
        """
        ini_blocks = [i for i, line in enumerate(lines) if line.startswith('[GRU_')]
        legacy_blocks = [
            i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line
        ]

        if ini_blocks:
            header = lines[:ini_blocks[0]]
            block_starts = ini_blocks + [len(lines)]
            blocks = [
                lines[block_starts[i]:block_starts[i + 1]]
                for i in range(len(block_starts) - 1)
            ]
        elif legacy_blocks:
            header = lines[:legacy_blocks[0]]
            block_starts = legacy_blocks + [len(lines)]
            blocks = [
                lines[block_starts[i]:block_starts[i + 1]]
                for i in range(len(block_starts) - 1)
            ]
        else:
            return False

        max_blocks = min(len(blocks), len(keep_mask))
        kept_blocks = [blocks[i] for i in range(max_blocks) if keep_mask[i]]

        if len(kept_blocks) != len(blocks):
            new_lines = header + [line for block in kept_blocks for line in block]
            content = '\n'.join(new_lines)
            if not content.endswith('\n'):
                content += '\n'
            with open(self._path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(
                f"Trimmed CLASS parameters to {len(kept_blocks)} GRU block(s)"
            )
            return True

        return False

    def remove_blocks_by_mask(self, keep_mask: np.ndarray) -> None:
        """Remove CLASS parameter blocks corresponding to removed GRUs.

        Args:
            keep_mask: Boolean array where True = keep GRU, False = remove
        """
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            block_starts = [
                i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line
            ]

            if not block_starts:
                self.logger.debug(
                    "No CLASS blocks found (looking for '05 5xFCAN/4xLAMX')"
                )
                return

            n_blocks = len(block_starts)
            n_mask = len(keep_mask)

            effective_mask = (
                keep_mask[:n_blocks]
                if n_mask >= n_blocks
                else np.pad(keep_mask, (0, n_blocks - n_mask), constant_values=True)
            )

            n_keep = int(effective_mask.sum())
            n_remove = n_blocks - n_keep

            if n_remove == 0:
                return

            self.logger.debug(
                f"Removing {n_remove} CLASS blocks "
                f"(keeping indices {[i for i, k in enumerate(effective_mask) if k]})"
            )

            footer_start = None
            for i, line in enumerate(lines):
                if re.search(r'^\s*0\s+0\s+0\s+0.*20\s', line):
                    footer_start = i
                    break

            footer = []
            if footer_start is not None:
                footer = lines[footer_start:]
                lines = lines[:footer_start]

            header = lines[:block_starts[0]]
            block_ends = block_starts[1:] + [len(lines)]
            blocks = [lines[block_starts[i]:block_ends[i]] for i in range(n_blocks)]

            kept_blocks = [
                blocks[i] for i in range(n_blocks) if effective_mask[i]
            ]

            new_lines = (
                header + [line for block in kept_blocks for line in block] + footer
            )
            content = '\n'.join(new_lines)
            if not content.endswith('\n'):
                content += '\n'

            with open(self._path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"Removed {n_remove} CLASS block(s), {n_keep} remaining")
            self.update_nm(n_keep)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to remove CLASS blocks: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Fix NM + optional trim (combined workflow)
    # ------------------------------------------------------------------

    def fix_nm(self, keep_mask: Optional[list] = None) -> None:
        """Fix CLASS NM parameter to match block count, optionally trimming blocks."""
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            block_count = sum(
                1 for line in lines
                if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_')
            )

            nm_from_class = self.read_nm_from_lines(lines)

            trimmed_class = False
            if keep_mask is not None:
                trimmed_class = self.trim_blocks_by_mask(lines, keep_mask)
                if trimmed_class:
                    with open(self._path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                    block_count = sum(
                        1 for line in lines
                        if 'XSLP/XDRAINH/MANN/KSAT/MID' in line
                        or line.startswith('[GRU_')
                    )
                    nm_from_class = self.read_nm_from_lines(lines)

            if nm_from_class != block_count:
                self.logger.warning(
                    f"CLASS NM ({nm_from_class}) != block count ({block_count})"
                )
                self.update_nm(block_count)
            else:
                self.logger.debug(
                    f"CLASS NM={nm_from_class} matches {block_count} blocks"
                )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix GRU count mismatch: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Vegetation parameters
    # ------------------------------------------------------------------

    def fix_vegetation_parameters(self) -> None:
        """Fix CLASS vegetation parameters for different GRU types.

        Adjusts LNZ0 (roughness length) and RSMN (stomatal resistance) based
        on vegetation class identified from the MID comment in line 13.
        """
        if not self._path.exists():
            return

        veg_corrections = {
            'needle': (-0.7, 145),
            'need_fore': (-0.7, 145),
            'broad': (-0.4, 150),
            'shru': (-1.8, 200),
            'grass': (-2.5, 200),
            'gras': (-2.5, 200),
            'crop': (-2.3, 120),
            'Crop': (-2.3, 120),
            'barren': (-4.6, 500),
            'urban': (-2.0, 200),
        }

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            gru_types = []
            for i, line in enumerate(lines):
                if 'XSLP/XDRAINH/MANN/KSAT/MID' in line:
                    gru_type = None
                    line_lower = line.lower()
                    for veg_key in veg_corrections:
                        if veg_key.lower() in line_lower:
                            gru_type = veg_key
                            break
                    gru_types.append((i, gru_type))

            if not gru_types:
                self.logger.debug("No GRU blocks found in CLASS file")
                return

            self.logger.debug(f"Found {len(gru_types)} GRU blocks: {gru_types}")

            modified = False
            new_lines = []

            block_starts = [
                i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line
            ]

            if len(block_starts) != len(gru_types):
                self.logger.warning(
                    f"Mismatch: {len(block_starts)} block starts vs "
                    f"{len(gru_types)} MID comments"
                )

            current_block = 0
            protected = self._ice_line_indices(lines)
            for i, line in enumerate(lines):
                if i in protected:
                    new_lines.append(line)
                    continue
                while (
                    current_block < len(block_starts) - 1
                    and i >= block_starts[current_block + 1]
                ):
                    current_block += 1

                if current_block < len(gru_types):
                    _, gru_type = gru_types[current_block]
                else:
                    gru_type = None

                if '5xLNZ0/4xLAMN' in line and gru_type:
                    if gru_type in veg_corrections:
                        lnz0_target = veg_corrections[gru_type][0]
                        line = line.replace('-1.300', f'{lnz0_target:.3f}')
                        modified = True
                        self.logger.debug(f"Set LNZ0={lnz0_target:.3f} for {gru_type}")

                if '4xRSMN/4xQA50' in line and gru_type:
                    if gru_type in veg_corrections:
                        rsmn_target = veg_corrections[gru_type][1]
                        line = line.replace('145.000', f'{rsmn_target:.3f}')
                        modified = True
                        self.logger.debug(
                            f"Set RSMN={rsmn_target:.3f} for {gru_type}"
                        )

                new_lines.append(line)

            if modified:
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                self.logger.info(
                    "Fixed CLASS vegetation parameters (LNZ0, RSMN) "
                    "for different GRU types"
                )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix CLASS vegetation parameters: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def fix_initial_conditions(
        self,
        time_window_fn: Optional[Callable[[], Optional[Tuple]]] = None,
        latitude: Optional[float] = None,
    ) -> None:
        """Fix CLASS initial conditions for proper snow simulation.

        Args:
            time_window_fn: Callback returning (start, end) datetime tuple
            latitude: Domain latitude for climate classification
        """
        if not self._path.exists():
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            time_window = time_window_fn() if time_window_fn else None
            start_month = time_window[0].month if time_window else 1

            snow_params = self._get_climate_adjusted_snow_params(
                start_month, latitude
            )

            initial_sno = snow_params['sno']
            initial_albs = snow_params['albs']
            initial_rhos = snow_params['rhos']
            initial_tsno = snow_params['tsno']
            initial_tcan = snow_params['tcan']

            climate_zone = (
                'arctic' if latitude and abs(latitude) >= 60
                else 'boreal' if latitude and abs(latitude) >= 50
                else 'temperate'
            )
            self.logger.info(
                f"Using {climate_zone} snow defaults (lat={latitude:.1f}°)"
                if latitude
                else "Using temperate snow defaults (latitude unknown)"
            )

            modified = False
            new_lines = []

            protected = self._ice_line_indices(lines)
            for line_index, line in enumerate(lines):
                if line_index in protected:
                    new_lines.append(line)
                    continue
                if '17 3xTBAR' in line or ('17' in line and 'TBAR' in line):
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            tbar1 = float(parts[0])
                            tbar2 = float(parts[1])
                            tbar3 = float(parts[2])
                            tpnd = float(parts[5])
                            new_line = (
                                f"  {tbar1:.3f}  {tbar2:.3f}  {tbar3:.3f}  "
                                f"{initial_tcan:.3f}  {initial_tsno:.3f}   "
                                f"{tpnd:.3f}  "
                                f"17 3xTBAR (or more)/TCAN/TSNO/TPND\n"
                            )
                            new_lines.append(new_line)
                            modified = True
                            continue
                        except (ValueError, IndexError) as e:
                            self.logger.debug(
                                f"Could not parse TBAR values from "
                                f"'{line.strip()}': {e}"
                            )

                if '19 RCAN/SCAN/SNO/ALBS/RHOS/GRO' in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            rcan = float(parts[0])
                            scan = float(parts[1])
                            gro = float(parts[5])
                            new_line = (
                                f"   {rcan:.3f}   {scan:.3f}   "
                                f"{initial_sno:.1f}   "
                                f"{initial_albs:.2f}   {initial_rhos:.1f}   "
                                f"{gro:.3f}  "
                                f"19 RCAN/SCAN/SNO/ALBS/RHOS/GRO\n"
                            )
                            new_lines.append(new_line)
                            modified = True
                            continue
                        except (ValueError, IndexError) as e:
                            self.logger.debug(
                                f"Could not parse RCAN/SCAN/SNO values from "
                                f"'{line.strip()}': {e}"
                            )

                new_lines.append(line)

            if modified:
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                self.logger.info(
                    f"Fixed CLASS initial conditions: SNO={initial_sno}mm, "
                    f"ALBS={initial_albs}, RHOS={initial_rhos}kg/m³"
                )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to fix CLASS initial conditions: {e}", exc_info=True)

    @staticmethod
    def _get_climate_adjusted_snow_params(
        start_month: int, latitude: Optional[float]
    ) -> dict:
        """Get snow initial conditions adjusted for climate zone and season."""
        is_winter = start_month in [11, 12, 1, 2, 3, 4]

        if latitude is None:
            climate = 'temperate'
        elif abs(latitude) >= 60:
            climate = 'arctic'
        elif abs(latitude) >= 50:
            climate = 'boreal'
        else:
            climate = 'temperate'

        params = {
            'arctic': {
                'winter': {'sno': 150.0, 'albs': 0.80, 'rhos': 200.0,
                           'tsno': -20.0, 'tcan': -15.0},
                'summer': {'sno': 50.0, 'albs': 0.70, 'rhos': 300.0,
                           'tsno': -5.0, 'tcan': 0.0},
            },
            'boreal': {
                'winter': {'sno': 100.0, 'albs': 0.75, 'rhos': 250.0,
                           'tsno': -10.0, 'tcan': -5.0},
                'summer': {'sno': 10.0, 'albs': 0.60, 'rhos': 350.0,
                           'tsno': -1.0, 'tcan': 5.0},
            },
            'temperate': {
                'winter': {'sno': 50.0, 'albs': 0.70, 'rhos': 300.0,
                           'tsno': -5.0, 'tcan': 0.0},
                'summer': {'sno': 0.0, 'albs': 0.50, 'rhos': 400.0,
                           'tsno': 0.0, 'tcan': 10.0},
            },
        }

        season = 'winter' if is_winter else 'summer'
        return params[climate][season]

    # ------------------------------------------------------------------
    # Config field overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_value(value: float) -> str:
        """Format a float using the same width heuristic as the calibrator."""
        av = abs(value)
        if av < 0.01:
            return f"{value:.4f}"
        if av < 1:
            return f"{value:.3f}"
        if av < 100:
            return f"{value:.2f}"
        return f"{value:.1f}"

    @staticmethod
    def _rewrite_line_values(line: str, num_values: int, replacements: dict) -> str:
        """Rewrite specific positional values on a fixed-format CLASS line.

        Values with index < ``num_values`` are written as right-justified
        8-char fields (matching the calibrator's formatting); any remaining
        tokens (comments, veg labels, line numbers) are preserved verbatim.

        Args:
            line: The raw CLASS line.
            num_values: Number of leading numeric fields on the line.
            replacements: Mapping ``{position_index: formatted_string}``.

        Returns:
            The rewritten line (without trailing newline).
        """
        parts = line.split()
        for pos, val in replacements.items():
            if 0 <= pos < len(parts):
                parts[pos] = val

        new_line = ""
        for i, part in enumerate(parts):
            if i < num_values:
                field = f"{part:>8}"
                if new_line and not field.startswith(" "):
                    field = " " + field
                new_line += field
            else:
                new_line += " " + part
        return new_line

    def apply_field_overrides(self, overrides: dict) -> None:
        """Override meshflow-derived CLASS fields from config.

        meshflow hard-derives regime-determining fields (soil texture, drainage
        density, initial states, vegetation parameters) from the input data.
        This applies user-configured overrides to those fields across ALL GRU
        blocks so the surface-runoff regime is controllable from config alone.

        Only keys present (non-None) in ``overrides`` are applied. Recognised
        keys: ``sand``/``clay``/``orgm`` (per-layer lists), ``dd`` (float),
        ``mid`` (int), ``tbar``/``thlq`` (per-layer lists), and the veg scalars
        ``cmas``/``qa50``/``vpda``/``vpdb``.

        Args:
            overrides: Dict of override values (see above).
        """
        if not self._path.exists():
            return

        active = {k: v for k, v in overrides.items() if v is not None}
        if not active:
            return

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            applied: set = set()
            protected = self._ice_line_indices(lines)
            for i, line in enumerate(lines):
                if i in protected:
                    continue
                new_line = self._apply_overrides_to_line(line, active, applied)
                if new_line is not None:
                    lines[i] = new_line + '\n'

            with open(self._path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            if applied:
                self.logger.info(
                    f"Applied CLASS field overrides from config: "
                    f"{', '.join(sorted(applied))}"
                )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to apply CLASS field overrides: {e}", exc_info=True)

    def _apply_overrides_to_line(
        self, line: str, active: dict, applied: set
    ) -> Optional[str]:
        """Return the rewritten line if it matches an override marker, else None."""
        fmt = self._fmt_value

        def _layer_replacements(values: list, count: int) -> dict:
            n = min(len(values), count)
            return {pos: fmt(float(values[pos])) for pos in range(n)}

        # Soil texture / initial-state lines (leading 3x-per-layer values)
        if '3xSAND' in line and 'sand' in active:
            vals = active['sand']
            applied.add('sand')
            return self._rewrite_line_values(line, len(vals), _layer_replacements(vals, len(vals)))
        if '3xCLAY' in line and 'clay' in active:
            vals = active['clay']
            applied.add('clay')
            return self._rewrite_line_values(line, len(vals), _layer_replacements(vals, len(vals)))
        if '3xORGM' in line and 'orgm' in active:
            vals = active['orgm']
            applied.add('orgm')
            return self._rewrite_line_values(line, len(vals), _layer_replacements(vals, len(vals)))
        if '3xTBAR' in line and 'tbar' in active:
            vals = active['tbar']
            applied.add('tbar')
            return self._rewrite_line_values(line, 6, _layer_replacements(vals, 3))
        if '3xTHLQ' in line and 'thlq' in active:
            vals = active['thlq']
            applied.add('thlq')
            return self._rewrite_line_values(line, 7, _layer_replacements(vals, 3))

        # DRN/SDEP/FARE/DD — DD at position 3
        if 'DRN/SDEP' in line and 'dd' in active:
            applied.add('dd')
            return self._rewrite_line_values(line, 4, {3: fmt(float(active['dd']))})

        # XSLP/XDRAINH/MANN/KSAT/MID — MID at position 4 (integer)
        if 'XSLP/XDRAINH' in line and 'mid' in active:
            applied.add('mid')
            return self._rewrite_line_values(line, 5, {4: str(int(active['mid']))})

        # Vegetation lines — override only the ACTIVE (currently non-zero)
        # column(s) in the 4x parameter group, leaving inactive categories at
        # zero. Writing the scalar into inactive (FCAN=0) columns activates
        # vegetation physics for categories that are not present and drives
        # CLASS into a snow-energy-balance crash, so we mirror meshflow's
        # single-active-column layout.
        if '5xALVC/4xCMAS' in line and 'cmas' in active:
            applied.add('cmas')
            return self._rewrite_line_values(
                line, 9, self._active_group_replacements(line, 5, 8, fmt(float(active['cmas'])))
            )
        if '4xRSMN/4xQA50' in line and 'qa50' in active:
            applied.add('qa50')
            return self._rewrite_line_values(
                line, 8, self._active_group_replacements(line, 4, 7, fmt(float(active['qa50'])))
            )
        if '4xVPDA/4xVPDB' in line and ('vpda' in active or 'vpdb' in active):
            repl = {}
            if 'vpda' in active:
                applied.add('vpda')
                repl.update(self._active_group_replacements(line, 0, 3, fmt(float(active['vpda']))))
            if 'vpdb' in active:
                applied.add('vpdb')
                repl.update(self._active_group_replacements(line, 4, 7, fmt(float(active['vpdb']))))
            return self._rewrite_line_values(line, 8, repl)

        return None

    @staticmethod
    def _active_group_replacements(line: str, lo: int, hi: int, value: str) -> dict:
        """Map the active (non-zero) columns in ``[lo, hi]`` to ``value``.

        The 4x vegetation groups carry one value per canopy category; only the
        category present in this GRU is non-zero. We override exactly those
        column(s), falling back to the last column when the group is all-zero.
        """
        parts = line.split()
        active_positions = []
        for pos in range(lo, hi + 1):
            if pos < len(parts):
                try:
                    if float(parts[pos]) != 0.0:
                        active_positions.append(pos)
                except ValueError:
                    continue
        if not active_positions:
            active_positions = [hi]
        return {pos: value for pos in active_positions}

    # ------------------------------------------------------------------
    # Elevation band blocks
    # ------------------------------------------------------------------

    def create_elevation_band_blocks(
        self,
        elevation_info: list,
        get_num_cells_fn: Callable[[], int],
    ) -> bool:
        """Create CLASS parameter blocks for elevation bands.

        Args:
            elevation_info: List of dicts with 'elevation' and 'fraction'
            get_num_cells_fn: Callable returning number of subbasins

        Returns:
            True if successful
        """
        if not self._path.exists():
            self.logger.warning(
                "CLASS file not found, cannot create elevation band blocks"
            )
            return False

        if not elevation_info:
            self.logger.warning("No elevation info provided")
            return False

        n_bands = len(elevation_info)
        self.logger.info(f"Creating {n_bands} CLASS blocks for elevation bands")

        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            header_lines = []
            block_start = None
            for i, line in enumerate(lines):
                if '04 DEGLAT/DEGLON' in line:
                    header_lines = lines[:i + 1]
                    block_start = i + 1
                    break

            if block_start is None:
                self.logger.warning("Could not find CLASS header (line 04)")
                return False

            footer_lines = []
            footer_start = None
            for i, line in enumerate(lines):
                if i > block_start and ('20 ' in line or line.strip().endswith('20')):
                    footer_start = i
                    footer_lines = lines[i:]
                    break

            if footer_start is None:
                self.logger.warning("Could not find CLASS footer (line 20)")
                return False

            n_cells = get_num_cells_fn()
            for i, line in enumerate(header_lines):
                if '04 DEGLAT/DEGLON' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        parts[7] = str(n_cells)
                        parts[8] = str(n_bands)
                        comment_idx = line.index('04 DEGLAT')
                        values = parts[:9]
                        header_lines[i] = (
                            '  '
                            + '  '.join(f'{v:>8s}' for v in values)
                            + '       '
                            + line[comment_idx:]
                        )
                    break

            block_lines = lines[block_start:footer_start]
            first_block = []
            in_block = False
            for line in block_lines:
                if '05 5xFCAN' in line or in_block:
                    in_block = True
                    first_block.append(line)
                    if '19 RCAN/SCAN' in line:
                        break

            if not first_block or len(first_block) < 15:
                first_block = self._get_default_class_block()

            new_blocks = []
            for i, elev_info in enumerate(elevation_info):
                elevation = elev_info['elevation']
                fraction = elev_info['fraction']

                block = self._create_elevation_adjusted_block(
                    first_block, i, elevation, fraction
                )
                new_blocks.extend(block)
                new_blocks.append('')

            new_lines = header_lines + [''] + new_blocks + footer_lines

            with open(self._path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))

            elev_str = ', '.join(
                [f"{e['elevation']:.0f}m" for e in elevation_info]
            )
            self.logger.info(
                f"Created {n_bands} elevation band CLASS blocks "
                f"(elevations: {elev_str})"
            )
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(
                f"Failed to create elevation band CLASS blocks: {e}"
            , exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def _create_elevation_adjusted_block(
        base_block: list, band_index: int, elevation: float, fraction: float
    ) -> list:
        """Create a CLASS block with elevation-adjusted parameters."""
        if CLASSFileManager._ice_line_indices(base_block):
            raise ValueError('Build elevation units before splitting glacier surfaces')
        block = []
        vegetation_cover = min(1.0, max(0.2, 1.0 - (elevation - 1500) / 3000))

        for line in base_block:
            new_line = line

            if '05 5xFCAN' in line:
                fcan_values = [0.0, 0.0, 0.0, vegetation_cover, 0.0]
                new_line = (
                    f"   {fcan_values[0]:.3f}   {fcan_values[1]:.3f}   "
                    f"{fcan_values[2]:.3f}   {fcan_values[3]:.3f}   "
                    f"{fcan_values[4]:.3f}   1.450   0.000   0.000   "
                    f"0.000     05 5xFCAN/4xLAMX"
                )
            elif '5xLNZ0/4xLAMN' in line:
                lnz0 = -1.8 + (elevation - 1500) / 3000 * 0.5
                new_line = (
                    f"   0.000   0.000   0.000  {lnz0:.3f}   0.000   "
                    f"0.000   0.000   0.000   1.200     "
                    f"06 5xLNZ0/4xLAMN"
                )
            elif '4xRSMN/4xQA50' in line:
                rsmn = 200.0 + (elevation - 1500) / 2000 * 100
                new_line = (
                    f"   0.000   0.000   0.000 {rsmn:.3f}           "
                    f"0.000   0.000   0.000  36.000     "
                    f"09 4xRSMN/4xQA50"
                )
            elif 'XSLP/XDRAINH/MANN/KSAT/MID' in line:
                mid = 200 + band_index
                parts = line.split()
                if len(parts) >= 5:
                    xslp = parts[0]
                    xdrainh = parts[1]
                    mann = parts[2]
                    ksat = parts[3]
                    new_line = (
                        f"   {xslp}   {xdrainh}   {mann}   {ksat}   "
                        f"{mid} ElevBand_{band_index+1}_{elevation:.0f}m"
                        f"                        "
                        f"13 XSLP/XDRAINH/MANN/KSAT/MID"
                    )
            elif '19 RCAN/SCAN' in line:
                sno = 50.0 + (elevation - 1500) / 100 * 5
                albs = min(0.85, 0.70 + (elevation - 1500) / 5000)
                rhos = 300.0
                new_line = (
                    f"   0.000   0.000   {sno:.1f}   {albs:.2f}   "
                    f"{rhos:.1f}   1.000  "
                    f"19 RCAN/SCAN/SNO/ALBS/RHOS/GRO"
                )

            block.append(new_line)

        return block

    @staticmethod
    def _get_default_class_block() -> list:
        """Get a default CLASS parameter block for grassland vegetation."""
        return [
            "   0.000   0.000   0.000   1.000   0.000   0.000   0.000   0.000   1.450     05 5xFCAN/4xLAMX",
            "   0.000   0.000   0.000  -1.800   0.000   0.000   0.000   0.000   1.200     06 5xLNZ0/4xLAMN",
            "   0.000   0.000   0.000   0.045   0.000   0.000   0.000   0.000   4.500     07 5xALVC/4xCMAS",
            "   0.000   0.000   0.000   0.160   0.000   0.000   0.000   0.000   1.090     08 5xALIC/4xROOT",
            "   0.000   0.000   0.000 200.000           0.000   0.000   0.000  36.000     09 4xRSMN/4xQA50",
            "   0.000   0.000   0.000   0.800           0.000   0.000   0.000   1.050     10 4xVPDA/4xVPDB",
            "   0.000   0.000   0.000 100.000           0.000   0.000   0.000   5.000     11 4xPSGA/4xPSGB",
            "   1.000   2.500   1.000  50.000                                             12 DRN/SDEP/FARE/DD",
            "   0.030   0.350   0.100   0.050   100 Default                               13 XSLP/XDRAINH/MANN/KSAT/MID",
            "  50.000  50.000  50.000                                                     14 3xSAND (or more)",
            "  20.000  20.000  20.000                                                     15 3xCLAY (or more)",
            "   0.000   0.000   0.000                                                     16 3xORGM (or more)",
            "  4.000  2.000  1.000  -5.000  -10.000   4.000  17 3xTBAR (or more)/TCAN/TSNO/TPND",
            "   0.250   0.150   0.040   0.000   0.000   0.000   0.000                     18 3xTHLQ (or more)/3xTHIC (or more)/ZPND",
            "   0.000   0.000   100.0   0.75   250.0   1.000  19 RCAN/SCAN/SNO/ALBS/RHOS/GRO",
        ]
