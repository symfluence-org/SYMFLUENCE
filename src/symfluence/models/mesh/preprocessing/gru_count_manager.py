# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GRU Count Manager

Coordinates GRU count alignment between the MESH drainage database (DDB)
and CLASS parameter file.  Orchestrates DDB trimming, CLASS block removal,
and GRU fraction renormalization.
"""
from __future__ import annotations

import logging

from symfluence.core.mixins import ConfigMixin

from .class_file_manager import CLASSFileManager
from .config_defaults import should_force_single_gru
from .ddb_file_manager import DDBFileManager


class GRUCountManager(ConfigMixin):
    """Manages GRU count alignment between DDB and CLASS files.

    Orchestrates the DDB manager and CLASS manager to ensure GRU counts
    are consistent (accounting for the MESH off-by-one issue) and that
    small GRUs are removed.

    Args:
        ddb_manager: DDBFileManager instance
        class_manager: CLASSFileManager instance
        config: Configuration dictionary
        logger: Logger instance
    """

    def __init__(
        self,
        ddb_manager: DDBFileManager,
        class_manager: CLASSFileManager,
        config: dict,
        logger: logging.Logger,
    ):
        self._ddb = ddb_manager
        self._class = class_manager
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fix_gru_count_mismatch(self) -> None:
        """Ensure CLASS NM matches GRU count and renormalize GRU fractions.

        MESH has an off-by-one issue where it reads NGRU-1 GRUs. This ensures:
        1. CLASS has NGRU-1 parameter blocks
        2. The first NGRU-1 GRU fractions sum to 1.0

        If lumped/point mode or MESH_FORCE_SINGLE_GRU is enabled, collapses
        to a single GRU.  This method is idempotent.
        """
        if should_force_single_gru(self.config_dict):
            self._collapse_to_single_gru()
            return

        # Remove small GRUs first
        self._remove_small_grus()

        current_ddb_gru_count = self._ddb.get_gru_count()
        class_block_count = self._class.get_block_count()

        if current_ddb_gru_count is None or class_block_count is None:
            min_total = float(self._get_config_value(
                lambda: self.config.model.mesh.gru_min_total,
                dict_key='MESH_GRU_MIN_TOTAL',
            ))
            keep_mask = self._ddb.trim_empty_gru_columns(min_total)
            self._class.fix_nm(keep_mask)
            self._ddb.ensure_gru_normalization()
            return

        expected_class_blocks = max(1, current_ddb_gru_count - 1)

        if class_block_count == expected_class_blocks:
            self.logger.debug(
                f"Already aligned: CLASS has {class_block_count} blocks, "
                f"DDB has {current_ddb_gru_count} NGRU "
                f"(MESH reads {expected_class_blocks})"
            )
            self._ddb.renormalize_active_grus(expected_class_blocks)
            return

        self.logger.warning(
            f"GRU count mismatch: CLASS has {class_block_count} blocks, "
            f"but MESH will read {expected_class_blocks} GRUs "
            f"(DDB NGRU={current_ddb_gru_count})"
        )

        if class_block_count > expected_class_blocks:
            self._class.trim_to_count(expected_class_blocks)
            self._class.update_nm(expected_class_blocks)
        elif class_block_count < expected_class_blocks:
            target_ddb_count = class_block_count + 1
            self.logger.info(
                f"Trimming DDB to {target_ddb_count} GRUs "
                f"so MESH reads {class_block_count}"
            )
            self._ddb.trim_to_active_grus(target_ddb_count)

        final_class_count = self._class.get_block_count() or expected_class_blocks
        self._ddb.renormalize_active_grus(final_class_count)

    # ------------------------------------------------------------------
    # Single-GRU collapse
    # ------------------------------------------------------------------

    def _collapse_to_single_gru(self) -> None:
        """Collapse all GRUs to a single dominant land cover class."""
        self._ddb.collapse_to_single_gru()
        self._class.trim_to_count(1)
        self._class.update_nm(1)
        self.logger.info("Single-GRU mode activated: 1 CLASS block, 1 active GRU")

    # ------------------------------------------------------------------
    # Small GRU removal
    # ------------------------------------------------------------------

    def _remove_small_grus(self) -> None:
        """Remove GRUs below the minimum fraction threshold."""
        min_fraction = float(self._get_config_value(
            lambda: self.config.model.mesh.gru_min_total,
            dict_key='MESH_GRU_MIN_TOTAL',
        ))

        keep_mask = self._ddb.remove_small_grus(min_fraction)
        if keep_mask is not None:
            self._class.remove_blocks_by_mask(keep_mask)
