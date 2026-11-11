# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base class for attribute processors.

Provides shared infrastructure for all attribute processing modules including:
- Configuration management
- Path resolution
- Catchment shapefile access
- Common utilities
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import geopandas as gpd
import numpy as np

from symfluence.core.config.coercion import coerce_config
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseAttributeProcessor(ConfigMixin):
    """Base class for all attribute processors."""

    def __init__(self, config: Union['SymfluenceConfig', Dict[str, Any]], logger: logging.Logger):
        """
        Initialize base attribute processor.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
        """
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        # Use ConfigMixin properties and methods with dict fallback
        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir,
            dict_key='SYMFLUENCE_DATA_DIR'
        ))
        self.logger.info(f'data dir: {self.data_dir}')

        # Get domain_name with dict fallback (set as instance attribute for subclass access)
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key='DOMAIN_NAME'
        )
        self.logger.info(f'domain name: {self.domain_name}')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()

        # Initialize results dictionary
        self.results: Dict[str, Any] = {}

    def _get_catchment_path(self) -> Path:
        """
        Get the path to the catchment shapefile.

        Uses the same convention as the rest of the framework
        (``PathManager`` / ``GeospatialStatistics``): the file is
        ``{domain}_HRUs_{discretization}.shp`` with "," replaced by "_" for
        combined discretizations, under
        ``shapefiles/catchment/{definition_method}/{experiment_id}/`` or the
        legacy flat ``shapefiles/catchment/`` directory.

        Returns:
            Path to catchment shapefile

        Raises:
            FileNotFoundError: when the shapefile does not exist. No other HRU
                shapefile is substituted: attributes computed for another
                discretization would be attached to the wrong HRUs.
        """
        catchment_path = self._get_config_value(
            lambda: self.config.paths.catchment_path,
            dict_key='CATCHMENT_PATH'
        )
        self.logger.info(f'catchment path: {catchment_path}')

        catchment_name = self._get_config_value(
            lambda: self.config.paths.catchment_name,
            dict_key='CATCHMENT_SHP_NAME'
        )
        self.logger.info(f'catchment name: {catchment_name}')

        if catchment_path in (None, 'default'):
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)

        if catchment_name in (None, 'default'):
            discretization = str(self._get_config_value(
                lambda: self.config.domain.discretization,
                default='',
                dict_key='SUB_GRID_DISCRETIZATION'
            )).replace(',', '_')
            catchment_file = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_file = catchment_name

        # Flat (legacy) location first, as PathManager.get_catchment_dir does
        direct = catchment_path / catchment_file
        if direct.exists():
            return direct

        # Organized location: {definition_method}/{experiment_id}/
        definition_method = self._get_config_value(
            lambda: self.config.domain.definition_method,
            default=None,
            dict_key='DOMAIN_DEFINITION_METHOD'
        )
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default=None,
            dict_key='EXPERIMENT_ID'
        )
        if definition_method and experiment_id:
            nested = catchment_path / str(definition_method) / str(experiment_id) / catchment_file
            if nested.exists():
                return nested

        # The same file name under another subdirectory (e.g. a different
        # experiment id) — only when that is unambiguous.
        matches = sorted(catchment_path.rglob(catchment_file)) if catchment_path.exists() else []
        if len(matches) == 1:
            self.logger.info(f'Using catchment shapefile found at {matches[0]}')
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Catchment shapefile '{catchment_file}' was not found at {direct} or in "
                f"the configured experiment directory, and {len(matches)} files with that "
                f"name exist under {catchment_path} ({', '.join(str(m) for m in matches[:5])}). "
                "Set CATCHMENT_PATH / CATCHMENT_SHP_NAME to select one."
            )

        raise FileNotFoundError(
            f"Catchment shapefile '{catchment_file}' not found under {catchment_path}. "
            "Run domain discretization first, or set CATCHMENT_PATH / CATCHMENT_SHP_NAME."
        )

    def _get_data_path(self, config_key: str, default_subfolder: str) -> Path:
        """
        Resolve a data path from config with default fallback.

        Args:
            config_key: Configuration key for the path
            default_subfolder: Default subfolder under project_dir

        Returns:
            Resolved path
        """
        path_value = self._get_config_value(lambda: None, dict_key=config_key)

        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subfolder

        return Path(path_value)

    def _is_lumped(self) -> bool:
        """
        Check if domain is lumped or distributed.

        Returns:
            True if lumped, False if distributed
        """
        return self._get_config_value(
            lambda: self.config.domain.definition_method,
            dict_key='DOMAIN_DEFINITION_METHOD'
        ) == 'lumped'

    def _get_hru_ids(self) -> Optional[list]:
        """
        Get list of HRU IDs from catchment shapefile (for distributed catchments).

        Returns:
            List of HRU IDs, or None for lumped catchments
        """
        if self._is_lumped():
            return None

        catchment = gpd.read_file(self.catchment_path)
        hru_id_field = self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            default='HRU_ID',
            dict_key='CATCHMENT_SHP_HRUID'
        )

        return catchment[hru_id_field].tolist()

    def _format_results_for_hrus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results with HRU prefixes if needed.

        Args:
            results: Raw results dictionary

        Returns:
            Formatted results (with HRU_ prefixes for distributed catchments)
        """
        if self._is_lumped():
            return results

        # For distributed catchments, results are already formatted by
        # individual processors with HRU prefixes
        return results

    def _write_results_csv(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        filename: str,
    ) -> Optional[Path]:
        """Write processor results as CSV for AttributesNetCDFBuilder.

        For lumped domains, writes a single-row CSV.  For distributed
        domains (results keyed as ``HRU_{id}_var.name``), reshapes into
        one row per HRU with columns matching the variable names — this
        is the format expected by transfer-function regionalization.

        Per-HRU tables carry the HRU id in an ``hru_id`` column and are
        ordered numerically (catchment order for non-numeric ids).  Readers
        must join on that column, never on row position.
        """
        if not results:
            return None

        import re

        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename

        numeric = {
            k: v for k, v in results.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        if not numeric:
            return None

        hru_keys = [k for k in numeric if k.startswith('HRU_')]
        if hru_keys:
            hru_pattern = re.compile(r'^HRU_(.+?)_(.+)$')
            hru_data: Dict[str, Dict[str, float]] = {}
            basin_data: Dict[str, float] = {}
            for k, v in numeric.items():
                m = hru_pattern.match(k)
                if m:
                    hru_id, var_name = m.group(1), m.group(2)
                    hru_data.setdefault(hru_id, {})[var_name] = v
                else:
                    basin_data[k] = v
            if hru_data:
                from symfluence.data.model_ready.hru_tables import (
                    HRU_ID_COLUMN,
                    normalize_hru_id,
                    order_hru_ids,
                )

                df = pd.DataFrame.from_dict(hru_data, orient='index')
                df.index = pd.Index([normalize_hru_id(i) for i in df.index])
                try:
                    catchment_order = self._get_hru_ids()
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    self.logger.debug(f"Could not read catchment HRU order: {exc}")
                    catchment_order = None
                df = df.reindex(order_hru_ids(df.index, catchment_order))
                for col, val in basin_data.items():
                    df[col] = val
                df.insert(0, HRU_ID_COLUMN, list(df.index))
                df.to_csv(out_path, index=False)
                self.logger.info(
                    f"Wrote {len(df.columns)} attributes x {len(df)} HRUs to {out_path}"
                )
                return out_path

        df = pd.DataFrame([numeric])
        df.to_csv(out_path, index=False)
        self.logger.info(f"Wrote {len(numeric)} attributes to {out_path}")
        return out_path
