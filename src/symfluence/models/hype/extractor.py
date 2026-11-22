# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE Result Extractor.

Handles extraction of simulation results from HYPE model outputs.
HYPE outputs are primarily text-based (timeCOUT.txt) but may also
use mizuRoute for routing.
"""
from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor

_logger = logging.getLogger(__name__)


def translate_reach_to_subid(reach: Any, map_path: Optional[Path], logger=None) -> Optional[str]:
    """HYPE subid (as a column name) for an original SIM_REACH_ID, via subid_map.csv.

    GeoData subids are shifted (+1) for 0-based input ids and replaced by band
    ids under elevation banding, so the configured id must be translated before
    it is looked up in HYPE output. Returns None when there is no map or no hit.
    """
    log = logger or _logger
    if reach is None or map_path is None or not Path(map_path).exists():
        return None
    try:
        mapping = pd.read_csv(map_path)
        hit = mapping.loc[mapping['original_id'] == int(reach), 'hype_subid']
    except (OSError, ValueError, KeyError) as e:
        log.warning(f"Could not read {map_path}: {e}")
        return None
    if not len(hit):
        return None
    return str(int(hit.iloc[0]))


def find_subid_map(output_file: Path) -> Optional[Path]:
    """Locate ``settings/HYPE/subid_map.csv`` in a project enclosing ``output_file``."""
    for parent in Path(output_file).resolve().parents:
        candidate = parent / 'settings' / 'HYPE' / 'subid_map.csv'
        if candidate.exists():
            return candidate
    return None


def select_outlet_column(df: pd.DataFrame, reach: Any = None,
                         map_path: Optional[Path] = None, logger=None) -> Optional[Hashable]:
    """Pick the outlet sub-basin column of a HYPE time file (DATE-indexed frame).

    Order: SIM_REACH_ID translated through subid_map.csv, the raw SIM_REACH_ID,
    then the column with the highest mean discharge (the outlet of a nested
    network). Mirrors ``HYPEPostProcessor._get_flow_column``.
    """
    log = logger or _logger
    columns = [str(c) for c in df.columns]
    translated = translate_reach_to_subid(reach, map_path, log)
    if translated is not None and translated in columns:
        if translated != str(reach):
            log.info(f"SIM_REACH_ID {reach} is HYPE subid {translated}")
        return df.columns[columns.index(translated)]
    if reach is not None and str(reach) in columns:
        return df.columns[columns.index(str(reach))]
    numeric = df.select_dtypes(include=['number'])
    if numeric.shape[1] == 0:
        return None
    if numeric.shape[1] == 1:
        return numeric.columns[0]
    return numeric.mean().idxmax()


class HYPEResultExtractor(ModelResultExtractor):
    """HYPE-specific result extraction.

    Handles HYPE's unique output characteristics:
    - File format: timeCOUT.txt (text file)
    - Variable naming: COUT (computed outflow)
    - Routing: Can use mizuRoute for additional routing
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for HYPE outputs."""
        return {
            'streamflow': [
                'timeCOUT.txt',
                'HYPE/timeCOUT.txt',
                '**/timeCOUT.txt',
                # Fallback to mizuRoute
                'mizuRoute/*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get HYPE variable names for different types."""
        variable_mapping = {
            'streamflow': ['COUT', 'streamflow', 'discharge'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from HYPE output.

        Args:
            output_file: Path to HYPE output file (txt or nc)
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable cannot be extracted
        """
        if output_file.suffix == '.txt':
            # HYPE text output
            return self._extract_from_text(
                output_file,
                sim_reach_id=kwargs.get('sim_reach_id'),
                subid_map_path=kwargs.get('subid_map_path'),
            )
        elif output_file.suffix == '.nc':
            # mizuRoute NetCDF output
            import xarray as xr

            with xr.open_dataset(output_file, engine='h5netcdf') as ds:
                # Use mizuRoute variables
                for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']:
                    if var_name in ds.variables:
                        var = ds[var_name]
                        if 'seg' in var.dims or 'reachID' in var.dims:
                            # Shared reach rule (SIM_REACH_ID, topology
                            # outlet, else highest mean flow)
                            from symfluence.core.modeling.utilities.reach_selection import (
                                select_outlet_series,
                                topology_candidates,
                            )
                            return select_outlet_series(
                                ds, var, kwargs.get('sim_reach_id'),
                                topology_candidates(near=output_file),
                            )

        raise ValueError(f"Could not extract {variable_type} from {output_file}")

    def _extract_from_text(self, output_file: Path, sim_reach_id=None,
                           subid_map_path: Optional[Path] = None) -> pd.Series:
        """Extract outlet streamflow from a HYPE ``timeCOUT.txt`` file.

        HYPE writes one comment line, then a ``DATE<TAB>subid...`` header, then
        one row per timestep. Each column is the outflow of one sub-basin, so
        the columns must never be summed (nested sub-basins would be counted
        several times). The outlet column is chosen as in
        ``HYPEPostProcessor._get_flow_column``: ``SIM_REACH_ID`` translated
        through ``settings/HYPE/subid_map.csv``, else the raw id, else the
        column with the highest mean discharge (the basin outlet).

        Args:
            output_file: Path to timeCOUT.txt
            sim_reach_id: Configured SIM_REACH_ID (original id), optional
            subid_map_path: Path to subid_map.csv; searched for in the
                ``settings/HYPE`` directory of the enclosing project if omitted

        Returns:
            Time series of outlet streamflow (m3/s) indexed by date
        """
        try:
            with open(output_file, encoding='utf-8') as f:
                first = f.readline()
            skip = 0 if first.strip().upper().startswith('DATE') else 1
            df = pd.read_csv(output_file, sep='\t', skiprows=skip, skipinitialspace=True)
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if c.upper() == 'DATE'), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = 'time'
            df = df.apply(pd.to_numeric, errors='coerce')
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ModelExecutionError(f"Failed to parse HYPE output file {output_file}: {e}") from e

        if df.empty or df.shape[1] == 0:
            raise ModelExecutionError(f"No sub-basin columns in HYPE output file {output_file}")

        if subid_map_path is None:
            subid_map_path = find_subid_map(output_file)
        column = select_outlet_column(df, sim_reach_id, subid_map_path)
        if column is None:
            raise ModelExecutionError(f"Could not select an outlet column from {output_file}")
        series = df[column]
        series.name = 'COUT'
        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """HYPE outputs are in m³/s."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """HYPE aggregates internally or uses routing."""
        return 'sum'  # Sum across subbasins if multiple
