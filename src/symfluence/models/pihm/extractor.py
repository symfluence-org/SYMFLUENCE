# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Result Extractor

Extracts results from PIHM output files:
- River flux from .rivflx files (channel discharge, m3/s)
- Groundwater head from .gwhead files (m)
- Surface water depth from .surf files (m)

PIHM output files are tab-delimited text with a time column (epoch seconds).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError
from symfluence.core.modeling.base.base_extractor import ModelResultExtractor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.result_extractors.add("PIHM")
class PIHMResultExtractor(ModelResultExtractor):
    """
    Extracts results from PIHM output files.

    Supports extraction of:
    - River flux (channel discharge) from .rivflx files
    - Groundwater head from .gwhead files
    - Surface water depth from .surf files
    """

    def __init__(self, model_name: str = 'PIHM'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating PIHM outputs."""
        return {
            'river_flux': ['*.river.flx1.txt'],
            'groundwater_head': ['*.gw.txt'],
            'surface': ['*.surf.txt'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from PIHM output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('river_flux', 'groundwater_head', 'baseflow')
            **kwargs: Additional args (start_date for time index)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        if variable_type in ('river_flux', 'baseflow'):
            return self._extract_river_flux(output_path, **kwargs)
        elif variable_type == 'groundwater_head':
            return self._extract_gwhead(output_path, **kwargs)
        else:
            raise ValueError(f"Unknown variable type: {variable_type}")

    def _parse_pihm_txt(self, filepath: Path, outlet_segment: int = 1) -> pd.Series:
        """Parse MM-PIHM .txt output file.

        Format: "YYYY-MM-DD HH:MM"<tab>value
        The timestamp is quoted and tab-separated from the numeric value.
        """
        times = []
        values = []
        if not isinstance(outlet_segment, int) or outlet_segment < 1:
            raise ValidationError("PIHM outlet_segment must be a positive one-based column")
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Split on tab: first part is quoted timestamp, second is value
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        ts_str = parts[0].strip('"')
                        # Column 1 is the outlet river segment's discharge. For a
                        # lumped mesh there is only one segment; for a distributed
                        # mesh the generator writes the outlet (DOWN=-3) as segment
                        # 1, so column 1 is total basin discharge either way.
                        # (parts[-1] would grab the LAST segment — a near-dry
                        # headwater on a distributed mesh.)
                        val = float(parts[outlet_segment])
                        times.append(pd.Timestamp(ts_str))
                        values.append(val)
                    except (ValueError, IndexError) as exc:
                        raise ValidationError(f"Invalid PIHM output row in {filepath}: {line}") from exc
        if not times:
            return pd.Series(dtype=float)
        result = pd.Series(values, index=pd.DatetimeIndex(times))
        if not result.index.is_unique or not result.index.is_monotonic_increasing or not np.isfinite(result).all():
            raise ValidationError(f"Invalid PIHM output timestamps or values in {filepath}")
        return result

    def _extract_river_flux(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract river flux time series from MM-PIHM output.

        Looks for .river.flx1.txt (downstream discharge) first.
        """
        if output_path.is_dir():
            # Try river flux file first
            files = sorted(output_path.glob("*.river.flx1.txt"))
            if not files:
                # Fallback to legacy rivflx pattern
                files = sorted(output_path.glob("*.rivflx*"))
            if not files:
                self.logger.warning(f"No river flux files in {output_path}")
                return pd.Series(dtype=float, name='river_flux_m3s')
            data_file = files[0]
        else:
            data_file = output_path

        series = self._parse_pihm_txt(data_file, outlet_segment=kwargs.get('outlet_segment', 1))
        if series.empty:
            return pd.Series(dtype=float, name='river_flux_m3s')

        # MM-PIHM PrintData writes interval means at the END of the interval.
        # Preserve the physical sign (DOWNSTREAM is positive out of the basin).
        interval = kwargs.get('output_interval')
        if interval is None:
            if len(series) < 2:
                raise ValidationError("Single PIHM output row requires output_interval")
            intervals = series.index.to_series().diff().dropna()
            interval = intervals.iloc[0]
            if not (intervals == interval).all():
                raise ValidationError("PIHM discharge output has time gaps")
        interval = pd.Timedelta(interval)
        if interval <= pd.Timedelta(0):
            raise ValidationError("PIHM output_interval must be positive")
        series.index -= interval
        series.name = 'river_flux_m3s'
        return series

    def _extract_gwhead(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract groundwater head time series from .gw.txt file."""
        if output_path.is_dir():
            files = sorted(output_path.glob("*.gw.txt"))
            if not files:
                files = sorted(output_path.glob("*.gwhead*"))
            if not files:
                self.logger.warning(f"No groundwater files in {output_path}")
                return pd.Series(dtype=float, name='gwhead_m')
            data_file = files[0]
        else:
            data_file = output_path

        series = self._parse_pihm_txt(data_file)
        series.name = 'gwhead_m'
        return series
