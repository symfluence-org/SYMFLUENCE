# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Result Extractor

Extracts results from ParFlow .pfb binary output files:
- Pressure head from *.out.press.*.pfb
- Saturation from *.out.satur.*.pfb
- Overland flow sum from *.out.overlandsum.*.pfb

Uses parflowio for .pfb reading (lazy import). Falls back to struct-based
reader when parflowio is not available.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.modeling.base.base_extractor import ModelResultExtractor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)

#: ParFlow ``TimingInfo.DumpInterval`` the preprocessors write when nothing
#: else is configured (hours).
DEFAULT_DUMP_INTERVAL_HOURS = 24.0


def resolve_dump_interval_hours(config, model_key: str = 'parflow') -> float:
    """Return the ParFlow output dump interval (hours) for ``model_key``.

    Single source of truth shared by the preprocessors (which write
    ``TimingInfo.DumpInterval``), the calibration targets, the
    postprocessors and the plotters. ``overlandsum`` output is a volume
    accumulated over one dump interval and the .pfb files are written once
    per dump interval, so every reader must use the same value or flow is
    scaled by ``dump/timestep`` and the time axis is compressed.

    Args:
        config: Typed SYMFLUENCE config (``config.model.<model_key>``) or
            ``None``.
        model_key: ``'parflow'`` or ``'clmparflow'``.
    """
    try:
        model_cfg = getattr(config.model, model_key)
        value = getattr(model_cfg, 'dump_interval_hours', None)
        if value is not None:
            return float(value)
    except (AttributeError, TypeError, ValueError):
        pass
    return DEFAULT_DUMP_INTERVAL_HOURS


def _read_pfb(filepath: Path) -> np.ndarray:
    """Read a ParFlow .pfb binary file.

    Tries parflowio first, falls back to manual struct parsing.

    Returns:
        3D numpy array (nz, ny, nx)
    """
    filepath = Path(filepath)
    try:
        from parflowio.pfb import PFBFile
        pfb = PFBFile(str(filepath))
        pfb.loadHeader()
        pfb.loadData()
        data = pfb.getDataAsArray()
        pfb.close()
        return data
    except ImportError:
        pass

    # Fallback: manual .pfb reader
    # PFB format: header (x0, y0, z0, nx, ny, nz, dx, dy, dz) + subgrid blocks
    with open(filepath, 'rb') as f:
        x0 = struct.unpack('>d', f.read(8))[0]  # noqa: F841
        y0 = struct.unpack('>d', f.read(8))[0]  # noqa: F841
        z0 = struct.unpack('>d', f.read(8))[0]  # noqa: F841
        nx = struct.unpack('>i', f.read(4))[0]
        ny = struct.unpack('>i', f.read(4))[0]
        nz = struct.unpack('>i', f.read(4))[0]
        dx = struct.unpack('>d', f.read(8))[0]  # noqa: F841
        dy = struct.unpack('>d', f.read(8))[0]  # noqa: F841
        dz = struct.unpack('>d', f.read(8))[0]  # noqa: F841
        n_subgrids = struct.unpack('>i', f.read(4))[0]

        data = np.zeros((nz, ny, nx), dtype=np.float64)

        for _ in range(n_subgrids):
            ix = struct.unpack('>i', f.read(4))[0]
            iy = struct.unpack('>i', f.read(4))[0]
            iz = struct.unpack('>i', f.read(4))[0]
            snx = struct.unpack('>i', f.read(4))[0]
            sny = struct.unpack('>i', f.read(4))[0]
            snz = struct.unpack('>i', f.read(4))[0]
            rx = struct.unpack('>i', f.read(4))[0]  # noqa: F841
            ry = struct.unpack('>i', f.read(4))[0]  # noqa: F841
            rz = struct.unpack('>i', f.read(4))[0]  # noqa: F841

            for k in range(snz):
                for j in range(sny):
                    for i in range(snx):
                        val = struct.unpack('>d', f.read(8))[0]
                        data[iz + k, iy + j, ix + i] = val

    return data


@R.result_extractors.add("PARFLOW")
class ParFlowResultExtractor(ModelResultExtractor):
    """
    Extracts results from ParFlow .pfb output files.

    Supports extraction of:
    - Pressure head from .pfb files -> water table depth
    - Saturation from .pfb files
    - Overland flow from .pfb files
    """

    def __init__(self, model_name: str = 'PARFLOW'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating ParFlow outputs."""
        return {
            'pressure': ['*.out.press.*.pfb'],
            'saturation': ['*.out.satur.*.pfb'],
            'overland': ['*.out.overlandsum.*.pfb'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from ParFlow output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('pressure', 'overland_flow', 'subsurface_drainage')
            **kwargs: Additional args (timestep_hours, start_date, dx, dy for unit conversion)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        if variable_type == 'pressure':
            return self._extract_pressure(output_path, **kwargs)
        elif variable_type in ('overland_flow', 'overland'):
            return self._extract_overland_flow(output_path, **kwargs)
        elif variable_type in ('subsurface_drainage', 'baseflow'):
            return self._extract_subsurface_drainage(output_path, **kwargs)
        else:
            raise ValueError(f"Unknown variable type: {variable_type}")

    def _find_pfb_files(
        self, output_path: Path, pattern: str
    ) -> List[Path]:
        """Find and sort PFB files by timestep number."""
        if output_path.is_dir():
            files = sorted(output_path.glob(pattern))
        else:
            files = [output_path]
        return files

    def _get_timestep_from_filename(self, filepath: Path) -> Optional[int]:
        """Extract timestep number from ParFlow output filename.

        ParFlow names: runname.out.press.00001.pfb -> timestep 1
        """
        stem = filepath.stem  # e.g., runname.out.press.00001
        parts = stem.split('.')
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return None

    def _build_time_index(
        self, n_steps: int, **kwargs
    ) -> pd.DatetimeIndex:
        """Build time index from timestep count and config."""
        dump_interval = kwargs.get('dump_interval_hours',
                                   kwargs.get('timestep_hours', 1.0))
        start_date = kwargs.get('start_date', '2000-01-01')
        freq_hours = int(dump_interval)
        return pd.date_range(
            start=start_date,
            periods=n_steps,
            freq=f'{freq_hours}h',
        )

    def _extract_pressure(
        self, output_path: Path, **kwargs
    ) -> pd.Series:
        """Extract pressure head time series from .pfb files.

        For lumped model (1x1x1), returns the single-cell pressure head.
        For multi-layer, returns pressure at the top cell (water table proxy).
        """
        files = self._find_pfb_files(output_path, '*.out.press.*.pfb')
        if not files:
            raise ValueError(f"No pressure .pfb files found in {output_path}")

        # Skip initial condition file (timestep 0) if present
        values = []
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                continue
            data = _read_pfb(f)
            # Top layer pressure (last z index = surface)
            values.append(data[-1, 0, 0])

        date_index = self._build_time_index(len(values), **kwargs)
        return pd.Series(values, index=date_index, name='pressure_head_m')

    def _extract_overland_flow(
        self, output_path: Path, **kwargs
    ) -> pd.Series:
        """Extract overland flow from .pfb files.

        ParFlow overland flow sum is in units of [L^3/T] per cell.
        For lumped mode with dx*dy area, convert to m3/hr, then to m3/s.
        """
        files = self._find_pfb_files(output_path, '*.out.overlandsum.*.pfb')
        if not files:
            self.logger.warning(
                "No overland flow .pfb files found. Returning zeros."
            )
            n = len(self._find_pfb_files(output_path, '*.out.press.*.pfb'))
            n = max(n - 1, 1)  # Exclude IC
            date_index = self._build_time_index(n, **kwargs)
            return pd.Series(0.0, index=date_index, name='overland_flow_m3s')

        values = []
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                continue
            data = _read_pfb(f)
            # Sum all cells for total overland flow
            total = float(np.sum(data))
            values.append(total)

        # ParFlow overlandsum is cumulative flow (m3) over the dump interval.
        # Convert to m3/s: divide by dump interval in seconds.
        dump_interval_hours = kwargs.get('dump_interval_hours',
                                         kwargs.get('timestep_hours', 1.0))
        dump_interval_s = dump_interval_hours * 3600.0
        values_m3s = [v / dump_interval_s for v in values]

        date_index = self._build_time_index(len(values_m3s), **kwargs)
        return pd.Series(values_m3s, index=date_index, name='overland_flow_m3s')

    def _extract_subsurface_drainage(
        self, output_path: Path, **kwargs
    ) -> pd.Series:
        """Compute subsurface drainage from pressure gradient at bottom.

        For lumped mode, estimates baseflow from pressure changes and
        hydraulic conductivity. This is a simplified estimate; for full
        budget tracking, ParFlow's water balance output is preferred.

        Returns drainage in m3/hr.
        """
        files = self._find_pfb_files(output_path, '*.out.press.*.pfb')
        if not files:
            raise ValueError(f"No pressure .pfb files found in {output_path}")

        k_sat = kwargs.get('k_sat', 5.0)
        dx = kwargs.get('dx', 1000.0)
        dy = kwargs.get('dy', 1000.0)
        dz = kwargs.get('dz', 100.0)

        values = []
        prev_pressure = None
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                data = _read_pfb(f)
                prev_pressure = data[-1, 0, 0]  # Top cell
                continue

            data = _read_pfb(f)
            pressure = data[-1, 0, 0]

            # Simplified Darcy flux estimate at bottom boundary
            # Positive pressure head -> saturated -> drainage possible
            if pressure > 0 and prev_pressure is not None:
                # Drainage flux ~ K * gradient (simplified)
                gradient = max(pressure / dz, 0.0)
                flux_m_hr = k_sat * gradient
                drainage_m3_hr = flux_m_hr * dx * dy
            else:
                drainage_m3_hr = 0.0

            values.append(drainage_m3_hr)
            prev_pressure = pressure

        date_index = self._build_time_index(len(values), **kwargs)
        return pd.Series(values, index=date_index, name='subsurface_drainage_m3hr')
