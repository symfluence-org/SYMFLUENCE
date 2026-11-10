# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Result Extractor.

Extracts results from GSFLOW output files including PRMS statvar
output and MODFLOW listing/head files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor

logger = logging.getLogger(__name__)


#: PRMS statvar fallback is only meaningful for PRMS-only GSFLOW runs: in
#: COUPLED mode basin_cfs is ~0 because flow is routed through MODFLOW SFR.
_CSV_FLOW_COLUMNS = ('streamout_q', 'streamflow_out', 'basinstreamflow_q')


def read_gsflow_csv_streamflow(csv_file: Path) -> Optional[pd.Series]:
    """Basin outlet streamflow (m3/s) from the GSFLOW budget CSV.

    Uses StreamOut_Q (flow leaving the SFR network, m3/day). Returns None
    when the file has no streamflow column or any budget column (``*_S``,
    ``*_Q``) is non-finite (a numerically failed run).
    """
    import numpy as np

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    flow_col = next((c for c in df.columns if c.lower() in _CSV_FLOW_COLUMNS), None)
    if flow_col is None:
        return None
    for column in [c for c in df if c == flow_col or c.endswith(('_S', '_Q'))]:
        if not np.isfinite(pd.to_numeric(df[column], errors='coerce')).all():
            logger.error(f"Non-finite GSFLOW budget variable {column} in {csv_file}")
            return None
    idx = pd.to_datetime(df[date_col])
    series = pd.Series(pd.to_numeric(df[flow_col], errors='coerce').values,
                       index=idx, name='GSFLOW_discharge_cms') / 86400.0
    return series.dropna()


def extract_gsflow_streamflow(
    sim_dir: Path, settings_dir: Optional[Path] = None,
) -> Optional[pd.Series]:
    """Basin outlet streamflow (m3/s) of a GSFLOW run.

    The one rule shared by the calibration worker and the postprocessor:
    the budget CSV (``gsflow.csv`` StreamOut_Q) when present -- the routed
    streamflow in COUPLED mode -- else the PRMS statvar via the shared PRMS
    parser (PRMS-only mode).

    Raises:
        ModelExecutionError: if a budget CSV exists but holds no valid streamflow
            (the run failed; do not silently fall back to statvar).
    """
    sim_dir = Path(sim_dir)
    csv_files = (sorted(sim_dir.glob('gsflow.csv'))
                 or sorted(sim_dir.glob('gsflow_*.csv'))
                 or sorted(sim_dir.glob('*.csv')))
    if csv_files:
        series = read_gsflow_csv_streamflow(csv_files[0])
        if series is None or len(series) == 0:
            raise ModelExecutionError(f'Invalid coupled streamflow/budget CSV: {csv_files[0]}')
        return series

    statvars = sorted(sim_dir.glob('statvar*'))
    if not statvars and settings_dir:
        statvars = sorted(Path(settings_dir).glob('statvar*'))
    if not statvars:
        return None
    from symfluence.models.prms.statvar import read_statvar_streamflow
    return read_statvar_streamflow(statvars[0]).rename('GSFLOW_discharge_cms')


class GSFLOWResultExtractor(ModelResultExtractor):
    """Extract results from GSFLOW output files."""

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Return file patterns for GSFLOW outputs."""
        return {
            'streamflow': ['statvar*', '*.csv', 'gsflow_*.csv'],
            'et': ['statvar*'],
            'soil_moisture': ['statvar*'],
            'heads': ['*.hds', '*.bhd'],
            'water_budget': ['*.lst', 'gsflow_*.out'],
        }

    _variable_name_map: Dict[str, List[str]] = {
        'streamflow': ['seg_outflow', 'basin_cfs', 'basin_cms', 'streamflow_cfs'],
        'et': ['hru_actet', 'potet', 'basin_actet'],
        'soil_moisture': ['soil_moist', 'soil_rechr', 'basin_soil_moist'],
        'heads': ['head'],
        'baseflow': ['gw_flow', 'sfr_flow'],
    }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Return GSFLOW variable names for a given type."""
        return self._variable_name_map.get(variable_type, [variable_type])

    def extract_variable(
        self, output_file: Path, variable_type: str, **kwargs
    ) -> pd.Series:
        """Extract a variable from GSFLOW output."""
        if not output_file.exists():
            raise ValueError(f"Output file not found: {output_file}")

        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                var_names = self.get_variable_names(variable_type)
                for var in var_names:
                    if var in df.columns:
                        return df[var]
            else:
                result = self._parse_statvar_file(output_file, variable_type)
                if result is not None:
                    return result
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Error extracting {variable_type}: {e}") from e

        raise ValueError(f"Variable '{variable_type}' not found in {output_file}")

    def _parse_statvar_file(
        self, statvar_file: Path, variable_type: str
    ) -> Optional[pd.Series]:
        """Parse PRMS/GSFLOW statvar.dat format (shared PRMS parser)."""
        from symfluence.models.prms.statvar import parse_statvar, read_statvar_streamflow
        try:
            if variable_type == 'streamflow':
                return read_statvar_streamflow(statvar_file).rename(f'GSFLOW_{variable_type}')
            header, df = parse_statvar(statvar_file)
            counts = dict(header)
            for name in self.get_variable_names(variable_type):
                if name in counts:
                    cols = [name] if counts[name] == 1 else [f"{name}_{i + 1}" for i in range(counts[name])]
                    return df[cols].mean(axis=1).rename(f'GSFLOW_{variable_type}')
        except (OSError, ModelExecutionError) as e:
            logger.error(f"Error parsing statvar: {e}", exc_info=True)
        return None

    def extract_streamflow(
        self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs
    ) -> pd.Series:
        """Extract streamflow from GSFLOW output."""
        result = extract_gsflow_streamflow(Path(output_dir), kwargs.get('settings_dir'))
        if result is None:
            raise FileNotFoundError(f"No GSFLOW output found in {output_dir}")
        return result

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """GSFLOW seg_outflow is converted to cms during extraction."""
        return variable_type not in ['streamflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type in ['streamflow']:
            return 'sum'
        return 'mean'
