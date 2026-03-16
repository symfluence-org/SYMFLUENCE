# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CAMELS Dataset Observation Handlers

Provides streamflow observation handlers and catchment attribute loaders for
the CAMELS (Catchment Attributes and Meteorology for Large-Sample Studies)
family of datasets:

- CAMELS-US:  671 basins across the contiguous United States
- CAMELS-BR:  897 basins in Brazil
- CAMELS-CL:  516 basins in Chile
- CAMELS-AUS: 222 basins in Australia (also known as CAMELS-OZ)
- CAMELS-GB:  671 basins in Great Britain

Each handler locates streamflow time series from the local CAMELS dataset
directory and processes them to the standard SYMFLUENCE format.

A shared ``load_camels_attributes`` function loads catchment attributes from
any CAMELS variant and maps column names to the standard regionalization
attribute names (``elev_m``, ``precip_mm_yr``, ``aridity``, etc.).

References:
    Addor, N., et al. (2017). The CAMELS data set. HESS, 21(10), 5293-5313.
    Chagas, V.B.P., et al. (2020). CAMELS-BR. ESSD, 12(3), 2075-2098.
    Alvarez-Garreton, C., et al. (2018). CAMELS-CL. HESS, 22(11), 5817-5846.
    Fowler, K.J.A., et al. (2021). CAMELS-AUS. ESSD, 13(8), 3847-3867.
    Coxon, G., et al. (2020). CAMELS-GB. ESSD, 12(4), 2459-2483.
"""
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

# ============================================================================
# Catchment attribute column mappings per CAMELS variant
# ============================================================================

# Maps CAMELS-specific column names → standard regionalization column names
_ATTR_COLUMN_MAP = {
    'camels_us': {
        'gauge_id': 'station_id',
        'elev_mean': 'elev_m',
        'p_mean': 'precip_mm_yr',      # mm/day → needs ×365
        'aridity': 'aridity',
        'frac_snow': 'snow_frac',
        'high_prec_freq': 'high_prec_freq',
        'area_gages2': 'area_km2',
        'gauge_lat': 'latitude',
        'gauge_lon': 'longitude',
    },
    'camels_br': {
        'gauge_id': 'station_id',
        'elev_mean': 'elev_m',
        'p_mean': 'precip_mm_yr',      # mm/day → needs ×365
        'aridity': 'aridity',
        'frac_snow': 'snow_frac',
        'area': 'area_km2',
        'gauge_lat': 'latitude',
        'gauge_lon': 'longitude',
    },
    'camels_cl': {
        'gauge_id': 'station_id',
        'elev_mean': 'elev_m',
        'p_mean_cr2met': 'precip_mm_yr',  # mm/day → needs ×365
        'aridity_cr2met': 'aridity',
        'frac_snow_cr2met': 'snow_frac',
        'area_km2': 'area_km2',
        'gauge_lat': 'latitude',
        'gauge_lon': 'longitude',
    },
    'camels_aus': {
        'station_id': 'station_id',
        'elev_mean': 'elev_m',
        'p_mean': 'precip_mm_yr',      # mm/day → needs ×365
        'aridity': 'aridity',
        'frac_snow': 'snow_frac',
        'catchment_area': 'area_km2',
        'lat_outlet': 'latitude',
        'long_outlet': 'longitude',
    },
    'camels_gb': {
        'gauge_id': 'station_id',
        'elev_mean': 'elev_m',
        'p_mean': 'precip_mm_yr',      # mm/day → needs ×365
        'aridity': 'aridity',
        'frac_snow': 'snow_frac',
        'area': 'area_km2',
        'gauge_lat': 'latitude',
        'gauge_lon': 'longitude',
    },
}

# CAMELS variants where p_mean is in mm/day and needs ×365 conversion
_P_MEAN_DAILY_VARIANTS = {'camels_us', 'camels_br', 'camels_cl', 'camels_aus', 'camels_gb'}


# ============================================================================
# Shared catchment attributes loader
# ============================================================================

def load_camels_attributes(
    camels_dir: Path,
    variant: str = 'camels_us',
    station_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load catchment attributes from a CAMELS dataset directory.

    Reads all available attribute files, merges them, and renames columns to
    the standard regionalization names (``elev_m``, ``precip_mm_yr``, etc.).

    Args:
        camels_dir: Root directory of the CAMELS dataset.
        variant: One of ``'camels_us'``, ``'camels_br'``, ``'camels_cl'``,
                 ``'camels_aus'``, ``'camels_gb'``.
        station_ids: Optional list of station IDs to filter to.

    Returns:
        DataFrame with one row per catchment, columns using standard names.
    """
    variant = variant.lower().replace('-', '_')

    # Locate attribute files based on variant
    attr_df = _read_attribute_files(camels_dir, variant)

    if attr_df is None or attr_df.empty:
        raise FileNotFoundError(
            f"No attribute files found for {variant} in {camels_dir}. "
            f"Expected directory structure varies by variant."
        )

    # Rename columns to standard names
    col_map = _ATTR_COLUMN_MAP.get(variant, {})
    for orig, standard in col_map.items():
        if orig in attr_df.columns:
            attr_df = attr_df.rename(columns={orig: standard})

    # Convert p_mean from mm/day to mm/yr if needed
    if variant in _P_MEAN_DAILY_VARIANTS and 'precip_mm_yr' in attr_df.columns:
        if attr_df['precip_mm_yr'].median() < 50:  # Likely mm/day
            attr_df['precip_mm_yr'] = attr_df['precip_mm_yr'] * 365.25

    # Derive temp_C from elevation if not present
    if 'temp_C' not in attr_df.columns and 'elev_m' in attr_df.columns:
        attr_df['temp_C'] = 15.0 - 0.0065 * attr_df['elev_m']

    # Filter to requested stations
    if station_ids is not None:
        id_col = 'station_id' if 'station_id' in attr_df.columns else attr_df.columns[0]
        attr_df[id_col] = attr_df[id_col].astype(str)
        attr_df = attr_df[attr_df[id_col].isin([str(s) for s in station_ids])]

    return attr_df.reset_index(drop=True)


def _read_attribute_files(camels_dir: Path, variant: str) -> Optional[pd.DataFrame]:
    """Read and merge attribute files for a CAMELS variant."""
    camels_dir = Path(camels_dir)

    if variant == 'camels_us':
        # CAMELS-US: camels_*/camels_*.txt files in subdirectories
        attr_dir = camels_dir / 'camels_attributes_v2.0'
        if not attr_dir.exists():
            # Try alternative layout
            attr_dir = camels_dir
        txt_files = list(attr_dir.glob('camels_*.txt'))
        if not txt_files:
            txt_files = list(camels_dir.rglob('camels_*.txt'))
        return _merge_attr_files(txt_files, sep=';', id_col='gauge_id')

    elif variant == 'camels_br':
        # CAMELS-BR: CSV files in subdirectories
        attr_files = list(camels_dir.rglob('camels_br_*.txt'))
        if not attr_files:
            attr_files = list(camels_dir.rglob('*.txt'))
        return _merge_attr_files(attr_files, sep='\t', id_col='gauge_id')

    elif variant == 'camels_cl':
        # CAMELS-CL: single or multiple txt files
        attr_files = list(camels_dir.rglob('camels_cl_*.txt'))
        if not attr_files:
            attr_files = list(camels_dir.rglob('*.txt'))
        return _merge_attr_files(attr_files, sep='\t', id_col='gauge_id')

    elif variant == 'camels_aus':
        # CAMELS-AUS: CSV files
        attr_files = list(camels_dir.rglob('CAMELS_AUS_Attributes*.csv'))
        if not attr_files:
            attr_files = list(camels_dir.rglob('*attributes*.csv'))
        if not attr_files:
            attr_files = list(camels_dir.rglob('*Attributes*.csv'))
        return _merge_attr_files(attr_files, sep=',', id_col='station_id')

    elif variant == 'camels_gb':
        # CAMELS-GB: CSV files
        attr_files = list(camels_dir.rglob('CAMELS_GB_*.csv'))
        if not attr_files:
            attr_files = list(camels_dir.rglob('*attributes*.csv'))
        return _merge_attr_files(attr_files, sep=',', id_col='gauge_id')

    return None


def _merge_attr_files(
    files: List[Path],
    sep: str = ',',
    id_col: str = 'gauge_id',
) -> Optional[pd.DataFrame]:
    """Merge multiple attribute files on a shared ID column."""
    if not files:
        return None

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, sep=sep, dtype={id_col: str})
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).str.strip()
            dfs.append(df)
        except Exception:  # noqa: BLE001 — skip unparseable files
            continue

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        if id_col in merged.columns and id_col in df.columns:
            # Drop duplicate columns before merge
            overlap = set(merged.columns) & set(df.columns) - {id_col}
            df = df.drop(columns=list(overlap), errors='ignore')
            merged = merged.merge(df, on=id_col, how='outer')
        else:
            # No shared ID column — just concat columns
            merged = pd.concat([merged, df], axis=1)

    return merged


# ============================================================================
# Base CAMELS streamflow handler
# ============================================================================

class _BaseCAMELSStreamflowHandler(BaseObservationHandler):
    """Base class for all CAMELS streamflow handlers.

    Subclasses set ``VARIANT``, ``SOURCE_INFO``, and override
    ``_find_streamflow_file`` and ``_parse_streamflow_file`` as needed.
    """

    obs_type = "streamflow"
    VARIANT: str = ""
    SOURCE_INFO: Dict[str, str] = {}

    def acquire(self) -> Path:
        """Locate the raw streamflow file for the configured station."""
        station_id = self._get_config_value(
            lambda: self.config.evaluation.streamflow.station_id,
            dict_key='STATION_ID'
        )
        camels_path_str = self._config_value(
            f'{self.VARIANT.upper()}_PATH', 'CAMELS_PATH',
            typed_path=lambda: getattr(self.config.data, f'{self.VARIANT.lower()}_path', None),
            default=None
        )

        if not station_id:
            raise ValueError(f"STATION_ID required for {self.source_name}")
        if not camels_path_str:
            raise ValueError(
                f"{self.VARIANT.upper()}_PATH (or CAMELS_PATH) required for {self.source_name}"
            )

        camels_dir = Path(camels_path_str)
        raw_file = self._find_streamflow_file(camels_dir, str(station_id))

        if raw_file is None or not raw_file.exists():
            raise FileNotFoundError(
                f"{self.source_name} streamflow file not found for station "
                f"{station_id} in {camels_dir}"
            )

        # Copy to project directory
        dest_dir = self.project_observations_dir / "streamflow" / "raw_data"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{self.VARIANT}_{station_id}_raw.csv"
        shutil.copy2(raw_file, dest_file)

        self.logger.info(f"Located {self.source_name} data → {dest_file}")
        return dest_file

    def process(self, input_path: Path) -> Path:
        """Process raw streamflow data to standard format."""
        self.logger.info(f"Processing {self.source_name} streamflow for {self.domain_name}")

        df = self._parse_streamflow_file(input_path)

        if df is None or df.empty:
            self.logger.warning(f"No data parsed from {input_path}")
            return input_path

        # Drop missing
        df = df.dropna(subset=['discharge_cms'])

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=30)

        # Filter to experiment period
        resampled = resampled.loc[self.start_date:self.end_date]

        # Save
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        resampled.to_csv(output_file, header=True, index_label='datetime')

        self.logger.info(f"{self.source_name} processing complete: {output_file}")
        return output_file

    # --- Hooks for subclasses -----------------------------------------------

    def _find_streamflow_file(self, camels_dir: Path, station_id: str) -> Optional[Path]:
        """Locate the streamflow time-series file for a station."""
        raise NotImplementedError

    def _parse_streamflow_file(self, path: Path) -> Optional[pd.DataFrame]:
        """Parse a raw streamflow file into a DataFrame with datetime index
        and ``discharge_cms`` column."""
        raise NotImplementedError


# ============================================================================
# CAMELS-US
# ============================================================================

@ObservationRegistry.register('camels_us_streamflow')
class CAMELSUSStreamflowHandler(_BaseCAMELSStreamflowHandler):
    """CAMELS-US streamflow handler (671 USGS basins).

    Configuration:
        STATION_ID: USGS gauge ID (e.g., '01013500')
        CAMELS_US_PATH: Root directory of the CAMELS-US dataset
    """

    source_name = "CAMELS_US"
    VARIANT = "camels_us"
    SOURCE_INFO = {
        'source': 'CAMELS-US',
        'source_doi': '10.5065/D6MW2F4D',
    }

    def _find_streamflow_file(self, camels_dir: Path, station_id: str) -> Optional[Path]:
        # CAMELS-US: basin_dataset_public_v1p2/usgs_streamflow/{basin_id}_streamflow_qc.txt
        station_id = station_id.zfill(8)
        candidates = [
            camels_dir / 'usgs_streamflow' / f'{station_id}_streamflow_qc.txt',
            camels_dir / 'basin_dataset_public_v1p2' / 'usgs_streamflow' / f'{station_id}_streamflow_qc.txt',
        ]
        # Also search recursively
        for c in candidates:
            if c.exists():
                return c
        found = list(camels_dir.rglob(f'*{station_id}*streamflow*.txt'))
        return found[0] if found else None

    def _parse_streamflow_file(self, path: Path) -> Optional[pd.DataFrame]:
        # Format: gauge_id  year  month  day  streamflow(cfs)  QC_flag
        df = pd.read_csv(path, sep=r'\s+', header=None,
                         names=['gauge_id', 'year', 'month', 'day', 'discharge_cfs', 'qc_flag'])
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.set_index('datetime', inplace=True)
        # Convert cfs → cms
        CFS_TO_CMS = 0.028316846592
        df['discharge_cms'] = df['discharge_cfs'] * CFS_TO_CMS
        df.loc[df['discharge_cfs'] < 0, 'discharge_cms'] = np.nan
        return df[['discharge_cms']]


# ============================================================================
# CAMELS-BR
# ============================================================================

@ObservationRegistry.register('camels_br_streamflow')
class CAMELSBRStreamflowHandler(_BaseCAMELSStreamflowHandler):
    """CAMELS-BR streamflow handler (897 Brazilian basins).

    Configuration:
        STATION_ID: ANA gauge ID (e.g., '10011000')
        CAMELS_BR_PATH: Root directory of the CAMELS-BR dataset
    """

    source_name = "CAMELS_BR"
    VARIANT = "camels_br"
    SOURCE_INFO = {
        'source': 'CAMELS-BR',
        'source_doi': '10.5281/zenodo.3709337',
    }

    def _find_streamflow_file(self, camels_dir: Path, station_id: str) -> Optional[Path]:
        # CAMELS-BR: 3_CAMELS_BR_streamflow_m3s/{station_id}_streamflow_m3s.txt
        candidates = [
            camels_dir / '3_CAMELS_BR_streamflow_m3s' / f'{station_id}_streamflow_m3s.txt',
            camels_dir / 'streamflow_m3s' / f'{station_id}_streamflow_m3s.txt',
        ]
        for c in candidates:
            if c.exists():
                return c
        found = list(camels_dir.rglob(f'*{station_id}*streamflow*.txt'))
        if not found:
            found = list(camels_dir.rglob(f'*{station_id}*streamflow*.csv'))
        return found[0] if found else None

    def _parse_streamflow_file(self, path: Path) -> Optional[pd.DataFrame]:
        # CAMELS-BR: date  streamflow_m3s (space or tab separated)
        for sep in [r'\s+', ',', ';']:
            try:
                df = pd.read_csv(path, sep=sep)
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            return None

        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'data' in col.lower():
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Find discharge column (already in m3/s)
        q_col = None
        for col in df.columns:
            if 'streamflow' in col.lower() or 'vazao' in col.lower() or 'discharge' in col.lower():
                q_col = col
                break
        if q_col is None:
            # Last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            q_col = numeric_cols[-1] if len(numeric_cols) > 0 else None

        if q_col is None:
            return None

        df['discharge_cms'] = pd.to_numeric(df[q_col], errors='coerce')
        df.loc[df['discharge_cms'] < 0, 'discharge_cms'] = np.nan
        return df[['discharge_cms']]


# ============================================================================
# CAMELS-CL
# ============================================================================

@ObservationRegistry.register('camels_cl_streamflow')
class CAMELSCLStreamflowHandler(_BaseCAMELSStreamflowHandler):
    """CAMELS-CL streamflow handler (516 Chilean basins).

    Configuration:
        STATION_ID: DGA gauge ID (e.g., '1010001')
        CAMELS_CL_PATH: Root directory of the CAMELS-CL dataset
    """

    source_name = "CAMELS_CL"
    VARIANT = "camels_cl"
    SOURCE_INFO = {
        'source': 'CAMELS-CL',
        'source_doi': '10.5281/zenodo.1135720',
    }

    def _find_streamflow_file(self, camels_dir: Path, station_id: str) -> Optional[Path]:
        # CAMELS-CL: 2_CAMELScl_streamflow_m3s/{station_id}_streamflow_m3s.txt
        candidates = [
            camels_dir / '2_CAMELScl_streamflow_m3s' / f'{station_id}_streamflow_m3s.txt',
            camels_dir / 'streamflow_m3s' / f'{station_id}_streamflow_m3s.txt',
        ]
        for c in candidates:
            if c.exists():
                return c
        found = list(camels_dir.rglob(f'*{station_id}*streamflow*.txt'))
        if not found:
            found = list(camels_dir.rglob(f'*{station_id}*.txt'))
        return found[0] if found else None

    def _parse_streamflow_file(self, path: Path) -> Optional[pd.DataFrame]:
        # CAMELS-CL format similar to CAMELS-BR
        for sep in [r'\s+', ',', ';']:
            try:
                df = pd.read_csv(path, sep=sep)
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            return None

        date_col = self._find_col(list(df.columns), ['date', 'fecha'])
        if date_col is None:
            date_col = df.columns[0]

        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        q_col = self._find_col(list(df.columns), ['streamflow', 'caudal', 'discharge'])
        if q_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            q_col = numeric_cols[-1] if len(numeric_cols) > 0 else None
        if q_col is None:
            return None

        df['discharge_cms'] = pd.to_numeric(df[q_col], errors='coerce')
        df.loc[df['discharge_cms'] < 0, 'discharge_cms'] = np.nan
        return df[['discharge_cms']]


# ============================================================================
# CAMELS-AUS
# ============================================================================

@ObservationRegistry.register('camels_aus_streamflow')
class CAMELSAUSStreamflowHandler(_BaseCAMELSStreamflowHandler):
    """CAMELS-AUS streamflow handler (222 Australian basins).

    Configuration:
        STATION_ID: BoM gauge ID (e.g., '912101A')
        CAMELS_AUS_PATH: Root directory of the CAMELS-AUS dataset
    """

    source_name = "CAMELS_AUS"
    VARIANT = "camels_aus"
    SOURCE_INFO = {
        'source': 'CAMELS-AUS',
        'source_doi': '10.5281/zenodo.4446138',
    }

    def _find_streamflow_file(self, camels_dir: Path, station_id: str) -> Optional[Path]:
        # CAMELS-AUS: 03_streamflow/{station_id}_streamflow_MLd.csv
        candidates = [
            camels_dir / '03_streamflow' / f'{station_id}_streamflow_MLd.csv',
            camels_dir / 'streamflow' / f'{station_id}_streamflow_MLd.csv',
        ]
        for c in candidates:
            if c.exists():
                return c
        found = list(camels_dir.rglob(f'*{station_id}*streamflow*.csv'))
        if not found:
            found = list(camels_dir.rglob(f'*{station_id}*.csv'))
        return found[0] if found else None

    def _parse_streamflow_file(self, path: Path) -> Optional[pd.DataFrame]:
        df = pd.read_csv(path)

        date_col = self._find_col(list(df.columns), ['date', 'year'])
        if date_col is None:
            # CAMELS-AUS sometimes has year/month/day columns
            if all(c in df.columns for c in ['year', 'month', 'day']):
                df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
            else:
                df['datetime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # CAMELS-AUS streamflow is in ML/day — convert to m3/s
        # 1 ML = 1000 m3, 1 day = 86400 s → ML/day × (1000/86400) = m3/s
        ML_DAY_TO_CMS = 1000.0 / 86400.0
        q_col = self._find_col(list(df.columns), ['streamflow', 'discharge'])
        if q_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            q_col = numeric_cols[-1] if len(numeric_cols) > 0 else None
        if q_col is None:
            return None

        df['discharge_cms'] = pd.to_numeric(df[q_col], errors='coerce') * ML_DAY_TO_CMS
        df.loc[df['discharge_cms'] < 0, 'discharge_cms'] = np.nan
        return df[['discharge_cms']]


# ============================================================================
# CAMELS-GB
# ============================================================================

@ObservationRegistry.register('camels_gb_streamflow')
class CAMELSGBStreamflowHandler(_BaseCAMELSStreamflowHandler):
    """CAMELS-GB streamflow handler (671 British basins).

    Configuration:
        STATION_ID: NRFA gauge ID (e.g., '33034')
        CAMELS_GB_PATH: Root directory of the CAMELS-GB dataset
    """

    source_name = "CAMELS_GB"
    VARIANT = "camels_gb"
    SOURCE_INFO = {
        'source': 'CAMELS-GB',
        'source_doi': '10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9',
    }

    def _find_streamflow_file(self, camels_dir: Path, station_id: str) -> Optional[Path]:
        # CAMELS-GB: data/timeseries/CAMELS_GB_hydromet_timeseries_{station_id}_*.csv
        candidates = [
            camels_dir / 'data' / 'timeseries' / f'CAMELS_GB_hydromet_timeseries_{station_id}_19701001-20150930.csv',
        ]
        for c in candidates:
            if c.exists():
                return c
        found = list(camels_dir.rglob(f'*{station_id}*.csv'))
        return found[0] if found else None

    def _parse_streamflow_file(self, path: Path) -> Optional[pd.DataFrame]:
        df = pd.read_csv(path)

        date_col = self._find_col(list(df.columns), ['date'])
        if date_col is None:
            date_col = df.columns[0]

        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # CAMELS-GB: discharge_vol (m3/s) or discharge_spec (mm/day)
        q_col = self._find_col(list(df.columns), ['discharge_vol', 'discharge_spec', 'discharge'])
        if q_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            q_col = numeric_cols[-1] if len(numeric_cols) > 0 else None
        if q_col is None:
            return None

        df['discharge_cms'] = pd.to_numeric(df[q_col], errors='coerce')
        df.loc[df['discharge_cms'] < 0, 'discharge_cms'] = np.nan
        return df[['discharge_cms']]
