# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Observations store builder for the model-ready data store.

Converts heterogeneous observation CSV files into a single grouped
NetCDF4 file at ``data/model_ready/observations/{domain}_observations.nc``.
Each observation type (streamflow, snow, ET, ...) lives in its own
NetCDF group with its own spatial dimension.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.modeling.model_ready.cf_conventions import CF_STANDARD_NAMES, build_global_attrs

from .source_metadata import SourceMetadata

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)


# Group definitions — maps group name to spatial dim & id variable.
OBSERVATION_GROUPS: Dict[str, Dict[str, str]] = {
    'streamflow':                {'spatial_dim': 'gauge',   'id_var': 'gauge_id'},
    'snow':                      {'spatial_dim': 'hru',     'id_var': 'hru_id'},
    'et':                        {'spatial_dim': 'hru',     'id_var': 'hru_id'},
    'soil_moisture':             {'spatial_dim': 'station', 'id_var': 'station_id'},
    'terrestrial_water_storage': {'spatial_dim': 'basin',   'id_var': 'basin_id'},
}


class ObservationsNetCDFBuilder:
    """Build grouped NetCDF observations from existing CSV files.

    Parameters
    ----------
    project_dir : Path
        Root of the SYMFLUENCE domain directory.
    domain_name : str
        Name of the hydrological domain.
    config : SymfluenceConfig or dict, optional
        Typed config or legacy flat dict.
    config_dict : dict, optional
        Deprecated. Use ``config`` instead.
    """

    def __init__(
        self,
        project_dir: Path,
        domain_name: str,
        config: Optional[Union['SymfluenceConfig', dict]] = None,
        config_dict: Optional[dict] = None,
    ) -> None:
        """Initialise the observations builder.

        Args:
            project_dir: Root of the SYMFLUENCE domain directory.
            domain_name: Name of the hydrological domain.
            config: Typed config or legacy flat dict.
            config_dict: Deprecated. Use *config* instead.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        self._config = config if config is not None else config_dict

        self.obs_dir = resolve_data_subdir(project_dir, 'observations')
        self.target_dir = project_dir / 'data' / 'model_ready' / 'observations'

    def _cfg(self, key: str, default=None):
        """Get config value from typed config or legacy dict."""
        cfg = self._config
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return cfg.get(key, default)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Optional[Path]:
        """Build the observations NetCDF file.

        Returns the output path on success, ``None`` if no observations
        could be found.
        """
        if not self.obs_dir.exists():
            logger.info("Skipping observations store: no observations/ dir")
            return None

        self.target_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.target_dir / f'{self.domain_name}_observations.nc'

        try:
            import netCDF4
        except ImportError:
            logger.warning("netCDF4 not available; cannot build observations store")
            return None

        groups_written = 0

        with netCDF4.Dataset(str(out_path), 'w', format='NETCDF4') as root:
            # Global attributes
            root.setncatts(build_global_attrs(
                domain_name=self.domain_name,
                title=f'{self.domain_name} observations',
                history='Built by ObservationsNetCDFBuilder',
            ))

            # Build each group
            if self._build_streamflow_group(root):
                groups_written += 1
            if self._build_snow_group(root):
                groups_written += 1
            if self._build_et_group(root):
                groups_written += 1
            if self._build_soil_moisture_group(root):
                groups_written += 1
            if self._build_tws_group(root):
                groups_written += 1

        if groups_written == 0:
            logger.info("No observation data found to include; removing empty file")
            out_path.unlink(missing_ok=True)
            return None

        logger.info(
            "Observations store built: %d groups in %s", groups_written, out_path
        )
        return out_path

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_streamflow_group(self, root) -> bool:
        """Build /streamflow/ group from preprocessed CSV(s)."""
        # Single-gauge path
        preprocessed = self.obs_dir / 'streamflow' / 'preprocessed'
        single_csv = preprocessed / f'{self.domain_name}_streamflow_processed.csv'

        # Multi-gauge path
        multi_dir_cfg = self._cfg('MULTI_GAUGE_OBS_DIR')
        multi_dir = Path(multi_dir_cfg) if multi_dir_cfg else None

        all_series: Dict[str, pd.Series] = {}

        if multi_dir and multi_dir.exists():
            for csv_file in sorted(multi_dir.glob('ID_*.csv')):
                gauge_id = csv_file.stem.replace('ID_', '')
                series = self._read_timeseries_csv(csv_file)
                if series is not None:
                    all_series[gauge_id] = series
        elif single_csv.exists():
            series = self._read_timeseries_csv(single_csv)
            if series is not None:
                all_series[self.domain_name] = series
        else:
            # Fallback: any CSV in preprocessed
            if preprocessed.exists():
                for csv_file in sorted(preprocessed.glob('*.csv')):
                    series = self._read_timeseries_csv(csv_file)
                    if series is not None:
                        all_series[csv_file.stem] = series
                        break  # Use first found

        if not all_series:
            return False

        self._write_group(
            root,
            group_name='streamflow',
            series_dict=all_series,
            var_name='discharge_cms',
            spatial_dim='gauge',
            id_var='gauge_id',
            source_meta=self._get_handler_source_meta('streamflow'),
        )
        return True

    def _build_snow_group(self, root) -> bool:
        """Build /snow/ group from SWE or SCA observations."""
        candidates = [
            self.obs_dir / 'snow' / 'swe' / 'processed',
            self.obs_dir / 'snow' / 'swe' / 'preprocessed',
            self.obs_dir / 'snow' / 'preprocessed',
            self.obs_dir / 'snow' / 'processed',
        ]
        return self._build_generic_group(
            root, 'snow', candidates, 'swe', 'hru', 'hru_id'
        )

    def _build_et_group(self, root) -> bool:
        """Build /et/ group from ET observations."""
        candidates = [
            self.obs_dir / 'et' / 'preprocessed',
            self.obs_dir / 'et' / 'processed',
        ]
        return self._build_generic_group(
            root, 'et', candidates, 'et', 'hru', 'hru_id'
        )

    def _build_soil_moisture_group(self, root) -> bool:
        """Build /soil_moisture/ group from SM observations."""
        candidates = [
            self.obs_dir / 'soil_moisture' / 'point' / 'processed',
            self.obs_dir / 'soil_moisture' / 'smap' / 'processed',
            self.obs_dir / 'soil_moisture' / 'processed',
        ]
        return self._build_generic_group(
            root, 'soil_moisture', candidates, 'soil_moisture', 'station', 'station_id'
        )

    def _build_tws_group(self, root) -> bool:
        """Build /terrestrial_water_storage/ group from GRACE TWS."""
        candidates = [
            self.obs_dir / 'storage' / 'grace',
            self.obs_dir / 'grace',
            self.obs_dir / 'groundwater' / 'grace' / 'processed',
        ]
        return self._build_generic_group(
            root, 'terrestrial_water_storage', candidates,
            'tws_anomaly', 'basin', 'basin_id'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_generic_group(
        self,
        root,
        group_name: str,
        candidate_dirs: List[Path],
        var_name: str,
        spatial_dim: str,
        id_var: str,
    ) -> bool:
        """Build a group from the first directory that has CSV data."""
        for d in candidate_dirs:
            if not d.exists():
                continue
            csvs = sorted(d.glob('*.csv'))
            if not csvs:
                continue

            series = self._read_timeseries_csv(csvs[0])
            if series is None:
                continue

            self._write_group(
                root,
                group_name=group_name,
                series_dict={self.domain_name: series},
                var_name=var_name,
                spatial_dim=spatial_dim,
                id_var=id_var,
                source_meta=self._get_handler_source_meta(group_name),
            )
            return True

        return False

    def _write_group(
        self,
        root,
        group_name: str,
        series_dict: Dict[str, pd.Series],
        var_name: str,
        spatial_dim: str,
        id_var: str,
        source_meta: Optional[SourceMetadata] = None,
    ) -> None:
        """Write a NetCDF group from a dict of named pandas Series."""
        import netCDF4

        # Build common time axis
        all_times = sorted(set().union(*(s.index for s in series_dict.values())))
        time_index = pd.DatetimeIndex(all_times)
        n_time = len(time_index)
        n_spatial = len(series_dict)

        grp = root.createGroup(group_name)

        # Dimensions
        grp.createDimension('time', n_time)
        grp.createDimension(spatial_dim, n_spatial)

        # Time coordinate
        t_var = grp.createVariable('time', 'f8', ('time',))
        t_var.units = 'days since 1970-01-01'
        t_var.calendar = 'gregorian'
        t_var.standard_name = 'time'
        t_var[:] = netCDF4.date2num(
            time_index.to_pydatetime(),
            units='days since 1970-01-01',
            calendar='gregorian',
        )

        # Spatial ID coordinate
        id_v = grp.createVariable(id_var, str, (spatial_dim,))
        ids = list(series_dict.keys())
        for i, sid in enumerate(ids):
            id_v[i] = sid

        # Data variable
        fill = -9999.0
        data = grp.createVariable(
            var_name, 'f4', ('time', spatial_dim),
            fill_value=fill,
        )

        # CF attributes
        if var_name in CF_STANDARD_NAMES:
            for k, v in CF_STANDARD_NAMES[var_name].items():
                data.setncattr(k, v)

        # Source provenance
        if source_meta:
            for k, v in source_meta.to_netcdf_attrs().items():
                data.setncattr(k, v)

        # Fill data matrix
        mat = np.full((n_time, n_spatial), fill, dtype='f4')
        for j, (sid, series) in enumerate(series_dict.items()):
            for i, t in enumerate(time_index):
                if t in series.index:
                    val = series.loc[t]
                    if not pd.isna(val):
                        mat[i, j] = float(val)
        data[:] = mat

    def _read_timeseries_csv(self, csv_path: Path) -> Optional[pd.Series]:
        """Read a single CSV and return a datetime-indexed Series."""
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    logger.debug("Cannot parse dates in %s", csv_path)
                    return None

            # Find the first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                return None

            series = df[numeric_cols[0]].dropna()
            series.name = numeric_cols[0]
            return series

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            logger.debug("Could not read %s: %s", csv_path, e, exc_info=True)
            return None

    def _get_handler_source_meta(self, group_name: str) -> Optional[SourceMetadata]:
        """Build minimal SourceMetadata from config context."""
        source_map = {
            'streamflow': self._cfg('STREAMFLOW_DATA_PROVIDER', 'unknown'),
            'snow': 'MODIS/CanSWE',
            'et': 'MODIS MOD16/GLEAM',
            'soil_moisture': 'SMAP/ISMN',
            'terrestrial_water_storage': 'GRACE/GRACE-FO',
        }
        source = source_map.get(group_name, 'unknown')
        return SourceMetadata(source=source, processing='preprocessed by SYMFLUENCE')
