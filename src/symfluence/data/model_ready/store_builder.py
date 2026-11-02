# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Orchestrator for building the complete model-ready data store.

Coordinates ``ForcingsStoreBuilder``, ``ObservationsNetCDFBuilder``,
and ``AttributesNetCDFBuilder`` into a single ``build_all()`` entry point.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from symfluence.core.mixins.project import resolve_data_subdir

from .attributes_builder import AttributesNetCDFBuilder
from .forcings_builder import ForcingsStoreBuilder
from .observations_builder import ObservationsNetCDFBuilder

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)


def _is_valid_netcdf(path: Path) -> bool:
    """Report whether *path* is an openable NetCDF file with any content."""
    try:
        import netCDF4
    except ImportError:
        return path.stat().st_size > 0
    try:
        with netCDF4.Dataset(str(path), 'r') as ds:
            return bool(ds.groups or ds.variables)
    except OSError:
        return False


class ModelReadyStoreBuilder:
    """Orchestrate building the complete model-ready data store.

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
        """Initialise the store builder.

        Args:
            project_dir: Root of the SYMFLUENCE domain directory.
            domain_name: Name of the hydrological domain.
            config: Typed config or legacy flat dict.
            config_dict: Deprecated. Use *config* instead.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        # Accept either typed config or legacy dict
        self._config = config if config is not None else config_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all(self) -> None:
        """Build forcings, observations, and attributes stores."""
        logger.info("Building model-ready data store for %s", self.domain_name)

        forcings = self.build_forcings()
        self._compute_climate_statistics()
        observations = self.build_observations()
        attributes = self.build_attributes()

        # Report what was actually materialized — a builder returning None means
        # its section was skipped or produced nothing, not that it succeeded.
        outcomes = ', '.join(
            f"{name}={'built' if result is not None else 'skipped'}"
            for name, result in (
                ('forcings', forcings),
                ('observations', observations),
                ('attributes', attributes),
            )
        )
        if self.is_store_complete():
            logger.info("Model-ready data store build complete (%s)", outcomes)
        else:
            logger.warning(
                "Model-ready data store is INCOMPLETE after build (%s); "
                "the build step will re-run on the next workflow invocation",
                outcomes,
            )

    def _cfg(self, key: str, default=None):
        """Get config value from typed config or legacy dict."""
        cfg = self._config
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        # Typed SymfluenceConfig — use .get() backward-compat layer
        return cfg.get(key, default)

    def build_forcings(self) -> Optional[Path]:
        """Build the forcings section. Skip if no basin_averaged_data."""
        forcing_dataset = self._cfg('FORCING_DATASET', 'ERA5')
        strategy = self._cfg('MODEL_READY_FORCING_STRATEGY', 'symlink')

        builder = ForcingsStoreBuilder(
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            forcing_dataset=forcing_dataset,
            strategy=strategy,
        )
        return builder.build()

    def build_observations(self) -> Optional[Path]:
        """Build the observations section. Skip if no observations dir."""
        builder = ObservationsNetCDFBuilder(
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            config=self._config,
        )
        return builder.build()

    def _compute_climate_statistics(self) -> None:
        """Compute per-HRU climate statistics from basin-averaged forcing.

        Derives mean precipitation, temperature, aridity index, and snow
        fraction from the forcing files and writes them to
        ``attributes/climate/climate_statistics.csv`` so the attributes
        builder can include a ``/climate/`` group in the model-ready store.
        """
        import glob

        import numpy as np
        import pandas as pd

        forcing_dir = self.project_dir / 'data' / 'forcing' / 'basin_averaged_data'
        if not forcing_dir.exists():
            return

        files = sorted(glob.glob(str(forcing_dir / '*.nc')))
        if not files:
            return

        try:
            import xarray as xr

            ds = xr.open_mfdataset(
                files, combine='nested', concat_dim='time',
                coords='minimal', compat='override', data_vars='minimal',
            ).sortby('time')

            n_hru = ds.sizes.get('hru', 1)
            if n_hru <= 1:
                ds.close()
                return

            precip_var = next((v for v in ['precipitation_flux', 'pptrate'] if v in ds), None)
            temp_var = next((v for v in ['air_temperature', 'airtemp'] if v in ds), None)

            if precip_var is None or temp_var is None:
                logger.debug("Climate stats: missing precip or temp variable")
                ds.close()
                return

            # Vectorized computation across all HRUs at once
            p_mean = ds[precip_var].mean(dim='time').values  # (n_hru,)
            p_mm_yr = p_mean * 86400 * 365.25

            t_all = ds[temp_var].values  # (time, n_hru)
            if np.nanmean(t_all) > 100:
                t_all = t_all - 273.15
            t_mean = np.nanmean(t_all, axis=0)  # (n_hru,)

            pet_daily = np.where(t_all > -5, 0.4 * (t_all + 5), 0.0)
            pet_annual = np.nanmean(pet_daily, axis=0) * 365.25  # (n_hru,)
            aridity = pet_annual / np.maximum(p_mm_yr, 1.0)

            p_all = ds[precip_var].values  # (time, n_hru)
            wet = p_all > 1e-7
            cold_wet = (t_all < 0) & wet
            wet_count = np.sum(wet, axis=0).astype(float)
            snow_frac = np.sum(cold_wet, axis=0) / np.maximum(wet_count, 1.0)

            ds.close()

            df = pd.DataFrame({
                'precip_mm_yr': p_mm_yr,
                'temp_C': t_mean,
                'aridity': aridity,
                'snow_frac': snow_frac,
            })

            # ── Enrich with mean elevation from catchment shapefile ──
            df['elev_m'] = float('nan')
            try:
                import geopandas as gpd

                shp_root = self.project_dir / 'shapefiles' / 'catchment'
                shp_file = None
                if shp_root.exists():
                    # Collect all HRU shapefiles, prefer one matching n_hru
                    candidates = []
                    for dirpath, _dirnames, filenames in os.walk(shp_root):
                        for fn in filenames:
                            if fn.endswith('.shp') and 'HRUs' in fn:
                                candidates.append(Path(dirpath) / fn)
                    # Try candidates with matching HRU count first
                    for candidate in candidates:
                        gdf = gpd.read_file(candidate)
                        if 'elev_mean' in gdf.columns and len(gdf) == n_hru:
                            shp_file = candidate
                            break

                if shp_file is not None:
                    df['elev_m'] = gdf['elev_mean'].values
                    logger.debug("Loaded elev_mean from %s", shp_file)
                else:
                    logger.debug(
                        "No HRU shapefile with elev_mean matching %d HRUs found",
                        n_hru,
                    )
            except Exception:  # noqa: BLE001
                logger.debug("Could not load elevation from shapefile")

            # ── Enrich with glacier fraction from domain_type intersection ──
            df['glacier_fraction'] = 0.0
            try:
                import geopandas as gpd

                glacier_shp = (
                    self.project_dir / 'shapefiles' / 'catchment_intersection'
                    / 'with_domain_type' / 'catchment_with_domain_type.shp'
                )
                if glacier_shp.exists():
                    gdf_gl = gpd.read_file(glacier_shp)
                    # domType_2=clean accum, domType_3=clean ablation, domType_4=debris
                    gl_cols = [c for c in gdf_gl.columns if c.startswith('domType_') and c != 'domType_1']
                    if gl_cols and len(gdf_gl) == n_hru:
                        df['glacier_fraction'] = gdf_gl[gl_cols].sum(axis=1).values
                        logger.debug("Loaded glacier_fraction from %s (mean=%.3f)", glacier_shp, df['glacier_fraction'].mean())
                    elif gl_cols:
                        logger.debug("Glacier shapefile row count mismatch (%d vs %d HRUs)", len(gdf_gl), n_hru)
            except Exception:  # noqa: BLE001
                logger.debug("Could not load glacier fraction")

            out_dir = self.project_dir / 'data' / 'attributes' / 'climate'
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / 'climate_statistics.csv', index=False)
            logger.info("Climate statistics computed for %d HRUs", n_hru)

        except Exception as e:  # noqa: BLE001
            logger.debug("Could not compute climate statistics: %s", e)

    def build_attributes(self) -> Optional[Path]:
        """Build the attributes section. Skip if no intersection shapefiles."""
        builder = AttributesNetCDFBuilder(
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            config=self._config,
        )
        return builder.build()

    def is_store_complete(self) -> bool:
        """Check whether every section the source inputs call for is materialized.

        The contract is source-driven, not existence-driven: an empty
        ``data/model_ready`` directory (created unconditionally at project init)
        is never complete, and a store that materialized only part of the
        basin-averaged forcing must report incomplete so the build re-runs.
        Observations and attributes are content-dependent (their builders may
        legitimately produce nothing), so they are required only to be *valid*
        when present — but at least one section must actually exist.
        """
        store_dir = self.project_dir / 'data' / 'model_ready'
        if not store_dir.exists():
            return False

        # Forcings: every basin-averaged source file must be present in the store.
        forcing_src = resolve_data_subdir(self.project_dir, 'forcing') / 'basin_averaged_data'
        src_names = {p.name for p in forcing_src.glob('*.nc')} if forcing_src.exists() else set()
        forcings_dir = store_dir / 'forcings'
        store_names = {p.name for p in forcings_dir.glob('*.nc')} if forcings_dir.exists() else set()
        if not src_names <= store_names:
            missing = sorted(src_names - store_names)
            logger.debug(
                "Model-ready store incomplete: %d forcing file(s) not materialized (e.g. %s)",
                len(missing), missing[:3],
            )
            return False

        obs_path = store_dir / 'observations' / f'{self.domain_name}_observations.nc'
        attrs_path = store_dir / 'attributes' / f'{self.domain_name}_attributes.nc'
        for path in (obs_path, attrs_path):
            if path.exists() and not _is_valid_netcdf(path):
                logger.debug("Model-ready store incomplete: %s is not a readable NetCDF", path)
                return False

        return bool(src_names) or obs_path.exists() or attrs_path.exists()

    def migrate_from_legacy(self) -> None:
        """Build model-ready store from existing legacy domain directory."""
        logger.info("Migrating legacy domain '%s' to model-ready store", self.domain_name)
        self.build_all()
