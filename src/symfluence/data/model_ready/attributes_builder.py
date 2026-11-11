# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Attributes store builder for the model-ready data store.

Reads intersection shapefiles from ``shapefiles/catchment_intersection/``
and writes a single grouped NetCDF4 file at
``data/model_ready/attributes/{domain}_attributes.nc`` with groups for
topology, terrain, soil, landcover, climate, and hydrogeology.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from symfluence.core.exceptions import ValidationError
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.modeling.model_ready.cf_conventions import CF_STANDARD_NAMES, build_global_attrs

from .source_metadata import SourceMetadata

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)


class AttributesNetCDFBuilder:
    """Build grouped attributes NetCDF from intersection shapefiles.

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

    # Groups that we attempt to build; each is optional.
    GROUPS = [
        'hru_identity',
        'topology',
        'terrain',
        'soil',
        'landcover',
        'climate',
        'hydrogeology',
        'cas',
        'community',
    ]

    def __init__(
        self,
        project_dir: Path,
        domain_name: str,
        config: Optional[Union['SymfluenceConfig', dict]] = None,
        config_dict: Optional[dict] = None,
    ) -> None:
        """Initialise the attributes builder.

        Args:
            project_dir: Root of the SYMFLUENCE domain directory.
            domain_name: Name of the hydrological domain.
            config: Typed config or legacy flat dict.
            config_dict: Deprecated. Use *config* instead.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        self._config = config if config is not None else config_dict

        self._config_fallback = (config_dict or {}) if config is None else {}

        self.intersect_dir = project_dir / 'shapefiles' / 'catchment_intersection'
        self.catchment_dir = project_dir / 'shapefiles' / 'catchment'
        self.basin_dir = project_dir / 'shapefiles' / 'river_basins'
        self.target_dir = project_dir / 'data' / 'model_ready' / 'attributes'
        self._catchment_hru_ids_cache: Optional[list] = None
        self._catchment_hru_ids_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default=None):
        """Get config value from typed config or legacy dict."""
        cfg = self._config
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return cfg.get(key, default)

    def build(self) -> Optional[Path]:
        """Build the attributes NetCDF file.

        Returns the output path, or ``None`` if no attribute data found.
        """
        try:
            import geopandas as gpd  # noqa: F401
        except ImportError:
            logger.warning("geopandas not available; cannot build attributes store")
            return None

        try:
            import netCDF4  # noqa: F401
        except ImportError:
            logger.warning("netCDF4 not available; cannot build attributes store")
            return None

        self.target_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.target_dir / f'{self.domain_name}_attributes.nc'

        groups_written = 0

        with netCDF4.Dataset(str(out_path), 'w', format='NETCDF4') as root:
            root.setncatts(build_global_attrs(
                domain_name=self.domain_name,
                title=f'{self.domain_name} catchment attributes',
                history='Built by AttributesNetCDFBuilder',
            ))

            if self._build_hru_identity_group(root):
                groups_written += 1
            if self._build_topology_group(root):
                groups_written += 1
            if self._build_terrain_group(root):
                groups_written += 1
            if self._build_soil_group(root):
                groups_written += 1
            if self._build_landcover_group(root):
                groups_written += 1
            if self._build_climate_group(root):
                groups_written += 1
            if self._build_hydrogeology_group(root):
                groups_written += 1
            if self._build_csv_group(root, 'soil_extended', 'soilclass'):
                groups_written += 1
            if self._build_csv_group(root, 'vegetation', 'vegetation'):
                groups_written += 1
            if self._build_csv_group(root, 'landcover_extended', 'landclass'):
                groups_written += 1
            # Community attribute layer (DATA_ACCESS: community): the CAS
            # AttributeBackend writes per-HRU stats to attributes/cas/, and the
            # entry-point plugin loop (climaclass, ...) to attributes/community/.
            if self._build_csv_group(root, 'cas', 'cas'):
                groups_written += 1
            if self._build_csv_group(root, 'community', 'community'):
                groups_written += 1

        if groups_written == 0:
            logger.info("No attribute data found; removing empty file")
            out_path.unlink(missing_ok=True)
            return None

        logger.info(
            "Attributes store built: %d groups in %s", groups_written, out_path
        )
        return out_path

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_hru_identity_group(self, root) -> bool:
        """Build /hru_identity/ group from catchment shapefile."""
        gdf = self._find_catchment_shapefile()
        if gdf is None:
            return False


        id_col = self._cfg('CATCHMENT_SHP_HRUID', 'HRU_ID')
        area_col = self._cfg('CATCHMENT_SHP_AREA', 'HRU_area')

        grp = root.createGroup('hru_identity')
        n_hru = len(gdf)
        grp.createDimension('hru', n_hru)

        # HRU ID
        id_v = grp.createVariable('hru_id', str, ('hru',))
        ids = gdf[id_col].astype(str).tolist() if id_col in gdf.columns else [str(i) for i in range(n_hru)]
        for i, sid in enumerate(ids):
            id_v[i] = sid

        # Area
        if area_col in gdf.columns:
            area_v = grp.createVariable('hru_area', 'f8', ('hru',))
            area_v[:] = gdf[area_col].values.astype('f8')
            if 'hru_area' in CF_STANDARD_NAMES:
                for k, v in CF_STANDARD_NAMES['hru_area'].items():
                    area_v.setncattr(k, v)

        # Centroid coordinates — project to equal-area CRS for accuracy, then back to WGS84
        try:
            projected = gdf.geometry.to_crs(epsg=6933)
            centroids = projected.centroid.to_crs(gdf.crs)
            lat_v = grp.createVariable('latitude', 'f8', ('hru',))
            lon_v = grp.createVariable('longitude', 'f8', ('hru',))
            lat_v[:] = centroids.y.values
            lon_v[:] = centroids.x.values
            for var, name in [(lat_v, 'latitude'), (lon_v, 'longitude')]:
                if name in CF_STANDARD_NAMES:
                    for k, v in CF_STANDARD_NAMES[name].items():
                        var.setncattr(k, v)
        except Exception:  # noqa: BLE001 — preprocessing resilience
            pass  # Geometry may not be available

        grp.setncattr('source_source', 'catchment shapefile')
        return True

    def _build_topology_group(self, root) -> bool:
        """Build /topology/ group from river basins shapefile."""
        gdf = self._find_basin_shapefile()
        if gdf is None:
            return False

        grp = root.createGroup('topology')
        n_gru = len(gdf)
        grp.createDimension('gru', n_gru)

        # GRU IDs
        gru_id_col = 'GRU_ID' if 'GRU_ID' in gdf.columns else gdf.columns[0]
        id_v = grp.createVariable('gru_id', str, ('gru',))
        for i, sid in enumerate(gdf[gru_id_col].astype(str).tolist()):
            id_v[i] = sid

        # Downstream ID
        if 'downstream_id' in gdf.columns:
            ds_v = grp.createVariable('downstream_id', str, ('gru',))
            for i, sid in enumerate(gdf['downstream_id'].astype(str).tolist()):
                ds_v[i] = sid

        # GRU area
        if 'GRU_area' in gdf.columns:
            area_v = grp.createVariable('gru_area', 'f8', ('gru',))
            area_v[:] = gdf['GRU_area'].values.astype('f8')
            area_v.units = 'm2'
            area_v.long_name = 'GRU area'

        # River attributes
        for col, nc_name, units in [
            ('river_length', 'river_length', 'm'),
            ('river_slope', 'river_slope', 'm m-1'),
        ]:
            if col in gdf.columns:
                v = grp.createVariable(nc_name, 'f8', ('gru',))
                v[:] = gdf[col].values.astype('f8')
                v.units = units

        grp.setncattr('source_source', 'river basins shapefile')
        return True

    def _build_terrain_group(self, root) -> bool:
        """Build /terrain/ group from DEM intersection shapefile."""
        shp = self.intersect_dir / 'with_dem' / 'catchment_with_dem.shp'
        gdf = self._read_shapefile(shp)
        if gdf is None:
            return False

        grp = root.createGroup('terrain')
        n = len(gdf)
        grp.createDimension('hru', n)
        self._write_hru_id(grp, gdf)

        if 'elev_mean' in gdf.columns:
            v = grp.createVariable('elev_mean', 'f8', ('hru',))
            v[:] = gdf['elev_mean'].values.astype('f8')
            if 'elev_mean' in CF_STANDARD_NAMES:
                for k, val in CF_STANDARD_NAMES['elev_mean'].items():
                    v.setncattr(k, val)

        meta = SourceMetadata(source='DEM', processing='zonal statistics over HRUs')
        grp.setncatts(meta.to_netcdf_attrs())
        return True

    def _build_soil_group(self, root) -> bool:
        """Build /soil/ group from soilgrids intersection shapefile."""
        shp = self.intersect_dir / 'with_soilgrids' / 'catchment_with_soilclass.shp'
        gdf = self._read_shapefile(shp)
        if gdf is None:
            return False

        grp = root.createGroup('soil')
        n = len(gdf)
        grp.createDimension('hru', n)
        self._write_hru_id(grp, gdf)

        # Look for USGS soil fraction columns
        soil_cols = [c for c in gdf.columns if c.startswith('USGS_')]
        n_class = len(soil_cols)
        if n_class > 0:
            grp.createDimension('soil_class', n_class)
            v = grp.createVariable('soil_fraction', 'f4', ('hru', 'soil_class'))
            v[:] = gdf[soil_cols].values.astype('f4')
            v.long_name = 'USGS soil class fraction per HRU'
            v.units = '1'

            cls_v = grp.createVariable('soil_class_name', str, ('soil_class',))
            for i, name in enumerate(soil_cols):
                cls_v[i] = name

        if 'soil_pixel_count' in gdf.columns:
            pc = grp.createVariable('soil_pixel_count', 'i4', ('hru',))
            pc[:] = gdf['soil_pixel_count'].values.astype('i4')

        meta = SourceMetadata(source='SoilGrids', processing='intersection with catchment')
        grp.setncatts(meta.to_netcdf_attrs())
        return True

    def _build_landcover_group(self, root) -> bool:
        """Build /landcover/ group from landclass intersection shapefile."""
        shp = self.intersect_dir / 'with_landclass' / 'catchment_with_landclass.shp'
        gdf = self._read_shapefile(shp)
        if gdf is None:
            return False

        grp = root.createGroup('landcover')
        n = len(gdf)
        grp.createDimension('hru', n)
        self._write_hru_id(grp, gdf)

        land_cols = [c for c in gdf.columns if c.startswith('IGBP_')]
        n_class = len(land_cols)
        if n_class > 0:
            grp.createDimension('land_class', n_class)
            v = grp.createVariable('land_fraction', 'f4', ('hru', 'land_class'))
            v[:] = gdf[land_cols].values.astype('f4')
            v.long_name = 'IGBP land cover fraction per HRU'
            v.units = '1'

            cls_v = grp.createVariable('land_class_name', str, ('land_class',))
            for i, name in enumerate(land_cols):
                cls_v[i] = name

        if 'land_pixel_count' in gdf.columns:
            pc = grp.createVariable('land_pixel_count', 'i4', ('hru',))
            pc[:] = gdf['land_pixel_count'].values.astype('i4')

        meta = SourceMetadata(source='MODIS IGBP', processing='intersection with catchment')
        grp.setncatts(meta.to_netcdf_attrs())
        return True

    def _build_climate_group(self, root) -> bool:
        """Build /climate/ group from climate attribute files."""
        climate_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'climate'
        if not climate_dir.exists():
            return False

        csvs = self._pick_csv(climate_dir, 'worldclim_attributes.csv')
        if not csvs:
            return False

        try:
            df, ids = self._combine_hru_tables(csvs, 'climate')
            if df is None or df.empty:
                return False

            grp = root.createGroup('climate')
            self._write_table_group(grp, df, ids)

            meta = SourceMetadata(source='climatological attributes')
            grp.setncatts(meta.to_netcdf_attrs())
            return True
        except ValidationError:
            raise
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            logger.debug("Could not build climate group: %s", e, exc_info=True)
            return False

    def _build_hydrogeology_group(self, root) -> bool:
        """Build /hydrogeology/ group from geology attribute files."""
        geo_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'geology'
        if not geo_dir.exists():
            return False

        csvs = self._pick_csv(geo_dir, 'glhymps_attributes.csv')
        if not csvs:
            return False

        try:
            df, ids = self._combine_hru_tables(csvs, 'hydrogeology')
            if df is None or df.empty:
                return False

            grp = root.createGroup('hydrogeology')
            self._write_table_group(grp, df, ids)

            meta = SourceMetadata(source='hydrogeological attributes')
            grp.setncatts(meta.to_netcdf_attrs())
            return True
        except ValidationError:
            raise
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            logger.debug("Could not build hydrogeology group: %s", e, exc_info=True)
            return False

    def _build_csv_group(self, root, group_name: str, subdir: str) -> bool:
        """Build a NetCDF group from CSV files in an attributes subdirectory."""
        attr_dir = resolve_data_subdir(self.project_dir, 'attributes') / subdir
        if not attr_dir.exists():
            return False

        csvs = list(attr_dir.glob('*_attributes.csv'))
        if not csvs:
            return False

        try:
            df, ids = self._combine_hru_tables(csvs, group_name)
            if df is None or df.empty:
                return False

            grp = root.createGroup(group_name)
            self._write_table_group(grp, df, ids)

            meta = SourceMetadata(source=f'{group_name} attributes')
            grp.setncatts(meta.to_netcdf_attrs())
            return True
        except ValidationError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not build %s group: %s", group_name, e)
            return False

    @staticmethod
    def _pick_csv(directory: Path, preferred: str) -> list:
        """The one CSV a single-file group is built from (deterministic)."""
        if (directory / preferred).exists():
            return [directory / preferred]
        return sorted(directory.glob('*.csv'))[:1]

    def _combine_hru_tables(self, csvs, group_name: str):
        """Combine the per-HRU CSVs of one group into a single table.

        Tables are joined on their ``hru_id`` column and ordered like the
        catchment shapefile (numerically when the catchment is unavailable).
        Tables without an id column cannot be joined to anything: a single
        one is passed through unchanged (and written without an ``hru_id``
        coordinate); next to id-keyed tables it is left out with a warning.

        Returns:
            ``(table, hru_ids)`` — *hru_ids* is None when the table carries
            no ids; ``(None, None)`` when there is nothing to write.

        Raises:
            ValidationError: for a processor-written CSV that predates the
                ``hru_id`` column and has too many rows to be trusted, for
                duplicate ids, and for several id-less tables that cannot be
                combined safely.
        """
        import pandas as pd

        from .hru_tables import (
            HRU_ID_COLUMN,
            LEGACY_POSITIONAL_MAX_ROWS,
            PROCESSOR_CSV_NAMES,
            legacy_table_error,
            normalize_hru_id,
            order_hru_ids,
            read_hru_table,
        )

        keyed = []
        unkeyed = []
        for csv in sorted(csvs):
            table = read_hru_table(csv)
            if table.empty:
                continue
            if HRU_ID_COLUMN in table.columns:
                if table[HRU_ID_COLUMN].duplicated().any():
                    raise ValidationError(
                        f"Per-HRU attribute table {csv} has duplicate '{HRU_ID_COLUMN}' values."
                    )
                keyed.append(table.set_index(HRU_ID_COLUMN))
            else:
                if csv.name in PROCESSOR_CSV_NAMES and len(table) > LEGACY_POSITIONAL_MAX_ROWS:
                    raise legacy_table_error(csv, len(table))
                unkeyed.append((csv, table))

        if keyed:
            for csv, table in unkeyed:
                logger.warning(
                    "Attribute table %s has no '%s' column and cannot be joined to the "
                    "id-keyed tables of group '%s'; leaving it out.",
                    csv, HRU_ID_COLUMN, group_name,
                )
            df = keyed[0]
            for other in keyed[1:]:
                new_cols = [c for c in other.columns if c not in df.columns]
                df = df.join(other[new_cols], how='outer')
            catchment_ids = self._catchment_hru_ids()
            if catchment_ids is not None and set(df.index) <= set(catchment_ids):
                order = [i for i in catchment_ids if i in set(df.index)]
            else:
                order = order_hru_ids(df.index, catchment_ids)
            df = df.reindex(order)
            ids = [normalize_hru_id(i) for i in df.index]
            return df.reset_index(drop=True), ids

        if not unkeyed:
            return None, None
        if len(unkeyed) == 1:
            csv, table = unkeyed[0]
            if len(table) > 1:
                logger.warning(
                    "Attribute table %s has no '%s' column; group '%s' is written "
                    "without HRU ids and cannot be joined to HRUs by id.",
                    csv, HRU_ID_COLUMN, group_name,
                )
            return table, None

        lengths = {len(table) for _, table in unkeyed}
        if len(lengths) != 1 or max(lengths) > LEGACY_POSITIONAL_MAX_ROWS:
            raise ValidationError(
                f"Attribute tables {[str(c) for c, _ in unkeyed]} for group "
                f"'{group_name}' have no '{HRU_ID_COLUMN}' column and "
                f"{sorted(lengths)} rows; they cannot be combined row by row safely. "
                "Regenerate the attributes (processor-written tables now carry HRU "
                f"ids) or add an '{HRU_ID_COLUMN}' column."
            )
        df = pd.concat([table for _, table in unkeyed], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        return df, None

    def _catchment_hru_ids(self):
        """Normalized HRU ids in catchment-shapefile order, or None."""
        if not self._catchment_hru_ids_loaded:
            from .hru_tables import normalize_hru_id

            ids = None
            gdf = self._find_catchment_shapefile()
            id_col = self._cfg('CATCHMENT_SHP_HRUID', 'HRU_ID')
            if gdf is not None and id_col in gdf.columns:
                ids = [normalize_hru_id(v) for v in gdf[id_col]]
            self._catchment_hru_ids_cache = ids
            self._catchment_hru_ids_loaded = True
        return self._catchment_hru_ids_cache

    @staticmethod
    def _write_table_group(grp, df, ids) -> None:
        """Write a per-HRU table (and its ``hru_id`` coordinate) to a group."""
        grp.createDimension('hru', len(df))
        if ids is not None:
            id_v = grp.createVariable('hru_id', str, ('hru',))
            for i, sid in enumerate(ids):
                id_v[i] = sid
        for col in df.select_dtypes(include=[np.number]).columns:
            v = grp.createVariable(col, 'f4', ('hru',))
            v[:] = df[col].values.astype('f4')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_catchment_shapefile(self):
        """Find and read the catchment/HRU shapefile."""
        import geopandas as gpd

        # Try organized path first, then flat path
        definition_method = self._cfg('DOMAIN_DEFINITION_METHOD', 'lumped')
        experiment_id = self._cfg('EXPERIMENT_ID', 'run_1')
        candidates = [
            self.catchment_dir / definition_method / experiment_id,
            self.catchment_dir,
        ]

        for d in candidates:
            if not d.exists():
                continue
            for shp in sorted(d.glob(f'{self.domain_name}_HRUs_*.shp')):
                return gpd.read_file(shp)
            for shp in sorted(d.glob('*.shp')):
                return gpd.read_file(shp)

        return None

    def _find_basin_shapefile(self):
        """Find and read the river basins shapefile."""
        import geopandas as gpd

        if not self.basin_dir.exists():
            return None

        for shp in sorted(self.basin_dir.glob(f'{self.domain_name}_riverBasins_*.shp')):
            return gpd.read_file(shp)
        for shp in sorted(self.basin_dir.glob('*.shp')):
            return gpd.read_file(shp)
        return None

    def _read_shapefile(self, path: Path):
        """Read a shapefile, returning None if it doesn't exist."""
        if not path.exists():
            return None
        try:
            import geopandas as gpd
            return gpd.read_file(path)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            logger.debug("Could not read %s: %s", path, e, exc_info=True)
            return None

    def _write_hru_id(self, grp, gdf) -> None:
        """Write an ``hru_id`` coordinate to a per-HRU group when available.

        Per-HRU groups (terrain, soil, landcover) are built from independent
        intersection shapefiles, so consumers must join on an explicit id
        rather than row position.  This stamps the catchment HRU id (when the
        column is present) so :class:`AttributesReader.per_hru_values` can
        return an order-independent ``{hru_id: value}`` mapping.
        """
        id_col = self._cfg('CATCHMENT_SHP_HRUID', 'HRU_ID')
        if id_col not in getattr(gdf, 'columns', []):
            return
        id_v = grp.createVariable('hru_id', str, ('hru',))
        for i, sid in enumerate(gdf[id_col].astype(str).tolist()):
            id_v[i] = sid
