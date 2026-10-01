# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GeoData management utilities for HYPE model.

This module provides the HYPEGeoDataManager class for generating HYPE's geographic
input files. It handles the creation of:

- **GeoData.txt**: Sub-basin properties (topology, area, location, SLC fractions)
- **GeoClass.txt**: Soil-Landcover Class definitions and soil layer depths
- **ForcKey.txt**: Mapping between sub-basin IDs and forcing file station IDs

The manager also performs topological sorting to ensure sub-basins are ordered
from upstream to downstream, which is required for HYPE's internal routing.

Example usage:
    >>> from symfluence.models.hype import HYPEGeoDataManager
    >>> manager = HYPEGeoDataManager(config, logger, output_path, geofabric_mapping)
    >>> land_uses = manager.create_geofiles(
    ...     gistool_output=Path('/path/to/gis_stats'),
    ...     subbasins_shapefile=Path('/path/to/basins.shp'),
    ...     rivers_shapefile=Path('/path/to/rivers.shp'),
    ...     frac_threshold=0.05
    ... )
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pint

from symfluence.core.exceptions import ValidationError

if TYPE_CHECKING:
    pass


def load_elevation_band_table(shapefile: Path | str) -> list[dict[str, float]]:
    """Read an elevation-banded HRU shapefile into an ordered band table.

    Returns a list of ``{'hru_id', 'elev_mean', 'area'}`` dicts sorted by
    elevation (lowest first). Robust to column-name variants. Shared by the
    GeoData manager (sub-basin expansion) and the forcing processor (per-band
    lapse expansion) so both key bands by the same HRU ids — the invariant that
    makes GeoData sub-basins line up with forcing columns.
    """
    gdf = gpd.read_file(shapefile)
    elev_col = next((c for c in ('elev_mean', 'elevMean', 'mean', 'ELEV_MEAN',
                                 'elevation', 'avg_elevcl') if c in gdf.columns), None)
    if elev_col is None:
        return []
    id_col = next((c for c in ('HRU_ID', 'hru_id', 'hruId', 'HRUID')
                   if c in gdf.columns), None)
    area_col = next((c for c in ('HRU_area', 'HRU_AREA', 'area', 'Area', 'AREA')
                     if c in gdf.columns), None)
    table: list[dict[str, float]] = []
    for i, (_, row) in enumerate(gdf.iterrows()):
        table.append({
            'hru_id': int(row[id_col]) if id_col is not None else i + 1,
            'elev_mean': float(row[elev_col]),
            'area': float(row[area_col]) if area_col is not None else 1.0,
        })
    table.sort(key=lambda d: d['elev_mean'])
    return table


class HYPEGeoDataManager:
    """
    Manager for HYPE geographic and classification data.

    This class handles the creation of HYPE's three geographic input files,
    which define the spatial structure of the model domain:

    **GeoData.txt** columns:
        - subid: Sub-basin identifier (must be > 0)
        - maindown: ID of downstream sub-basin (0 for outlets)
        - area: Sub-basin area (m²)
        - rivlen: River length through sub-basin (m)
        - slope_mean: Mean river slope (m/m)
        - latitude/longitude: Centroid coordinates
        - elev_mean: Mean elevation (m)
        - SLC_1, SLC_2, ...: Fractions of each Soil-Landcover Class

    **GeoClass.txt** columns:
        - SLC: Soil-Landcover Class ID
        - LULC: Land use/cover type (IGBP class 1-17)
        - SOIL TYPE: Soil type ID
        - Vegetation type, soil layer depths, etc.

    **ForcKey.txt** columns:
        - subid: Sub-basin ID
        - stationid: Forcing file station ID (typically same as subid)

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance for status messages.
        output_path: Path to HYPE settings directory.
        geofabric_mapping: Field name mappings for geospatial inputs.
        ureg: Pint unit registry for unit conversions.

    Note:
        HYPE requires sub-basin IDs > 0. If input data uses 0-based IDs,
        this manager automatically shifts all IDs by +1.
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger | Any | None,
        output_path: Path | str,
        geofabric_mapping: dict[str, Any]
    ) -> None:
        """
        Initialize the HYPE GeoData manager.

        Args:
            config: Configuration dictionary containing domain settings.
            logger: Logger instance for status messages. If None, creates
                a module-level logger.
            output_path: Path to the HYPE settings directory where geographic
                files will be written.
            geofabric_mapping: Dictionary mapping input field names to HYPE
                concepts. Expected keys:
                - 'basinID': {'in_varname': str} - Sub-basin ID field
                - 'nextDownID': {'in_varname': str} - Downstream ID field
                - 'area': {'in_varname': str, 'in_units': str, 'out_units': str}
                - 'rivlen': {'in_varname': str, 'in_units': str, 'out_units': str}
        """
        self.config = config
        self.logger = logger if logger else logging.getLogger(__name__)
        self.output_path = Path(output_path)
        self.geofabric_mapping = geofabric_mapping
        self.ureg: pint.UnitRegistry = pint.UnitRegistry()

    def create_geofiles(
        self,
        gistool_output: Path,
        subbasins_shapefile: Path,
        rivers_shapefile: Path,
        frac_threshold: float,
        intersect_base_path: Optional[Path] = None,
        elevation_band_shapefile: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create GeoData.txt, GeoClass.txt, and ForcKey.txt files.

        Args:
            gistool_output: Path to GIS statistics CSVs
            subbasins_shapefile: Path to catchment shapefile
            rivers_shapefile: Path to river network shapefile
            frac_threshold: Minimum landcover fraction to consider
            intersect_base_path: Optional path to intersection shapefiles
            elevation_band_shapefile: Optional path to an elevation-banded HRU
                shapefile (``{domain}_HRUs_elevation.shp``). When provided, each
                sub-basin is expanded into a vertical cascade of elevation-band
                sub-basins so HYPE can resolve elevation-dependent snow/glacier
                melt timing instead of running as a single lumped bucket.

        Returns:
            Array of unique land use IDs for parameter file generation
        """
        self.logger.debug("Generating HYPE geographic files...")
        gistool_output = Path(gistool_output)
        subbasins_shapefile = Path(subbasins_shapefile)
        rivers_shapefile = Path(rivers_shapefile)

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 1. Build base topology from river network
        basin_id_col = self.geofabric_mapping['basinID']['in_varname']
        next_down_col = self.geofabric_mapping['nextDownID']['in_varname']

        if rivers_shapefile.exists():
            riv = gpd.read_file(rivers_shapefile)
        else:
            riv = gpd.read_file(subbasins_shapefile)
            if next_down_col not in riv.columns:
                riv[next_down_col] = 0

        base_df = pd.DataFrame({
            'subid': riv[basin_id_col],
            'maindown': riv[next_down_col]
        })

        # Regional groundwater routing follows the surface water path
        base_df['grwdown'] = base_df['maindown']

        # 2. River properties
        rivlen_info = self.geofabric_mapping['rivlen']
        if rivlen_info['in_varname'] in riv.columns:
            lengths = riv[rivlen_info['in_varname']].values * self.ureg(rivlen_info['in_units'])
            base_df['rivlen'] = lengths.to(rivlen_info['out_units']).magnitude
        else:
            base_df['rivlen'] = 0

        if 'Slope' in riv.columns:
            base_df['slope_mean'] = riv['Slope']
        else:
            base_df['slope_mean'] = 0.001

        # 3. Catchment properties
        cat = gpd.read_file(subbasins_shapefile)
        area_info = self.geofabric_mapping['area']

        # Calculate centroids in projected CRS
        centroids = self._get_projected_centroids(cat)

        # Calculate area from geometry using equal-area projection (more reliable than stored attributes)
        # This addresses issues where stored area attributes may not match actual geometry
        geometry_area_m2 = self._calculate_geometry_area(cat)

        # Check for significant mismatch between stored and calculated area
        if area_info['in_varname'] in cat.columns:
            stored_area = cat[area_info['in_varname']].values * self.ureg(area_info['in_units']).to('m^2').magnitude
            area_ratio = geometry_area_m2 / stored_area
            if abs(area_ratio.mean() - 1.0) > 0.1:  # More than 10% difference
                self.logger.warning(
                    f"Significant area mismatch detected: stored area attribute differs from geometry by "
                    f"{(area_ratio.mean() - 1.0) * 100:.1f}%. Using geometry-calculated area for accuracy."
                )

        # Convert to output units
        area_values = geometry_area_m2 * self.ureg('m^2').to(area_info['out_units']).magnitude

        cat_props = pd.DataFrame({
            basin_id_col: cat[basin_id_col],
            'area': area_values,
            'latitude': centroids.y,
            'longitude': centroids.x
        }).set_index(basin_id_col)

        # 4. Load GIS stats
        soil_data, landcover_data, elevation_data = self._load_gis_stats(
            gistool_output, intersect_base_path, basin_id_col
        )

        # Canonical statistics may be keyed by elevation HRU while topology
        # still contains parent basins. Aggregate before building parent SLCs.
        if elevation_band_shapefile is not None:
            parent_bands = self._load_elevation_bands(Path(elevation_band_shapefile), base_df)
            soil_data, landcover_data, elevation_data = (
                self._aggregate_band_stats(frame, parent_bands)
                for frame in (soil_data, landcover_data, elevation_data)
            )

        # 5. SLC processing
        glacier_fraction = self.config.get('HYPE_GLACIER_FRACTION')
        if glacier_fraction is not None:
            if len(base_df) != 1:
                raise ValidationError('HYPE_GLACIER_FRACTION currently requires one parent basin')
            landcover_data = landcover_data.copy()
            lc_cols = [c for c in landcover_data if isinstance(c, str) and c.startswith('IGBP_') and c[5:].isdigit()]
            if not lc_cols:
                lc_cols = [c for c in landcover_data if isinstance(c, str) and c.startswith('frac_') and c[5:].isdigit()]
            if not lc_cols:
                raise ValidationError('Cannot apply glacier fraction without land-cover class fractions')
            ice_col = ('IGBP_' if lc_cols[0].startswith('IGBP_') else 'frac_') + '15'
            other_cols = [c for c in lc_cols if c != ice_col]
            total = landcover_data[other_cols].sum(axis=1)
            if (total <= 0).any() and glacier_fraction < 1:
                raise ValidationError('No non-glacier land cover available for fraction override')
            landcover_data[other_cols] = landcover_data[other_cols].div(total.replace(0, 1), axis=0) * (1 - glacier_fraction)
            landcover_data[ice_col] = glacier_fraction
            frac_threshold = 0.0  # Retain measured fractions without renormalizing away small classes.
        slc_df, base_df = self._process_slc(base_df, landcover_data, soil_data, frac_threshold)

        # 6. Final merging
        base_df = base_df.join(cat_props, on='subid')

        # Positional fallback for id-label mismatch between the river network
        # and the catchment shapefile. For lumped/single-subbasin domains the
        # river reach id (e.g. LINKNO=1) often differs from the catchment id
        # (e.g. GRU_ID=<domain>), so the id-based join above leaves area/lat/lon
        # as NaN and the only subbasin gets dropped -> empty GeoData -> HYPE
        # fails. When the row counts match (1:1, the lumped case), assign the
        # catchment properties positionally instead of by id.
        join_cols = ['area', 'latitude', 'longitude']
        if base_df[join_cols].isna().any(axis=None) and len(base_df) == len(cat_props):
            self.logger.warning(
                "Catchment id-join left %d/%d sub-basins without geometry; "
                "river/catchment ids differ — filling positionally (1:1 lumped).",
                int(base_df[join_cols].isna().any(axis=1).sum()), len(base_df),
            )
            for col in join_cols:
                base_df[col] = cat_props[col].to_numpy()
            # For a single lumped basin, also reconcile the subbasin id to the
            # catchment id. The forcing/observation files (Pobs/Tobs/Qobs) are
            # keyed by the catchment id, so a river-reach subid (e.g. 1) would
            # make HYPE fail to match forcing/obs ("halt: loading observations").
            # maindown is the outlet (0) here, so topology is unaffected.
            if len(base_df) == 1:
                base_df['subid'] = [int(cat_props.index[0])]

        # Robust elevation mapping
        elev_col = 'mean' if 'mean' in elevation_data.columns else 'elev_mean'

        def get_elevation(subid):
            if subid in elevation_data.index:
                result = elevation_data.loc[subid, elev_col]
                if isinstance(result, pd.Series):
                    return float(result.iloc[0])
                return float(result)
            elif len(elevation_data) == 1:
                return elevation_data[elev_col].iloc[0]
            return 0.0

        base_df['elev_mean'] = base_df['subid'].apply(get_elevation)

        # Load glacier fraction from domain_type intersection shapefile
        glacier_shp = None
        if intersect_base_path:
            glacier_shp = Path(intersect_base_path).parent / 'with_domain_type' / 'catchment_with_domain_type.shp'
        if glacier_shp is None or not glacier_shp.exists():
            glacier_shp = Path(str(subbasins_shapefile)).parents[1] / 'catchment_intersection' / 'with_domain_type' / 'catchment_with_domain_type.shp'
        if glacier_shp.exists():
            try:
                gl_gdf = gpd.read_file(glacier_shp)
                gl_cols = [c for c in gl_gdf.columns if c.startswith('domType_') and c != 'domType_1']
                if gl_cols:
                    gl_frac = gl_gdf[gl_cols].sum(axis=1).values
                    if len(gl_frac) == len(cat):
                        gl_map = dict(zip(cat[basin_id_col].values, gl_frac))
                        base_df['glacier_fraction'] = base_df['subid'].map(gl_map).fillna(0.0)
                        self.logger.debug("Loaded glacier_fraction for %d sub-basins (mean=%.3f)", len(base_df), base_df['glacier_fraction'].mean())
            except Exception as e:  # noqa: BLE001
                self.logger.debug("Could not load glacier fraction: %s", e)
        if 'glacier_fraction' not in base_df.columns:
            base_df['glacier_fraction'] = 0.0

        # Normalize SLC fractions
        slc_cols = [col for col in base_df.columns if col.startswith('SLC_')]
        if slc_cols:
            base_df[slc_cols] = base_df[slc_cols].div(base_df[slc_cols].sum(axis=1), axis=0).fillna(0)

        # Keep glacier metadata consistent with the classes actually simulated.
        glacier_slc_cols = [f'SLC_{int(r.SLC)}' for r in slc_df.itertuples() if r.landcover == 15]
        base_df['glacier_fraction'] = base_df[glacier_slc_cols].sum(axis=1)

        # 7. Drop sub-basins with missing geometry (no area/lat/lon from join)
        required_cols = ['area', 'latitude', 'longitude']
        missing_mask = base_df[required_cols].isna().any(axis=1)
        if missing_mask.any():
            dropped_ids = base_df.loc[missing_mask, 'subid'].tolist()
            self.logger.warning(
                f"Dropping {missing_mask.sum()} sub-basins with missing geometry: {dropped_ids}"
            )
            base_df = base_df[~missing_mask].copy()
            # Remove any maindown references to dropped sub-basins
            base_df.loc[base_df['maindown'].isin(dropped_ids), 'maindown'] = 0

        # 7b. Expand into elevation-band sub-basins (semi-distributed HYPE).
        # Done after geometry/SLC/glacier are resolved per original sub-basin so
        # each band inherits clean parent attributes; gated on the caller passing
        # an elevation-band HRU shapefile (i.e. SUB_GRID_DISCRETIZATION=elevation).
        if elevation_band_shapefile is not None and Path(elevation_band_shapefile).exists():
            bands_by_parent = self._load_elevation_bands(
                Path(elevation_band_shapefile), base_df
            )
            if bands_by_parent:
                n_before = len(base_df)
                base_df = self._build_banded_geodata(base_df, bands_by_parent, glacier_slc_cols)
                self.logger.info(
                    "Elevation banding: expanded %d sub-basin(s) into %d "
                    "elevation-band sub-basins (elev range %.0f-%.0f m).",
                    n_before, len(base_df),
                    base_df['elev_mean'].min(), base_df['elev_mean'].max(),
                )

        # 8. Handle ID shifting (HYPE requires IDs > 0)
        base_df = self._shift_ids_if_needed(base_df)

        # 9. Sort and save
        sorted_df = self.sort_geodata(base_df)
        sorted_df = self.apply_river_length_overrides(sorted_df)
        volume_correction = sorted_df.pop('_glacier_logvolcor') if '_glacier_logvolcor' in sorted_df else None
        sorted_df.to_csv(self.output_path / 'GeoData.txt', sep='\t', index=False)

        # 9. Write ForcKey.txt (required for readobsid=y)
        self._write_forckey(sorted_df)

        # 10. Write GeoClass.txt
        self._write_geoclass(slc_df)

        glacier_type = self.config.get('HYPE_GLACIER_TYPE')
        if glacier_type is not None:
            glacier_cols = [f'SLC_{int(r.SLC)}' for r in slc_df.itertuples() if r.landcover == 15]
            glacier_rows = sorted_df.loc[sorted_df[glacier_cols].sum(axis=1) > 0, ['subid']].copy()
            glacier_rows['glactype'] = glacier_type
            if volume_correction is not None:
                glacier_rows['logvolcor'] = volume_correction.loc[glacier_rows.index]
            glacier_rows.to_csv(self.output_path / 'GlacierData.txt', sep='\t', index=False)

        self.logger.debug("GeoData.txt, GeoClass.txt, and ForcKey.txt created successfully")

        # Return land use information for parameter file generation
        return slc_df['landcover'].unique()

    def _get_projected_centroids(self, gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """
        Return centroids as GEOGRAPHIC (lat/lon, EPSG:4326) points.

        Centroids are computed in a projected CRS (to avoid the geographic-
        centroid warning) and then reprojected to degrees. The previous version
        returned centroids in the *projected* CRS when the input was projected
        (e.g. EPSG:3057), so HYPE's latitude/longitude fields were filled with
        projected metres instead of degrees — corrupting latitude-dependent PET.
        """
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        if gdf.crs.is_geographic:
            centroids = gdf.to_crs(epsg=3857).geometry.centroid
        else:
            centroids = gdf.geometry.centroid  # already projected → accurate
        return centroids.to_crs(epsg=4326)

    def _calculate_geometry_area(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Calculate area from geometry using an equal-area projection.

        This method calculates the true geodetic area of each polygon by projecting
        to an appropriate equal-area coordinate system. This is more reliable than
        using stored area attributes, which may have been calculated incorrectly or
        in a different CRS.

        Args:
            gdf: GeoDataFrame with polygon geometries

        Returns:
            numpy array of area values in square meters (m²)

        Note:
            Uses an Albers Equal Area projection centered on the data extent
            for accurate area calculations regardless of input CRS.
        """
        if gdf.crs is None:
            self.logger.warning("GeoDataFrame has no CRS, assuming EPSG:4326")
            gdf = gdf.set_crs(epsg=4326)

        # Derive the Albers parallels/centre from GEOGRAPHIC bounds. The input
        # may be in a projected CRS (e.g. EPSG:3057, bounds in metres); using
        # those metre values as +lat_1/+lat_2 produces an invalid projection
        # that silently fell back to EPSG:3857 — which inflates area by
        # ~1/cos^2(lat) (≈6x at 66°N). Reproject to 4326 first to get degrees.
        gdf_geo = gdf.to_crs(epsg=4326)
        bounds = gdf_geo.total_bounds  # [minlon, minlat, maxlon, maxlat] in degrees
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2

        aea_proj = (
            f"+proj=aea +lat_1={bounds[1]} +lat_2={bounds[3]} "
            f"+lat_0={center_lat} +lon_0={center_lon} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )

        try:
            areas = gdf_geo.to_crs(aea_proj).geometry.area.values
        except Exception as e:  # noqa: BLE001 — model execution resilience
            # Last resort: a global equal-area CRS (still correct, just less
            # locally optimal) — NOT Web Mercator, which distorts area badly.
            self.logger.warning(f"Local equal-area projection failed ({e}); using EPSG:6933", exc_info=True)
            areas = gdf_geo.to_crs(epsg=6933).geometry.area.values

        return areas

    def _load_gis_stats(
        self,
        gistool_output: Path,
        intersect_base_path: Optional[Path],
        basin_id_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Robustly load GIS statistics from the model-ready attributes store,
        falling back to gistool CSVs or intersection shapefiles."""
        store_stats = self._load_gis_stats_from_store(basin_id_col)
        if store_stats is not None:
            return store_stats

        def find_data(pattern: str, fallback_shp_path: Optional[str] = None) -> Optional[pd.DataFrame]:
            files = list(gistool_output.glob(pattern))
            if files:
                df = pd.read_csv(files[0])
                # Be robust with index name
                idx_col = basin_id_col if basin_id_col in df.columns else ('ID' if 'ID' in df.columns else df.columns[0])
                return df.set_index(idx_col)

            if intersect_base_path and fallback_shp_path:
                shp_files = list(Path(intersect_base_path).parent.glob(fallback_shp_path))
                if shp_files:
                    gdf = gpd.read_file(shp_files[0])
                    idx_col = basin_id_col if basin_id_col in gdf.columns else ('ID' if 'ID' in gdf.columns else gdf.columns[0])
                    return gdf.set_index(idx_col)
            return None

        soil = find_data('*stats_soil_classes.csv', 'with_soilgrids/*soilclass.shp')
        land = find_data('*stats_*landcover*.csv', 'with_landclass/*landclass.shp')
        elev = find_data('*stats_elv.csv', 'with_dem/*dem.shp')

        if soil is None or land is None or elev is None:
            raise FileNotFoundError(
                f"Required geospatial statistics not found. "
                f"Checked {gistool_output} and {intersect_base_path}"
            )

        return soil, land, elev

    @staticmethod
    def _coerce_basin_ids(ids: list) -> list:
        """Coerce store HRU ids to integers where possible (basin ids are ints)."""
        out = []
        for hid in ids:
            try:
                out.append(int(float(hid)))
            except (ValueError, TypeError):
                out.append(hid)
        return out

    def _store_class_frame(
        self, reader, group: str, frac_var: str, name_var: str, index_name: str
    ) -> Optional[pd.DataFrame]:
        """Build a per-basin class-fraction DataFrame from an attributes group.

        Columns are the class names (``USGS_<id>`` / ``IGBP_<id>``) and the index
        is the basin id, matching the gistool/shapefile layout ``_process_slc``
        consumes. Returns ``None`` when the group is unavailable.
        """
        if not reader.has_group(group):
            return None
        ids = reader.hru_ids(group)
        if not ids:
            return None
        with reader.group(group) as ds:
            if frac_var not in ds.variables or name_var not in ds.variables:
                return None
            frac = np.asarray(ds[frac_var].values)
            names = [str(x) for x in np.atleast_1d(ds[name_var].values)]
        if frac.ndim != 2 or frac.shape[0] != len(ids):
            return None
        df = pd.DataFrame(frac, columns=names, index=self._coerce_basin_ids(ids))
        df.index.name = index_name
        return df

    def _load_gis_stats_from_store(
        self, basin_id_col: str
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Build (soil, landcover, elevation) stats from the model-ready
        attributes store, or ``None`` when unavailable.

        No drift: the store's soil/landcover/terrain values originate from the
        same intersection shapefiles the CSV/shapefile fallback reads.
        """
        try:
            from symfluence.core.modeling.model_ready import open_canonical_attributes

            data_dir = self.config.get('SYMFLUENCE_DATA_DIR')
            domain = self.config.get('DOMAIN_NAME')
            if not data_dir or not domain:
                return None
            project_dir = Path(data_dir) / f'domain_{domain}'
            reader = open_canonical_attributes(project_dir, domain)
            if reader is None:
                return None

            soil = self._store_class_frame(
                reader, 'soil', 'soil_fraction', 'soil_class_name', basin_id_col)
            land = self._store_class_frame(
                reader, 'landcover', 'land_fraction', 'land_class_name', basin_id_col)

            elev = None
            if reader.has_group('terrain'):
                ids = reader.hru_ids('terrain')
                elev_vals = reader.variable('terrain', 'elev_mean')
                if ids and elev_vals is not None and len(elev_vals) == len(ids):
                    elev = pd.DataFrame(
                        {'elev_mean': np.asarray(elev_vals)},
                        index=self._coerce_basin_ids(ids),
                    )
                    elev.index.name = basin_id_col

            if soil is None or land is None or elev is None:
                return None
            self.logger.info(
                "Loaded HYPE GIS statistics from the model-ready attributes store")
            return soil, land, elev
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(
                f"Could not load GIS stats from attributes store: {e}", exc_info=True)
            return None

    @staticmethod
    def _aggregate_band_stats(frame: pd.DataFrame, bands_by_parent: dict) -> pd.DataFrame:
        """Area-weight HRU statistics to parents; preserve categorical soil IDs."""
        ids = [int(b['hru_id']) for bands in bands_by_parent.values() for b in bands]
        if not ids or set(frame.index) != set(ids):
            return frame
        rows = {}
        for parent, bands in bands_by_parent.items():
            selected = frame.loc[[int(b['hru_id']) for b in bands]]
            weights = np.asarray([b['area_frac'] for b in bands], dtype=float)
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValidationError('Invalid elevation-band areas for GIS aggregation')
            weights /= weights.sum()
            values = selected.select_dtypes(include=np.number)
            row = values.mul(weights, axis=0).sum().to_dict()
            if 'majority' in selected:
                vote = pd.Series(weights).groupby(selected['majority'].to_numpy()).sum()
                row['majority'] = vote.idxmax()
            rows[parent] = row
        result = pd.DataFrame.from_dict(rows, orient='index')
        result.index.name = frame.index.name
        return result

    def _process_slc(
        self,
        base_df: pd.DataFrame,
        landcover_data: pd.DataFrame,
        soil_data: pd.DataFrame,
        threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate SLC combinations and fractions."""
        combinations_set: Set[Tuple[int, int]] = set()

        # Land-class columns are ONLY those with an integer class id after the
        # prefix (e.g. IGBP_10, frac_16). Plain-named fractions such as
        # 'frac_snow' are catchment attributes, NOT land classes — including
        # them breaks int() parsing and used to corrupt class ids to [1,2,3]
        # (which then matched no IGBP_<id> column → all SLC fractions = 0 →
        # zero discharge).
        def _lc_class_id(col: str) -> Optional[int]:
            for prefix in ('IGBP_', 'frac_'):
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    if suffix.isdigit():
                        return int(suffix)
            return None

        lc_cols = [col for col in landcover_data.columns if _lc_class_id(col) is not None]

        for basin_id in landcover_data.index:
            # Robust landcover retrieval
            if basin_id in landcover_data.index:
                basin_lc = landcover_data.loc[[basin_id]]
            elif len(landcover_data) == 1:
                basin_lc = landcover_data.iloc[[0]]
            else:
                continue

            active_lc = [col for col in lc_cols if basin_lc[col].values[0] > threshold]

            # Class id is the integer suffix (guaranteed present by _lc_class_id).
            lc_values = [_lc_class_id(col) for col in active_lc]

            # Robust soil retrieval
            if basin_id in soil_data.index:
                basin_soil_data = soil_data.loc[[basin_id]]
            elif len(soil_data) == 1:
                basin_soil_data = soil_data.iloc[[0]]
            else:
                basin_soil_data = None

            if basin_soil_data is not None and 'majority' in basin_soil_data.columns:
                soil_value = [basin_soil_data['majority'].values[0]]
            elif basin_soil_data is not None:
                usgs_cols = [col for col in basin_soil_data.columns if col.startswith('USGS_')]
                if usgs_cols:
                    soil_value = [int(basin_soil_data[usgs_cols].idxmax(axis=1).values[0].split('_')[1])]
                else:
                    soil_value = [1]
            else:
                soil_value = [1]

            combinations_set.update(product(lc_values, soil_value))

        slc_df = pd.DataFrame(list(combinations_set), columns=['landcover', 'soil'])
        # HYPE requires soil types >= 1, so remap 0 to 1
        slc_df['soil'] = slc_df['soil'].replace(0, 1)
        slc_df['SLC'] = range(1, len(slc_df) + 1)

        # Calculate SLC fractions for each basin
        for basin_id in base_df['subid']:
            # Robust landcover row retrieval
            if basin_id in landcover_data.index:
                basin_lc = landcover_data.loc[[basin_id]]
            elif len(landcover_data) == 1:
                basin_lc = landcover_data.iloc[[0]]
            else:
                basin_lc = None

            # Robust soil row retrieval
            if basin_id in soil_data.index:
                basin_soil_data = soil_data.loc[[basin_id]]
            elif len(soil_data) == 1:
                basin_soil_data = soil_data.iloc[[0]]
            else:
                basin_soil_data = None

            if basin_soil_data is not None and 'majority' in basin_soil_data.columns:
                basin_soil = basin_soil_data['majority'].values[0]
            elif basin_soil_data is not None:
                usgs_cols = [col for col in basin_soil_data.columns if col.startswith('USGS_')]
                basin_soil = int(basin_soil_data[usgs_cols].idxmax(axis=1).values[0].split('_')[1]) if usgs_cols else 1
            else:
                basin_soil = 1
            # Match the SLC table remap: soil type 0 (Water/NoData) → 1
            if basin_soil == 0:
                basin_soil = 1

            for slc_idx, (lc, soil) in enumerate(zip(slc_df['landcover'], slc_df['soil']), 1):
                lc_val = 0
                if basin_lc is not None:
                    for prefix in ['IGBP_', 'frac_']:
                        col = f'{prefix}{lc}'
                        if col in basin_lc.columns:
                            lc_val = basin_lc[col].values[0]
                            break

                if lc_val > threshold and basin_soil == soil:
                    base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = lc_val
                else:
                    base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = 0

        return slc_df, base_df

    # Multiplier used to derive band sub-basin IDs (parent*MULT + band_index).
    # Must exceed the maximum number of elevation bands per sub-basin.
    _BAND_ID_MULTIPLIER = 100

    def _load_elevation_bands(
        self, shapefile: Path, base_df: pd.DataFrame
    ) -> dict[int, list[dict[str, float]]]:
        """Read an elevation-banded HRU shapefile into per-parent band lists.

        Returns a mapping ``parent_subid -> [{'hru_id', 'elev_mean', 'area_frac'}]``.
        The ``hru_id`` becomes the band sub-basin id, so GeoData sub-basins line
        up with the forcing columns (which the forcing processor keys by the same
        HRU id). Robust to column-name variants.
        """
        try:
            table = load_elevation_band_table(shapefile)
        except (OSError, ValueError, KeyError) as e:
            self.logger.warning(
                "Could not read elevation-band shapefile %s: %s", shapefile, e,
                exc_info=True)
            return {}
        if not table:
            self.logger.warning(
                "Elevation-band shapefile %s has no elevation column; skipping banding.",
                shapefile)
            return {}

        single_parent = len(base_df) == 1
        if single_parent:
            pid = int(base_df['subid'].iloc[0])
            return {pid: [{'hru_id': b['hru_id'], 'elev_mean': b['elev_mean'],
                           'area_frac': b['area']} for b in table]}

        # Multi-sub-basin: group bands by their parent GRU id.
        gdf = gpd.read_file(shapefile)
        parent_col = next((c for c in ('GRU_ID', 'gru_id', 'GRU', 'COMID', 'gruId')
                           if c in gdf.columns), None)
        if parent_col is None:
            self.logger.warning(
                "Multi-sub-basin banding needs a parent (GRU) column in %s; skipping.",
                shapefile)
            return {}
        id_col = next((c for c in ('HRU_ID', 'hru_id', 'hruId', 'HRUID')
                       if c in gdf.columns), None)
        elev_col = next((c for c in ('elev_mean', 'elevMean', 'mean', 'ELEV_MEAN',
                                     'elevation', 'avg_elevcl') if c in gdf.columns), None)
        area_col = next((c for c in ('HRU_area', 'HRU_AREA', 'area', 'Area', 'AREA')
                         if c in gdf.columns), None)
        bands_by_parent: dict[int, list[dict[str, float]]] = {}
        for i, (_, row) in enumerate(gdf.iterrows()):
            pid = int(row[parent_col])
            bands_by_parent.setdefault(pid, []).append({
                'hru_id': int(row[id_col]) if id_col is not None else i + 1,
                'elev_mean': float(row[elev_col]),
                'area_frac': float(row[area_col]) if area_col is not None else 1.0,
            })
        return bands_by_parent

    def _build_banded_geodata(
        self,
        base_df: pd.DataFrame,
        bands_by_parent: dict[int, list[dict[str, float]]],
        glacier_slc_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Expand each sub-basin into a vertical cascade of elevation bands.

        Pure DataFrame transform (no I/O) so it is unit-testable. Per band:
        - ``subid``: the band's ``hru_id`` when available (so it matches the
          forcing columns), else ``parent*MULT + j`` (j = 1..N, lowest first).
        - ``maindown``: band drains to the next lower band; the lowest band drains
          to the parent's downstream sub-basin's outlet band (or 0 at the domain
          outlet) — water flows high -> low -> downstream.
        - ``area``: parent area * band area fraction.
        - ``elev_mean``: the band's mean elevation (drives HYPE's per-sub-basin
          temperature/precipitation lapse — the point of banding).
        - ``glacier_fraction``: the parent's glacier *area* concentrated into the
          highest band(s), which is where alpine ice actually sits.
        - With glacier SLC columns supplied, actual glacier classes follow the
          redistributed ice area; non-glacier classes retain their relative shares.
          Otherwise SLC fractions are inherited for backward compatibility.
        """
        mult = self._BAND_ID_MULTIPLIER
        # Use real HRU ids as sub-basin ids only when every band carries one.
        use_hru_id = all(
            all(b.get('hru_id') is not None for b in bands)
            for bands in bands_by_parent.values() if bands)

        def band_subid(pid: int, rank: int, band: dict[str, float]) -> int:
            # rank is 1-based position by ascending elevation.
            return int(band['hru_id']) if use_hru_id else pid * mult + rank

        # Each parent's outlet (lowest-elevation) band, for downstream routing.
        parent_outlet: dict[int, int] = {}
        for pid, bands in bands_by_parent.items():
            if bands:
                low = sorted(bands, key=lambda b: b['elev_mean'])[0]
                parent_outlet[pid] = band_subid(pid, 1, low)

        out_rows: list[dict[str, Any]] = []
        for _, prow in base_df.iterrows():
            pid = int(prow['subid'])
            bands = bands_by_parent.get(pid)
            if not bands:
                out_rows.append(prow.to_dict())
                continue
            n = len(bands)
            if not use_hru_id and n >= mult:
                self.logger.warning(
                    "Sub-basin %d has %d elevation bands (>= id multiplier %d); "
                    "keeping it lumped to avoid id collisions.", pid, n, mult)
                out_rows.append(prow.to_dict())
                continue

            bands_sorted = sorted(bands, key=lambda b: b['elev_mean'])
            total_frac = sum(b['area_frac'] for b in bands_sorted) or 1.0
            parent_area = float(prow['area'])
            band_areas = [b['area_frac'] / total_frac * parent_area for b in bands_sorted]
            subids = [band_subid(pid, k + 1, b) for k, b in enumerate(bands_sorted)]

            # Distribute the parent's glacier area from the top band downward.
            parent_glacier_fraction = (float(prow[glacier_slc_cols].sum())
                                       if glacier_slc_cols is not None
                                       else float(prow.get('glacier_fraction', 0.0)))
            remaining_glac = parent_glacier_fraction * parent_area
            band_glac = [0.0] * n
            for j in range(n - 1, -1, -1):
                take = min(band_areas[j], remaining_glac)
                band_glac[j] = take
                remaining_glac -= take
                if remaining_glac <= 1e-9:
                    break

            parent_maindown = int(prow['maindown'])
            parent_rivlen = float(prow.get('rivlen', 0.0))
            for k, band in enumerate(bands_sorted):
                row = prow.to_dict()
                row['subid'] = subids[k]
                if k == 0:
                    row['maindown'] = (0 if parent_maindown <= 0
                                       else parent_outlet.get(parent_maindown, 0))
                else:
                    row['maindown'] = subids[k - 1]
                row['grwdown'] = row['maindown']
                row['area'] = band_areas[k]
                row['elev_mean'] = band['elev_mean']
                row['glacier_fraction'] = (
                    band_glac[k] / band_areas[k] if band_areas[k] > 0 else 0.0)
                if self.config.get('HYPE_PRESERVE_ICECAP_VOLUME', False):
                    if self.config.get('HYPE_GLACIER_TYPE') != 1:
                        raise ValidationError('HYPE_PRESERVE_ICECAP_VOLUME requires glacier type 1')
                    # HYPE defaults: V = exp(logvolcor) * 1.701 * A**1.25.
                    # Each band retains the parent's mean ice thickness; the
                    # inverse area relationship consequently retains its area.
                    row['_glacier_logvolcor'] = (
                        0.25 * np.log(parent_glacier_fraction * parent_area / band_glac[k])
                        if band_glac[k] > 0 else 0.0)
                # HYPE consumes SLC fractions, not the glacier_fraction metadata.
                # Redistribute actual glacier classes and preserve the relative
                # shares of non-glacier classes, conserving every class's area.
                if glacier_slc_cols:
                    target = row['glacier_fraction']
                    for col in (c for c in base_df.columns if c.startswith('SLC_')):
                        if col in glacier_slc_cols:
                            row[col] = (float(prow[col]) * target / parent_glacier_fraction
                                        if parent_glacier_fraction > 0 else 0.0)
                        else:
                            row[col] = (float(prow[col]) * (1 - target) / (1 - parent_glacier_fraction)
                                        if parent_glacier_fraction < 1 else 0.0)
                # Full channel length only for the valley (outlet) band; internal
                # band-to-band links are vertical, not channel reaches.
                if k > 0:
                    row['rivlen'] = min(parent_rivlen, 100.0)
                out_rows.append(row)

        result = pd.DataFrame(out_rows).reset_index(drop=True)
        # Keep id/routing columns integer so GeoData.txt never writes "100.0"
        # (HYPE parses sub-basin ids as integers).
        for col in ('subid', 'maindown', 'grwdown'):
            if col in result.columns:
                result[col] = result[col].astype(int)
        return result

    def _shift_ids_if_needed(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Shift IDs if they start from 0 (HYPE requires > 0).

        Note: HYPEForcingProcessor also shifts forcing IDs, so this must be consistent.
        """
        if base_df['subid'].min() == 0:
            self.logger.debug("Shifting subids +1 for HYPE compatibility (0-based to 1-based)")

            # Get original IDs for checking connectivity
            original_ids = set(base_df['subid'])

            # Shift subids
            base_df['subid'] = base_df['subid'] + 1

            # Update maindown: map valid connections to shifted ID, set others (outlets) to 0
            def update_downstream(val):
                if val in original_ids:
                    return val + 1
                return 0  # Outlet

            base_df['maindown'] = base_df['maindown'].apply(update_downstream)

        return base_df

    def _write_forckey(self, sorted_df: pd.DataFrame) -> None:
        """
        Write ForcKey.txt (required for readobsid=y).

        Maps subid to the station id in forcing files (which we set to subid).
        """
        forckey_df = pd.DataFrame({
            'subid': sorted_df['subid'],
            'stationid': sorted_df['subid']
        })
        forckey_df.to_csv(self.output_path / 'ForcKey.txt', sep='\t', index=False)
        self.logger.debug("ForcKey.txt created")

    def sort_geodata(self, geodata: pd.DataFrame) -> pd.DataFrame:
        """
        Sort sub-basins from upstream to downstream using topological sorting.

        HYPE requires basins to be ordered such that all upstream basins
        appear before their downstream basins. This uses networkx's
        topological sort which guarantees this ordering.
        """
        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("networkx not installed, skipping topological sort")
            return geodata

        # Create directed graph from subid -> maindown relationships
        G = nx.DiGraph()
        all_subids = set(geodata['subid'].tolist())

        for _, row in geodata.iterrows():
            subid = row['subid']
            maindown = row['maindown']
            # Add all nodes to ensure isolated nodes are included
            G.add_node(subid)
            if maindown > 0 and maindown in all_subids:
                # Edge from upstream to downstream
                G.add_edge(subid, maindown)

        # Find and break cycles if they exist
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                self.logger.warning(f"Found {len(cycles)} circular reference(s) in the network")
                for cycle in cycles:
                    # Find the node in the cycle with the most downstream connections
                    max_downstream = max(cycle, key=lambda n: len(list(nx.descendants(G, n))))
                    cycle_idx = cycle.index(max_downstream)
                    from_node = cycle[cycle_idx - 1]
                    G.remove_edge(from_node, max_downstream)
                    self.logger.warning(f"Breaking cycle at edge: {from_node} -> {max_downstream}")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not check for cycles: {e}", exc_info=True)

        try:
            # Use networkx topological sort - this guarantees upstream before downstream
            # Since edges go from upstream to downstream, topological_sort gives correct order
            final_order = list(nx.topological_sort(G))

            # Handle nodes that weren't in the graph (shouldn't happen, but safety check)
            missing_subids = geodata[~geodata['subid'].isin(final_order)]['subid'].tolist()
            if missing_subids:
                self.logger.warning(f"Found {len(missing_subids)} basins not in network, adding at start")
                final_order = missing_subids + final_order

            # Create a mapping from subid to desired position
            position_map = {subid: pos for pos, subid in enumerate(final_order)}

            # Sort geodata based on the position map
            geodata = geodata.copy()
            geodata['sort_idx'] = geodata['subid'].map(position_map)
            geodata = geodata.sort_values('sort_idx', ignore_index=True)
            geodata = geodata.drop(columns=['sort_idx'])

            # Verify the sorting
            errors = 0
            for i, row in geodata.iterrows():
                if row['maindown'] > 0:
                    downstream_rows = geodata[geodata['subid'] == row['maindown']]
                    if not downstream_rows.empty:
                        downstream_idx = downstream_rows.index[0]
                        if downstream_idx < i:
                            errors += 1
                            if errors <= 3:  # Only log first few
                                self.logger.warning(
                                    f"Basin {row['subid']} (idx={i}) appears after its "
                                    f"downstream basin {row['maindown']} (idx={downstream_idx})"
                                )

            if errors > 0:
                self.logger.error(f"Topological sort failed: {errors} ordering violations found")
            else:
                self.logger.debug("Topological sort successful: all basins correctly ordered")

            return geodata

        except nx.NetworkXUnfeasible:
            self.logger.error("Graph has cycles that could not be resolved")
            return geodata
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error during topological sorting: {str(e)}", exc_info=True)
            return geodata

    def apply_river_length_overrides(self, geodata: pd.DataFrame) -> pd.DataFrame:
        """Apply explicit, validated reach lengths to final model subbasin IDs."""
        overrides = self.config.get('HYPE_RIVER_LENGTH_OVERRIDES') or {}
        result = geodata.copy()
        for subid, length in overrides.items():
            if not np.isfinite(length) or length <= 0:
                raise ValidationError('HYPE river length overrides must be finite and positive')
            mask = result.subid == int(subid)
            if mask.sum() != 1:
                raise ValidationError(f'River length override subid {subid} does not identify one subbasin')
            result.loc[mask, 'rivlen'] = float(length)
        return result

    def _write_geoclass(self, slc_df: pd.DataFrame) -> None:
        """Write GeoClass.txt file with full metadata and specific formatting."""
        combination = slc_df.copy()
        combination = combination.rename(columns={'landcover': 'LULC', 'soil': 'SOIL TYPE'})
        combination = combination[['SLC', 'LULC', 'SOIL TYPE']]

        combination['Main crop cropid'] = 0
        combination['Second crop cropid'] = 0
        combination['Crop rotation group'] = 0
        combination['Vegetation type'] = 1
        # HYPE's glacier_model is 3; code 2 is an internal lake.
        combination['Special class code'] = combination['LULC'].apply(lambda x: 3 if x == 15 else 0)
        soil_depths = self.config.get('HYPE_SOIL_LAYER_DEPTHS')
        if soil_depths and len(soil_depths) == 3:
            d1, d2, d3 = [float(d) for d in soil_depths]
        else:
            d1, d2, d3 = 0.091, 0.493, 2.296
        combination['Tile depth'] = 0
        combination['Stream depth'] = d3
        combination['Number of soil layers'] = 3
        combination['Soil layer depth 1'] = d1
        combination['Soil layer depth 2'] = d2
        combination['Soil layer depth 3'] = d3

        # newline='' + an explicit lineterminator, or this file is silently
        # corrupt on Windows. Without newline='', pandas writes os.linesep
        # ('\r\n' here) and Python's text layer translates the '\n' again, so
        # every data row ends '\r\r\n' while the f.write header ends '\r\n'.
        # HYPE then parses only the header, reports nclass=1, and skips the
        # remaining land classes with "slc larger than nclass skipped" — on
        # this catchment that silently removed 85% of the area from runoff
        # generation and collapsed every calibration score to ~0.
        # Pinning '\n' also makes the file byte-identical across platforms,
        # which a reproduction campaign depends on.
        with open(self.output_path / 'GeoClass.txt', 'w', encoding='utf-8',
                  newline='') as f:
            f.write(
                "!          SLC\tLULC\tSOIL TYPE\tMain crop cropid\tSecond crop cropid\t"
                "Crop rotation group\tVegetation type\tSpecial class code\tTile depth\t"
                "Stream depth\tNumber of soil layers\tSoil layer depth 1\tSoil layer depth 2\t"
                "Soil layer depth 3 \n"
            )
            combination.to_csv(f, sep='\t', index=False, header=False,
                               lineterminator='\n')

        self.logger.debug("GeoClass.txt created")
