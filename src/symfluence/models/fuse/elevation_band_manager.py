# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Elevation band management for FUSE model.

This module contains the FuseElevationBandManager class, which writes the FUSE
elevation-band file from the real hypsometry of each spatial unit, and the
helpers that read that file back for the runner and the calibration optimizer.

Band definition
---------------
Each spatial unit (the whole catchment in lumped mode, one subcatchment per
forcing cell in distributed mode) is cut into ``FUSE_N_ELEVATION_BANDS``
equal-area bands of its own hypsometric curve. A band's elevation is the mean
elevation of the area inside it, so ``sum(area_frac * mean_elev)`` is the
unit's true mean elevation -- the elevation the unit's forcing represents and
the value FUSE uses as the forcing elevation.

The hypsometric curve comes from, in order:

1. the DEM pixels inside the unit's polygons;
2. the unit's HRU table (``elev_mean`` with HRU areas) when it has more than
   one HRU, e.g. an elevation-band discretisation;
3. otherwise only a mean elevation is known: a single band at that elevation
   is used and this is logged. No elevation range is ever invented.

A unit with no elevation information at all is an error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.modeling.spatial_modes import SpatialMode

_BAND_VARS = ('area_frac', 'mean_elev', 'prec_frac')
_AREA_COLUMNS = ('HRU_area', 'GRU_area', 'area', 'Area')


def equal_area_bands(
    elevations: Sequence[float],
    weights: Optional[Sequence[float]],
    n_bands: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut an elevation sample into ``n_bands`` equal-area bands, lowest first.

    Args:
        elevations: elevation of each sample (DEM pixel or HRU).
        weights: area of each sample; None for equal areas (DEM pixels).
        n_bands: number of bands.

    Returns:
        ``(mean_elev, area_frac)``: the area-weighted mean elevation of each
        band and its area fraction (``1 / n_bands`` each). A sample straddling
        a band boundary is split between the two bands in proportion, so the
        fractions are exact and ``sum(area_frac * mean_elev)`` equals the
        sample's area-weighted mean elevation.
    """
    z = np.asarray(elevations, dtype=float).ravel()
    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=float).ravel()
    keep = np.isfinite(z) & np.isfinite(w) & (w > 0)
    z, w = z[keep], w[keep]
    if z.size == 0:
        raise ModelExecutionError("No valid elevation samples to build elevation bands from")
    order = np.argsort(z, kind='stable')
    z, w = z[order], w[order]

    # Cumulative area (0..1) at the lower and upper edge of each sample.
    upper = np.cumsum(w) / w.sum()
    lower = np.concatenate([[0.0], upper[:-1]])
    upper[-1] = 1.0
    edges = np.linspace(0.0, 1.0, n_bands + 1)

    mean_elev = np.empty(n_bands)
    for i in range(n_bands):
        # Share of each sample's area lying inside band i
        overlap = np.clip(np.minimum(upper, edges[i + 1]) - np.maximum(lower, edges[i]), 0.0, None)
        mean_elev[i] = float(np.sum(overlap * z) / np.sum(overlap))
    return mean_elev, np.full(n_bands, 1.0 / n_bands)


def read_band_parameters(path: Path) -> Dict[str, Any]:
    """Elevation-band parameters FUSE takes as scalars, read from a bands file.

    Returns a dict with ``n_bands`` (int), ``mean_elev`` and ``area_frac``
    (1-D, one value per band) and ``z_forcing`` -- the area-weighted mean
    elevation ``sum(area_frac * mean_elev)``, i.e. the elevation the unit's
    forcing represents.

    A file holding several spatial units carries one band set per unit, but
    FUSE's parameter files hold one scalar per band; the first unit's bands
    are returned for them.
    """
    with xr.open_dataset(path) as ds:
        n_bands = int(ds.sizes.get('elevation_band', 1))

        def first_unit(name: str) -> np.ndarray:
            da = ds[name]
            if 'elevation_band' in da.dims:
                da = da.transpose('elevation_band', ...)
            return np.asarray(da.values, dtype=float).reshape(n_bands, -1)[:, 0]

        mean_elev = first_unit('mean_elev')
        area_frac = first_unit('area_frac')
    return {
        'n_bands': n_bands,
        'mean_elev': mean_elev,
        'area_frac': area_frac,
        'z_forcing': float(np.sum(mean_elev * area_frac)),
    }


def write_unit_bands(source: Path, target: Path, unit_dim: str, unit_id: float) -> Path:
    """Write the elevation bands of one spatial unit of *source* to *target*.

    The unit is the cell of ``unit_dim`` whose coordinate equals ``unit_id``
    (the subcatchment id). The spatial dimension is kept with length 1, as in
    the per-subcatchment forcing file.
    """
    with xr.open_dataset(source) as ds:
        if unit_dim not in ds.dims:
            raise ModelExecutionError(f"Elevation bands file {source} has no '{unit_dim}' dimension")
        matches = np.flatnonzero(np.isclose(np.asarray(ds[unit_dim].values, dtype=float), float(unit_id)))
        if matches.size != 1:
            raise ModelExecutionError(
                f"Subcatchment {unit_id} not found (or not unique) along '{unit_dim}' in {source}"
            )
        unit = ds.isel({unit_dim: slice(int(matches[0]), int(matches[0]) + 1)}).load()
    unit.to_netcdf(target, encoding=_band_encoding(unit))
    return target


def _band_encoding(ds: xr.Dataset) -> Dict[str, Dict[str, Any]]:
    """Strict encoding FUSE expects: float32 band variables, float64 coordinates."""
    encoding: Dict[str, Dict[str, Any]] = {
        name: {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False} for name in _BAND_VARS
    }
    for coord in ds.coords:
        encoding[str(coord)] = {'dtype': 'float64', '_FillValue': None}
    return encoding


class FuseElevationBandManager(ConfigMixin):
    """
    Manager for FUSE elevation band creation in lumped and distributed modes.

    One builder serves both modes; they differ only in how the catchment
    shapefile's features are grouped into spatial units. See the module
    docstring for how the bands are defined.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        project_dir: Root project directory
        forcing_fuse_path: Path to FUSE forcing output directory
        catchment_path: Path to catchment shapefile
        domain_name: Domain name for file naming
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        project_dir: Path,
        forcing_fuse_path: Path,
        catchment_path: Path,
        domain_name: str,
        calculate_catchment_centroid_callback
    ):
        """
        Initialize the elevation band manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Root project directory
            forcing_fuse_path: Path to FUSE forcing output directory
            catchment_path: Path to catchment shapefile
            domain_name: Domain name for file naming
            calculate_catchment_centroid_callback: Callback to parent's calculate_catchment_centroid method
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.project_dir = Path(project_dir)
        self.forcing_fuse_path = Path(forcing_fuse_path)
        self.catchment_path = Path(catchment_path)
        self.domain_name = domain_name
        self.calculate_catchment_centroid = calculate_catchment_centroid_callback

    def create_elevation_bands(self) -> Path:
        """
        Create the elevation bands file for the configured spatial mode.

        Returns:
            Path to created elevation bands file

        Raises:
            ModelExecutionError: unknown spatial mode, spatial units that cannot
                be matched to the catchment shapefile, or a unit without any
                elevation information.
        """
        spatial_mode = self._get_config_value(lambda: self.config.model.fuse.spatial_mode, default='lumped', dict_key='FUSE_SPATIAL_MODE')
        if spatial_mode == SpatialMode.LUMPED:
            mode = 'lumped'
        elif spatial_mode in (SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED):
            mode = 'distributed'
        else:
            raise ModelExecutionError(f"Unknown FUSE spatial mode: {spatial_mode}")
        self.logger.info(f"Creating elevation bands for {spatial_mode} mode")

        try:
            catchment = gpd.read_file(self.catchment_path)
        except (FileNotFoundError, OSError, PermissionError) as e:
            self.logger.error(f"Error creating {mode} elevation bands: {str(e)}")
            raise

        spatial_dims, spatial_coords = self._spatial_layout(catchment)
        unit_shape = tuple(len(spatial_coords[dim]) for dim in spatial_dims)
        units = self._unit_rows(catchment, spatial_dims, spatial_coords, mode)

        n_bands = int(self._get_config_value(lambda: self.config.model.fuse.n_elevation_bands, default=5, dict_key='FUSE_N_ELEVATION_BANDS'))
        samples = self._unit_hypsometry_samples(catchment, units)
        if all(np.ptp(z) == 0 for z, _ in samples):
            # Only a mean elevation per unit: several bands would all sit at it.
            if n_bands > 1:
                self.logger.warning(
                    "No elevation distribution available (no DEM pixels and no multi-HRU "
                    f"elevation table): using a single elevation band at the mean elevation "
                    f"instead of the {n_bands} configured"
                )
            n_bands = 1

        mean_elev = np.empty((n_bands, len(units)))
        area_frac = np.empty((n_bands, len(units)))
        for i, (z, w) in enumerate(samples):
            mean_elev[:, i], area_frac[:, i] = equal_area_bands(z, w, n_bands)

        # FUSE (Fortran) reads data with nf90_get_var using count=(/nSpat1,nSpat2,n_bands/)
        # In Fortran, dimensions are reversed from C/NetCDF order, so the NetCDF
        # dimension order must be (elevation_band, lat, lon) so that Fortran sees
        # (lon, lat, elevation_band) matching (/nSpat1, nSpat2, n_bands/)
        band_dims = ['elevation_band'] + spatial_dims
        band_shape = (n_bands,) + unit_shape

        coords = {dim: (dim, spatial_coords[dim]) for dim in spatial_dims}
        coords['elevation_band'] = ('elevation_band', np.arange(1, n_bands + 1))
        ds = xr.Dataset(coords=coords, attrs={
            'title': f'FUSE elevation bands ({mode} mode)',
            'source': 'Generated by SYMFLUENCE',
            'history': f'Created {pd.Timestamp.now()}',
            'spatial_mode': mode,
            'n_bands': n_bands
        })
        for name, values, attrs in (
            ('area_frac', area_frac, {
                'long_name': 'Fraction of catchment area in each elevation band', 'units': '-'}),
            ('mean_elev', mean_elev, {
                'long_name': 'Mean elevation of each (equal-area) elevation band', 'units': 'm',
                'standard_name': 'height_above_reference_ellipsoid'}),
            # Precipitation is uniform per unit area, so each band receives its area share.
            ('prec_frac', area_frac, {
                'long_name': 'Fraction of catchment precipitation that falls on each elevation band',
                'units': '-'}),
        ):
            ds[name] = xr.DataArray(
                values.reshape(band_shape), dims=band_dims, attrs=attrs
            ).astype('float32')

        output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
        try:
            ds.to_netcdf(output_file, encoding=_band_encoding(ds))
        except (OSError, PermissionError) as e:
            self.logger.error(f"Error creating {mode} elevation bands: {str(e)}")
            raise

        self.logger.info(
            f"Created {mode} elevation bands file with {n_bands} band(s) for "
            f"{len(units)} spatial unit(s): {output_file}"
        )
        return output_file

    def _get_forcing_spatial_info(self) -> Optional[Tuple[list, Dict[str, np.ndarray]]]:
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        if not forcing_file.exists():
            return None

        try:
            with xr.open_dataset(forcing_file) as ds:
                data_var = next(iter(ds.data_vars.values()), None)
                if data_var is None:
                    return None

                spatial_dims: List[Any] = []
                spatial_coords: Dict[str, np.ndarray] = {}

                for dim in data_var.dims:
                    if dim == 'time':
                        continue
                    dim_str = str(dim)
                    spatial_dims.append(dim_str)
                    if dim in ds.coords:
                        spatial_coords[dim_str] = ds[dim].values
                    else:
                        spatial_coords[dim_str] = np.arange(ds.sizes[dim])

                return spatial_dims, spatial_coords
        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            self.logger.warning(f"Unable to read spatial info from {forcing_file}: {e}")
            return None

    def _spatial_layout(self, catchment: gpd.GeoDataFrame) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Spatial dimensions and coordinates of the bands file.

        Taken from the FUSE forcing file so the two match; without a forcing
        file, a single cell at the catchment centroid.
        """
        spatial_info = self._get_forcing_spatial_info()
        if spatial_info:
            return spatial_info
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        return ['latitude', 'longitude'], {
            'latitude': np.array([float(mean_lat)]),
            'longitude': np.array([float(mean_lon)]),
        }

    def _unit_rows(
        self,
        catchment: gpd.GeoDataFrame,
        spatial_dims: List[str],
        spatial_coords: Dict[str, np.ndarray],
        mode: str,
    ) -> List[np.ndarray]:
        """Row positions of *catchment* making up each spatial unit, in file order.

        A single-cell layout (always the case in lumped mode) is one unit made
        of every feature. Otherwise the subcatchments lie along the one spatial
        dimension longer than 1, whose coordinate holds the subcatchment ids
        (``GRU_ID``); each unit is the features with that id.
        """
        sizes = {dim: len(spatial_coords[dim]) for dim in spatial_dims}
        n_units = int(np.prod(list(sizes.values()))) if sizes else 1
        all_rows = np.arange(len(catchment))
        if n_units == 1:
            return [all_rows]
        if mode == 'lumped':
            raise ModelExecutionError(
                f"FUSE lumped mode expects a single forcing cell, but the forcing has {sizes}"
            )
        long_dims = [dim for dim, size in sizes.items() if size > 1]
        if len(long_dims) != 1:
            raise ModelExecutionError(
                f"Cannot match FUSE spatial units to subcatchments: forcing dimensions {sizes}"
            )
        ids = spatial_coords[long_dims[0]]
        if 'GRU_ID' in catchment.columns:
            gru_ids = catchment['GRU_ID'].to_numpy(dtype=float)
            units = [np.flatnonzero(np.isclose(gru_ids, float(unit_id))) for unit_id in ids]
            missing = [float(unit_id) for unit_id, rows in zip(ids, units) if rows.size == 0]
            if missing:
                raise ModelExecutionError(
                    f"FUSE subcatchment id(s) {missing[:10]} in the forcing file have no feature "
                    f"with that GRU_ID in {self.catchment_path.name}"
                )
            return units
        if len(catchment) != len(ids):
            raise ModelExecutionError(
                f"FUSE forcing has {len(ids)} subcatchments but {self.catchment_path.name} has "
                f"{len(catchment)} features and no GRU_ID column to match them by"
            )
        return [np.array([i]) for i in all_rows]

    def _unit_hypsometry_samples(
        self, catchment: gpd.GeoDataFrame, units: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """``(elevations, weights)`` describing each unit's hypsometry.

        DEM pixels inside the unit when available (weights None), else the
        unit's HRU table (``elev_mean`` weighted by HRU area). A unit with
        neither raises.
        """
        dem_samples = self._dem_samples(catchment, units)
        area_col = next((c for c in _AREA_COLUMNS if c in catchment.columns), None)
        samples: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        from_table = 0
        for i, rows in enumerate(units):
            if dem_samples is not None and dem_samples[i].size:
                samples.append((dem_samples[i], None))
                continue
            z = (catchment['elev_mean'].to_numpy(dtype=float)[rows]
                 if 'elev_mean' in catchment.columns else np.array([]))
            if z.size == 0 or not np.isfinite(z).any():
                raise ModelExecutionError(
                    f"No elevation information for FUSE spatial unit {i + 1}: no valid DEM pixels "
                    f"({self._dem_path()}) and no 'elev_mean' in {self.catchment_path.name}. "
                    "FUSE elevation bands cannot be built without it."
                )
            samples.append((z, self._hru_areas(catchment, area_col)[rows] if rows.size > 1 else None))
            from_table += 1
        if from_table:
            self.logger.warning(
                f"FUSE elevation bands: {from_table} of {len(units)} spatial unit(s) have no DEM "
                "pixels; using the 'elev_mean' of their HRUs instead"
            )
        return samples

    @staticmethod
    def _hru_areas(catchment: gpd.GeoDataFrame, area_col: Optional[str]) -> np.ndarray:
        """Area of each feature: the area column if present, else its geometry."""
        if area_col is not None:
            return catchment[area_col].to_numpy(dtype=float)
        geometries = catchment.geometry
        if catchment.crs is not None and catchment.crs.is_geographic:
            geometries = geometries.to_crs(catchment.estimate_utm_crs())
        return geometries.area.to_numpy(dtype=float)

    def _dem_path(self) -> Path:
        return (resolve_data_subdir(self.project_dir, 'attributes') / 'elevation' / 'dem'
                / f"domain_{self.domain_name}_elv.tif")

    def _dem_samples(
        self, catchment: gpd.GeoDataFrame, units: List[np.ndarray]
    ) -> Optional[List[np.ndarray]]:
        """Valid DEM pixel elevations inside each unit, or None without a DEM."""
        dem_path = self._dem_path()
        if not dem_path.exists():
            self.logger.info(f"No DEM at {dem_path}; FUSE elevation bands use the HRU elevation table")
            return None

        import rasterio
        import rasterio.mask

        out: List[np.ndarray] = []
        with rasterio.open(dem_path) as src:
            geometries = catchment.geometry
            if catchment.crs is not None and src.crs is not None and catchment.crs != src.crs:
                geometries = geometries.to_crs(src.crs)
            fill_value = src.nodata if src.nodata is not None else 0
            for rows in units:
                try:
                    image, _ = rasterio.mask.mask(
                        src, geometries.iloc[rows], crop=True, nodata=fill_value
                    )
                except ValueError:
                    # rasterio: "Input shapes do not overlap raster."
                    out.append(np.array([], dtype=float))
                    continue
                values = image[0][image[0] != fill_value].astype(float)
                out.append(values[np.isfinite(values)])
        return out
