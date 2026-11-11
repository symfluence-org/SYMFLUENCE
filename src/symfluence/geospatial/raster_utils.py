# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Raster processing utilities for geospatial analysis.

Provides functions for land cover mode calculation, aspect and radiation
derivation from DEMs, and raster value analysis for discretization.
"""
from __future__ import annotations

import glob
import logging
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pvlib
import rasterio
from scipy import stats


def _scipy_mode_compat(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute mode with scipy version compatibility.

    Handles scipy <1.11 (no keepdims) and >=1.11 (keepdims parameter).

    Args:
        data: Input array to compute mode over
        axis: Axis along which to compute the mode

    Returns:
        Array of mode values with the specified axis removed
    """
    try:
        # scipy >= 1.11.0
        result = stats.mode(data, axis=axis, keepdims=False)
        return result.mode
    except TypeError:
        # scipy 1.10.x - no keepdims parameter
        result = stats.mode(data, axis=axis)
        # Old scipy returns ModeResult with shape including axis dimension
        return np.squeeze(result.mode, axis=axis)


def calculate_landcover_mode(
    input_dir: Path,
    output_file: Path,
    start_year: int,
    end_year: int,
    domain_name: str
) -> None:
    """
    Calculate the temporal mode of land cover data across multiple years.

    This function computes the most frequently occurring land cover class for each
    pixel across a time series of annual land cover rasters. The mode calculation
    reduces year-to-year variability caused by classification noise, creating a
    more representative land cover map for hydrological modeling.

    Algorithm:
        1. Search for land cover rasters matching the domain and year pattern
        2. Load all matching rasters into a 3D array (time, height, width)
        3. Compute the statistical mode along the time axis using scipy.stats.mode
        4. Handle edge cases (single year, scipy version differences)
        5. Write the mode raster with the same georeference as inputs

    File Pattern:
        Primary: ``{input_dir}/domain_{domain_name}_*_{year}*.tif``
        Fallback: ``{input_dir}/domain_{domain_name}_*.tif`` if no year match

    Args:
        input_dir: Directory containing the yearly land cover GeoTIFF files
        output_file: Path where the output mode raster will be saved
        start_year: Start year for mode calculation (inclusive)
        end_year: End year for mode calculation (inclusive)
        domain_name: Domain name used in file pattern matching

    Returns:
        None. Writes the mode raster to output_file.

    Raises:
        FileNotFoundError: If no land cover files match the search pattern

    Notes:
        - Uses scipy.stats.mode which returns the smallest value if multimodal
        - Handles scipy version differences (pre/post keepdims parameter)
        - Creates output directory if it doesn't exist
        - Preserves input CRS, transform, and dtype in output
        - Sets nodata value to 0 in output
        - If only one year of data found, returns that year's data directly

    Example:
        >>> calculate_landcover_mode(
        ...     Path('/data/landcover'),
        ...     Path('/output/landcover_mode.tif'),
        ...     2015, 2020, 'bow_river'
        ... )
    """

    # Create a list to store the data from each year
    yearly_data = []
    meta = None

    # Get a list of files matching the pattern for the specified years
    file_pattern = f"{input_dir}/domain_{domain_name}_*_{start_year}*.tif"
    files = glob.glob(str(file_pattern))

    if not files:
        # If no files match the start year, try to find any files in the directory
        file_pattern = f"{input_dir}/domain_{domain_name}_*.tif"
        files = glob.glob(str(file_pattern))

    if not files:
        raise FileNotFoundError(f"No land cover files found matching pattern: {file_pattern}")

    # Read metadata from the first file
    with rasterio.open(files[0]) as src:
        meta = src.meta.copy()
        shape = (src.height, src.width)
        input_nodata = meta.get('nodata', None)

    # Read data for each year
    for year in range(start_year, end_year + 1):
        pattern = f"{input_dir}/domain_{domain_name}_*_{year}*.tif"
        year_files = glob.glob(str(pattern))

        if year_files:
            with rasterio.open(year_files[0]) as src:
                # Read the data and append to our list
                data = src.read(1)
                yearly_data.append(data)

    if not yearly_data:
        # If no yearly data was found, use the first file we found
        with rasterio.open(files[0]) as src:
            data = src.read(1)
            yearly_data.append(data)

    # Check if we have only one year of data
    if len(yearly_data) == 1:
        # Just use that single year's data
        mode_data = yearly_data[0]
    else:
        # Stack the arrays
        stacked_data = np.stack(yearly_data, axis=0)

        # Calculate the mode along the year axis (axis=0)
        # Use compatibility wrapper for scipy version differences
        mode_data = _scipy_mode_compat(stacked_data, axis=0)

    # Update the metadata for the output file
    # Preserve input nodata or use dtype-appropriate value to avoid conflicts
    # with valid land cover classes (e.g., 0 often represents water or background)
    if input_nodata is not None:
        output_nodata = input_nodata
    elif meta['dtype'] == 'uint8':
        output_nodata = 255  # Standard nodata for uint8 land cover
    else:
        output_nodata = -9999

    meta.update({
        'count': 1,
        'nodata': output_nodata
    })

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the result
    logger = logging.getLogger(__name__)

    with rasterio.open(output_file, 'w', **meta) as dst:
        # Make sure mode_data has the right shape
        if mode_data.ndim == 1 or mode_data.shape != shape:
            # If the shape doesn't match, reshape it to the expected dimensions
            if mode_data.size == shape[0] * shape[1]:
                logger.warning(
                    f"Mode data shape {mode_data.shape} reshaped to {shape}. "
                    "Verify output raster for correctness."
                )
                mode_data = mode_data.reshape(shape)
            else:
                logger.warning(
                    f"Mode data shape {mode_data.shape} does not match expected {shape}. "
                    f"Padding with zeros - DATA MAY BE INCOMPLETE."
                )
                # Create a new array with the correct shape
                new_mode_data = np.zeros(shape, dtype=meta['dtype'])

                # If mode_data is 1D but should be 2D
                if mode_data.ndim == 1:
                    # Take as many values as we can from mode_data
                    size = min(mode_data.size, shape[0] * shape[1])
                    new_mode_data.flat[:size] = mode_data[:size]
                else:
                    # If dimensions don't match but we can copy partial data
                    min_h = min(mode_data.shape[0], shape[0])
                    min_w = min(mode_data.shape[1], shape[1])
                    new_mode_data[:min_h, :min_w] = mode_data[:min_h, :min_w]

                mode_data = new_mode_data

        # Now write the data
        dst.write(mode_data, 1)


#: Metres per degree of latitude on the WGS84 sphere of equal area
_METRES_PER_DEGREE = 111_194.93


def terrain_slope_aspect(
    dem: np.ndarray,
    transform,
    crs=None,
    nodata=None,
):
    """
    Slope and downslope compass aspect of a DEM, from metric gradients.

    This is the single terrain-derivative implementation for the package.

    Conventions:
        - ``aspect_deg`` is the compass bearing the surface FACES (the downslope
          direction): 0 = north, 90 = east, 180 = south, 270 = west. A plane
          rising towards the north therefore has aspect 180.
        - ``slope_rad`` is the angle from horizontal, from gradients in metres
          per metre. For a geographic CRS the pixel size is converted with
          ~111.2 km per degree and ``cos(latitude)`` per row for the east-west
          spacing; for a projected CRS the CRS linear unit is used.
        - Nodata / non-finite cells are excluded: they are NaN in both outputs
          and never enter a neighbour's finite difference (one-sided
          differences are used next to them).

    Args:
        dem: 2-D elevation array (rows north to south for a north-up raster).
        transform: Affine transform of the array (rotation is not supported).
        crs: rasterio CRS (None is treated as projected, in metres).
        nodata: Nodata value of the DEM, if any.

    Returns:
        ``(slope_rad, aspect_deg, valid)`` — float arrays and the boolean mask
        of cells with data. ``aspect_deg`` is 0 where the surface is flat.
    """
    valid = valid_pixel_mask(dem, nodata)
    z = np.where(valid, np.ma.getdata(dem).astype(float), np.nan)

    def _nan_gradient(arr: np.ndarray, axis: int) -> np.ndarray:
        """Central difference; one-sided next to NaN/edges; 0 if isolated."""
        arr = np.moveaxis(arr, axis, 0)
        fwd = np.full_like(arr, np.nan)
        bwd = np.full_like(arr, np.nan)
        if arr.shape[0] > 1:
            diff = arr[1:] - arr[:-1]
            fwd[:-1] = diff
            bwd[1:] = diff
        both = np.isfinite(fwd) & np.isfinite(bwd)
        grad = np.where(both, 0.5 * (fwd + bwd), np.where(np.isfinite(fwd), fwd, bwd))
        grad = np.where(np.isfinite(grad), grad, 0.0)
        return np.moveaxis(grad, 0, axis)

    dz_drow = _nan_gradient(z, 0)
    dz_dcol = _nan_gradient(z, 1)

    n_rows = z.shape[0]
    if crs is not None and getattr(crs, 'is_geographic', False):
        row_lat = transform.f + (np.arange(n_rows) + 0.5) * transform.e
        metres_per_row = transform.e * _METRES_PER_DEGREE  # signed: negative when north-up
        cos_lat = np.maximum(np.cos(np.radians(row_lat)), 1e-6)
        metres_per_col = (transform.a * _METRES_PER_DEGREE * cos_lat)[:, np.newaxis]
    else:
        unit = 1.0
        if crs is not None:
            try:
                unit = float(crs.linear_units_factor[1])
            except (AttributeError, TypeError, ValueError, rasterio.errors.CRSError):
                unit = 1.0
        metres_per_row = transform.e * unit
        metres_per_col = transform.a * unit

    # ``y`` of the transform increases northward, so d/dnorth = d/drow / e
    dz_dnorth = dz_drow / metres_per_row
    dz_deast = dz_dcol / metres_per_col

    slope_rad = np.arctan(np.hypot(dz_deast, dz_dnorth))
    # Bearing of the downslope vector (-dz/deast, -dz/dnorth)
    aspect_deg = np.degrees(np.arctan2(-dz_deast, -dz_dnorth)) % 360.0

    slope_rad = np.where(valid, slope_rad, np.nan)
    aspect_deg = np.where(valid, aspect_deg, np.nan)
    return slope_rad, aspect_deg, valid


def calculate_aspect(dem_raster: Path, aspect_raster: Path, aspect_class_number: int, logger: Logger) -> Optional[Path]:
    """
    Calculate aspect (slope direction) from DEM and classify into directional classes.

    Aspect is the compass direction the slope faces (0 = north, 90 = east; see
    :func:`terrain_slope_aspect`). Classes: 0 = flat, 1..N clockwise from
    north; -9999 = nodata.

    Args:
        dem_raster: Path to the DEM raster
        aspect_raster: Path where the aspect raster will be saved
        aspect_class_number: Number of aspect classes to create
        logger: Logger object

    Returns:
        Path to the created aspect raster or None if failed
    """
    logger.info(f"Calculating aspect from DEM: {dem_raster}")

    try:
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        slope_rad, aspect_deg, valid = terrain_slope_aspect(dem, transform, crs, nodata)

        # Flat areas have no aspect
        flat_threshold = 1e-6  # m/m
        flat_mask = np.tan(np.where(valid, slope_rad, 0.0)) < flat_threshold

        # Classify aspect into directional classes
        classified_aspect = classify_aspect_into_classes(
            np.where(valid, aspect_deg, 0.0), flat_mask, aspect_class_number
        )

        # Nodata cells of the DEM have no aspect
        classified_aspect[~valid] = -9999

        # Save the classified aspect raster
        aspect_raster.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(aspect_raster, 'w', driver='GTiff',
                        height=classified_aspect.shape[0], width=classified_aspect.shape[1],
                        count=1, dtype=classified_aspect.dtype,
                        crs=crs, transform=transform, nodata=-9999) as dst:
            dst.write(classified_aspect, 1)

        logger.info(f"Aspect raster saved to: {aspect_raster}")
        logger.info(f"Aspect classes: {np.unique(classified_aspect[classified_aspect != -9999])}")
        return aspect_raster

    except Exception as e:  # noqa: BLE001 — plotting resilience
        logger.error(f"Error calculating aspect: {str(e)}", exc_info=True)
        return None

def classify_aspect_into_classes(aspect_deg: np.ndarray, flat_mask: np.ndarray,
                                num_classes: int) -> np.ndarray:
    """
    Classify aspect degrees into directional classes.

    Args:
        aspect_deg: Aspect in degrees (0-360)
        flat_mask: Boolean mask for flat areas
        num_classes: Number of aspect classes to create

    Returns:
        Classified aspect array
    """
    classified = np.zeros_like(aspect_deg, dtype=int)

    if num_classes == 8:
        # Standard 8-direction classification
        # N, NE, E, SE, S, SW, W, NW
        bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 1]  # Last one wraps to North

        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # Last bin (337.5 to 360)
                mask = (aspect_deg >= bins[i]) & (aspect_deg <= bins[i+1])
            else:
                mask = (aspect_deg >= bins[i]) & (aspect_deg < bins[i+1])
            classified[mask] = labels[i]

    elif num_classes == 4:
        # 4-direction classification (N, E, S, W)
        bins = [0, 45, 135, 225, 315, 360]
        labels = [1, 2, 3, 4, 1]  # N, E, S, W, N

        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # Last bin
                mask = (aspect_deg >= bins[i]) & (aspect_deg <= bins[i+1])
            else:
                mask = (aspect_deg >= bins[i]) & (aspect_deg < bins[i+1])
            classified[mask] = labels[i]

    else:
        # Custom number of classes - divide 360 degrees evenly
        class_width = 360.0 / num_classes
        for i in range(num_classes):
            lower = i * class_width
            upper = (i + 1) * class_width

            if i == num_classes - 1:  # Last class includes 360
                mask = (aspect_deg >= lower) & (aspect_deg <= upper)
            else:
                mask = (aspect_deg >= lower) & (aspect_deg < upper)
            classified[mask] = i + 1

    # Set flat areas to a special class (0)
    classified[flat_mask] = 0

    # Set areas that don't fall into any class to -9999 (shouldn't happen but safety)
    classified[classified == 0] = 0  # Keep flat areas as 0

    return classified

#: Tag written to radiation rasters produced by :func:`calculate_annual_radiation`
RADIATION_RASTER_METHOD = 'clearsky_poa_hourly_monthly_v2'
RADIATION_NODATA = -9999.0


def radiation_raster_is_current(radiation_raster: Path) -> bool:
    """True if *radiation_raster* was written by the current radiation method."""
    with rasterio.open(radiation_raster) as src:
        return src.tags().get('SYMFLUENCE_RADIATION_METHOD') == RADIATION_RASTER_METHOD


def calculate_annual_radiation(dem_raster: Path, radiation_raster: Path, logger: Logger) -> Optional[Path]:
    """
    Calculate annual clear-sky radiation on the terrain surface from a DEM.

    For the 15th of each month the clear-sky irradiance on every pixel's
    plane (slope/aspect from :func:`terrain_slope_aspect`, isotropic sky
    diffuse) is evaluated every hour of the UTC day — a full 24 h, so every
    longitude sees its whole daylight period — summed to a daily total and
    weighted by the number of days in the month. Sun position and clear-sky
    irradiance are those of the raster centre; terrain shading by neighbouring
    cells is not modelled.

    Output: float32 raster of annual irradiation in kWh m-2 yr-1, nodata
    -9999 where the DEM has no data.

    Args:
        dem_raster: Path to the DEM raster
        radiation_raster: Path where the radiation raster will be saved
        logger: Logger object

    Returns:
        Path to the created radiation raster or None if failed
    """
    logger.info(f"Calculating annual radiation from DEM: {dem_raster}")

    try:
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            nodata = src.nodata

        slope_rad, aspect_deg, valid = terrain_slope_aspect(dem, transform, crs, nodata)
        if not np.any(valid):
            raise ValueError(f"DEM {dem_raster} has no valid cells")

        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        if crs is not None and not crs.is_geographic:
            from rasterio.warp import transform as warp_transform
            lons, lats = warp_transform(crs, 'EPSG:4326', [center_x], [center_y])
            center_lon, center_lat = float(lons[0]), float(lats[0])
        else:
            center_lon, center_lat = float(center_x), float(center_y)

        mean_altitude = float(np.mean(np.ma.getdata(dem)[valid]))
        location = pvlib.location.Location(
            latitude=center_lat, longitude=center_lon, altitude=mean_altitude, tz='UTC'
        )

        # Hourly samples (centred on the half hour) of the 15th of each month
        year = 2019
        times = pd.DatetimeIndex(np.concatenate([
            pd.date_range(f'{year}-{month:02d}-15 00:30', periods=24, freq='h', tz='UTC').values
            for month in range(1, 13)
        ])).tz_localize('UTC')
        day_weights = np.repeat(
            [pd.Period(f'{year}-{month:02d}').days_in_month for month in range(1, 13)], 24
        )

        solar_pos = location.get_solarposition(times=times)
        clearsky = location.get_clearsky(times=times)
        zenith = solar_pos['apparent_zenith'].values
        azimuth = solar_pos['azimuth'].values
        dni = clearsky['dni'].values
        ghi = clearsky['ghi'].values
        dhi = clearsky['dhi'].values

        tilt = np.degrees(slope_rad[valid])
        surface_azimuth = aspect_deg[valid]

        logger.info(
            f"Calculating radiation for {tilt.size} pixels at "
            f"{int(np.sum(zenith < 90))} daylight time steps..."
        )

        # Wh m-2 per representative-day hour, weighted by days in the month
        total_wh = np.zeros(tilt.size, dtype=float)
        for t_idx in range(len(times)):
            if not zenith[t_idx] < 90:  # night (or NaN)
                continue
            irrad = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt,
                surface_azimuth=surface_azimuth,
                solar_zenith=zenith[t_idx],
                solar_azimuth=azimuth[t_idx],
                dni=dni[t_idx],
                ghi=ghi[t_idx],
                dhi=dhi[t_idx],
            )
            poa = np.nan_to_num(np.asarray(irrad['poa_global'], dtype=float), nan=0.0)
            total_wh += np.maximum(poa, 0.0) * day_weights[t_idx]

        radiation = np.full(dem.shape, RADIATION_NODATA, dtype='float32')
        radiation[valid] = (total_wh / 1000.0).astype('float32')

        # Save the radiation raster
        radiation_raster.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(radiation_raster, 'w', driver='GTiff',
                        height=radiation.shape[0], width=radiation.shape[1],
                        count=1, dtype='float32',
                        crs=crs, transform=transform, nodata=RADIATION_NODATA) as dst:
            dst.write(radiation, 1)
            dst.update_tags(
                SYMFLUENCE_RADIATION_METHOD=RADIATION_RASTER_METHOD,
                units='kWh m-2 yr-1',
            )

        logger.info(f"Radiation raster saved to: {radiation_raster}")
        return radiation_raster

    except Exception as e:  # noqa: BLE001 — plotting resilience
        logger.error(f"Error calculating annual radiation: {str(e)}", exc_info=True)
        return None


def quantile_class_thresholds(
    raster_path: Path,
    n_classes: int,
    logger: Optional[Logger] = None,
) -> np.ndarray:
    """
    Class boundaries splitting a raster's valid values into *n_classes* quantile classes.

    Used for radiation discretization (``RADIATION_CLASS_NUMBER`` is a class
    COUNT). Thresholds are computed over the whole raster so a class means the
    same radiation range in every GRU. Returns ``n_classes + 1`` boundaries
    (fewer when the data have too few distinct values), from the minimum to
    the maximum valid value.
    """
    n_classes = max(int(n_classes), 1)
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        values = data[valid_pixel_mask(data, src.nodata)].astype(float)
    if values.size == 0:
        raise ValueError(f"No valid data found in raster {raster_path}")
    thresholds = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_classes + 1)))
    if thresholds.size < 2:
        thresholds = np.array([thresholds[0], thresholds[0]])
    if logger:
        logger.info(
            f"{len(thresholds) - 1} quantile classes for {raster_path}: "
            f"{np.array2string(thresholds, precision=2)}"
        )
    return thresholds


def valid_pixel_mask(data, nodata=None):
    """Boolean mask of usable raster pixels.

    Excludes masked-array cells (e.g. outside a ``rasterio.mask`` geometry),
    the ``nodata`` value (NaN-aware), and non-finite values for ANY float
    dtype -- float32 DEMs included, not just float64.
    """
    arr = np.ma.getdata(data)
    valid = ~np.ma.getmaskarray(data)
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            pass  # covered by the finite check below
        else:
            valid &= arr != nodata
    if np.issubdtype(arr.dtype, np.floating):
        valid &= np.isfinite(arr)
    return np.asarray(valid, dtype=bool)


def analyze_raster_values(
    raster_path: Path,
    band_size: Optional[float] = None,
    logger: Optional[Logger] = None
) -> np.ndarray:
    """
    Analyze raster values to determine discretization thresholds.
    Reads raster in chunks to handle large files efficiently.

    Args:
        raster_path: Path to the raster file
        band_size: Optional band size for discretization (for continuous data)
        logger: Optional logger

    Returns:
        Array of threshold values
    """
    # Process raster in chunks
    CHUNK_SIZE = 1024  # Adjust based on available memory
    valid_data = []

    if logger:
        logger.info(f"Analyzing raster values from: {raster_path}")

    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        nodata = src.nodata

        if logger:
            logger.info(f"Raster info: {width}x{height} pixels, nodata={nodata}")

        for y in range(0, height, CHUNK_SIZE):
            for x in range(0, width, CHUNK_SIZE):
                window = rasterio.windows.Window(
                    x, y, min(CHUNK_SIZE, width - x), min(CHUNK_SIZE, height - y)
                )
                chunk = src.read(1, window=window)

                # Filter out nodata values (and NaN for any float dtype,
                # including float32 DEMs)
                valid_chunk = chunk[valid_pixel_mask(chunk, nodata)]

                if len(valid_chunk) > 0:
                    valid_data.extend(valid_chunk.flatten())

    if len(valid_data) == 0:
        raise ValueError("No valid data found in raster")

    valid_data = np.array(valid_data)
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)

    if logger:
        logger.info(f"Valid data range: {data_min:.2f} to {data_max:.2f}")
        logger.info(f"Total valid pixels: {len(valid_data)}")

    # Calculate thresholds based on the data
    if band_size is not None:
        # For elevation-based or radiation-based discretization
        # Ensure thresholds cover the full data range
        min_val = data_min
        max_val = data_max

        # Create bands that fully cover the data range
        thresholds = np.arange(min_val, max_val + band_size, band_size)

        # Ensure the last threshold covers the maximum value
        if thresholds[-1] < max_val:
            thresholds = np.append(thresholds, thresholds[-1] + band_size)

        if logger:
            logger.info(f"Created {len(thresholds)-1} bands with size {band_size}")
            logger.info(
                f"Threshold range: {thresholds[0]:.2f} to {thresholds[-1]:.2f}"
            )
    else:
        # For soil or land class-based discretization
        thresholds = np.unique(valid_data)
        if logger:
            logger.info(f"Found {len(thresholds)} unique classes")

    return thresholds
