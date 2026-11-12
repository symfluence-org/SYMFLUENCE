# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Combined multi-attribute discretization for HRU creation.

Enables domain discretization using combinations of attributes such as
elevation, soil class, land cover, aspect, and radiation simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import MultiPolygon, shape
from shapely.validation import make_valid

from symfluence.core.exceptions import DiscretizationError
from symfluence.geospatial.raster_utils import (
    calculate_annual_radiation,
    calculate_aspect,
    quantile_class_thresholds,
)

from ..hru_utils import fill_nodata_from_nearest, read_zone_window
from .radiation import warn_if_stale_radiation_raster

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer", attributes: List[str]):
    """
    Discretize the domain based on a combination of geospatial attributes.

    Args:
        attributes: List of attribute names to combine (e.g., ['elevation', 'landclass'])
    """
    discretizer.logger.info(
        f"Starting combined discretization with attributes: {attributes}"
    )

    # Get GRU shapefile
    default_name = f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp"
    delineate_coastal = discretizer._get_config_value(
        lambda: discretizer.config.domain.delineation.delineate_coastal_watersheds,
        default=False
    )
    if delineate_coastal:
        default_name = f"{discretizer.domain_name}_riverBasins_with_coastal.shp"

    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=default_name,
    )

    # Generate output filename with backward-compatible catchment subpath
    method_suffix = "_".join(attributes)
    default_name = f"{discretizer.domain_name}_HRUs_{method_suffix}.shp"
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath=discretizer._get_catchment_subpath(default_name),
        default_name=default_name,
    )

    # Get raster paths and thresholds for each attribute
    raster_info = _get_raster_info_for_attributes(discretizer, attributes)

    # Read GRU data
    gru_gdf = discretizer._read_shapefile(gru_shapefile)

    # Create combined HRUs
    hru_gdf = _create_combined_attribute_hrus(
        discretizer, gru_gdf, raster_info, attributes
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf, gru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Combined attribute HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None


def _get_raster_info_for_attributes(
    discretizer: "DomainDiscretizer", attributes: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Get raster paths and classification information for each attribute.

    Args:
        attributes: List of attribute names

    Returns:
        Dictionary containing raster path and classification info for each attribute
    """
    raster_info = {}

    for attr in attributes:
        attr_lower = attr.lower()

        if attr_lower == "elevation":
            raster_path = discretizer._get_file_path(
                path_key="DEM_PATH",
                name_key="DEM_NAME",
                default_subpath="attributes/elevation/dem",
                default_name=f"domain_{discretizer.domain_name}_elv.tif"
            )
            band_size = float(discretizer._get_config_value(
                lambda: discretizer.config.domain.elevation_band_size,
                default=200.0
            ))

            raster_info[attr] = {
                "path": raster_path,
                "type": "continuous",
                "band_size": band_size,
                "class_name": "elevClass",
            }

        elif attr_lower == "soilclass":
            raster_path = discretizer._get_file_path(
                path_key="SOIL_CLASS_PATH",
                name_key="SOIL_CLASS_NAME",
                default_subpath="attributes/soilclass",
                default_name=f"domain_{discretizer.domain_name}_soil_classes.tif",
            )
            raster_info[attr] = {
                "path": raster_path,
                "type": "discrete",
                "class_name": "soilClass",
            }

        elif attr_lower == "landclass":
            raster_path = discretizer._get_file_path(
                path_key="LAND_CLASS_PATH",
                name_key="LAND_CLASS_NAME",
                default_subpath="attributes/landclass",
                default_name=f"domain_{discretizer.domain_name}_land_classes.tif",
            )
            raster_info[attr] = {
                "path": raster_path,
                "type": "discrete",
                "class_name": "landClass",
            }

        elif attr_lower == "radiation":
            radiation_raster = discretizer._get_file_path(
                path_key="RADIATION_PATH",
                name_key="RADIATION_NAME",
                default_subpath="attributes/radiation",
                default_name="annual_radiation.tif",
            )

            # Calculate radiation if it doesn't exist
            if not radiation_raster.exists():
                discretizer.logger.info(
                    "Annual radiation raster not found. Calculating radiation..."
                )
                dem_raster = discretizer._get_file_path(
                    path_key="DEM_PATH",
                    name_key="DEM_NAME",
                    default_subpath="attributes/elevation/dem",
                    default_name=f"domain_{discretizer.domain_name}_elv.tif"
                )
                radiation_raster = calculate_annual_radiation(
                    dem_raster, radiation_raster, discretizer.logger
                )
                if radiation_raster is None:
                    raise ValueError("Failed to calculate annual radiation")
            else:
                warn_if_stale_radiation_raster(discretizer, radiation_raster)

            radiation_class_number = int(discretizer._get_config_value(
                lambda: discretizer.config.domain.radiation_class_number,
                default=1
            ))

            raster_info[attr] = {
                "path": radiation_raster,
                "type": "continuous",
                # RADIATION_CLASS_NUMBER quantile classes of the whole raster —
                # the same thresholds the single-attribute path uses.
                "thresholds": quantile_class_thresholds(
                    radiation_raster, radiation_class_number, discretizer.logger
                ),
                "class_name": "radiationClass",
            }
        elif attr_lower == "aspect":
            aspect_raster = discretizer._get_file_path(
                path_key="ASPECT_PATH",
                name_key="ASPECT_NAME",
                default_subpath="attributes/aspect",
                default_name="aspect.tif"
            )

            # Calculate aspect if it doesn't exist
            if not aspect_raster.exists():
                discretizer.logger.info("Aspect raster not found. Calculating aspect...")
                dem_raster = discretizer._get_file_path(
                    path_key="DEM_PATH",
                    name_key="DEM_NAME",
                    default_subpath="attributes/elevation/dem",
                    default_name=f"domain_{discretizer.domain_name}_elv.tif"
                )
                aspect_class_number = int(discretizer._get_config_value(
                    lambda: discretizer.config.domain.aspect_class_number,
                    dict_key='ASPECT_CLASS_NUMBER',
                ))
                aspect_raster = calculate_aspect(
                    dem_raster,
                    aspect_raster,
                    aspect_class_number,
                    discretizer.logger,
                )
                if aspect_raster is None:
                    raise ValueError("Failed to calculate aspect")

            raster_info[attr] = {
                "path": aspect_raster,
                "type": "discrete",
                "class_name": "aspectClass",
            }

        else:
            raise ValueError(f"Unsupported attribute for discretization: {attr}")

    return raster_info


def _open_on_reference_grid(path, ref, categorical: bool):
    """Open *path* so that it reads on the grid of the open dataset *ref*.

    Returns ``(dataset, to_close)``. A raster already on the reference grid is
    read directly; any other resolution, origin or CRS is resampled on the fly
    with a ``WarpedVRT`` (nearest neighbour for classes, bilinear otherwise).
    """
    src = rasterio.open(path)
    same_grid = (
        src.crs == ref.crs
        and src.width == ref.width
        and src.height == ref.height
        and src.transform.almost_equals(ref.transform)
    )
    if same_grid:
        return src, [src]
    vrt = WarpedVRT(
        src,
        crs=ref.crs,
        transform=ref.transform,
        width=ref.width,
        height=ref.height,
        resampling=Resampling.nearest if categorical else Resampling.bilinear,
    )
    return vrt, [vrt, src]


def _create_combined_attribute_hrus(
    discretizer: "DomainDiscretizer",
    gru_gdf,
    raster_info: Dict[str, Dict[str, Any]],
    attributes: List[str],
):
    """
    Create HRUs based on unique combinations of multiple attributes within each GRU.

    The first attribute's raster is the reference grid: the GRUs are
    reprojected to its CRS and every other raster is read onto that grid
    (resampled when its resolution, origin or CRS differ), so the attribute
    layers are always co-registered. The HRUs are returned in the CRS of
    *gru_gdf*.

    Args:
        gru_gdf: GeoDataFrame containing GRU data
        raster_info: Dictionary containing raster information for each attribute
        attributes: List of attribute names

    Returns:
        GeoDataFrame containing combined attribute HRUs

    Raises:
        DiscretizationError: if an attribute raster cannot be read for a GRU.
    """
    discretizer.logger.info(
        f"Creating combined attribute HRUs within {len(gru_gdf)} GRUs"
    )

    all_hrus = []
    hru_id_counter = 1
    output_crs = gru_gdf.crs
    to_close: List[Any] = []

    try:
        ref = rasterio.open(raster_info[attributes[0]]["path"])
        to_close.append(ref)
        datasets = {attributes[0]: ref}
        for attr in attributes[1:]:
            info = raster_info[attr]
            ds, opened = _open_on_reference_grid(
                info["path"], ref, categorical=info["type"] == "discrete"
            )
            to_close.extend(opened)
            if len(opened) > 1:
                discretizer.logger.info(
                    f"{attr} raster {info['path']} is not on the grid of the "
                    f"{attributes[0]} raster; resampling it onto that grid "
                    f"({'nearest' if info['type'] == 'discrete' else 'bilinear'})."
                )
            datasets[attr] = ds

        if ref.crs is not None and gru_gdf.crs is not None and gru_gdf.crs != ref.crs:
            discretizer.logger.info(
                f"Reprojecting GRUs from {gru_gdf.crs} to raster CRS {ref.crs} for masking")
            gru_gdf = gru_gdf.to_crs(ref.crs)

        # Process each GRU individually
        for gru_pos, (gru_idx, gru_row) in enumerate(gru_gdf.iterrows()):
            discretizer.logger.info(f"Processing GRU {gru_pos + 1}/{len(gru_gdf)}")

            gru_geometry = gru_row.geometry
            gru_id = gru_row.get("GRU_ID", gru_idx)

            # Extract all raster data for this GRU (same window on every raster)
            raster_data = {}
            common_transform = None
            inside_mask = None

            for attr in attributes:
                try:
                    data, inside, valid, out_transform = read_zone_window(
                        datasets[attr], gru_geometry, all_touched=True
                    )
                except (rasterio.RasterioIOError, ValueError, RuntimeError) as e:
                    raise DiscretizationError(
                        f"Could not extract {attr} raster data for GRU {gru_id}: {e}. "
                        "Every attribute of a combined discretization must cover every GRU."
                    ) from e

                if common_transform is None:
                    common_transform = out_transform
                    inside_mask = inside
                elif data.shape != inside_mask.shape:  # type: ignore[union-attr]
                    raise DiscretizationError(
                        f"{attr} raster window for GRU {gru_id} has shape {data.shape}, "
                        f"expected {inside_mask.shape}; rasters are not co-registered."  # type: ignore[union-attr]
                    )
                raster_data[attr] = {"data": data, "valid": valid}

            assert inside_mask is not None

            # Nodata holes inside the GRU: fill each layer from its nearest
            # valid pixel so the HRUs tile the GRU. A layer with no valid
            # pixel in the GRU cannot be filled.
            empty_layers = [a for a in attributes if not raster_data[a]["valid"].any()]
            if empty_layers:
                discretizer.logger.warning(
                    f"No valid {', '.join(empty_layers)} pixels found in GRU {gru_id}; "
                    "no HRUs created for it."
                )
                continue

            n_inside = int(inside_mask.sum())
            for attr in attributes:
                layer = raster_data[attr]
                layer["data"], n_filled = fill_nodata_from_nearest(
                    layer["data"], layer["valid"], inside_mask
                )
                if n_filled:
                    discretizer.logger.warning(
                        f"GRU {gru_id}: {n_filled} of {n_inside} raster pixels "
                        f"({100.0 * n_filled / n_inside:.1f}%) have no {attr} data; assigned "
                        "the nearest valid pixel's value so the HRUs cover the whole GRU."
                    )
            combined_valid_mask = inside_mask

            # Classify each attribute and find unique combinations
            classified_data = {}
            for attr in attributes:
                raster_array = raster_data[attr]["data"]
                attr_info = raster_info[attr]

                if attr_info["type"] == "continuous":
                    # Classify continuous data into bands
                    classified_data[attr] = _classify_continuous_data(
                        raster_array,
                        combined_valid_mask,
                        attr_info.get("band_size"),
                        thresholds=attr_info.get("thresholds"),
                    )
                else:
                    # Use discrete values directly
                    classified_data[attr] = raster_array

            # Find unique combinations of classified values
            unique_combinations = _find_unique_combinations(
                classified_data, combined_valid_mask
            )

            # Create HRUs for each unique combination
            gru_hrus = _create_hrus_from_combinations(
                discretizer,
                unique_combinations,
                classified_data,
                combined_valid_mask,
                common_transform,
                gru_geometry,
                gru_row,
                hru_id_counter,
                attributes,
                class_names={a: raster_info[a]["class_name"] for a in attributes},
            )

            all_hrus.extend(gru_hrus)
            hru_id_counter += len(gru_hrus)
    finally:
        for handle in to_close:
            handle.close()

    discretizer.logger.info(
        f"Created {len(all_hrus)} combined attribute HRUs across all GRUs"
    )
    hrus = gpd.GeoDataFrame(all_hrus, geometry="geometry", crs=gru_gdf.crs) if all_hrus \
        else gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
    if output_crs is not None and hrus.crs is not None and hrus.crs != output_crs:
        hrus = hrus.to_crs(output_crs)
    return hrus


def _classify_continuous_data(
    raster_array: np.ndarray,
    valid_mask: np.ndarray,
    band_size: Optional[float] = None,
    thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Classify continuous raster data into discrete bands.

    Args:
        raster_array: Input raster array
        valid_mask: Boolean mask for valid pixels
        band_size: Width of the bands, counted from the minimum valid value of
            this array (elevation bands, in metres)
        thresholds: Explicit class boundaries (radiation classes); takes
            precedence over *band_size*. Values outside them go to the first /
            last class.

    Returns:
        Classified array with discrete class values (1-based; 0 = not valid)
    """
    valid_data = raster_array[valid_mask]

    if len(valid_data) == 0:
        return np.zeros(raster_array.shape, dtype=int)

    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=float)
        n_classes = max(len(thresholds) - 1, 1)
        # interior boundaries; right-open bands, last band closed
        classes = np.searchsorted(thresholds[1:-1], raster_array, side="right") + 1
        classes = np.clip(classes, 1, n_classes)
        return np.where(valid_mask, classes, 0).astype(int)

    if band_size is None or band_size <= 0:
        raise ValueError(f"band_size must be positive, got {band_size}")

    data_min = np.min(valid_data)
    data_max = np.max(valid_data)

    # Fixed band width from the minimum value
    thresholds = np.arange(data_min, data_max + band_size, band_size)
    if thresholds[-1] < data_max:
        thresholds = np.append(thresholds, thresholds[-1] + band_size)

    # Classify the data
    classified = np.zeros(raster_array.shape, dtype=int)
    for i in range(len(thresholds) - 1):
        lower, upper = thresholds[i : i + 2]
        if i == len(thresholds) - 2:  # Last band
            mask = valid_mask & (raster_array >= lower) & (raster_array <= upper)
        else:
            mask = valid_mask & (raster_array >= lower) & (raster_array < upper)
        classified[mask] = i + 1

    return classified


def _find_unique_combinations(
    classified_data: Dict[str, np.ndarray], valid_mask: np.ndarray
) -> List[Tuple]:
    """
    Find unique combinations of classified values across all attributes.

    Args:
        classified_data: Dictionary of classified raster arrays for each attribute
        valid_mask: Boolean mask for valid pixels

    Returns:
        List of unique value combinations
    """
    # Stack all classified arrays
    stacked_data = []
    for attr in sorted(classified_data.keys()):
        stacked_data.append(classified_data[attr][valid_mask])

    # Find unique combinations
    combined_array = np.column_stack(stacked_data)
    unique_combinations = [tuple(row) for row in np.unique(combined_array, axis=0)]

    return unique_combinations


def _create_hrus_from_combinations(
    discretizer: "DomainDiscretizer",
    unique_combinations: List[Tuple],
    classified_data: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    transform: Any,
    gru_geometry: Any,
    gru_row: Any,
    start_hru_id: int,
    attributes: List[str],
    class_names: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Create HRUs for each unique combination of attribute values.

    Args:
        unique_combinations: List of unique value combinations
        classified_data: Dictionary of classified raster arrays
        valid_mask: Boolean mask for valid pixels
        transform: Raster transform
        gru_geometry: GRU geometry
        gru_row: GRU data row
        start_hru_id: Starting HRU ID
        attributes: List of attribute names

    Returns:
        List of HRU dictionaries
    """
    hrus = []
    current_hru_id = start_hru_id

    for combination in unique_combinations:
        # Create mask for this combination
        combination_mask = valid_mask.copy()

        for i, attr in enumerate(sorted(classified_data.keys())):
            attr_value = combination[i]
            combination_mask &= classified_data[attr] == attr_value

        if not np.any(combination_mask):
            continue

        # Create HRU from this combination
        hru = _create_hru_from_combination_mask(
            discretizer,
            combination_mask,
            transform,
            classified_data,
            gru_geometry,
            gru_row,
            current_hru_id,
            attributes,
            combination,
            class_names=class_names,
        )

        if hru:
            hrus.append(hru)
            current_hru_id += 1

    return hrus


def _create_hru_from_combination_mask(
    discretizer: "DomainDiscretizer",
    combination_mask: np.ndarray,
    transform: Any,
    classified_data: Dict[str, np.ndarray],
    gru_geometry: Any,
    gru_row: Any,
    hru_id: int,
    attributes: List[str],
    combination: Tuple,
    class_names: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """
    Create a single HRU from a combination mask.

    Args:
        combination_mask: Boolean mask for the combination
        transform: Raster transform
        classified_data: Dictionary of classified raster arrays
        gru_geometry: GRU geometry
        gru_row: GRU data row
        hru_id: HRU ID
        attributes: List of attribute names
        combination: Tuple of attribute values for this combination

    Returns:
        Dictionary representing the HRU or None if creation fails
    """
    try:
        # Extract shapes from the mask
        shapes = list(
            rasterio.features.shapes(
                combination_mask.astype(np.uint8),
                mask=combination_mask,
                transform=transform,
                connectivity=4,
            )
        )

        if not shapes:
            return None

        # Create polygons from shapes
        polygons = []
        for shp, _ in shapes:
            try:
                geom = shape(shp)
                if geom.is_valid and not geom.is_empty and geom.area > 0:
                    polygons.append(geom)
            except (ValueError, TypeError, AttributeError):
                # Invalid shape geometry - skip and continue
                continue

        if not polygons:
            return None

        # Create final geometry
        if len(polygons) == 1:
            final_geometry = polygons[0]
        else:
            final_geometry = MultiPolygon(polygons)

        # Clean the geometry
        if not final_geometry.is_valid:
            final_geometry = make_valid(final_geometry)

        if final_geometry.is_empty or not final_geometry.is_valid:
            return None

        # Ensure it's within the GRU boundary
        clipped_geometry = final_geometry.intersection(gru_geometry)

        if clipped_geometry.is_empty or not clipped_geometry.is_valid:
            return None

        # Create HRU data with combination attributes
        hru_data = {
            "geometry": clipped_geometry,
            "GRU_ID": gru_row.get("GRU_ID", gru_row.name),
            "HRU_ID": hru_id,
            "hru_type": f'combined_{"_".join(attributes)}',
        }

        # Add individual attribute values (every attribute, aspect included)
        default_names = {
            "elevation": "elevClass",
            "soilclass": "soilClass",
            "landclass": "landClass",
            "radiation": "radiationClass",
            "aspect": "aspectClass",
        }
        for i, attr in enumerate(sorted(attributes)):
            column = (class_names or {}).get(attr) or default_names.get(attr.lower())
            if column:
                hru_data[column] = combination[i]

        # Add combined attribute identifier
        combined_id = "_".join([str(val) for val in combination])
        combined_name = f"combined_{'_'.join(attributes)}"
        hru_data[combined_name] = combined_id

        # Copy relevant GRU attributes (excluding geometry)
        for col in gru_row.index:
            if col not in ["geometry", "GRU_ID"] and col not in hru_data:
                hru_data[col] = gru_row[col]

        return hru_data

    except Exception as e:  # noqa: BLE001 — geospatial resilience
        discretizer.logger.warning(
            f"Error creating HRU for combination {combination}: {str(e)}"
        )
        return None
