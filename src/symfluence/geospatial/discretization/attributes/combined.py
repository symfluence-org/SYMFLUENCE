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
from rasterio.mask import mask
from shapely.geometry import MultiPolygon, shape
from shapely.validation import make_valid

from symfluence.geospatial.raster_utils import (
    calculate_annual_radiation,
    calculate_aspect,
)

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
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf)
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

            radiation_class_number = int(discretizer._get_config_value(
                lambda: discretizer.config.domain.radiation_class_number,
                default=1
            ))

            raster_info[attr] = {
                "path": radiation_raster,
                "type": "continuous",
                "band_size": radiation_class_number,
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
                    default=8
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


def _create_combined_attribute_hrus(
    discretizer: "DomainDiscretizer",
    gru_gdf,
    raster_info: Dict[str, Dict[str, Any]],
    attributes: List[str],
):
    """
    Create HRUs based on unique combinations of multiple attributes within each GRU.

    Args:
        gru_gdf: GeoDataFrame containing GRU data
        raster_info: Dictionary containing raster information for each attribute
        attributes: List of attribute names

    Returns:
        GeoDataFrame containing combined attribute HRUs
    """
    discretizer.logger.info(
        f"Creating combined attribute HRUs within {len(gru_gdf)} GRUs"
    )

    all_hrus = []
    hru_id_counter = 1

    # Process each GRU individually
    for gru_idx, gru_row in gru_gdf.iterrows():
        discretizer.logger.info(f"Processing GRU {gru_idx + 1}/{len(gru_gdf)}")

        gru_geometry = gru_row.geometry
        gru_id = gru_row.get("GRU_ID", gru_idx + 1)

        # Extract all raster data for this GRU
        raster_data = {}
        common_transform = None
        common_shape = None

        for attr in attributes:
            attr_info = raster_info[attr]
            raster_path = attr_info["path"]

            try:
                with rasterio.open(raster_path) as src:
                    out_image, out_transform = mask(
                        src,
                        [gru_geometry],
                        crop=True,
                        all_touched=True,
                        filled=False,
                    )
                    out_image = out_image[0]
                    nodata_value = src.nodata

                    # Store raster data and metadata
                    raster_data[attr] = {
                        "data": out_image,
                        "nodata": nodata_value,
                        "info": attr_info,
                    }

                    # Set common transform and shape from first raster
                    if common_transform is None:
                        common_transform = out_transform
                        common_shape = out_image.shape

            except Exception as e:  # noqa: BLE001 — geospatial resilience
                discretizer.logger.warning(
                    f"Could not extract {attr} raster data for GRU {gru_id}: {str(e)}"
                )
                continue

        if not raster_data:
            discretizer.logger.warning(f"No valid raster data found for GRU {gru_id}")
            continue

        # Create combined valid mask (pixels that are valid in all rasters)
        combined_valid_mask = np.full(common_shape, True, dtype=bool)  # type: ignore[type-var]
        for attr, data in raster_data.items():
            raster_array = data["data"]
            nodata_value = data["nodata"]

            if nodata_value is not None:
                valid_mask = raster_array != nodata_value
            else:
                valid_mask = (
                    ~np.isnan(raster_array)
                    if raster_array.dtype == np.float64 or raster_array.dtype == np.float32
                    else np.ones_like(raster_array, dtype=bool)
                )

            combined_valid_mask = np.logical_and(combined_valid_mask, valid_mask)

        if not np.any(combined_valid_mask):
            discretizer.logger.warning(f"No valid pixels found in GRU {gru_id}")
            continue

        # Classify each attribute and find unique combinations
        classified_data = {}
        for attr in attributes:
            data_info = raster_data[attr]
            raster_array = data_info["data"]
            attr_info = data_info["info"]

            if attr_info["type"] == "continuous":
                # Classify continuous data into bands
                classified_data[attr] = _classify_continuous_data(
                    raster_array, combined_valid_mask, attr_info["band_size"]
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
        )

        all_hrus.extend(gru_hrus)
        hru_id_counter += len(gru_hrus)

    discretizer.logger.info(
        f"Created {len(all_hrus)} combined attribute HRUs across all GRUs"
    )
    return gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)


def _classify_continuous_data(
    raster_array: np.ndarray, valid_mask: np.ndarray, band_size: float
) -> np.ndarray:
    """
    Classify continuous raster data into discrete bands.

    Args:
        raster_array: Input raster array
        valid_mask: Boolean mask for valid pixels
        band_size: Size of bands (for elevation) or number of classes (for radiation)

    Returns:
        Classified array with discrete class values
    """
    valid_data = raster_array[valid_mask]

    if len(valid_data) == 0:
        return raster_array.copy()

    data_min = np.min(valid_data)
    data_max = np.max(valid_data)

    # Create classification based on band_size
    if isinstance(band_size, int) and band_size < 50:  # Assume it's number of classes for radiation
        # Use quantile-based classification
        quantiles = np.linspace(0, 1, band_size + 1)
        thresholds = np.quantile(valid_data, quantiles)
    else:
        # Use fixed band size for elevation
        thresholds = np.arange(data_min, data_max + band_size, band_size)
        if thresholds[-1] < data_max:
            thresholds = np.append(thresholds, thresholds[-1] + band_size)

    # Classify the data
    classified = np.zeros_like(raster_array, dtype=int)
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

        # Add individual attribute values
        for i, attr in enumerate(sorted(attributes)):
            attr_name = attr.lower()
            if attr_name == "elevation":
                hru_data["elevClass"] = combination[i]
            elif attr_name == "soilclass":
                hru_data["soilClass"] = combination[i]
            elif attr_name == "landclass":
                hru_data["landClass"] = combination[i]
            elif attr_name == "radiation":
                hru_data["radiationClass"] = combination[i]

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
