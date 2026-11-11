# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geology attribute processor.

Handles geological and hydrogeological attributes including:
- GLHYMPS data (permeability, porosity)
- Lithology processing
- Structural features
- Derived hydrogeological properties
"""
from __future__ import annotations

# Security rationale: used for caching geology lookup tables, not untrusted data
import pickle  # nosec B403
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np

from symfluence.core.geometry_utils import feature_areas_m2

from .base import BaseAttributeProcessor


class GeologyProcessor(BaseAttributeProcessor):
    """Processor for geological and hydrogeological attributes."""

    def process(self) -> Dict[str, Any]:
        """
        Process geological attributes.

        Returns:
            Dictionary of geological attributes
        """
        results: Dict[str, Any] = {}

        # Process GLHYMPS data for permeability and porosity
        glhymps_results = self._process_glhymps_data()
        results.update(glhymps_results)

        # Process additional lithology data if available
        litho_results = self._process_lithology_data()
        results.update(litho_results)

        # Enhance with derived hydrogeological properties
        results = self._enhance_hydrogeological_attributes(results)

        return results

    def _process_glhymps_data(self) -> Dict[str, Any]:
        """
        Process GLHYMPS (Global Hydrogeology MaPS) data for permeability and porosity.
        Optimized for performance with spatial filtering and simplified geometries.

        Returns:
            Dict[str, Any]: Dictionary of hydrogeological attributes
        """
        results: Dict[str, Any] = {}

        # Define path to GLHYMPS data (configurable via GLHYMPS_PATH)
        glhymps_path = self._find_glhymps_file()

        if glhymps_path is None:
            self.logger.warning("GLHYMPS data not found")
            return results

        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'glhymps'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_glhymps_results.pickle"

        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached GLHYMPS results from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    # Security rationale: loading trusted local cache files
                    return pickle.load(f)  # nosec B301
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error loading cached GLHYMPS results: {str(e)}")

        self.logger.info("Processing GLHYMPS data for hydrogeological attributes")

        try:
            # Get catchment bounding box first to filter GLHYMPS data
            catchment = gpd.read_file(self.catchment_path)
            catchment_bbox = catchment.total_bounds

            # Add buffer to bounding box (e.g., 1% of width/height)
            bbox_width = catchment_bbox[2] - catchment_bbox[0]
            bbox_height = catchment_bbox[3] - catchment_bbox[1]
            buffer_x = bbox_width * 0.01
            buffer_y = bbox_height * 0.01

            # Create expanded bbox
            expanded_bbox = (
                catchment_bbox[0] - buffer_x,
                catchment_bbox[1] - buffer_y,
                catchment_bbox[2] + buffer_x,
                catchment_bbox[3] + buffer_y
            )

            # Load only GLHYMPS data within the bounding box and only necessary columns
            self.logger.info("Loading GLHYMPS data within catchment bounding box")
            glhymps = gpd.read_file(
                str(glhymps_path),
                bbox=expanded_bbox,
                # Only read necessary columns if they exist
                columns=['geometry', 'porosity', 'logK_Ice', 'Porosity', 'Permeabi_1']
            )

            if glhymps.empty:
                self.logger.warning("No GLHYMPS data within catchment bounding box")
                return results

            # Rename columns if they exist with shortened names
            if ('Porosity' in glhymps.columns) and ('Permeabi_1' in glhymps.columns):
                glhymps.rename(columns={'Porosity': 'porosity', 'Permeabi_1': 'logK_Ice'}, inplace=True)

            # Check for missing required columns
            required_columns = ['porosity', 'logK_Ice']
            missing_columns = [col for col in required_columns if col not in glhymps.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns in GLHYMPS: {', '.join(missing_columns)}")
                return results

            # Simplify geometries for faster processing
            self.logger.info("Simplifying geometries for faster processing")
            simplify_tolerance = 0.001  # Adjust based on your data units
            glhymps['geometry'] = glhymps.geometry.simplify(simplify_tolerance)
            catchment['geometry'] = catchment.geometry.simplify(simplify_tolerance)

            # Calculate area in equal area projection for weighting
            # (local equal-area projection: valid anywhere, not only North America)
            glhymps['New_area_m2'] = feature_areas_m2(glhymps)

            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self._is_lumped()

            if is_lumped:
                # For lumped catchment, intersect with catchment boundary
                try:
                    # Ensure CRS match
                    if glhymps.crs != catchment.crs:
                        glhymps = glhymps.to_crs(catchment.crs)

                    self.logger.info("Intersecting GLHYMPS with catchment boundary")
                    # Intersect GLHYMPS with catchment
                    intersection = gpd.overlay(glhymps, catchment, how='intersection')

                    if not intersection.empty:
                        # Convert to equal area for proper area calculation
                        intersection['New_area_m2'] = feature_areas_m2(intersection)
                        total_area = intersection['New_area_m2'].sum()

                        if total_area > 0:
                            # Calculate area-weighted averages for porosity and permeability
                            # Porosity
                            porosity_mean = (intersection['porosity'] * intersection['New_area_m2']).sum() / total_area
                            # Standard deviation calculation (area-weighted)
                            porosity_variance = ((intersection['New_area_m2'] *
                                                (intersection['porosity'] - porosity_mean)**2).sum() /
                                                total_area)
                            porosity_std = np.sqrt(porosity_variance)

                            # Get min and max values
                            porosity_min = intersection['porosity'].min()
                            porosity_max = intersection['porosity'].max()

                            # Add porosity results
                            results["geology.porosity_mean"] = porosity_mean
                            results["geology.porosity_std"] = porosity_std
                            results["geology.porosity_min"] = porosity_min
                            results["geology.porosity_max"] = porosity_max

                            # Permeability (log10 values)
                            logk_mean = (intersection['logK_Ice'] * intersection['New_area_m2']).sum() / total_area
                            # Standard deviation calculation (area-weighted)
                            logk_variance = ((intersection['New_area_m2'] *
                                            (intersection['logK_Ice'] - logk_mean)**2).sum() /
                                            total_area)
                            logk_std = np.sqrt(logk_variance)

                            # Get min and max values
                            logk_min = intersection['logK_Ice'].min()
                            logk_max = intersection['logK_Ice'].max()

                            # Add permeability results
                            results["geology.log_permeability_mean"] = logk_mean
                            results["geology.log_permeability_std"] = logk_std
                            results["geology.log_permeability_min"] = logk_min
                            results["geology.log_permeability_max"] = logk_max

                            # Calculate derived hydraulic properties

                            # Convert log permeability to hydraulic conductivity
                            # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                            hyd_cond = 10**(logk_mean) * (10**7)  # m/s
                            results["geology.hydraulic_conductivity_m_per_s"] = hyd_cond

                            # Calculate transmissivity assuming 100m aquifer thickness
                            # T = K * b where b is aquifer thickness
                            transmissivity = hyd_cond * 100  # m²/s
                            results["geology.transmissivity_m2_per_s"] = transmissivity
                    else:
                        self.logger.warning("No intersection between GLHYMPS and catchment boundary")
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.error(f"Error processing GLHYMPS for lumped catchment: {str(e)}")
            else:
                # For distributed catchment, process each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self._get_config_value(lambda: self.config.paths.catchment_hruid, default='HRU_ID', dict_key='CATCHMENT_SHP_HRUID')

                # Ensure CRS match
                if glhymps.crs != catchment.crs:
                    glhymps = glhymps.to_crs(catchment.crs)

                # Create spatial index for GLHYMPS
                self.logger.info("Creating spatial index for faster intersections")
                import rtree
                spatial_index = rtree.index.Index()
                for idx, geom in enumerate(glhymps.geometry):
                    spatial_index.insert(idx, geom.bounds)

                # Process in chunks for better performance
                chunk_size = 10  # Adjust based on your data
                for i in range(0, len(catchment), chunk_size):
                    self.logger.info(f"Processing HRUs {i} to {min(i+chunk_size, len(catchment))-1}")
                    hru_chunk = catchment.iloc[i:min(i+chunk_size, len(catchment))]

                    for j, hru in hru_chunk.iterrows():
                        try:
                            hru_id = hru[hru_id_field]
                            prefix = f"HRU_{hru_id}_"

                            # Create a GeoDataFrame with just this HRU
                            hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)

                            # Use spatial index to filter candidates
                            hru_bounds = hru.geometry.bounds
                            potential_matches_idx = list(spatial_index.intersection(hru_bounds))

                            if not potential_matches_idx:
                                self.logger.debug(f"No potential GLHYMPS matches for HRU {hru_id}")
                                continue

                            # Subset GLHYMPS using potential matches
                            glhymps_subset = glhymps.iloc[potential_matches_idx]

                            # Intersect with GLHYMPS
                            intersection = gpd.overlay(glhymps_subset, hru_gdf, how='intersection')

                            if not intersection.empty:
                                # Convert to equal area for proper area calculation
                                intersection['New_area_m2'] = feature_areas_m2(intersection)
                                total_area = intersection['New_area_m2'].sum()

                                if total_area > 0:
                                    # Calculate area-weighted averages for porosity and permeability
                                    # Porosity
                                    porosity_mean = (intersection['porosity'] * intersection['New_area_m2']).sum() / total_area
                                    # Standard deviation calculation (area-weighted)
                                    porosity_variance = ((intersection['New_area_m2'] *
                                                        (intersection['porosity'] - porosity_mean)**2).sum() /
                                                        total_area)
                                    porosity_std = np.sqrt(porosity_variance)

                                    # Get min and max values
                                    porosity_min = intersection['porosity'].min()
                                    porosity_max = intersection['porosity'].max()

                                    # Add porosity results
                                    results[f"{prefix}geology.porosity_mean"] = porosity_mean
                                    results[f"{prefix}geology.porosity_std"] = porosity_std
                                    results[f"{prefix}geology.porosity_min"] = porosity_min
                                    results[f"{prefix}geology.porosity_max"] = porosity_max

                                    # Permeability (log10 values)
                                    logk_mean = (intersection['logK_Ice'] * intersection['New_area_m2']).sum() / total_area
                                    # Standard deviation calculation (area-weighted)
                                    logk_variance = ((intersection['New_area_m2'] *
                                                    (intersection['logK_Ice'] - logk_mean)**2).sum() /
                                                    total_area)
                                    logk_std = np.sqrt(logk_variance)

                                    # Get min and max values
                                    logk_min = intersection['logK_Ice'].min()
                                    logk_max = intersection['logK_Ice'].max()

                                    # Add permeability results
                                    results[f"{prefix}geology.log_permeability_mean"] = logk_mean
                                    results[f"{prefix}geology.log_permeability_std"] = logk_std
                                    results[f"{prefix}geology.log_permeability_min"] = logk_min
                                    results[f"{prefix}geology.log_permeability_max"] = logk_max

                                    # Calculate derived hydraulic properties

                                    # Convert log permeability to hydraulic conductivity
                                    # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                                    hyd_cond = 10**(logk_mean) * (10**7)  # m/s
                                    results[f"{prefix}geology.hydraulic_conductivity_m_per_s"] = hyd_cond

                                    # Calculate transmissivity assuming 100m aquifer thickness
                                    # T = K * b where b is aquifer thickness
                                    transmissivity = hyd_cond * 100  # m²/s
                                    results[f"{prefix}geology.transmissivity_m2_per_s"] = transmissivity
                            else:
                                self.logger.debug(f"No intersection between GLHYMPS and HRU {hru_id}")
                        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                            self.logger.error(f"Error processing GLHYMPS for HRU {hru_id}: {str(e)}")

            # Cache results
            try:
                self.logger.info(f"Caching GLHYMPS results to {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error caching GLHYMPS results: {str(e)}")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing GLHYMPS data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

        return results

    def _find_glhymps_file(self):
        """Find GLHYMPS file, checking acquisition output then legacy paths."""
        from pathlib import Path

        # Acquisition output: domain_{name}_glhymps.gpkg
        acq_dir = self.project_dir / 'data' / 'attributes' / 'geology' / 'glhymps'
        for ext in ('*.gpkg', '*.shp'):
            matches = list(acq_dir.glob(f"*glhymps{ext[1:]}")) if acq_dir.exists() else []
            if matches:
                return matches[0]

        # Config-specified path
        config_path = self._get_config_value(
            lambda: None, default=None, dict_key='GLHYMPS_PATH'
        )
        if config_path and config_path != 'default':
            p = Path(config_path)
            if p.exists():
                return p

        # Legacy path
        legacy = self.data_dir / 'geospatial' / 'glhymps' / 'raw' / 'glhymps.shp'
        if legacy.exists():
            return legacy

        return None

    def _process_lithology_data(self) -> Dict[str, Any]:
        """
        Process lithology classification data from geological maps.

        Returns:
            Dict[str, Any]: Dictionary of lithology attributes
        """
        results: Dict[str, Any] = {}

        # Define path to geological map data (configurable via GEOLOGY_DIR)
        # This could be GMNA (Geological Map of North America) or similar dataset
        geo_map_path = Path(self._get_config_value(lambda: None, default=str(self.data_dir / 'geospatial' / 'geology' / 'raw'), dict_key='GEOLOGY_DIR')
        )

        # Check if geological map directory exists
        if not geo_map_path.exists():
            self.logger.warning(f"Geological map directory not found: {geo_map_path}")
            return results

        self.logger.info("Processing lithology data from geological maps")

        try:
            # Search for potential geological shapefiles
            shapefile_patterns = ["*geologic*.shp", "*lithology*.shp", "*bedrock*.shp", "*rock*.shp"]
            geo_files = []

            for pattern in shapefile_patterns:
                geo_files.extend(list(geo_map_path.glob(pattern)))

            if not geo_files:
                self.logger.warning(f"No geological map files found in {geo_map_path}")
                return results

            # Use the first matching file
            geo_file = geo_files[0]
            self.logger.info(f"Using geological map file: {geo_file}")

            # Load geological map shapefile
            geo_map = gpd.read_file(str(geo_file))

            # Identify potential lithology classification columns
            litho_columns = []
            for col in geo_map.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['lith', 'rock', 'geol', 'form', 'unit']):
                    litho_columns.append(col)

            if not litho_columns:
                self.logger.warning(f"No lithology classification columns found in {geo_file}")
                return results

            # Use the first matching column
            litho_column = litho_columns[0]
            self.logger.info(f"Using lithology classification column: {litho_column}")

            # Define rock type categories for classification
            rock_categories = {
                'igneous': ['igneous', 'volcanic', 'plutonic', 'basalt', 'granite', 'diorite', 'gabbro', 'rhyolite', 'andesite'],
                'metamorphic': ['metamorphic', 'gneiss', 'schist', 'quartzite', 'marble', 'amphibolite', 'slate', 'phyllite'],
                'sedimentary': ['sedimentary', 'limestone', 'sandstone', 'shale', 'conglomerate', 'dolomite', 'chalk', 'siltstone'],
                'unconsolidated': ['alluvium', 'colluvium', 'sand', 'gravel', 'clay', 'silt', 'soil', 'till', 'glacial']
            }

            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self._is_lumped()

            if is_lumped:
                # For lumped catchment, intersect with catchment boundary
                catchment = gpd.read_file(self.catchment_path)

                # Ensure CRS match
                if geo_map.crs != catchment.crs:
                    geo_map = geo_map.to_crs(catchment.crs)

                # Intersect geological map with catchment
                intersection = gpd.overlay(geo_map, catchment, how='intersection')

                if not intersection.empty:
                    # Convert to equal area for proper area calculation
                    # (local equal-area projection: valid anywhere, not only North America)
                    intersection_equal_area = intersection.copy()

                    # Calculate area for each lithology class
                    intersection_equal_area['area_km2'] = feature_areas_m2(intersection) / 10**6  # m² to km²

                    # Calculate lithology class distribution
                    litho_areas = intersection_equal_area.groupby(litho_column)['area_km2'].sum()
                    total_area = litho_areas.sum()

                    if total_area > 0:
                        # Calculate fraction for each lithology class
                        for litho_type, area in litho_areas.items():
                            if litho_type and str(litho_type).strip():  # Skip empty values
                                fraction = area / total_area
                                # Clean name for attribute key
                                clean_name = self._clean_attribute_name(str(litho_type))
                                results[f"geology.{clean_name}_fraction"] = fraction

                        # Identify dominant lithology class
                        dominant_litho = litho_areas.idxmax()
                        if dominant_litho and str(dominant_litho).strip():
                            results["geology.dominant_lithology"] = str(dominant_litho)
                            results["geology.dominant_lithology_fraction"] = litho_areas[dominant_litho] / total_area

                        # Calculate lithology diversity (Shannon entropy)
                        shannon_entropy = 0
                        for _, area in litho_areas.items():
                            if area > 0:
                                p = area / total_area
                                shannon_entropy -= p * np.log(p)
                        results["geology.lithology_diversity"] = shannon_entropy

                    # Classify by broad rock types
                    litho_by_category = {category: 0 for category in rock_categories}

                    # Iterate through each geological unit
                    for _, unit in intersection_equal_area.iterrows():
                        area = unit['area_km2']
                        unit_desc = str(unit.get(litho_column, '')).lower()

                        # Classify the unit into a category based on keywords
                        for category, keywords in rock_categories.items():
                            if any(keyword in unit_desc for keyword in keywords):
                                litho_by_category[category] += area
                                break

                    # Calculate percentages for rock categories
                    category_total = sum(litho_by_category.values())
                    if category_total > 0:
                        for category, area in litho_by_category.items():
                            results[f"geology.{category}_fraction"] = area / category_total

                        # Identify dominant category
                        dominant_category = max(litho_by_category.items(), key=lambda x: x[1])[0]
                        results["geology.dominant_category"] = dominant_category
                        results["geology.dominant_category_fraction"] = litho_by_category[dominant_category] / category_total
                else:
                    self.logger.warning("No intersection between geological map and catchment boundary")
            else:
                # For distributed catchment, process each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self._get_config_value(lambda: self.config.paths.catchment_hruid, default='HRU_ID', dict_key='CATCHMENT_SHP_HRUID')

                # Ensure CRS match
                if geo_map.crs != catchment.crs:
                    geo_map = geo_map.to_crs(catchment.crs)

                for i, hru in catchment.iterrows():
                    try:
                        hru_id = hru[hru_id_field]
                        prefix = f"HRU_{hru_id}_"

                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)

                        # Intersect with geological map
                        intersection = gpd.overlay(geo_map, hru_gdf, how='intersection')

                        if not intersection.empty:
                            # Convert to equal area for proper area calculation
                            # (local equal-area projection: valid anywhere, not only North America)
                            intersection_equal_area = intersection.copy()

                            # Calculate area for each lithology class
                            intersection_equal_area['area_km2'] = feature_areas_m2(intersection) / 10**6  # m² to km²

                            # Calculate lithology class distribution
                            litho_areas = intersection_equal_area.groupby(litho_column)['area_km2'].sum()
                            total_area = litho_areas.sum()

                            if total_area > 0:
                                # Calculate fraction for each lithology class
                                for litho_type, area in litho_areas.items():
                                    if litho_type and str(litho_type).strip():  # Skip empty values
                                        fraction = area / total_area
                                        # Clean name for attribute key
                                        clean_name = self._clean_attribute_name(str(litho_type))
                                        results[f"{prefix}geology.{clean_name}_fraction"] = fraction

                                # Identify dominant lithology class
                                dominant_litho = litho_areas.idxmax()
                                if dominant_litho and str(dominant_litho).strip():
                                    results[f"{prefix}geology.dominant_lithology"] = str(dominant_litho)
                                    results[f"{prefix}geology.dominant_lithology_fraction"] = litho_areas[dominant_litho] / total_area

                                # Calculate lithology diversity (Shannon entropy)
                                shannon_entropy = 0
                                for _, area in litho_areas.items():
                                    if area > 0:
                                        p = area / total_area
                                        shannon_entropy -= p * np.log(p)
                                results[f"{prefix}geology.lithology_diversity"] = shannon_entropy

                            # Classify by broad rock types
                            litho_by_category = {category: 0 for category in rock_categories}

                            # Iterate through each geological unit
                            for _, unit in intersection_equal_area.iterrows():
                                area = unit['area_km2']
                                unit_desc = str(unit.get(litho_column, '')).lower()

                                # Classify the unit into a category based on keywords
                                for category, keywords in rock_categories.items():
                                    if any(keyword in unit_desc for keyword in keywords):
                                        litho_by_category[category] += area
                                        break

                            # Calculate percentages for rock categories
                            category_total = sum(litho_by_category.values())
                            if category_total > 0:
                                for category, area in litho_by_category.items():
                                    results[f"{prefix}geology.{category}_fraction"] = area / category_total

                                # Identify dominant category
                                dominant_category = max(litho_by_category.items(), key=lambda x: x[1])[0]
                                results[f"{prefix}geology.dominant_category"] = dominant_category
                        else:
                            self.logger.debug(f"No intersection between geological map and HRU {hru_id}")
                    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                        self.logger.error(f"Error processing lithology for HRU {hru_id}: {str(e)}")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing lithology data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

        return results

    def _clean_attribute_name(self, name: str) -> str:
        """
        Clean a string to be usable as an attribute name.

        Args:
            name: Original string

        Returns:
            String usable as an attribute name
        """
        if not name:
            return "unknown"

        # Convert to string and strip whitespace
        name_str = str(name).strip()
        if not name_str:
            return "unknown"

        # Replace spaces and special characters
        cleaned = name_str.lower()
        cleaned = cleaned.replace(' ', '_')
        cleaned = cleaned.replace('-', '_')
        cleaned = cleaned.replace('.', '_')
        cleaned = cleaned.replace('(', '')
        cleaned = cleaned.replace(')', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace('/', '_')
        cleaned = cleaned.replace('\\', '_')
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('%', 'percent')

        # Remove any remaining non-alphanumeric characters
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')

        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'x' + cleaned

        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50]

        # Handle empty result
        if not cleaned:
            return "unknown"

        return cleaned

    def _enhance_hydrogeological_attributes(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance existing geological attributes with derived hydraulic properties.

        Args:
            current_results: Current geological attributes

        Returns:
            Dict[str, Any]: Enhanced geological attributes
        """
        results = dict(current_results)  # Create a copy to avoid modifying the original

        try:
            # Check if we have log permeability values
            if "geology.log_permeability_mean" in results:
                log_k = results["geology.log_permeability_mean"]

                # Convert log permeability to hydraulic conductivity in m/s
                # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                hydraulic_conductivity = 10**(log_k) * (10**7)
                results["geology.hydraulic_conductivity_m_per_s"] = hydraulic_conductivity

                # Convert to more common units (cm/hr, m/day)
                results["geology.hydraulic_conductivity_cm_per_hr"] = hydraulic_conductivity * 3600 * 100
                results["geology.hydraulic_conductivity_m_per_day"] = hydraulic_conductivity * 86400

                # Calculate transmissivity assuming typical aquifer thickness values
                # Shallow aquifer (10m)
                results["geology.transmissivity_shallow_m2_per_day"] = hydraulic_conductivity * 10 * 86400

                # Deep aquifer (100m)
                results["geology.transmissivity_deep_m2_per_day"] = hydraulic_conductivity * 100 * 86400

                # Calculate hydraulic diffusivity if we have porosity data
                if "geology.porosity_mean" in results:
                    porosity = results["geology.porosity_mean"]
                    if porosity > 0:
                        specific_storage = 1e-6  # Typical value for confined aquifer (1/m)
                        storativity = specific_storage * 100  # For 100m thick aquifer

                        # Calculate specific yield (typically 0.1-0.3 of porosity)
                        specific_yield = 0.2 * porosity
                        results["geology.specific_yield"] = specific_yield

                        # Hydraulic diffusivity (m²/s)
                        diffusivity = hydraulic_conductivity / storativity
                        results["geology.hydraulic_diffusivity_m2_per_s"] = diffusivity

                        # Groundwater response time (days) for 1km travel distance
                        if diffusivity > 0:
                            response_time = (1000**2) / (diffusivity * 86400)  # days
                            results["geology.groundwater_response_time_days"] = response_time

            # Derive aquifer properties from soil depth if available
            if "soil.regolith_thickness_mean" in results or "soil.sedimentary_thickness_mean" in results:
                # Use available thickness data or default to a typical value
                regolith_thickness = results.get("soil.regolith_thickness_mean", 0)
                sedimentary_thickness = results.get("soil.sedimentary_thickness_mean", 0)

                # Estimate total aquifer thickness
                aquifer_thickness = max(regolith_thickness, sedimentary_thickness)
                if aquifer_thickness == 0:
                    aquifer_thickness = 50  # Default value if no data available

                results["geology.estimated_aquifer_thickness_m"] = aquifer_thickness

                # If we have hydraulic conductivity, calculate transmissivity
                if "geology.hydraulic_conductivity_m_per_s" in results:
                    transmissivity = results["geology.hydraulic_conductivity_m_per_s"] * aquifer_thickness
                    results["geology.transmissivity_m2_per_s"] = transmissivity

            # Extract bedrock depth information from soil data if available
            if "soil.regolith_thickness_mean" in results:
                results["geology.bedrock_depth_m"] = results["soil.regolith_thickness_mean"]

            return results

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error enhancing hydrogeological attributes: {str(e)}")
            return current_results  # Return the original results if there was an error
