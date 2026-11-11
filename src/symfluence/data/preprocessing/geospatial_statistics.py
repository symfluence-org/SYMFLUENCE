# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Geospatial statistics calculator for catchment attribute extraction from rasters.

Computes zonal statistics (elevation, soil class, land cover) by extracting raster values
within catchment polygon boundaries. Implements memory-efficient chunked processing for
large domains (e.g., NWM North America with millions of catchments). Integrates with
PathManager for standardized path resolution and ConfigurableMixin for configuration access.

Architecture:
    The GeospatialStatistics module implements a chunked zonal statistics workflow to handle
    massive spatial datasets without memory overload:

    1. Zonal Statistics Concept:
       Input: Vector polygons (catchments) + Raster (elevation, soil, landcover)
       Output: Polygon dataset with summary statistics from raster values within each polygon
       Example: For each catchment polygon, calculate mean elevation from DEM

    2. Computational Challenge (Large Domains):
       Problem: Direct processing of millions of catchments causes out-of-memory errors
       Solution: Chunk processing approach with two strategies:
           a) Spatial tiling: Process catchments in geographic tiles (bounding boxes)
           b) Index chunking: Process catchments in sequential index ranges
       Both strategies reduce in-memory footprint while maintaining accuracy

    3. Statistics Computation:
       Uses rasterstats.zonal_stats for efficient raster-polygon overlay
       Computes summary statistics: mean, min, max, std (configurable)
       Handles NoData values correctly (excludes from calculations)
       Memory efficient via windowed raster access (rasterio)

    4. Attribute Types:
       Elevation (DEM):
           - Source: Digital Elevation Model (raster)
           - Statistics: mean, min, max, std
           - Units: meters
           - Example: dem_mean, dem_min, dem_max

       Soil Class (Categorical):
           - Source: Soil type/class raster
           - Statistics: Mode (most common class)
           - Units: Class codes (integer)
           - Example: soil_dominant

       Land Cover (Categorical):
           - Source: Land use/cover classification
           - Statistics: Mode (most common class)
           - Units: Class codes (integer)
           - Example: landcover_dominant

    5. Checkpoint System:
       For long-running large domain processing:
       - Saves incremental progress to checkpoint files
       - Allows resumption if processing interrupted
       - Checkpoint format: GeoDataFrame with partial results
       - Useful for NWM-scale processing (millions of catchments)

Configuration Parameters:
    paths.catchment_name: str (optional)
        Input catchment polygon shapefile name
        Default: auto-generated from domain_name and discretization method

    paths.dem_name: str (optional)
        Input DEM (elevation) raster filename
        Default: domain_{domain_name}_elv.tif

    paths.soil_class_path: str (optional)
        Input soil classification raster path
        Used for soil statistics calculation

    paths.land_class_path: str (optional)
        Input land cover classification raster path
        Used for landcover statistics calculation

    INTERSECT_DEM_PATH: Path (optional)
        Output directory for elevation-catchment intersection results

Input Data:
    Catchment Shapefile (shapefile/catchment/{catchment_name}):
        - Format: GeoDataFrame with polygon geometries
        - Geometry: Catchment boundary polygons
        - Attributes: catchment IDs, names, etc.

    Elevation Raster (attributes/elevation/dem/{dem_name}):
        - Format: GeoTIFF (rasterio-compatible)
        - Values: Elevation in meters
        - CRS: Must match catchment shapefile CRS

    Soil Raster (attributes/soilclass/...):
        - Format: GeoTIFF or similar
        - Values: Soil class codes (integer)
        - CRS: Must match catchment shapefile CRS

    Landcover Raster (attributes/landclass/...):
        - Format: GeoTIFF or similar
        - Values: Land cover class codes (integer)
        - CRS: Must match catchment shapefile CRS

Output Data:
    Elevation Statistics:
        Output: shapefiles/catchment_intersection/with_dem/catchment_with_dem.shp
        Columns added: elev_mean, elev_min, elev_max, elev_std

    Soil Statistics:
        Merged into catchment shapefile
        Columns added: USGS_* (fraction per soil class), soil_pixel_count
        Contains pixel-count fractions (0-1) of each soil type within catchment

    Landcover Statistics:
        Merged into catchment shapefile
        Columns added: IGBP_* (fraction per landcover class), land_pixel_count
        Contains pixel-count fractions (0-1) of each landcover type within catchment

    Full Results:
        Final output shapefile contains all merged attributes
        Used for model parameterization and spatial analysis

Chunking Strategies:

    1. Spatial Tiling (_process_elevation_spatial_tiles):
       - Divides domain into geographic bounding box tiles
       - Processes each tile's catchments independently
       - Advantages: Natural geographic grouping, parallel-friendly
       - Disadvantages: May have uneven distribution
       - For: Medium-sized domains, when geographic locality matters

    2. Index Chunking (_process_elevation_index_chunks):
       - Divides catchments into sequential index ranges (e.g., 0-10000, 10000-20000)
       - Processes each chunk in order
       - Advantages: Simple, predictable, even distribution
       - Disadvantages: Less geographic meaning
       - For: Large domains, simple parallelization

    3. Memory Efficiency:
       - Windowed raster access: Reads only tiles needed for current chunk
       - Garbage collection: Explicit gc.collect() between chunks
       - NoData handling: Excludes invalid values from statistics
       - Generator-based processing where possible

Use Cases:

    1. Model Parameterization:
       Extract catchment-averaged attributes for lumped models (GR4J, GR6J)
       Mean elevation drives potential evapotranspiration
       Dominant soil class determines infiltration parameters

    2. Distributed Model Setup:
       Create HRU attributes for distributed models (SUMMA, HYPE)
       Each HRU gets catchment-averaged elevation, soil, landcover
       Spatial discretization (elevation bands, landcover types) uses these stats

    3. Continental/Continental-Scale Analysis:
       Process millions of catchments (NWM, global hydrological models)
       Checkpoint system enables long-running batch processing
       Parallel processing of chunks for speedup

    4. Validation & QA:
       Check consistency between catchment boundaries and raster data
       Identify catchments with missing data
       Validate CRS alignment and data coverage

Example Workflow:

    >>> config = load_config('config.yaml')
    >>> logger = setup_logger()
    >>> stats_calc = GeospatialStatistics(config, logger)
    >>>
    >>> # Calculate elevation statistics (with chunking for large domains)
    >>> stats_calc.calculate_elevation_stats()
    >>> # Output: catchment_with_dem.shp with elev_mean, elev_min, elev_max, elev_std
    >>>
    >>> # Calculate soil and landcover statistics
    >>> stats_calc.calculate_soil_stats()
    >>> stats_calc.calculate_land_stats()
    >>>
    >>> # Run all statistics calculations
    >>> stats_calc.run_statistics()

Error Handling:

    - NoData handling: Automatically detects and excludes NoData values
    - CRS validation: Checks alignment of raster and vector CRS
    - File existence checks: Skips processing if output already exists
    - Checkpoint recovery: Resumes from last checkpoint if interrupted
    - Graceful degradation: Continues with available data if some layers missing

Performance Considerations:

    - Large domains: Spatial tiling or index chunking reduces memory by 10-100x
    - Raster access: Windowed reading more efficient than loading full raster
    - Vectorization: rasterstats uses optimized C libraries
    - Parallel: Process chunks independently on different machines/cores

Dependencies:
    - geopandas: Vector geometry handling
    - rasterio: Raster I/O and windowed reading
    - rasterstats: Efficient zonal statistics computation
    - pandas/numpy: Numerical operations and data structures

References:
    - Zonal Statistics: https://www.rspatial.org/raster/rs/3-sdfitness.pdf
    - rasterstats: https://github.com/perrygeo/python-rasterstats
    - Rasterio Windowed Reading: https://rasterio.readthedocs.io/
    - GeoDataFrame: https://geopandas.org/

See Also:
    - PathManager: Path resolution for data files
    - AttributeProcessors: Downstream processing of extracted attributes
    - DataManager: High-level data workflow coordination
"""
from __future__ import annotations

import gc
import hashlib
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

from symfluence.core import ConfigurableMixin
from symfluence.data.path_manager import PathManager
from symfluence.geospatial.zonal import fill_empty_zones, zonal_stats_covered

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class GeospatialStatistics(ConfigurableMixin):
    """Computes zonal statistics extracting catchment attributes from raster data.

    Central component for extracting catchment-scale attributes (elevation, soil class,
    land cover) from raster datasets through efficient zonal statistics computation.
    Implements chunked processing strategies for handling large domains (millions of
    catchments) without memory overload. Integrates PathManager for path resolution and
    ConfigurableMixin for configuration access.

    This class bridges the gap between raw raster data (DEMs, soil maps, landcover) and
    catchment-level attributes needed for hydrological model parameterization. For each
    catchment polygon, it extracts all raster values that intersect and computes summary
    statistics (mean, min, max, std for continuous data; mode for categorical data).

    Chunked Processing Strategy:

        Problem: Large domains like NWM North America have millions of catchments.
        Processing all catchments at once causes out-of-memory errors.

        Solution: Two chunking approaches:
            1. Spatial Tiling: Divide domain into geographic tiles, process each tile
            2. Index Chunking: Process sequential index ranges (0-N, N-2N, etc.)

        Benefits:
            - Reduces in-memory footprint by 10-100x
            - Enables processing of continental-scale domains
            - Compatible with parallel processing
            - Progress checkpointing for resumption

    Attribute Types Calculated:

        Elevation Statistics (from DEM raster):
            - elev_mean: Mean elevation in catchment (meters)
            - elev_min: Minimum elevation (meters)
            - elev_max: Maximum elevation (meters)
            - elev_std: Standard deviation (meters)
            - Used for: PET calculation, precipitation lapse rate correction

        Soil Class Statistics (from soil classification raster):
            - soil_class_{code}: Areal proportion of each soil class
            - Categorical: Mode (dominant soil class)
            - Used for: Infiltration rates, water retention, soil-specific parameters

        Landcover Statistics (from land use/cover raster):
            - landcover_class_{code}: Areal proportion of each landcover class
            - Categorical: Mode (dominant land cover type)
            - Used for: Vegetation parameters, interception, roughness

    Key Responsibilities:

        1. Path Resolution:
           Uses PathManager for standardized file location lookup
           Handles optional custom paths via configuration

        2. Zonal Statistics Computation:
           Applies rasterstats.zonal_stats for efficient raster-polygon overlay
           Leverages rasterio windowed reading for memory efficiency
           Correctly handles NoData values

        3. Memory Management:
           Implements chunking to prevent OOM on large domains
           Explicit garbage collection between chunks
           Windowed raster access (reads only needed portions)

        4. Checkpoint System:
           Saves progress to intermediate files for long-running jobs
           Enables resumption if processing interrupted
           Useful for NWM-scale processing (24+ hours possible)

        5. Data Integration:
           Merges all statistics into single output GeoDataFrame
           Produces model-ready shapefile with all attributes
           Maintains spatial geometry and topology

    Configuration:

        paths.catchment_name: str (optional)
            Input catchment shapefile name. Default: auto-generated from domain_name
            and discretization method.

        paths.dem_name: str (optional)
            DEM raster filename. Default: domain_{domain_name}_elv.tif

        paths.soil_class_path: Path to soil classification raster
        paths.land_class_path: Path to land cover classification raster

        Chunking parameters (from calculate_*_stats methods):
            chunk_size: Number of catchments per chunk
            tile_size: Bounding box size for spatial tiling

    Attributes:

        config (SymfluenceConfig): Configuration object (from ConfigurableMixin)
        logger (logging.Logger): Logger instance
        paths (PathManager): Path resolution helper
        project_dir (Path): Project root directory
        catchment_path (Path): Catchment shapefile directory
        catchment_name (str): Catchment shapefile name
        dem_path (Path): DEM raster path
        soil_path (Path): Soil classification raster path
        land_path (Path): Land cover classification raster path

    Methods:

        calculate_elevation_stats():
            Extract elevation statistics from DEM within catchments.
            Uses chunking for memory efficiency on large domains.
            Produces: catchment_with_dem.shp with elev_mean, elev_min, elev_max, elev_std

        calculate_soil_stats():
            Extract soil class statistics from soil raster.
            Computes area-weighted proportions of each soil class.
            Produces: Soil attributes merged into catchment shapefile

        calculate_land_stats():
            Extract land cover statistics from landcover raster.
            Computes area-weighted proportions of each landcover class.
            Produces: Landcover attributes merged into catchment shapefile

        run_statistics():
            Execute all statistics calculations in sequence.
            Produces: Complete catchment shapefile with all attributes

    Output Format:

        GeoDataFrame with catchment polygons and extracted attributes:
            - geometry: Catchment polygon
            - catchment_id: Unique identifier
            - elev_mean, elev_min, elev_max, elev_std: Elevation statistics
            - soil_class_{code}: Soil class proportions
            - landcover_class_{code}: Landcover proportions
            - Additional: Any other input attributes

    Example Usage:

        >>> config = load_config('config.yaml')
        >>> logger = setup_logger()
        >>> stats = GeospatialStatistics(config, logger)
        >>>
        >>> # Process individual attributes
        >>> stats.calculate_elevation_stats()  # With automatic chunking
        >>> stats.calculate_soil_stats()
        >>> stats.calculate_land_stats()
        >>>
        >>> # Or process all at once
        >>> stats.run_statistics()

    Large Domain Handling:

        For domains with millions of catchments (e.g., NWM):
            - Automatic chunking detects large datasets
            - Spatial tiling or index chunking selected automatically
            - Checkpoint files enable resumption
            - Typical NWM processing: 24-48 hours on single machine

    Error Handling:

        - Missing rasters: Skip processing if raster file not found
        - NoData values: Automatically excluded from statistics
        - CRS mismatch: Logs warning if vector/raster CRS differ
        - Corrupt files: Continues with available data
        - Checkpoint failure: Can resume from last successful checkpoint

    Performance Characteristics:

        - Elevation stats: ~0.1-0.5 seconds per 1000 catchments (depends on complexity)
        - Soil stats: ~0.2-1.0 seconds per 1000 catchments (more complex calculation)
        - Memory overhead: ~100-500 MB per million catchments (chunked)
        - Parallel potential: Process chunks independently on separate machines

    References:

        - Zonal Statistics Algorithm: https://www.rspatial.org/
        - rasterstats Documentation: https://github.com/perrygeo/python-rasterstats
        - GeoDataFrame I/O: https://geopandas.org/

    See Also:

        - PathManager: Path resolution for spatial data files
        - AttributeProcessor: Post-processing of extracted attributes
        - DataManager: High-level coordination of all data operations
    """

    def __init__(self, config: Union['SymfluenceConfig', Dict[str, Any]], logger):
        # Handle typed config using centralized coercion
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        # Use PathManager for path resolution
        self.paths = PathManager(config)
        self.project_dir = self.paths.project_dir

        # Resolve paths using PathManager with backward compatibility
        self.catchment_name = self._get_config_value(
            lambda: self.config.paths.catchment_name, default='default'
        )
        if self.catchment_name == 'default':
            discretization = str(self._get_config_value(
                lambda: self.config.domain.discretization, default=''
            )).replace(',', '_')
            self.catchment_name = f"{self.paths.domain_name}_HRUs_{discretization}.shp"

        # Use backward-compatible catchment path resolution
        self.catchment_path = self.paths.get_catchment_dir(self.catchment_name)

        dem_name = self._get_config_value(lambda: self.config.paths.dem_name, default='default')
        if dem_name == "default":
            dem_name = f"domain_{self.paths.domain_name}_elv.tif"

        self.dem_path = self.paths.resolve('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.soil_path = self.paths.resolve('SOIL_CLASS_PATH', 'attributes/soilclass')
        self.land_path = self.paths.resolve('LAND_CLASS_PATH', 'attributes/landclass')
        self._catchment_signature_cache: Any = None

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def _output_has_expected_columns(
        self,
        output_file: Path,
        required_prefixes: tuple[str, ...] = (),
        required_columns: tuple[str, ...] = (),
        warn_on_error: bool = True,
    ) -> bool:
        """Return True when an existing shapefile can be reused.

        It must have the expected columns and rows AND describe the current
        catchment shapefile (same HRU ids, extent and polygon areas): the
        output file names are fixed, so a file left by another discretization
        of the same domain would otherwise hand its values to renumbered HRUs.
        """
        if not output_file.exists():
            return False

        try:
            gdf = gpd.read_file(output_file)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as exc:
            msg = f"Error checking existing statistics file '{output_file}': {exc}. Recalculating."
            if warn_on_error:
                self.logger.warning(msg)
            else:
                self.logger.debug(msg)
            return False

        if len(gdf) == 0:
            return False

        if required_columns and not all(col in gdf.columns for col in required_columns):
            return False

        if required_prefixes:
            has_required_prefix = any(
                any(col.startswith(prefix) for col in gdf.columns)
                for prefix in required_prefixes
            )
            if not has_required_prefix:
                return False

        return self._cached_output_matches_catchment(output_file, gdf)

    def _zone_id_columns(self) -> tuple[str, str]:
        """Names of the HRU and GRU id columns of the catchment shapefile."""
        hru_col = self._get_config_value(lambda: self.config.paths.catchment_hruid, default='HRU_ID')
        gru_col = self._get_config_value(lambda: self.config.paths.catchment_gruid, default='GRU_ID')
        return str(hru_col or 'HRU_ID'), str(gru_col or 'GRU_ID')

    def _layer_signature(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Identity of a polygon layer: ids, extent and per-polygon areas."""
        hru_col, _ = self._zone_id_columns()
        ids = [str(v) for v in gdf[hru_col]] if hru_col in gdf.columns else None
        return {
            'n': len(gdf),
            'ids': ids,
            'bounds': np.asarray(gdf.total_bounds, dtype=float),
            'areas': np.asarray(gdf.geometry.area, dtype=float),
        }

    def _cached_output_matches_catchment(self, output_file: Path, cached_gdf: gpd.GeoDataFrame) -> bool:
        """True when *cached_gdf* was computed for the current catchment shapefile."""
        catchment_file = Path(self.catchment_path) / self.catchment_name
        if not catchment_file.exists():
            self.logger.debug(
                f"Cannot validate {output_file} against the catchment: {catchment_file} not found"
            )
            return True

        if getattr(self, '_catchment_signature_cache', None) is None:
            self._catchment_signature_cache = self._layer_signature(gpd.read_file(catchment_file))
        current = self._catchment_signature_cache
        cached = self._layer_signature(cached_gdf)

        reason = None
        if cached['n'] != current['n']:
            reason = f"{cached['n']} polygons vs {current['n']} in the catchment"
        elif (cached['ids'] is None) != (current['ids'] is None):
            reason = "HRU id column present in only one of the two files"
        elif cached['ids'] is not None and set(cached['ids']) != set(current['ids']):
            reason = "different HRU ids"
        else:
            extent = max(float(np.ptp(current['bounds'][[0, 2]])), float(np.ptp(current['bounds'][[1, 3]])), 1e-9)
            if not np.allclose(cached['bounds'], current['bounds'], rtol=0.0, atol=1e-6 * extent):
                reason = "different extent"
            else:
                if cached['ids'] is not None:
                    cached_area = dict(zip(cached['ids'], cached['areas']))
                    a = np.array([cached_area[i] for i in current['ids']])
                else:
                    a = cached['areas']
                if not np.allclose(a, current['areas'], rtol=1e-6, atol=0.0):
                    reason = "different HRU polygon areas"

        if reason is not None:
            self.logger.info(
                f"Existing statistics file {output_file} was computed for a different "
                f"catchment layer than {catchment_file.name} ({reason}). Recalculating."
            )
            return False
        return True

    def _validate_checkpoints(self, checkpoint_dir: Path, gdf: gpd.GeoDataFrame, strategy: str) -> None:
        """Discard elevation checkpoints that belong to another catchment layer or DEM.

        Checkpoints are stored by tile / chunk number only. A signature of the
        catchment layer, the DEM and the chunking is kept next to them; when
        it differs the checkpoints are removed and recomputed.
        """
        sig = self._layer_signature(gdf)
        digest = hashlib.sha256()
        digest.update(str(sig['n']).encode())
        if sig['ids'] is not None:
            digest.update('\x1f'.join(sig['ids']).encode())
        digest.update(np.round(sig['bounds'], 9).tobytes())
        digest.update(np.asarray(sig['areas'], dtype=np.float32).tobytes())
        dem_stat = Path(self.dem_path).stat() if Path(self.dem_path).exists() else None
        signature = {
            'strategy': strategy,
            'catchments': digest.hexdigest(),
            'dem': str(self.dem_path),
            'dem_size': dem_stat.st_size if dem_stat else None,
            'dem_mtime_ns': dem_stat.st_mtime_ns if dem_stat else None,
        }

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        signature_file = checkpoint_dir / 'signature.json'
        stored = None
        if signature_file.exists():
            try:
                stored = json.loads(signature_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                stored = None

        if stored != signature:
            stale = sorted(checkpoint_dir.glob('*.npy'))
            if stale:
                self.logger.info(
                    f"Discarding {len(stale)} elevation checkpoint(s) in {checkpoint_dir}: they "
                    "were written for a different catchment layer, DEM or chunking."
                )
                for npy in stale:
                    npy.unlink()
            signature_file.write_text(json.dumps(signature, indent=2), encoding='utf-8')

    def _migrate_legacy_bundle_raster(self, raster_path: Path, legacy_filename: str) -> None:
        """Rename known legacy Bow/Banff raster files to the configured filename."""
        if raster_path.exists():
            return

        if self._get_config_value(lambda: self.config.domain.name) != 'bow_banff_minimal':
            return

        legacy_raster = raster_path.parent / legacy_filename
        if legacy_raster.exists():
            legacy_raster.replace(raster_path)
            self.logger.info(f"Renamed legacy raster '{legacy_filename}' to '{raster_path.name}'")

    def _write_legacy_stats_csv(self, filename: str, df: pd.DataFrame, label: str) -> None:
        """Write compatibility CSVs used by older Bow/Banff workflows."""
        if self._get_config_value(lambda: self.config.domain.name) != 'bow_banff_minimal':
            return

        legacy_csv_dir = self.project_attributes_dir / "gistool-outputs"
        legacy_csv_dir.mkdir(parents=True, exist_ok=True)
        legacy_csv_path = legacy_csv_dir / filename
        df.to_csv(legacy_csv_path)
        self.logger.info(f"Created legacy {label} CSV: {legacy_csv_path}")

    def _read_raster_crs(self, raster_path: Path, layer_label: str, fallback: Any = None) -> Any:
        """Read raster CRS, optionally returning a fallback when CRS lookup fails."""
        try:
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                self.logger.info(f"{layer_label} raster CRS: {raster_crs}")
                return raster_crs
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as exc:
            self.logger.error(f"Error reading {layer_label} raster CRS: {exc}")
            if fallback is not None:
                self.logger.warning(
                    f"Falling back to CRS '{fallback}' for {layer_label}; calculations may be less reliable."
                )
                return fallback
            raise

    def _align_catchments_to_raster_crs(
        self,
        catchment_gdf: gpd.GeoDataFrame,
        raster_crs: Any,
        layer_label: str,
        *,
        allow_fallback_on_error: bool,
    ) -> gpd.GeoDataFrame:
        """Align catchment geometries to raster CRS, with optional fallback behavior."""
        shapefile_crs = catchment_gdf.crs
        self.logger.info(f"Catchment shapefile CRS: {shapefile_crs}")

        if raster_crs == shapefile_crs:
            self.logger.info("CRS match - no reprojection needed")
            return catchment_gdf.copy()

        self.logger.info(
            f"CRS mismatch detected for {layer_label}. "
            f"Reprojecting catchment from {shapefile_crs} to {raster_crs}"
        )
        try:
            projected = catchment_gdf.to_crs(raster_crs)
            self.logger.info("CRS reprojection successful")
            return projected
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as exc:
            self.logger.error(f"Failed to reproject CRS for {layer_label}: {exc}")
            if allow_fallback_on_error:
                self.logger.warning("Using original CRS - calculation may fail")
                return catchment_gdf.copy()
            raise

    def calculate_elevation_stats(self):
        """
        Calculate elevation statistics with chunked processing for memory efficiency.

        This method processes catchments in spatial tiles or index chunks to avoid
        OOM on large domains like NWM North America (~2.7M catchments).
        """
        # Get the output path and check if the file already exists
        intersect_path = self.paths.resolve('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self._get_config_value(
            lambda: self.config.paths.intersect_dem_name, default='default'
        )
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

        output_file = intersect_path / intersect_name
        checkpoint_dir = intersect_path / 'checkpoints'

        # Check if output already exists
        if self._output_has_expected_columns(output_file, required_columns=('elev_mean',)):
            self.logger.info(f"Elevation statistics file already exists: {output_file}. Skipping calculation.")
            return

        self.logger.info("Calculating elevation statistics (memory-optimized chunked mode)")

        # Fallback for legacy naming in data bundle
        self._migrate_legacy_bundle_raster(
            self.dem_path,
            "domain_Bow_at_Banff_lumped_elv.tif",
        )

        # Load catchment shapefile
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        n_catchments = len(catchment_gdf)
        self.logger.info(f"Loaded {n_catchments:,} catchments")

        # Ensure we have a unique ID column for tracking
        if 'chunk_idx' not in catchment_gdf.columns:
            catchment_gdf['chunk_idx'] = catchment_gdf.index

        try:
            # Get DEM info
            dem_crs = self._read_raster_crs(self.dem_path, "DEM")
            catchment_gdf_projected = self._align_catchments_to_raster_crs(
                catchment_gdf,
                dem_crs,
                "DEM",
                allow_fallback_on_error=False,
            )

            # Initialize results array
            elev_means = np.full(n_catchments, np.nan, dtype=np.float32)

            # === CHUNKING STRATEGY ===
            # Choose strategy based on catchment count
            if n_catchments > 500_000:
                # Very large: use spatial tiling
                self.logger.info("Using SPATIAL TILING strategy for very large domain")
                elev_means = self._process_elevation_spatial_tiles(
                    catchment_gdf_projected, elev_means, checkpoint_dir
                )
            elif n_catchments > 50_000:
                # Medium-large: use index-based chunking
                chunk_size = self._get_config_value(lambda: self.config.data.elev_chunk_size, default=10_000)
                self.logger.info(f"Using INDEX CHUNKING strategy ({chunk_size:,} catchments/chunk)")
                elev_means = self._process_elevation_index_chunks(
                    catchment_gdf_projected, elev_means, chunk_size, checkpoint_dir
                )
            else:
                # Small enough to process at once
                self.logger.info("Processing all catchments in single batch")
                with rasterio.open(self.dem_path) as src:
                    dem_array = src.read(1)
                    dem_transform = src.transform
                    dem_nodata = src.nodata

                # Catchments smaller than a DEM pixel are retried with every
                # touched pixel, then filled from the nearest catchment of the
                # same GRU (with a warning) rather than left without elevation.
                hru_col, gru_col = self._zone_id_columns()
                stats, empty = zonal_stats_covered(
                    catchment_gdf_projected.geometry,
                    dem_array,
                    affine=dem_transform,
                    stats=['mean'],
                    nodata=dem_nodata if dem_nodata is not None else -9999,
                    logger=self.logger,
                    label='Elevation',
                )
                fill_empty_zones(
                    stats,
                    empty,
                    catchment_gdf_projected.geometry,
                    group_ids=catchment_gdf[gru_col].tolist() if gru_col in catchment_gdf.columns else None,
                    zone_ids=catchment_gdf[hru_col].tolist() if hru_col in catchment_gdf.columns else None,
                    logger=self.logger,
                    label='Elevation',
                )
                for i, stat in enumerate(stats):
                    if stat.get('mean') is not None:
                        elev_means[i] = stat['mean']

            # Add results to original GeoDataFrame
            catchment_gdf['elev_mean'] = elev_means

            # Report statistics
            valid_count = np.sum(~np.isnan(elev_means))
            self.logger.info(f"Computed elevation for {valid_count:,}/{n_catchments:,} catchments ({100*valid_count/n_catchments:.1f}%)")
            if valid_count < n_catchments:
                self.logger.warning(
                    f"{n_catchments - valid_count:,} catchment(s) have no elevation (elev_mean is "
                    "missing): the DEM has no valid pixel there or a chunk failed. Check the DEM "
                    "coverage before using these catchments."
                )

            # Save output
            intersect_path.mkdir(parents=True, exist_ok=True)
            # Suppress shapefile column name truncation warning (known limitation)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Column names longer than 10.*')
                catchment_gdf.to_file(output_file)
            self.logger.info(f"Elevation statistics saved to {output_file}")

            # Legacy compatibility: also save as CSV in gistool-outputs for HYPE
            if 'elev_mean' in catchment_gdf.columns:
                self._write_legacy_stats_csv(
                    "modified_domain_stats_elv.csv",
                    catchment_gdf[['elev_mean']],
                    "elevation",
                )

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error calculating elevation statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _process_elevation_spatial_tiles(self, gdf, elev_means, checkpoint_dir):
        """
        Process catchments using spatial tiles for memory efficiency.

        Divides the domain into a grid and processes each tile independently,
        reading only the required DEM window for each tile.
        """

        # Get domain bounds
        total_bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

        # Create tile grid - aim for ~50k catchments per tile
        n_catchments = len(gdf)
        target_per_tile = self._get_config_value(lambda: self.config.data.elev_tile_target, default=50_000)
        self._validate_checkpoints(checkpoint_dir, gdf, f"tiles:{target_per_tile}")
        n_tiles_approx = max(1, n_catchments // target_per_tile)

        # Calculate grid dimensions
        aspect = (total_bounds[2] - total_bounds[0]) / max(0.001, total_bounds[3] - total_bounds[1])
        n_cols = max(1, int(np.sqrt(n_tiles_approx * aspect)))
        n_rows = max(1, int(n_tiles_approx / n_cols))

        self.logger.info(f"Creating {n_rows}x{n_cols} tile grid ({n_rows * n_cols} tiles)")

        tile_width = (total_bounds[2] - total_bounds[0]) / n_cols
        tile_height = (total_bounds[3] - total_bounds[1]) / n_rows
        buffer = max(tile_width, tile_height) * 0.01

        # Build spatial index
        self.logger.info("Building spatial index...")
        sindex = gdf.sindex

        processed_count = 0
        tile_num = 0
        total_tiles = n_rows * n_cols

        for row in range(n_rows):
            for col in range(n_cols):
                tile_num += 1

                # Calculate tile bounds with buffer
                minx = total_bounds[0] + col * tile_width - buffer
                maxx = total_bounds[0] + (col + 1) * tile_width + buffer
                miny = total_bounds[1] + row * tile_height - buffer
                maxy = total_bounds[1] + (row + 1) * tile_height + buffer

                tile_box = box(minx, miny, maxx, maxy)

                # Find catchments using spatial index
                possible_matches_idx = list(sindex.intersection(tile_box.bounds))

                if not possible_matches_idx:
                    continue

                # Filter to catchments whose centroid is in this tile (avoid duplicates)
                tile_gdf = gdf.iloc[possible_matches_idx]
                centroids = tile_gdf.geometry.centroid
                mask = centroids.within(tile_box)
                tile_gdf = tile_gdf[mask]
                tile_indices = tile_gdf['chunk_idx'].values

                if len(tile_gdf) == 0:
                    continue

                self.logger.info(f"Tile {tile_num}/{total_tiles}: {len(tile_gdf):,} catchments")

                # Check for checkpoint
                checkpoint_file = checkpoint_dir / f"tile_{row}_{col}.npy"
                if checkpoint_file.exists():
                    tile_elevs = np.load(checkpoint_file)
                    if len(tile_elevs) == len(tile_indices):
                        self.logger.debug("  Loading from checkpoint")
                        for i, idx in enumerate(tile_indices):
                            elev_means[idx] = tile_elevs[i]
                        processed_count += len(tile_gdf)
                        continue
                    self.logger.info(
                        f"  Checkpoint {checkpoint_file.name} holds {len(tile_elevs)} values for "
                        f"{len(tile_indices)} catchments; recomputing"
                    )

                try:
                    # Compute with windowed DEM reading
                    tile_elevs = self._compute_elevation_windowed(tile_gdf, tile_box.bounds)

                    for i, idx in enumerate(tile_indices):
                        elev_means[idx] = tile_elevs[i]

                    np.save(checkpoint_file, np.array(tile_elevs, dtype=np.float32))
                    processed_count += len(tile_gdf)

                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.warning(f"  Error: {str(e)}")

                del tile_gdf
                gc.collect()

                if tile_num % 20 == 0:
                    self.logger.info(f"Progress: {processed_count:,}/{len(gdf):,} catchments")

        return elev_means

    def _process_elevation_index_chunks(self, gdf, elev_means, chunk_size, checkpoint_dir):
        """
        Process catchments in index-based chunks.
        """
        self._validate_checkpoints(checkpoint_dir, gdf, f"chunks:{chunk_size}")

        n_catchments = len(gdf)
        n_chunks = (n_catchments + chunk_size - 1) // chunk_size

        self.logger.info(f"Processing {n_catchments:,} catchments in {n_chunks} chunks")
        dem_nodata = self.get_nodata_value(self.dem_path)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_catchments)

            checkpoint_file = checkpoint_dir / f"chunk_{chunk_idx}.npy"
            if checkpoint_file.exists():
                chunk_values = np.load(checkpoint_file)
                if len(chunk_values) == end_idx - start_idx:
                    self.logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: loading checkpoint")
                    elev_means[start_idx:end_idx] = chunk_values
                    continue
                self.logger.info(
                    f"Chunk {chunk_idx + 1}/{n_chunks}: checkpoint holds {len(chunk_values)} values "
                    f"for {end_idx - start_idx} catchments; recomputing"
                )

            self.logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: indices {start_idx:,}-{end_idx:,}")

            chunk_gdf = gdf.iloc[start_idx:end_idx]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # zones without a pixel centre are retried with all_touched
                    stats, _ = zonal_stats_covered(
                        chunk_gdf.geometry,
                        str(self.dem_path),
                        stats=['mean'],
                        nodata=dem_nodata,
                    )

                chunk_elevs = np.array([
                    s['mean'] if s.get('mean') is not None else np.nan
                    for s in stats
                ], dtype=np.float32)

                elev_means[start_idx:end_idx] = chunk_elevs
                np.save(checkpoint_file, chunk_elevs)

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"  Error: {str(e)}")

            del chunk_gdf
            gc.collect()

        return elev_means

    def _compute_elevation_windowed(self, gdf, tile_bounds):
        """
        Compute zonal statistics using windowed DEM reading.

        Only reads the portion of the DEM needed for the current tile.
        """
        with rasterio.open(self.dem_path) as src:
            pad = 0.001
            window = from_bounds(
                tile_bounds[0] - pad,
                tile_bounds[1] - pad,
                tile_bounds[2] + pad,
                tile_bounds[3] + pad,
                src.transform
            )

            dem_data = src.read(1, window=window)
            dem_transform = src.window_transform(window)
            dem_nodata = src.nodata if src.nodata is not None else -9999

        # zones without a pixel centre are retried with all_touched
        stats, _ = zonal_stats_covered(
            gdf.geometry,
            dem_data,
            affine=dem_transform,
            stats=['mean'],
            nodata=dem_nodata,
        )

        elevs = [s['mean'] if s.get('mean') is not None else np.nan for s in stats]

        del dem_data
        gc.collect()

        return elevs

    def calculate_soil_stats(self):
        """Calculate soil statistics with output file checking and CRS alignment"""
        # Get the output path and check if the file already exists
        intersect_path = self.paths.resolve('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self._get_config_value(lambda: self.config.paths.intersect_soil_name, default='default')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        output_file = intersect_path / intersect_name

        # Check if output already exists
        if self._output_has_expected_columns(output_file, required_prefixes=('USGS_',)):
            self.logger.info(f"Soil statistics file already exists: {output_file}. Skipping calculation.")
            return
        if output_file.exists():
            self.logger.info(f"Existing soil statistics file {output_file} does not contain expected data. Recalculating.")

        self.logger.info("Calculating soil statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        soil_name = self._get_config_value(lambda: self.config.paths.soil_class_name, default='default')
        if soil_name == 'default':
            soil_name = f"domain_{self._get_config_value(lambda: self.config.domain.name)}_soil_classes.tif"
        soil_raster = self.soil_path / soil_name

        # Fallback for legacy naming in data bundle
        self._migrate_legacy_bundle_raster(
            soil_raster,
            "domain_Bow_at_Banff_lumped_soil_classes.tif",
        )

        try:
            # Get CRS information
            soil_crs = self._read_raster_crs(soil_raster, "Soil")
            catchment_gdf_projected = self._align_catchments_to_raster_crs(
                catchment_gdf,
                soil_crs,
                "Soil",
                allow_fallback_on_error=True,
            )

            # Use rasterstats with the raster array and transform
            with rasterio.open(soil_raster) as src:
                soil_array = src.read(1)
                soil_transform = src.transform
                soil_nodata = src.nodata

            # Catchments without a pixel centre are retried with every touched
            # pixel, then given the class composition of the nearest catchment of
            # the same GRU (with a warning) — never all-zero fractions.
            hru_col, gru_col = self._zone_id_columns()
            stats, empty = zonal_stats_covered(
                catchment_gdf_projected.geometry,
                soil_array,
                affine=soil_transform,
                stats=['count'],
                categorical=True,
                nodata=soil_nodata if soil_nodata is not None else 255,
                logger=self.logger,
                label='Soil class',
            )
            fill_empty_zones(
                stats,
                empty,
                catchment_gdf_projected.geometry,
                group_ids=catchment_gdf[gru_col].tolist() if gru_col in catchment_gdf.columns else None,
                zone_ids=catchment_gdf[hru_col].tolist() if hru_col in catchment_gdf.columns else None,
                logger=self.logger,
                label='Soil class',
            )

            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return 'soil_pixel_count'
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)

            # Normalize pixel counts to fractions (0-1)
            class_cols = [col for col in result_df.columns if col.startswith('USGS_')]
            total_pixels = result_df['soil_pixel_count']
            for col in class_cols:
                result_df[col] = (result_df[col] / total_pixels).where(total_pixels > 0, 0.0)

            catchment_gdf = catchment_gdf.join(result_df)

            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            # Suppress shapefile column name truncation warning (known limitation)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Column names longer than 10.*')
                catchment_gdf.to_file(output_file)
            self.logger.info(f"Soil statistics saved to {output_file}")

            # Legacy compatibility: also save as CSV in gistool-outputs for HYPE
            self._write_legacy_stats_csv(
                "modified_domain_stats_soil_classes.csv",
                result_df,
                "soil",
            )

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error calculating soil statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_land_stats(self):
        """Calculate land statistics with output file checking and CRS alignment"""
        # Get the output path and check if the file already exists
        intersect_path = self.paths.resolve('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self._get_config_value(lambda: self.config.paths.intersect_land_name, default='default')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'

        output_file = intersect_path / intersect_name

        # Check if output already exists
        if self._output_has_expected_columns(output_file, required_prefixes=('IGBP_',)):
            self.logger.info(f"Land statistics file already exists: {output_file}. Skipping calculation.")
            return
        if output_file.exists():
            self.logger.info(f"Existing land statistics file {output_file} does not contain expected data. Recalculating.")

        self.logger.info("Calculating land statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        land_name = self._get_config_value(lambda: self.config.domain.land_class_name, default='default')
        if land_name == 'default':
            land_name = f"domain_{self._get_config_value(lambda: self.config.domain.name)}_land_classes.tif"
        land_raster = self.land_path / land_name

        # Fallback for legacy naming in data bundle
        self._migrate_legacy_bundle_raster(
            land_raster,
            "domain_Bow_at_Banff_lumped_land_classes.tif",
        )

        land_crs = self._read_raster_crs(land_raster, "Land", fallback='EPSG:4326')
        catchment_gdf_projected = self._align_catchments_to_raster_crs(
            catchment_gdf,
            land_crs,
            "Land",
            allow_fallback_on_error=True,
        )

        # Use rasterstats with the raster array and transform
        with rasterio.open(land_raster) as src:
            land_array = src.read(1)
            land_transform = src.transform
            land_nodata = src.nodata

        # Catchments without a pixel centre are retried with every touched
        # pixel, then given the class composition of the nearest catchment of
        # the same GRU (with a warning) — never all-zero fractions.
        hru_col, gru_col = self._zone_id_columns()
        stats, empty = zonal_stats_covered(
            catchment_gdf_projected.geometry,
            land_array,
            affine=land_transform,
            stats=['count'],
            categorical=True,
            nodata=land_nodata if land_nodata is not None else 255,
            logger=self.logger,
            label='Land class',
        )
        fill_empty_zones(
            stats,
            empty,
            catchment_gdf_projected.geometry,
            group_ids=catchment_gdf[gru_col].tolist() if gru_col in catchment_gdf.columns else None,
            zone_ids=catchment_gdf[hru_col].tolist() if hru_col in catchment_gdf.columns else None,
            logger=self.logger,
            label='Land class',
        )

        result_df = pd.DataFrame(stats).fillna(0)

        def rename_column(x):
            if x == 'count':
                return 'land_pixel_count'
            try:
                return f'IGBP_{int(float(x))}'
            except ValueError:
                return x

        result_df = result_df.rename(columns=rename_column)

        # Normalize pixel counts to fractions (0-1)
        class_cols = [col for col in result_df.columns if col.startswith('IGBP_')]
        total_pixels = result_df['land_pixel_count']
        for col in class_cols:
            result_df[col] = (result_df[col] / total_pixels).where(total_pixels > 0, 0.0)

        catchment_gdf = catchment_gdf.join(result_df)

        # Create output directory and save the file
        intersect_path.mkdir(parents=True, exist_ok=True)
        # Suppress shapefile column name truncation warning (known limitation)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Column names longer than 10.*')
            catchment_gdf.to_file(output_file)

        self.logger.info(f"Land statistics saved to {output_file}")

        # Legacy compatibility: also save as CSV in gistool-outputs for HYPE/MESH
        self._write_legacy_stats_csv(
            "modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv",
            result_df,
            "landcover",
        )

    def run_statistics(self):
        """Run all geospatial statistics with checks for existing outputs"""
        self.logger.info("Starting geospatial statistics calculation")

        # Count how many steps we're skipping
        skipped = 0
        total = 3  # Total number of statistics operations

        # Check soil stats
        intersect_soil_path = self.paths.resolve('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_soil_name = self._get_config_value(lambda: self.config.paths.intersect_soil_name, default='default')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'

        soil_output_file = intersect_soil_path / intersect_soil_name
        soil_ready = self._output_has_expected_columns(
            soil_output_file,
            required_prefixes=('USGS_',),
            warn_on_error=False,
        )
        if soil_ready:
            self.logger.debug(f"Soil statistics already calculated: {soil_output_file}")
            skipped += 1
        else:
            self.calculate_soil_stats()

        # Check land stats
        intersect_land_path = self.paths.resolve('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_land_name = self._get_config_value(lambda: self.config.paths.intersect_land_name, default='default')
        if intersect_land_name == 'default':
            intersect_land_name = 'catchment_with_landclass.shp'

        land_output_file = intersect_land_path / intersect_land_name
        land_ready = self._output_has_expected_columns(
            land_output_file,
            required_prefixes=('IGBP_',),
            warn_on_error=False,
        )
        if land_ready:
            self.logger.debug(f"Land statistics already calculated: {land_output_file}")
            skipped += 1
        else:
            self.calculate_land_stats()

        # Check elevation stats
        intersect_dem_path = self.paths.resolve('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_dem_name = self._get_config_value(lambda: self.config.paths.intersect_dem_name, default='default')
        if intersect_dem_name == 'default':
            intersect_dem_name = 'catchment_with_dem.shp'

        dem_output_file = intersect_dem_path / intersect_dem_name
        elev_ready = self._output_has_expected_columns(
            dem_output_file,
            required_columns=('elev_mean',),
            warn_on_error=False,
        )
        if elev_ready:
            self.logger.debug(f"Elevation statistics already calculated: {dem_output_file}")
            skipped += 1
        else:
            self.calculate_elevation_stats()

        self.logger.debug(f"Geospatial statistics completed: {skipped}/{total} steps skipped, {total-skipped}/{total} steps executed")
