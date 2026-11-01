# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base topology generator for routing models.

Provides shared algorithms for network topology generation: cycle detection/fix,
headwater basin detection, synthetic network creation, pour point snapping,
downstream reference validation, outlet enforcement, and topological sort.

Each routing model subclasses this to produce its own topology format.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)


class TopologyPreprocessorProtocol(Protocol):
    """Protocol defining what the base topology generator needs from a preprocessor."""

    logger: Any

    @property
    def project_dir(self) -> Path: ...
    @property
    def domain_name(self) -> str: ...
    @property
    def domain_definition_method(self) -> str: ...
    @property
    def river_segid_col(self) -> str: ...
    @property
    def river_downsegid_col(self) -> str: ...
    @property
    def river_length_col(self) -> str: ...
    @property
    def river_slope_col(self) -> str: ...
    @property
    def basin_gruid_col(self) -> str: ...
    @property
    def basin_hru_to_seg_col(self) -> str: ...
    @property
    def basin_area_col(self) -> str: ...

    def _get_config_value(self, getter: Any, default: Any = None, dict_key: str = '') -> Any: ...
    def calculate_feature_centroids(self, gdf: Any) -> Any: ...


@dataclass
class TopologyData:
    """Intermediate representation of network topology."""

    seg_ids: np.ndarray
    down_seg_ids: np.ndarray
    slopes: np.ndarray
    lengths: np.ndarray
    hru_ids: np.ndarray
    hru_to_seg_ids: np.ndarray
    hru_areas: np.ndarray
    elevations: Optional[np.ndarray] = None
    num_seg: int = 0
    num_hru: int = 0
    domain_type: str = 'distributed'
    summa_uses_gru_runoff: bool = False
    needs_remap_lumped_distributed: bool = False
    subcatchment_weights: Optional[np.ndarray] = None
    subcatchment_gru_ids: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.num_seg == 0:
            self.num_seg = len(self.seg_ids)
        if self.num_hru == 0:
            self.num_hru = len(self.hru_ids)


class BaseTopologyGenerator(ABC):
    """
    Base class for routing model topology generators.

    Provides shared algorithms for network topology construction. Subclasses
    implement write_topology_file() to produce model-specific output formats.
    """

    def __init__(self, preprocessor: TopologyPreprocessorProtocol):
        self.pp = preprocessor

    # =========================================================================
    # Abstract methods — subclasses must implement
    # =========================================================================

    @abstractmethod
    def write_topology_file(self, topology_data: TopologyData, output_path: Path) -> None:
        """Write topology data in the model-specific format."""

    @abstractmethod
    def get_topology_output_path(self) -> Path:
        """Return the output path for the topology file."""

    # =========================================================================
    # Domain type detection
    # =========================================================================

    def detect_domain_type(self) -> str:
        """Detect the domain type from configuration."""
        method = self.pp.domain_definition_method
        if method == 'distribute':
            return 'grid'
        if method == 'point':
            return 'point'

        routing_delineation = self.pp._get_config_value(
            lambda: self.pp.config.domain.delineation.routing,
            default='lumped',
            dict_key='ROUTING_DELINEATION',
        )
        if method == 'lumped' and routing_delineation == 'river_network':
            return 'lumped_to_distributed'
        return 'distributed'

    # =========================================================================
    # Network topology construction
    # =========================================================================

    def build_topology(self) -> TopologyData:
        """
        Build topology data from shapefiles based on the detected domain type.

        Returns TopologyData with all network information populated.
        """
        domain_type = self.detect_domain_type()

        if domain_type == 'grid':
            return self._build_grid_topology()
        if domain_type == 'point':
            return self._build_point_topology()
        if domain_type == 'lumped_to_distributed':
            return self._build_lumped_to_distributed_topology()
        return self._build_distributed_topology()

    def _build_distributed_topology(self) -> TopologyData:
        """Build topology from river network and basin shapefiles."""
        shp_river, shp_basin = self._load_river_and_basin_shapefiles()

        # Check for SUMMA attributes with multi-HRU per GRU
        attributes_path = self.pp.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
        summa_uses_gru_runoff = False

        if attributes_path.exists():
            import netCDF4 as nc4
            with nc4.Dataset(attributes_path, 'r') as attrs:
                n_hrus = len(attrs.dimensions['hru'])
                n_grus = len(attrs.dimensions['gru'])

                if n_hrus > n_grus:
                    summa_uses_gru_runoff = True
                    self.pp.logger.info(f"Distributed SUMMA with {n_hrus} HRUs across {n_grus} GRUs")

                    gru_ids = attrs.variables['gruId'][:].astype(int)
                    hru2gru = attrs.variables['hru2gruId'][:].astype(int)
                    hru_areas_all = attrs.variables['HRUarea'][:].astype(float)

                    gru_areas = np.zeros(n_grus)
                    for i, gru_id in enumerate(gru_ids):
                        gru_mask = hru2gru == gru_id
                        gru_areas[i] = hru_areas_all[gru_mask].sum()

                    hru_ids = gru_ids
                    hru_to_seg_ids = gru_ids
                    hru_areas = gru_areas
                    num_hru = n_grus
                else:
                    hru_ids, hru_to_seg_ids, hru_areas, num_hru = self._extract_basin_topology(
                        shp_river, shp_basin
                    )
        else:
            hru_ids, hru_to_seg_ids, hru_areas, num_hru = self._extract_basin_topology(
                shp_river, shp_basin
            )

        num_seg = len(shp_river)
        self._enforce_minimum_values(shp_river)
        self._enforce_outlets(shp_river)
        self._validate_downstream_refs(shp_river)
        hru_to_seg_ids = self._validate_hru_to_seg_refs(
            hru_ids, hru_to_seg_ids, shp_river
        )

        seg_ids = shp_river[self.pp.river_segid_col].values.astype(int)
        down_seg_ids = shp_river[self.pp.river_downsegid_col].values.astype(int)
        slopes = shp_river[self.pp.river_slope_col].values.astype(float)
        lengths = shp_river[self.pp.river_length_col].values.astype(float)

        return TopologyData(
            seg_ids=seg_ids,
            down_seg_ids=down_seg_ids,
            slopes=slopes,
            lengths=lengths,
            hru_ids=hru_ids,
            hru_to_seg_ids=hru_to_seg_ids,
            hru_areas=hru_areas,
            num_seg=num_seg,
            num_hru=num_hru,
            domain_type='distributed',
            summa_uses_gru_runoff=summa_uses_gru_runoff,
        )

    def _build_lumped_to_distributed_topology(self) -> TopologyData:
        """Build topology for lumped domain with distributed routing."""
        import geopandas as gpd

        shp_river, shp_basin = self._load_river_and_basin_shapefiles(
            routing_suffix='delineate'
        )

        catchment_filename = f"{self.pp.domain_name}_catchment_delineated.shp"
        catchment_path = self.pp.project_dir / 'shapefiles' / 'catchment' / catchment_filename
        if not catchment_path.exists():
            # Search subdirectories (discretization may place it in catchment/{method}/{experiment}/)
            candidates = list((self.pp.project_dir / 'shapefiles' / 'catchment').rglob(catchment_filename))
            if candidates:
                catchment_path = candidates[0]
            else:
                raise FileNotFoundError(f"Delineated catchment shapefile not found: {catchment_path}")

        shp_catchments = gpd.read_file(catchment_path)
        self.pp.logger.info(f"Loaded {len(shp_catchments)} delineated subcatchments")

        hru_ids = shp_catchments['GRU_ID'].values.astype(int)

        if self.check_if_headwater_basin(shp_river):
            shp_river = self.create_synthetic_river_network(shp_river, hru_ids)

        num_seg = len(shp_river)
        num_hru = len(shp_catchments)
        hru_to_seg_ids = shp_catchments['GRU_ID'].values.astype(int)

        total_basin_area = shp_basin[self.pp.basin_area_col].sum()
        hru_areas = shp_catchments['avg_subbas'].values * total_basin_area

        subcatchment_weights = shp_catchments['avg_subbas'].values
        subcatchment_gru_ids = hru_ids

        self._enforce_minimum_values(shp_river)
        self._enforce_outlets(shp_river)
        self._validate_downstream_refs(shp_river)

        seg_ids = shp_river[self.pp.river_segid_col].values.astype(int)
        down_seg_ids = shp_river[self.pp.river_downsegid_col].values.astype(int)
        slopes = shp_river[self.pp.river_slope_col].values.astype(float)
        lengths = shp_river[self.pp.river_length_col].values.astype(float)

        return TopologyData(
            seg_ids=seg_ids,
            down_seg_ids=down_seg_ids,
            slopes=slopes,
            lengths=lengths,
            hru_ids=hru_ids,
            hru_to_seg_ids=hru_to_seg_ids,
            hru_areas=hru_areas,
            num_seg=num_seg,
            num_hru=num_hru,
            domain_type='lumped_to_distributed',
            summa_uses_gru_runoff=True,
            needs_remap_lumped_distributed=True,
            subcatchment_weights=subcatchment_weights,
            subcatchment_gru_ids=subcatchment_gru_ids,
        )

    def _build_grid_topology(self) -> TopologyData:
        """Build topology from grid-based distributed domain."""
        import geopandas as gpd

        grid_path = (
            self.pp.project_dir / 'shapefiles' / 'river_basins'
            / f"{self.pp.domain_name}_riverBasins_distribute.shp"
        )
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid basins shapefile not found: {grid_path}")

        grid_gdf = gpd.read_file(grid_path)
        num_cells = len(grid_gdf)
        self.pp.logger.info(f"Loaded {num_cells} grid cells from {grid_path}")

        seg_ids = grid_gdf['GRU_ID'].values.astype(int)

        # D8 downstream topology
        if 'downstream_id' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream_id'].values.astype(int)
        elif 'downstream' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream'].values.astype(int)
        elif 'DSLINKNO' in grid_gdf.columns:
            down_seg_ids = grid_gdf['DSLINKNO'].values.astype(int)
        else:
            self.pp.logger.warning("No D8 topology found, setting all cells as outlets")
            down_seg_ids = np.zeros(num_cells, dtype=int)

        slopes = grid_gdf['slope'].values.astype(float) if 'slope' in grid_gdf.columns else np.full(num_cells, 0.01)
        slopes = np.maximum(slopes, 0.001)

        elevations = grid_gdf['elev_mean'].values.astype(float) if 'elev_mean' in grid_gdf.columns else np.zeros(num_cells)

        down_seg_ids = self.fix_routing_cycles(seg_ids, down_seg_ids, elevations)
        self._validate_downstream_refs_arrays(seg_ids, down_seg_ids)

        grid_cell_size = self.pp._get_config_value(
            lambda: self.pp.config.model.mizuroute.grid_cell_size,  # type: ignore[attr-defined]
            default=1000.0
        )
        lengths = np.full(num_cells, float(grid_cell_size))

        hru_ids = seg_ids.copy()
        hru_to_seg_ids = seg_ids.copy()
        hru_areas = (
            grid_gdf['GRU_area'].values.astype(float)
            if 'GRU_area' in grid_gdf.columns
            else np.full(num_cells, float(grid_cell_size) ** 2)
        )

        return TopologyData(
            seg_ids=seg_ids,
            down_seg_ids=down_seg_ids,
            slopes=slopes,
            lengths=lengths,
            hru_ids=hru_ids,
            hru_to_seg_ids=hru_to_seg_ids,
            hru_areas=hru_areas,
            elevations=elevations,
            num_seg=num_cells,
            num_hru=num_cells,
            domain_type='grid',
            summa_uses_gru_runoff=True,
        )

    def _build_point_topology(self) -> TopologyData:
        """Build minimal single-segment topology for point-scale domains."""
        self.pp.logger.info("Creating point-scale network topology")

        return TopologyData(
            seg_ids=np.array([1]),
            down_seg_ids=np.array([0]),
            slopes=np.array([0.01]),
            lengths=np.array([100.0]),
            hru_ids=np.array([1]),
            hru_to_seg_ids=np.array([1]),
            hru_areas=np.array([10000.0]),
            num_seg=1,
            num_hru=1,
            domain_type='point',
            summa_uses_gru_runoff=True,
        )

    # =========================================================================
    # Shapefile loading
    # =========================================================================

    def _load_river_and_basin_shapefiles(
        self, routing_suffix: Optional[str] = None
    ) -> Tuple['gpd.GeoDataFrame', 'gpd.GeoDataFrame']:
        """Load river network and basin shapefiles."""
        import geopandas as gpd

        if routing_suffix is None:
            routing_suffix = self._get_method_suffix()

        river_network_path = self.pp._get_config_value(
            lambda: self.pp.config.paths.river_network_path,
            default='default'
        )
        river_network_name = self.pp._get_config_value(
            lambda: self.pp.config.paths.river_network_name,
            default='default'
        )

        if river_network_name == 'default':
            river_network_name = f"{self.pp.domain_name}_riverNetwork_{routing_suffix}.shp"
        if river_network_path == 'default':
            river_network_path = self.pp.project_dir / 'shapefiles/river_network'
        else:
            river_network_path = Path(river_network_path)

        river_basin_path = self.pp._get_config_value(
            lambda: self.pp.config.paths.river_basins_path,  # type: ignore[attr-defined]
            default='default'
        )
        river_basin_name = self.pp._get_config_value(
            lambda: self.pp.config.paths.river_basins_name,  # type: ignore[attr-defined]
            default='default'
        )

        if river_basin_name == 'default':
            river_basin_name = f"{self.pp.domain_name}_riverBasins_{routing_suffix}.shp"
        if river_basin_path == 'default':
            river_basin_path = self.pp.project_dir / 'shapefiles/river_basins'
        else:
            river_basin_path = Path(river_basin_path)

        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)
        return shp_river, shp_basin

    def _get_method_suffix(self) -> str:
        """Get the shapefile method suffix from config."""
        method = self.pp.domain_definition_method
        if method in ('lumped', 'point'):
            return 'lumped'
        return method

    def _extract_basin_topology(
        self, shp_river: 'gpd.GeoDataFrame', shp_basin: 'gpd.GeoDataFrame'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Extract HRU topology from basin shapefile."""
        closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)

        if len(shp_basin) == 1:
            shp_basin.loc[shp_basin.index[0], self.pp.basin_hru_to_seg_col] = closest_segment_id
            self.pp.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")

        hru_ids = shp_basin[self.pp.basin_gruid_col].values.astype(int)
        hru_to_seg_ids = shp_basin[self.pp.basin_hru_to_seg_col].values.astype(int)
        hru_areas = shp_basin[self.pp.basin_area_col].values.astype(float)
        return hru_ids, hru_to_seg_ids, hru_areas, len(shp_basin)

    # =========================================================================
    # Headwater basin handling
    # =========================================================================

    def check_if_headwater_basin(self, shp_river: 'gpd.GeoDataFrame') -> bool:
        """Check if this is a headwater basin with None/invalid river network data."""
        seg_id_col = self.pp.river_segid_col
        downseg_id_col = self.pp.river_downsegid_col

        if seg_id_col in shp_river.columns and downseg_id_col in shp_river.columns:
            seg_ids_null = shp_river[seg_id_col].isna().all()
            downseg_ids_null = shp_river[downseg_id_col].isna().all()

            if seg_ids_null and downseg_ids_null:
                self.pp.logger.info("Detected headwater basin: all river network IDs are None/null")
                return True

            if shp_river[seg_id_col].dtype == 'object':
                seg_ids_none_str = (shp_river[seg_id_col] == 'None').all()
                downseg_ids_none_str = (shp_river[downseg_id_col] == 'None').all()
                if seg_ids_none_str and downseg_ids_none_str:
                    self.pp.logger.info("Detected headwater basin: all river network IDs are 'None' strings")
                    return True

        return False

    def create_synthetic_river_network(
        self, shp_river: 'gpd.GeoDataFrame', hru_ids: np.ndarray
    ) -> 'gpd.GeoDataFrame':
        """Create a synthetic single-segment river network for headwater basins."""
        import geopandas as gpd

        self.pp.logger.info("Creating synthetic river network for headwater basin")

        synthetic_seg_id = int(hru_ids[0]) if len(hru_ids) > 0 else 1

        synthetic_data = {
            self.pp.river_segid_col: synthetic_seg_id,
            self.pp.river_downsegid_col: 0,
            self.pp.river_length_col: 1000.0,
            self.pp.river_slope_col: 0.001,
        }

        geom_col = shp_river.geometry.name
        if not shp_river.empty and shp_river.geometry.iloc[0] is not None:
            synthetic_geom = self.pp.calculate_feature_centroids(shp_river.iloc[[0]]).iloc[0]
        else:
            from shapely.geometry import Point
            synthetic_geom = Point(0, 0)

        synthetic_data[geom_col] = synthetic_geom
        return gpd.GeoDataFrame([synthetic_data], crs=shp_river.crs)

    # =========================================================================
    # Pour point and segment lookup
    # =========================================================================

    def _find_closest_segment_to_pour_point(self, shp_river: 'gpd.GeoDataFrame') -> int:
        """Find the river segment closest to the pour point."""
        import geopandas as gpd

        pour_point_dir = self.pp.project_dir / 'shapefiles' / 'pour_point'
        pour_point_files = list(pour_point_dir.glob('*.shp'))

        if not pour_point_files:
            return self._fallback_outlet_segment(shp_river)

        try:
            shp_pour_point = gpd.read_file(pour_point_files[0])

            if shp_river.crs != shp_pour_point.crs:
                shp_pour_point = shp_pour_point.to_crs(shp_river.crs)

            shp_river_proj = shp_river.to_crs(shp_river.estimate_utm_crs())
            pour_point_centroids = self.pp.calculate_feature_centroids(shp_pour_point.iloc[[0]])
            pour_point_proj = pour_point_centroids.to_crs(shp_river_proj.crs)
            distances = shp_river_proj.geometry.distance(pour_point_proj.iloc[0])

            closest_idx = distances.idxmin()
            closest_segment_id = shp_river.loc[closest_idx, self.pp.river_segid_col]
            self.pp.logger.info(f"Closest segment to pour point: {closest_segment_id}")
            return closest_segment_id

        except Exception as e:  # noqa: BLE001
            self.pp.logger.error(f"Error finding closest segment: {e}")
            return self._fallback_outlet_segment(shp_river)

    def _fallback_outlet_segment(self, shp_river: 'gpd.GeoDataFrame') -> int:
        """Find an outlet or first segment as fallback."""
        outlet_mask = shp_river[self.pp.river_downsegid_col] == 0
        if outlet_mask.any():
            return shp_river.loc[outlet_mask, self.pp.river_segid_col].iloc[0]
        return shp_river[self.pp.river_segid_col].iloc[0]

    # =========================================================================
    # Cycle detection and fixing
    # =========================================================================

    @staticmethod
    def fix_routing_cycles(
        seg_ids: np.ndarray,
        down_seg_ids: np.ndarray,
        elevations: np.ndarray,
    ) -> np.ndarray:
        """
        Detect and fix cycles in the routing graph.

        For each cycle found, the node with the lowest elevation is forced
        to be an outlet (downSegId = 0).

        This is a static method so it can be reused by external code (e.g.,
        dRoute's network_adapter) without instantiating the full generator.
        """
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        adj = {}
        for i, down_sid in enumerate(down_seg_ids):
            if down_sid in id_to_idx:
                adj[i] = id_to_idx[down_sid]
            else:
                adj[i] = -1

        visited: Set[int] = set()
        path_set: Set[int] = set()
        cycles_found = 0
        fixed_down_ids = down_seg_ids.copy()

        for start_node_idx in range(len(seg_ids)):
            if start_node_idx in visited:
                continue

            stack = [(start_node_idx, 0)]

            while stack:
                u, state = stack[-1]

                if state == 0:
                    visited.add(u)
                    path_set.add(u)
                    stack[-1] = (u, 1)

                    v = adj.get(u, -1)
                    if v != -1:
                        if v in path_set:
                            cycles_found += 1
                            cycle_indices = []
                            for node, _ in reversed(stack):
                                cycle_indices.append(node)
                                if node == v:
                                    break

                            min_elev = float('inf')
                            sink_idx = -1
                            for idx in cycle_indices:
                                if elevations[idx] < min_elev:
                                    min_elev = elevations[idx]
                                    sink_idx = idx

                            fixed_down_ids[sink_idx] = 0
                            adj[sink_idx] = -1
                        elif v not in visited:
                            stack.append((v, 0))
                else:
                    path_set.remove(u)
                    stack.pop()

        if cycles_found > 0:
            logger.warning(f"Detected and fixed {cycles_found} cycles in routing topology.")
        else:
            logger.info("No cycles detected in routing topology.")

        return fixed_down_ids

    # =========================================================================
    # Topological sort
    # =========================================================================

    @staticmethod
    def topological_sort(seg_ids: np.ndarray, down_seg_ids: np.ndarray) -> List[int]:
        """
        Topological sort of segment indices (Kahn's algorithm).

        Returns indices in headwater-first (upstream-to-downstream) order.
        """
        n = len(seg_ids)
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        in_degree = np.zeros(n, dtype=int)
        children: dict = {i: [] for i in range(n)}

        for i, down_sid in enumerate(down_seg_ids):
            if down_sid in id_to_idx:
                downstream_idx = id_to_idx[down_sid]
                in_degree[downstream_idx] += 1
                children[i].append(downstream_idx)

        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Append any remaining nodes (disconnected/cyclic) at end
        if len(order) < n:
            remaining = [i for i in range(n) if i not in set(order)]
            order.extend(remaining)

        return order

    # =========================================================================
    # Validation and enforcement
    # =========================================================================

    def _enforce_minimum_values(self, shp_river: 'gpd.GeoDataFrame') -> None:
        """Ensure minimum values for length and slope."""
        length_col = self.pp.river_length_col
        if length_col in shp_river.columns:
            shp_river[length_col] = shp_river[length_col].fillna(0)
            shp_river.loc[shp_river[length_col] == 0, length_col] = 1

        slope_col = self.pp.river_slope_col
        if slope_col in shp_river.columns:
            shp_river[slope_col] = shp_river[slope_col].fillna(0.001)
            shp_river.loc[shp_river[slope_col] == 0, slope_col] = 0.001

    def _enforce_outlets(self, shp_river: 'gpd.GeoDataFrame') -> None:
        """Force configured segments to be outlets (downSegId = 0)."""
        make_outlet = self.pp._get_config_value(
            lambda: self.pp.config.model.mizuroute.make_outlet,  # type: ignore[attr-defined]
            default='n/a',
            dict_key='SETTINGS_MIZU_MAKE_OUTLET',
        )
        if not make_outlet or make_outlet == 'n/a':
            return

        seg_id_col = self.pp.river_segid_col
        downseg_id_col = self.pp.river_downsegid_col

        for outlet_id in [int(x) for x in make_outlet.split(',')]:
            if outlet_id in shp_river[seg_id_col].values:
                shp_river.loc[shp_river[seg_id_col] == outlet_id, downseg_id_col] = 0
            else:
                self.pp.logger.warning(f"Outlet ID {outlet_id} not found in river network")

    def _validate_downstream_refs(self, shp_river: 'gpd.GeoDataFrame') -> None:
        """Fix invalid downstream segment references in a GeoDataFrame."""
        seg_id_col = self.pp.river_segid_col
        downseg_id_col = self.pp.river_downsegid_col
        valid_seg_ids = set(shp_river[seg_id_col].values.astype(int))

        invalid_refs = []
        for idx, row in shp_river.iterrows():
            seg_id = int(row[seg_id_col])
            down_seg_id = int(row[downseg_id_col])
            if down_seg_id not in valid_seg_ids and down_seg_id != -9999 and down_seg_id != 0:
                invalid_refs.append((seg_id, down_seg_id))
                shp_river.loc[idx, downseg_id_col] = 0

        if invalid_refs:
            self.pp.logger.warning(f"Fixed {len(invalid_refs)} invalid downstream segment references")

    def _validate_downstream_refs_arrays(
        self, seg_ids: np.ndarray, down_seg_ids: np.ndarray
    ) -> None:
        """Fix invalid downstream refs in numpy arrays (modifies in place)."""
        valid_seg_ids = set(seg_ids)
        invalid_count = 0
        for i, down_seg_id in enumerate(down_seg_ids):
            if down_seg_id not in valid_seg_ids and down_seg_id != -9999 and down_seg_id != 0:
                invalid_count += 1
                down_seg_ids[i] = 0
        if invalid_count > 0:
            self.pp.logger.warning(f"Fixed {invalid_count} invalid downstream segment references")

    def _validate_hru_to_seg_refs(
        self,
        hru_ids: np.ndarray,
        hru_to_seg_ids: np.ndarray,
        shp_river: 'gpd.GeoDataFrame',
    ) -> np.ndarray:
        """Fix invalid HRU-to-segment references."""
        valid_seg_ids = set(shp_river[self.pp.river_segid_col].values.astype(int))
        fixed = hru_to_seg_ids.copy()

        for i, hru_to_seg in enumerate(fixed):
            if hru_to_seg not in valid_seg_ids:
                closest_seg = min(valid_seg_ids, key=lambda x: abs(x - hru_to_seg))
                fixed[i] = closest_seg
                self.pp.logger.warning(
                    f"HRU {hru_ids[i]} had invalid segment ref {hru_to_seg} -> set to {closest_seg}"
                )

        return fixed
