# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NWS NextGen Hydrofabric Acquisition Handler

Downloads the CONUS NextGen Hydrofabric from the Community Hydrofabric S3
bucket, then subsets locally using the network topology to extract only the
upstream catchments and flowpaths for the domain.

Workflow:
    1. Download conus_nextgen.tar.gz (cached in ~/.symfluence/hydrofabric/)
    2. Extract to conus_nextgen.gpkg
    3. Locate pour point catchment via spatial query on divides layer
    4. Trace upstream network using the network table (toid graph)
    5. Subset divides and flowpaths layers for output

Source:
    Community Hydrofabric S3 (public, anonymous):
    https://communityhydrofabric.s3.us-east-1.amazonaws.com/
        hydrofabrics/community/conus_nextgen.tar.gz

Column Convention (NextGen v2.x):
    divide_id, toid

References:
    - Johnson et al. (2023). National Hydrologic Geospatial Fabric (hydrofabric)
      for the Next Generation (NextGen) Hydrologic Modeling Framework
    - https://github.com/CIROH-UA/NGIAB_data_preprocess
"""

import sqlite3
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set, Tuple

import requests

if TYPE_CHECKING:
    import geopandas as gpd

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session

# Community Hydrofabric S3 — public bucket, anonymous access
_S3_BUCKET = "communityhydrofabric"
_S3_REGION = "us-east-1"
_S3_KEY = "hydrofabrics/community/conus_nextgen.tar.gz"
_HYDROFABRIC_URL = (
    f"https://{_S3_BUCKET}.s3.{_S3_REGION}.amazonaws.com/{_S3_KEY}"
)

# Shared cache directory for the CONUS hydrofabric (avoids re-downloading per project)
_CACHE_DIR = Path.home() / ".symfluence" / "hydrofabric" / "v2.2"

# CONUS VPU (HUC-2) approximate bounding boxes — used only for CONUS validation
# Format: {vpu_code: (lat_min, lon_min, lat_max, lon_max)}
_NWS_VPU_BBOXES = {
    "01": (40.0, -80.5, 48.0, -66.5),     # New England
    "02": (36.0, -81.0, 45.5, -72.0),     # Mid-Atlantic
    "03N": (29.0, -91.0, 37.0, -75.5),    # South Atlantic-Gulf (North)
    "03S": (24.0, -90.0, 31.0, -79.5),    # South Atlantic-Gulf (South)
    "03W": (28.5, -91.5, 35.0, -84.0),    # South Atlantic-Gulf (West)
    "04": (40.5, -93.0, 49.5, -74.0),     # Great Lakes
    "05": (34.5, -90.5, 43.5, -77.5),     # Ohio
    "06": (35.0, -91.5, 43.5, -83.0),     # Tennessee
    "07": (37.0, -98.0, 49.5, -84.0),     # Upper Mississippi
    "08": (28.0, -98.0, 37.0, -88.0),     # Lower Mississippi
    "09": (43.5, -99.0, 49.5, -88.0),     # Souris-Red-Rainy
    "10L": (36.0, -105.0, 49.5, -95.5),   # Missouri (Lower)
    "10U": (37.0, -114.0, 49.5, -96.0),   # Missouri (Upper)
    "11": (27.5, -106.5, 40.0, -93.5),    # Arkansas-White-Red
    "12": (25.5, -107.0, 37.0, -93.0),    # Texas-Gulf
    "13": (31.0, -109.5, 38.0, -103.0),   # Rio Grande
    "14": (35.5, -113.0, 43.5, -105.5),   # Upper Colorado
    "15": (31.0, -115.0, 38.0, -106.5),   # Lower Colorado
    "16": (34.0, -120.5, 44.0, -109.0),   # Great Basin
    "17": (42.0, -125.0, 49.5, -110.5),   # Pacific Northwest
    "18": (32.0, -125.0, 43.0, -114.0),   # California
}


@AcquisitionRegistry.register('NWS_HYDROFABRIC')
@AcquisitionRegistry.register('NWS')
class NWSHydrofabricAcquirer(BaseAcquisitionHandler, RetryMixin):
    """NWS NextGen Hydrofabric acquisition with local subsetting.

    Downloads the full CONUS NextGen hydrofabric once (cached in
    ~/.symfluence/hydrofabric/v2.2/), then subsets locally for the domain
    using upstream network tracing from the pour point.

    Config Keys:
        NWS_HYDROFABRIC_VERSION: Version string (default: 'v2.2')

    Output Files:
        nws_catchments.gpkg — subset catchment polygons (divides layer)
        nws_flowpaths.gpkg — subset river network lines (flowpaths layer)
    """

    def download(self, output_dir: Path) -> Path:
        """Download and subset NWS Hydrofabric data for the domain.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the directory containing subset geofabric files
        """
        geofabric_dir = self._attribute_dir("geofabric") / "nws_hydrofabric"
        geofabric_dir.mkdir(parents=True, exist_ok=True)

        cat_path = geofabric_dir / "nws_catchments.gpkg"
        riv_path = geofabric_dir / "nws_flowpaths.gpkg"

        if self._skip_if_exists(cat_path) and self._skip_if_exists(riv_path):
            return geofabric_dir

        lat, lon = self._get_pour_point_coords()
        self.logger.info(
            f"Preparing NWS Hydrofabric for pour point ({lat:.4f}, {lon:.4f})"
        )

        # Validate CONUS coverage
        if not self._is_within_conus(lat, lon):
            raise ValueError(
                f"Pour point ({lat}, {lon}) does not fall within CONUS. "
                "NWS Hydrofabric is CONUS-only."
            )

        # Step 1: Ensure CONUS hydrofabric is downloaded (shared cache)
        gpkg_path = self._ensure_conus_gpkg()

        # Step 2: Find pour point catchment and trace upstream
        upstream_ids = self._trace_upstream_network(gpkg_path, lat, lon)
        self.logger.info(
            f"Found {len(upstream_ids)} upstream features for pour point"
        )

        # Step 3: Subset and save
        self._subset_and_save(gpkg_path, upstream_ids, cat_path, riv_path)

        self.logger.info(f"NWS Hydrofabric subset saved to: {geofabric_dir}")
        return geofabric_dir

    # =========================================================================
    # CONUS Hydrofabric Download & Cache
    # =========================================================================

    def _ensure_conus_gpkg(self) -> Path:
        """Ensure the CONUS hydrofabric gpkg is available in the shared cache.

        Returns:
            Path to conus_nextgen.gpkg
        """
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gpkg_path = _CACHE_DIR / "conus_nextgen.gpkg"

        if gpkg_path.exists():
            self.logger.info(f"Using cached CONUS hydrofabric: {gpkg_path}")
            return gpkg_path

        tar_path = _CACHE_DIR / "conus_nextgen.tar.gz"

        # Download the tar.gz
        if not tar_path.exists():
            self._download_conus_archive(tar_path)

        # Extract gpkg from tar.gz
        self._extract_conus_archive(tar_path, gpkg_path)

        return gpkg_path

    def _download_conus_archive(self, tar_path: Path):
        """Download the CONUS hydrofabric tar.gz from S3.

        Args:
            tar_path: Path to save the tar.gz file
        """
        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        def do_download():
            self.logger.info(
                f"Downloading CONUS NextGen hydrofabric from {_HYDROFABRIC_URL}"
            )
            self.logger.info(
                "This is a large download (~1.6 GB) and will be cached for "
                f"future use at {_CACHE_DIR}"
            )
            part_path = tar_path.with_suffix('.part')
            with session.get(_HYDROFABRIC_URL, stream=True, timeout=3600) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                downloaded = 0
                with open(part_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0 and downloaded % (100 * 1024 * 1024) < 1024 * 1024:
                                pct = downloaded * 100 // total
                                self.logger.info(
                                    f"Download progress: {downloaded // (1024*1024)} MB "
                                    f"/ {total // (1024*1024)} MB ({pct}%)"
                                )
            part_path.rename(tar_path)
            self.logger.info(f"Download complete: {tar_path}")

        self.execute_with_retry(
            do_download,
            max_retries=3,
            base_delay=30,
            backoff_factor=2.0,
            retryable_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError,
                IOError,
            ),
        )

    def _extract_conus_archive(self, tar_path: Path, gpkg_path: Path):
        """Extract conus_nextgen.gpkg from the tar.gz archive.

        Args:
            tar_path: Path to the tar.gz file
            gpkg_path: Target path for the extracted gpkg
        """
        self.logger.info(f"Extracting {tar_path.name}...")

        part_path = gpkg_path.with_suffix('.part')

        with tarfile.open(tar_path, 'r:gz') as tar:
            # Find the gpkg member
            gpkg_member = None
            for member in tar.getmembers():
                if member.name.endswith('.gpkg'):
                    gpkg_member = member
                    break

            if gpkg_member is None:
                raise FileNotFoundError(
                    f"No .gpkg file found in {tar_path}"
                )

            self.logger.info(
                f"Extracting {gpkg_member.name} "
                f"({gpkg_member.size // (1024*1024)} MB)..."
            )
            source = tar.extractfile(gpkg_member)
            if source is None:
                raise FileNotFoundError(
                    f"Cannot extract {gpkg_member.name} from archive"
                )
            with open(part_path, 'wb') as f:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        part_path.rename(gpkg_path)
        self.logger.info(f"Extracted CONUS hydrofabric: {gpkg_path}")

        # Remove tar.gz to save disk space
        tar_path.unlink()
        self.logger.info("Removed archive to save disk space")

    # =========================================================================
    # Network Tracing (upstream from pour point)
    # =========================================================================

    def _trace_upstream_network(
        self, gpkg_path: Path, lat: float, lon: float
    ) -> Set[str]:
        """Find all upstream catchment IDs from the pour point.

        Uses SQLite spatial queries on the CONUS gpkg to:
        1. Find the catchment containing the pour point
        2. Build the upstream network graph from the network table
        3. Trace all upstream nodes

        Args:
            gpkg_path: Path to conus_nextgen.gpkg
            lat: Pour point latitude (EPSG:4326)
            lon: Pour point longitude (EPSG:4326)

        Returns:
            Set of upstream feature IDs (divide_id values like 'cat-NNNN')
        """
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point

        # The CONUS hydrofabric uses EPSG:5070 (CONUS Albers)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        x, y = transformer.transform(lon, lat)
        point = Point(x, y)

        self.logger.info("Locating pour point catchment in CONUS hydrofabric...")

        # Find the catchment containing the pour point using spatial query
        # Read only a spatial window around the point to avoid loading all of CONUS
        bbox_buffer = 50000  # 50km buffer in EPSG:5070 meters
        bbox = (x - bbox_buffer, y - bbox_buffer, x + bbox_buffer, y + bbox_buffer)
        divides = gpd.read_file(gpkg_path, layer='divides', bbox=bbox)

        if divides.empty:
            raise ValueError(
                f"No catchments found near pour point ({lat}, {lon}). "
                "Check that coordinates are within CONUS."
            )

        # Find exact containing catchment
        containing = divides[divides.geometry.contains(point)]
        if containing.empty:
            # Fall back to nearest
            divides['_dist'] = divides.geometry.distance(point)
            nearest = divides.loc[divides['_dist'].idxmin()]
            cat_id = nearest['divide_id']
            self.logger.warning(
                f"Pour point not inside any catchment, using nearest: {cat_id}"
            )
        else:
            cat_id = containing.iloc[0]['divide_id']

        self.logger.info(f"Pour point catchment: {cat_id}")

        # Build upstream network from the network table using SQLite
        upstream_ids = self._trace_upstream_sql(gpkg_path, cat_id)

        return upstream_ids

    def _trace_upstream_sql(
        self, gpkg_path: Path, cat_id: str
    ) -> Set[str]:
        """Trace upstream network using SQL on the network table.

        The network table has columns: id, toid, divide_id.
        Edges go from id -> toid (downstream). We trace upstream by
        finding all nodes that eventually flow into cat_id.

        Args:
            gpkg_path: Path to conus_nextgen.gpkg
            cat_id: Starting catchment ID (e.g. 'cat-1643991')

        Returns:
            Set of all upstream feature IDs (waterbodies + nexuses)
        """
        conn = sqlite3.connect(str(gpkg_path))
        try:
            # Load the full network topology (id -> toid)
            cursor = conn.execute("SELECT id, toid, divide_id FROM network")
            rows = cursor.fetchall()

            # Build reverse adjacency: toid -> [ids that flow into it]
            # This lets us traverse upstream from any node
            reverse_adj: dict[str, list[str]] = {}
            id_to_divide: dict[str, str] = {}
            for row_id, toid, divide_id in rows:
                if toid:
                    reverse_adj.setdefault(toid, []).append(row_id)
                if divide_id:
                    id_to_divide[row_id] = divide_id

            # Find the network node for this catchment
            # cat_id is like 'cat-NNNN', network id is like 'wb-NNNN'
            numeric_part = cat_id.replace('cat-', '')
            wb_id = f'wb-{numeric_part}'

            # Also find the downstream nexus so we include the outlet
            start_nodes = set()
            if wb_id in {r[0] for r in rows}:
                start_nodes.add(wb_id)
                # Find the nexus this wb flows to (outlet nexus)
                for row_id, toid, _ in rows:
                    if row_id == wb_id and toid:
                        start_nodes.add(toid)
                        break

            if not start_nodes:
                # Try matching by divide_id directly
                for row_id, _, divide_id in rows:
                    if divide_id == cat_id:
                        start_nodes.add(row_id)
                        break

            if not start_nodes:
                raise ValueError(
                    f"Could not find network node for catchment {cat_id}"
                )

            # BFS upstream from start nodes
            visited = set()
            queue = list(start_nodes)
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                for upstream_node in reverse_adj.get(node, []):
                    if upstream_node not in visited:
                        queue.append(upstream_node)

            # Collect all divide_ids from visited nodes
            divide_ids = set()
            for node in visited:
                if node in id_to_divide:
                    divide_ids.add(id_to_divide[node])
                # Also include the node id itself for subsetting flowpaths
                divide_ids.add(node)

            # Ensure the starting catchment is included
            divide_ids.add(cat_id)

            self.logger.info(
                f"Traced {len(visited)} network nodes, "
                f"{len([d for d in divide_ids if d.startswith('cat-')])} catchments"
            )

            return divide_ids

        finally:
            conn.close()

    # =========================================================================
    # Subsetting & Output
    # =========================================================================

    def _subset_and_save(
        self,
        gpkg_path: Path,
        upstream_ids: Set[str],
        cat_path: Path,
        riv_path: Path,
    ):
        """Subset divides and flowpaths layers and save as separate gpkg files.

        Args:
            gpkg_path: Path to conus_nextgen.gpkg
            upstream_ids: Set of all upstream IDs from network tracing
            cat_path: Output path for catchments
            riv_path: Output path for flowpaths
        """

        # Subset catchment IDs (cat-NNNN)
        cat_ids = {i for i in upstream_ids if i.startswith('cat-')}
        # Subset waterbody/flowpath IDs (wb-NNNN)
        wb_ids = {i for i in upstream_ids if i.startswith('wb-')}

        # Extract divides (catchment polygons)
        self.logger.info(f"Subsetting {len(cat_ids)} catchments from divides layer...")
        divides = self._read_by_ids(gpkg_path, 'divides', 'divide_id', cat_ids)
        if divides is not None and not divides.empty:
            divides.to_file(cat_path, driver='GPKG')
            self.logger.info(
                f"Saved {len(divides)} catchments to {cat_path}"
            )
        else:
            self.logger.warning("No catchments found in subset")

        # Extract flowpaths (river network lines)
        self.logger.info(f"Subsetting {len(wb_ids)} flowpaths...")
        flowpaths = self._read_by_ids(gpkg_path, 'flowpaths', 'id', wb_ids)
        if flowpaths is not None and not flowpaths.empty:
            flowpaths.to_file(riv_path, driver='GPKG')
            self.logger.info(
                f"Saved {len(flowpaths)} flowpaths to {riv_path}"
            )
        else:
            self.logger.warning("No flowpaths found in subset")

    def _read_by_ids(
        self,
        gpkg_path: Path,
        layer: str,
        id_col: str,
        ids: Set[str],
    ) -> Optional['gpd.GeoDataFrame']:
        """Read a gpkg layer filtered by a set of IDs using SQL.

        Uses SQL WHERE IN for efficient subsetting of the large CONUS gpkg.

        Args:
            gpkg_path: Path to the GeoPackage
            layer: Layer name to read
            id_col: Column name to filter on
            ids: Set of ID values to include

        Returns:
            Filtered GeoDataFrame, or None on error
        """
        import geopandas as gpd

        if not ids:
            return None

        # Build SQL query with parameterized IN clause
        id_list = ",".join(f"'{i}'" for i in ids)
        sql = f'SELECT * FROM "{layer}" WHERE "{id_col}" IN ({id_list})'

        try:
            return gpd.read_file(gpkg_path, sql=sql)
        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error(f"Failed to subset {layer}: {e}")
            return None

    # =========================================================================
    # Coordinate Helpers
    # =========================================================================

    def _get_pour_point_coords(self) -> Tuple[float, float]:
        """Extract pour point lat/lon from config.

        Returns:
            Tuple of (lat, lon)
        """
        pour_point_str = self._get_config_value(
            lambda: self.config.domain.pour_point_coords, default=None
        )
        if pour_point_str:
            parts = str(pour_point_str).replace('/', ',').split(',')
            return float(parts[0].strip()), float(parts[1].strip())

        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
        return lat, lon

    def _is_within_conus(self, lat: float, lon: float) -> bool:
        """Check if coordinates fall within any CONUS VPU bounding box.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if within CONUS
        """
        for lat_min, lon_min, lat_max, lon_max in _NWS_VPU_BBOXES.values():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return True
        return False
