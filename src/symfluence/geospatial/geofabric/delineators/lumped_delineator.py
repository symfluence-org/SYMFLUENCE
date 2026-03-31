# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Lumped watershed delineation using TauDEM.

Delineates a single lumped watershed based on DEM and pour point.
Uses TauDEM for watershed delineation and creates simplified river network.

Refactored from geofabric_utils.py (2026-01-01)
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from ....geospatial.delineation_registry import DelineationRegistry
from ..base.base_delineator import BaseGeofabricDelineator
from ..processors.gdal_processor import GDALProcessor
from ..processors.taudem_executor import TauDEMExecutor


@DelineationRegistry.register('lumped')
class LumpedWatershedDelineator(BaseGeofabricDelineator):
    """
    Delineates lumped watersheds using TauDEM.

    A lumped watershed is a single basin with a single pour point,
    suitable for simple hydrological modeling.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize lumped watershed delineator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        self.output_dir = self.project_dir / "shapefiles/tempdir"
        self.interim_dir = self.project_dir / "taudem-interim-files" / "lumped"
        self.delineation_method = 'TauDEM'

        self.taudem = TauDEMExecutor(config, logger, self.taudem_dir)
        self.gdal = GDALProcessor(logger)

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files."""
        return "lumped"

    def delineate_lumped_watershed(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Delineate a lumped watershed.

        Returns:
            Tuple of (river_network_path, river_basins_path)
        """
        self.logger.info(f"Delineating lumped watershed: {self.domain_name}")

        pour_point_path = self._get_pour_point_path()
        self.pour_point_path = pour_point_path

        method_suffix = self._get_method_suffix()
        river_basins_path = (
            self.project_dir / "shapefiles" / "river_basins" /
            f"{self.domain_name}_riverBasins_{method_suffix}.shp"
        )
        river_network_path = (
            self.project_dir / "shapefiles" / "river_network" /
            f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
        )

        river_basins_path.parent.mkdir(parents=True, exist_ok=True)
        river_network_path.parent.mkdir(parents=True, exist_ok=True)

        basin_result = self._delineate_with_taudem()
        if basin_result is None:
            self.logger.error("Lumped watershed delineation failed; _delineate_with_taudem returned None")
            return None, None

        if not river_basins_path.exists():
            self.logger.error(
                f"Lumped watershed delineation failed; river basin shapefile was not created: {river_basins_path}"
            )
            return None, None

        self._create_river_network(pour_point_path, river_network_path)
        self._ensure_required_fields(river_basins_path, river_network_path)

        return river_network_path, river_basins_path

    def _create_river_network(self, pour_point_path: Path, river_network_path: Path) -> None:
        """
        Create a simple river network shapefile based on the pour point.

        Args:
            pour_point_path: Path to the pour point shapefile
            river_network_path: Path to save the river network shapefile
        """
        try:
            pour_point_gdf = gpd.read_file(pour_point_path)
            river_network = pour_point_gdf.copy()

            river_network['LINKNO'] = 1
            river_network['DSLINKNO'] = 0
            river_network['Length'] = 100.0
            river_network['Slope'] = 0.01
            river_network['GRU_ID'] = 1

            river_network.to_file(river_network_path)
            self.logger.debug(f"Created river network shapefile at: {river_network_path}")

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error creating river network: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _ensure_required_fields(self, river_basins_path: Path, river_network_path: Path) -> None:
        """
        Ensure that all required fields are present in both shapefiles.

        Args:
            river_basins_path: Path to the river basins shapefile
            river_network_path: Path to the river network shapefile
        """
        try:
            basins_gdf = gpd.read_file(river_basins_path)

            required_basin_fields = {
                'GRU_ID': 1,
                'gru_to_seg': 1
            }

            if basins_gdf.crs is None:
                basins_gdf = basins_gdf.set_crs("EPSG:4326")
                self.logger.info("Assigned EPSG:4326 to lumped river basins before area computation")

            if 'GRU_area' not in basins_gdf.columns:
                original_crs = basins_gdf.crs
                utm_crs = basins_gdf.estimate_utm_crs()
                basins_utm = basins_gdf.to_crs(utm_crs)
                basins_gdf['GRU_area'] = basins_utm.geometry.area
                if original_crs is not None:
                    basins_gdf = basins_gdf.to_crs(original_crs)

            for field, default_value in required_basin_fields.items():
                if field not in basins_gdf.columns:
                    basins_gdf[field] = default_value

            basins_gdf.to_file(river_basins_path)
            self.logger.debug(f"Updated river basins shapefile with required fields at: {river_basins_path}")

            if river_network_path.exists():
                network_gdf = gpd.read_file(river_network_path)

                required_network_fields = {
                    'LINKNO': 1,
                    'DSLINKNO': 0,
                    'Length': 100.0,
                    'Slope': 0.01,
                    'GRU_ID': 1
                }

                for field, default_value in required_network_fields.items():
                    if field not in network_gdf.columns:
                        network_gdf[field] = default_value

                network_gdf.to_file(river_network_path)
                self.logger.debug(f"Updated river network shapefile with required fields at: {river_network_path}")

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error ensuring required fields: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    @staticmethod
    def _normalize_id(value):
        """Normalize topology IDs to comparable integers when possible."""
        if pd.isna(value):
            return None
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            try:
                return int(float(value))
            except Exception:  # noqa: BLE001
                return str(value)

    def _build_upstream_id_set_from_rivers(
        self,
        rivers_gdf: gpd.GeoDataFrame,
        outlet_id,
        river_id_col: str,
        downstream_col: str
    ):
        """Collect outlet segment and all upstream segments using river topology."""
        upstream_ids = set()
        pending = [outlet_id]

        while pending:
            current = pending.pop()
            if current in upstream_ids or current is None:
                continue
            upstream_ids.add(current)

            upstream_matches = rivers_gdf[
                rivers_gdf[downstream_col].apply(self._normalize_id) == current
            ]
            for _, row in upstream_matches.iterrows():
                next_id = self._normalize_id(row[river_id_col])
                if next_id is not None and next_id not in upstream_ids:
                    pending.append(next_id)

        return upstream_ids

    def _read_taudem_tree(self, tree_path: Path) -> Optional[pd.DataFrame]:
        """Read TauDEM basin-tree.dat using a tolerant parser."""
        if not tree_path.exists() or tree_path.stat().st_size == 0:
            return None

        try:
            with open(tree_path, "r") as f:
                first_nonempty = ""
                for line in f:
                    if line.strip():
                        first_nonempty = line.strip()
                        break

            has_header = any(ch.isalpha() for ch in first_nonempty)

            if has_header:
                df = pd.read_csv(tree_path, sep=r"\s+", comment="#", engine="python")
            else:
                df = pd.read_csv(tree_path, sep=r"\s+", header=None, comment="#", engine="python")
                if df.shape[1] >= 4:
                    rename_map = {
                        0: "LINKNO",
                        1: "DSLINKNO",
                        2: "USLINKNO1",
                        3: "USLINKNO2",
                    }
                    df = df.rename(columns=rename_map)

            return df
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Could not parse TauDEM tree file {tree_path}: {e}")
            return None

    def _build_upstream_id_set_from_tree(self, tree_df: pd.DataFrame, outlet_id):
        """Build upstream set from TauDEM tree.dat."""
        if tree_df is None or tree_df.empty:
            return set()

        river_id_col = "LINKNO" if "LINKNO" in tree_df.columns else None
        downstream_col = "DSLINKNO" if "DSLINKNO" in tree_df.columns else None
        us1_col = "USLINKNO1" if "USLINKNO1" in tree_df.columns else None
        us2_col = "USLINKNO2" if "USLINKNO2" in tree_df.columns else None

        if river_id_col is None:
            return set()

        upstream_ids = set()
        pending = [outlet_id]

        while pending:
            current = pending.pop()
            if current is None or current in upstream_ids:
                continue
            upstream_ids.add(current)

            row_match = tree_df[tree_df[river_id_col].apply(self._normalize_id) == current]
            if not row_match.empty:
                row = row_match.iloc[0]
                for col in [us1_col, us2_col]:
                    if col is not None and col in row.index:
                        nxt = self._normalize_id(row[col])
                        if nxt is not None and nxt not in upstream_ids and nxt != 0:
                            pending.append(nxt)

            if downstream_col is not None:
                upstream_matches = tree_df[
                    tree_df[downstream_col].apply(self._normalize_id) == current
                ]
                for _, row in upstream_matches.iterrows():
                    nxt = self._normalize_id(row[river_id_col])
                    if nxt is not None and nxt not in upstream_ids:
                        pending.append(nxt)

        return upstream_ids

    def _pick_outlet_polygon(self, basins_gdf: gpd.GeoDataFrame, gauges_gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, int]:
        """Pick the watershed polygon containing the snapped gauge, or nearest if none contains it."""
        gauge_geom = gauges_gdf.geometry.iloc[0]

        containing = basins_gdf[basins_gdf.geometry.contains(gauge_geom)]
        if not containing.empty:
            idx = containing.index[0]
            return basins_gdf.loc[[idx]].copy(), idx

        touching = basins_gdf[basins_gdf.geometry.intersects(gauge_geom)]
        if not touching.empty:
            idx = touching.index[0]
            return basins_gdf.loc[[idx]].copy(), idx

        distances = basins_gdf.geometry.distance(gauge_geom)
        idx = distances.idxmin()
        return basins_gdf.loc[[idx]].copy(), idx

    def _polygonize_valid_streamnet_watershed_mask(self, streamnet_watershed_raster: Path) -> gpd.GeoDataFrame:
        """
        Polygonize all valid non-nodata cells in elv-watersheds.tif.

        This is a fallback for small basins where the streamnet watershed raster
        is spatially meaningful but positive-ID filtering removed everything.
        """
        if not streamnet_watershed_raster.exists():
            raise RuntimeError(f"Streamnet watershed raster not found: {streamnet_watershed_raster}")

        with rasterio.open(streamnet_watershed_raster) as src:
            arr = src.read(1)
            nodata = src.nodata

            valid_mask = np.ones(arr.shape, dtype=bool)
            if nodata is not None:
                valid_mask &= arr != nodata

            if np.issubdtype(arr.dtype, np.floating):
                valid_mask &= ~np.isnan(arr)

            if not np.any(valid_mask):
                raise RuntimeError("No valid non-nodata cells found in streamnet watershed raster")

            geoms = []
            for geom, val in shapes(valid_mask.astype(np.uint8), mask=valid_mask, transform=src.transform):
                if int(val) == 1:
                    geoms.append(shape(geom))

            if not geoms:
                raise RuntimeError("Valid-mask polygonization of streamnet watershed raster produced no polygons")

            gdf = gpd.GeoDataFrame({"mask_id": [1] * len(geoms)}, geometry=geoms, crs=src.crs)

        if len(gdf) > 1:
            gdf["dissolve_key"] = 1
            gdf = gdf.dissolve(by="dissolve_key").reset_index(drop=True)
            gdf = gdf.drop(columns=["dissolve_key"], errors="ignore")

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        return gdf

    def _ensure_nonempty_streamnet_watersheds(
        self,
        interim_watershed_shp: Path,
        streamnet_watershed_raster: Path,
    ) -> Optional[Path]:
        """
        Make sure streamnet watershed polygons exist and are non-empty.
        If run_gdal_processing did not leave a usable basin-watersheds.shp,
        explicitly polygonize elv-watersheds.tif.
        """
        if interim_watershed_shp.exists():
            try:
                gdf = gpd.read_file(interim_watershed_shp)
                if len(gdf) > 0:
                    return interim_watershed_shp
                self.logger.warning(
                    f"Existing streamnet watershed shapefile is empty: {interim_watershed_shp}; "
                    "will re-polygonize elv-watersheds.tif"
                )
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"Could not read existing streamnet watershed shapefile ({e}); "
                    "will re-polygonize elv-watersheds.tif"
                )

        if not streamnet_watershed_raster.exists():
            return None

        try:
            self.gdal.raster_to_polygon(streamnet_watershed_raster, interim_watershed_shp)
            gdf = gpd.read_file(interim_watershed_shp)
            if len(gdf) > 0:
                self.logger.info(
                    f"Re-polygonized non-empty streamnet watershed shapefile from {streamnet_watershed_raster}"
                )
                return interim_watershed_shp
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                f"Explicit polygonization of streamnet watershed raster failed ({e})"
            )

        return None

    def _select_lumped_basin_from_streamnet(
        self,
        interim_watershed_shp: Path,
        interim_river_shp: Path,
        gauges_shp: Path
    ) -> gpd.GeoDataFrame:
        """
        Select the correct full outlet drainage area from streamnet outputs by:
        1) locating the outlet segment at the snapped gauge,
        2) traversing all upstream segments,
        3) selecting matching watershed polygons,
        4) dissolving them to one lumped basin.
        """
        basins_gdf = gpd.read_file(interim_watershed_shp)
        rivers_gdf = gpd.read_file(interim_river_shp)
        gauges_gdf = gpd.read_file(gauges_shp)

        if basins_gdf.crs is None:
            basins_gdf = basins_gdf.set_crs("EPSG:4326")
        if gauges_gdf.crs is None:
            gauges_gdf = gauges_gdf.set_crs("EPSG:4326")
        if rivers_gdf.crs is None:
            rivers_gdf = rivers_gdf.set_crs(gauges_gdf.crs if gauges_gdf.crs is not None else "EPSG:4326")

        if basins_gdf.crs != gauges_gdf.crs:
            basins_gdf = basins_gdf.to_crs(gauges_gdf.crs)
        if rivers_gdf.crs != gauges_gdf.crs:
            rivers_gdf = rivers_gdf.to_crs(gauges_gdf.crs)

        if gauges_gdf.empty:
            raise RuntimeError("Snapped gauge shapefile is empty")
        if rivers_gdf.empty:
            raise RuntimeError("Streamnet river shapefile is empty")
        if basins_gdf.empty:
            raise RuntimeError("Streamnet watershed shapefile is empty")

        river_id_col = 'WSNO' if 'WSNO' in rivers_gdf.columns else ('LINKNO' if 'LINKNO' in rivers_gdf.columns else None)
        if river_id_col is None:
            raise RuntimeError("Could not find river ID column in streamnet river shapefile")

        downstream_col = 'DSLINKNO' if 'DSLINKNO' in rivers_gdf.columns else None
        if downstream_col is None:
            raise RuntimeError("Could not find downstream river ID column in streamnet river shapefile")

        gauge_geom = gauges_gdf.geometry.iloc[0]

        touching = rivers_gdf[rivers_gdf.geometry.intersects(gauge_geom)]
        if touching.empty:
            distances = rivers_gdf.geometry.distance(gauge_geom)
            touching = rivers_gdf.loc[[distances.idxmin()]]

        outlet_row = touching.iloc[0]
        outlet_id = self._normalize_id(outlet_row[river_id_col])

        upstream_ids = self._build_upstream_id_set_from_rivers(rivers_gdf, outlet_id, river_id_col, downstream_col)
        if not upstream_ids:
            raise RuntimeError("Failed to build upstream river ID set for lumped basin")

        candidate_cols = [c for c in ['WSNO', 'LINKNO', 'DN', 'GRIDCODE', 'ID'] if c in basins_gdf.columns]
        if not candidate_cols:
            raise RuntimeError("Could not find a basin ID column in streamnet watershed shapefile")

        best_col = None
        best_overlap = -1
        for col in candidate_cols:
            basin_ids = {self._normalize_id(v) for v in basins_gdf[col].tolist()}
            basin_ids.discard(None)
            overlap = len(basin_ids.intersection(upstream_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col

        if best_col is None or best_overlap <= 0:
            raise RuntimeError("Could not match streamnet watershed polygons to upstream river IDs")

        selected = basins_gdf[
            basins_gdf[best_col].apply(self._normalize_id).isin(upstream_ids)
        ].copy()

        if selected.empty:
            raise RuntimeError("Upstream streamnet watershed subset is empty")

        if len(selected) > 1:
            self.logger.info(f"Dissolving {len(selected)} upstream streamnet polygons into single lumped basin")
            selected['dissolve_key'] = 1
            selected = selected.dissolve(by='dissolve_key').reset_index(drop=True)
            selected = selected.drop(columns=['dissolve_key'], errors='ignore')
            self.logger.info(f"Dissolved to {len(selected)} feature(s)")

        if selected.crs is None:
            selected = selected.set_crs("EPSG:4326")

        return selected

    def _select_lumped_basin_from_tree_only(
        self,
        interim_watershed_shp: Path,
        gauges_shp: Path,
        tree_path: Path
    ) -> gpd.GeoDataFrame:
        """
        Fallback when basin-streams.shp is empty:
        use snapped outlet polygon + basin-tree.dat to select the upstream basin set
        before dissolving.
        """
        basins_gdf = gpd.read_file(interim_watershed_shp)
        gauges_gdf = gpd.read_file(gauges_shp)

        if basins_gdf.crs is None:
            basins_gdf = basins_gdf.set_crs("EPSG:4326")
        if gauges_gdf.crs is None:
            gauges_gdf = gauges_gdf.set_crs("EPSG:4326")
        if basins_gdf.crs != gauges_gdf.crs:
            basins_gdf = basins_gdf.to_crs(gauges_gdf.crs)

        if basins_gdf.empty:
            raise RuntimeError("Streamnet watershed shapefile is empty")
        if gauges_gdf.empty:
            raise RuntimeError("Snapped gauge shapefile is empty")

        outlet_polygon_gdf, _ = self._pick_outlet_polygon(basins_gdf, gauges_gdf)
        outlet_row = outlet_polygon_gdf.iloc[0]

        candidate_cols = [c for c in ['WSNO', 'LINKNO', 'DN', 'GRIDCODE', 'ID'] if c in basins_gdf.columns]
        if not candidate_cols:
            raise RuntimeError("Could not find a basin ID column in streamnet watershed shapefile")

        tree_df = self._read_taudem_tree(tree_path)
        if tree_df is None or tree_df.empty:
            raise RuntimeError("Could not read a usable basin-tree.dat for fallback topology selection")

        tree_ids = set()
        if 'LINKNO' in tree_df.columns:
            tree_ids = {self._normalize_id(v) for v in tree_df['LINKNO'].tolist()}
            tree_ids.discard(None)

        best_col = None
        best_overlap = -1
        for col in candidate_cols:
            basin_ids = {self._normalize_id(v) for v in basins_gdf[col].tolist()}
            basin_ids.discard(None)
            overlap = len(basin_ids.intersection(tree_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col

        if best_col is None:
            raise RuntimeError("Could not determine fallback basin ID column")

        outlet_id = self._normalize_id(outlet_row[best_col])
        upstream_ids = self._build_upstream_id_set_from_tree(tree_df, outlet_id)

        if not upstream_ids:
            raise RuntimeError("Fallback basin-tree topology produced an empty upstream ID set")

        selected = basins_gdf[
            basins_gdf[best_col].apply(self._normalize_id).isin(upstream_ids)
        ].copy()

        if selected.empty:
            raise RuntimeError("Fallback basin-tree topology selected no watershed polygons")

        if len(selected) > 1:
            self.logger.info(f"Dissolving {len(selected)} fallback upstream polygons into single lumped basin")
            selected['dissolve_key'] = 1
            selected = selected.dissolve(by='dissolve_key').reset_index(drop=True)
            selected = selected.drop(columns=['dissolve_key'], errors='ignore')
            self.logger.info(f"Dissolved to {len(selected)} feature(s)")

        if selected.crs is None:
            selected = selected.set_crs("EPSG:4326")

        return selected

    def _warn_if_fallback_area_looks_inconsistent(
        self,
        watershed_gdf: gpd.GeoDataFrame,
        gauges_shp: Path,
        ad8_raster: Path
    ) -> None:
        """
        Compare fallback gagewatershed polygon area against outlet aread8 contributing area.

        Logs a warning when the gagewatershed fallback produces a materially different
        area than what elv-ad8.tif indicates at the snapped outlet.
        """
        try:
            if watershed_gdf is None or watershed_gdf.empty:
                self.logger.warning("Skipping fallback area validation because watershed GeoDataFrame is empty")
                return

            if not gauges_shp.exists():
                self.logger.warning(f"Skipping fallback area validation because gauges shapefile is missing: {gauges_shp}")
                return

            if not ad8_raster.exists():
                self.logger.warning(f"Skipping fallback area validation because aread8 raster is missing: {ad8_raster}")
                return

            gauges_gdf = gpd.read_file(gauges_shp)
            if gauges_gdf.empty:
                self.logger.warning("Skipping fallback area validation because gauges shapefile is empty")
                return

            with rasterio.open(ad8_raster) as src:
                raster_crs = src.crs
                if raster_crs is None:
                    self.logger.warning("Skipping fallback area validation because aread8 raster CRS is undefined")
                    return

                gauges_in_raster_crs = gauges_gdf
                if gauges_in_raster_crs.crs is None:
                    gauges_in_raster_crs = gauges_in_raster_crs.set_crs("EPSG:4326")

                if gauges_in_raster_crs.crs != raster_crs:
                    gauges_in_raster_crs = gauges_in_raster_crs.to_crs(raster_crs)

                outlet_geom = gauges_in_raster_crs.geometry.iloc[0]
                outlet_x = outlet_geom.x
                outlet_y = outlet_geom.y

                sampled_values = list(src.sample([(outlet_x, outlet_y)]))
                if not sampled_values:
                    self.logger.warning("Skipping fallback area validation because aread8 sampling returned no values")
                    return

                outlet_ad8_value = sampled_values[0][0]
                if np.isnan(outlet_ad8_value) or outlet_ad8_value <= 0:
                    self.logger.warning(
                        f"Skipping fallback area validation because outlet aread8 value is invalid: {outlet_ad8_value}"
                    )
                    return

                cell_area = abs(src.transform.a * src.transform.e)
                if cell_area <= 0:
                    self.logger.warning(
                        f"Skipping fallback area validation because computed raster cell area is invalid: {cell_area}"
                    )
                    return

                expected_area_from_ad8 = float(outlet_ad8_value) * float(cell_area)

                watershed_in_raster_crs = watershed_gdf
                if watershed_in_raster_crs.crs is None:
                    watershed_in_raster_crs = watershed_in_raster_crs.set_crs("EPSG:4326")

                if watershed_in_raster_crs.crs != raster_crs:
                    watershed_in_raster_crs = watershed_in_raster_crs.to_crs(raster_crs)

                fallback_polygon_area = float(watershed_in_raster_crs.geometry.area.sum())
                if fallback_polygon_area <= 0:
                    self.logger.warning(
                        f"Skipping fallback area validation because fallback polygon area is invalid: {fallback_polygon_area}"
                    )
                    return

                area_ratio = fallback_polygon_area / expected_area_from_ad8
                relative_diff = abs(fallback_polygon_area - expected_area_from_ad8) / expected_area_from_ad8

                if relative_diff > 0.25:
                    self.logger.warning(
                        "gagewatershed fallback area differs materially from outlet aread8 contributing area: "
                        f"fallback_area={fallback_polygon_area:.6f}, "
                        f"expected_from_ad8={expected_area_from_ad8:.6f}, "
                        f"ratio={area_ratio:.3f}, "
                        f"outlet_ad8_cells={float(outlet_ad8_value):.3f}"
                    )
                else:
                    self.logger.info(
                        "gagewatershed fallback area is broadly consistent with outlet aread8 contributing area: "
                        f"fallback_area={fallback_polygon_area:.6f}, "
                        f"expected_from_ad8={expected_area_from_ad8:.6f}, "
                        f"ratio={area_ratio:.3f}"
                    )

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Fallback area validation failed: {e}")

    def _delineate_with_taudem(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using TauDEM.

        Returns:
            Path to the delineated watershed shapefile, or None if delineation fails
        """
        try:
            if not self.pour_point_path.is_file():
                self.logger.error(f"Pour point file not found: {self.pour_point_path}")
                return None

            dem_input = self._condition_dem()

            self.interim_dir.mkdir(parents=True, exist_ok=True)

            threshold = self._get_config_value(
                lambda: self.config.domain.delineation.stream_threshold,
                default=5000,
                dict_key='STREAM_THRESHOLD'
            )

            max_distance = self._get_config_value(
                lambda: self.config.domain.delineation.move_outlets_max_distance,
                default=200,
                dict_key='MOVE_OUTLETS_MAX_DISTANCE'
            )

            steps = [
                f"{self.taudem_dir}/pitremove -z {dem_input} -fel {self.interim_dir}/elv-fel.tif",
                f"{self.taudem_dir}/d8flowdir -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -sd8 {self.interim_dir}/elv-sd8.tif",
                f"{self.taudem_dir}/aread8 -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -nc",
                f"{self.taudem_dir}/gridnet -p {self.interim_dir}/elv-fdir.tif -plen {self.interim_dir}/elv-plen.tif -tlen {self.interim_dir}/elv-tlen.tif -gord {self.interim_dir}/elv-gord.tif",
                f"{self.taudem_dir}/threshold -ssa {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -thresh {threshold}",
                f"{self.taudem_dir}/moveoutletstostreams -p {self.interim_dir}/elv-fdir.tif -src {self.interim_dir}/elv-src.tif -o {self.pour_point_path} -om {self.interim_dir}/gauges.shp -md {max_distance}",
                f"{self.taudem_dir}/streamnet -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -ord {self.interim_dir}/elv-ord.tif -tree {self.interim_dir}/basin-tree.dat -coord {self.interim_dir}/basin-coord.dat -net {self.interim_dir}/basin-streams.shp -o {self.interim_dir}/gauges.shp -w {self.interim_dir}/elv-watersheds.tif",
                f"{self.taudem_dir}/gagewatershed -p {self.interim_dir}/elv-fdir.tif -o {self.interim_dir}/gauges.shp -gw {self.interim_dir}/watershed.tif -id {self.interim_dir}/watershed_id.txt"
            ]

            for step in steps:
                self.taudem.run_command(step)
                self.logger.debug(f"Completed TauDEM step: {step}")

            method_suffix = self._get_method_suffix()
            watershed_shp_path = (
                self.project_dir / "shapefiles" / "river_basins" /
                f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            )
            watershed_shp_path.parent.mkdir(parents=True, exist_ok=True)

            self.gdal.run_gdal_processing(self.interim_dir)

            interim_watershed_shp = self.interim_dir / "basin-watersheds.shp"
            interim_river_shp = self.interim_dir / "basin-streams.shp"
            gauges_shp = self.interim_dir / "gauges.shp"
            tree_path = self.interim_dir / "basin-tree.dat"
            streamnet_watershed_raster = self.interim_dir / "elv-watersheds.tif"
            gage_watershed_raster = self.interim_dir / "watershed.tif"
            ad8_raster = self.interim_dir / "elv-ad8.tif"

            usable_streamnet_watershed_shp = self._ensure_nonempty_streamnet_watersheds(
                interim_watershed_shp=interim_watershed_shp,
                streamnet_watershed_raster=streamnet_watershed_raster,
            )

            if usable_streamnet_watershed_shp is not None:
                use_streamnet_selection = False
                if interim_river_shp.exists():
                    try:
                        rivers_preview = gpd.read_file(interim_river_shp)
                        use_streamnet_selection = len(rivers_preview) > 0
                    except Exception as e:  # noqa: BLE001
                        self.logger.warning(
                            f"Could not read streamnet river shapefile ({e}); will try basin-tree fallback."
                        )

                if use_streamnet_selection:
                    watershed_gdf = self._select_lumped_basin_from_streamnet(
                        interim_watershed_shp=usable_streamnet_watershed_shp,
                        interim_river_shp=interim_river_shp,
                        gauges_shp=gauges_shp
                    )
                    self.logger.info("Selected lumped basin from streamnet topology before dissolve")
                else:
                    self.logger.warning(
                        "Streamnet river shapefile is empty; trying basin-tree-based upstream selection before any fallback."
                    )
                    watershed_gdf = self._select_lumped_basin_from_tree_only(
                        interim_watershed_shp=usable_streamnet_watershed_shp,
                        gauges_shp=gauges_shp,
                        tree_path=tree_path
                    )
                    self.logger.info("Selected lumped basin from basin-tree topology before dissolve")
            else:
                try:
                    self.logger.warning(
                        "Positive-ID streamnet watershed polygonization is unavailable; trying valid-mask polygonization of elv-watersheds.tif before gagewatershed fallback."
                    )
                    watershed_gdf = self._polygonize_valid_streamnet_watershed_mask(streamnet_watershed_raster)
                    self.logger.info("Recovered lumped basin from valid non-nodata streamnet watershed mask")
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(
                        f"Valid-mask streamnet watershed recovery failed ({e}); falling back to snapped gagewatershed raster polygonization."
                    )
                    self.gdal.raster_to_polygon(gage_watershed_raster, watershed_shp_path)
                    watershed_gdf = gpd.read_file(watershed_shp_path)

                    if watershed_gdf.empty:
                        raise RuntimeError("Fallback gagewatershed polygonization produced an empty lumped basin") from None

                    if watershed_gdf.crs is None:
                        watershed_gdf = watershed_gdf.set_crs("EPSG:4326")
                        self.logger.info("Assigned EPSG:4326 to fallback lumped watershed polygons")

                    if len(watershed_gdf) > 1:
                        self.logger.info(
                            f"Dissolving {len(watershed_gdf)} fallback watershed polygons into single lumped basin"
                        )
                        watershed_gdf["dissolve_key"] = 1
                        watershed_gdf = watershed_gdf.dissolve(by="dissolve_key").reset_index(drop=True)
                        watershed_gdf = watershed_gdf.drop(columns=["dissolve_key"], errors="ignore")
                        self.logger.info(f"Dissolved to {len(watershed_gdf)} feature(s)")

                    self._warn_if_fallback_area_looks_inconsistent(
                        watershed_gdf=watershed_gdf,
                        gauges_shp=gauges_shp,
                        ad8_raster=ad8_raster
                    )

            if watershed_gdf.crs is None:
                watershed_gdf = watershed_gdf.set_crs("EPSG:4326")
                self.logger.info("Assigned EPSG:4326 to lumped watershed polygons after selection")

            if 'GRU_ID' not in watershed_gdf.columns:
                watershed_gdf['GRU_ID'] = 1

            if 'gru_to_seg' not in watershed_gdf.columns:
                watershed_gdf['gru_to_seg'] = 1

            if 'GRU_area' not in watershed_gdf.columns:
                utm_crs = watershed_gdf.estimate_utm_crs()
                watershed_utm = watershed_gdf.to_crs(utm_crs)
                watershed_gdf['GRU_area'] = watershed_utm.geometry.area
                watershed_gdf = watershed_gdf.to_crs('EPSG:4326')

            if watershed_gdf is None:
                raise RuntimeError("Selected lumped watershed GeoDataFrame is None")

            if watershed_gdf.empty:
                raise RuntimeError("Selected lumped watershed GeoDataFrame is empty")

            if watershed_gdf.crs is None:
                watershed_gdf = watershed_gdf.set_crs("EPSG:4326")
                self.logger.info("Assigned EPSG:4326 to lumped watershed polygons before writing")

            self.logger.info(
                f"Writing lumped river basin shapefile with {len(watershed_gdf)} feature(s) to: {watershed_shp_path}"
            )

            watershed_gdf.to_file(watershed_shp_path)

            if not watershed_shp_path.exists():
                raise RuntimeError(
                    f"Lumped watershed shapefile write appeared to succeed but file does not exist: {watershed_shp_path}"
                )

            self.logger.debug(f"Updated watershed shapefile at: {watershed_shp_path}")

            if self._get_config_value(lambda: self.config.domain.delineation.cleanup_intermediate_files, default=True, dict_key='CLEANUP_INTERMEDIATE_FILES'):
                shutil.rmtree(self.interim_dir, ignore_errors=True)
                self.logger.debug(f"Cleaned up intermediate files: {self.interim_dir}")

            return watershed_shp_path

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error during TauDEM watershed delineation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
