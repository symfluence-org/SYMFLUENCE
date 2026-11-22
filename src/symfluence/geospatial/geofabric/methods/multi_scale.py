# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Multi-scale hierarchical stream delineation method.

Delineates streams at multiple threshold scales and combines them to create
a hierarchical stream network capturing both main stems and headwaters.

Extracted from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from symfluence.core.mixins import ConfigMixin


class MultiScaleMethod(ConfigMixin):
    """
    Multi-scale hierarchical stream identification.

    Delineates streams at multiple threshold scales and combines them.
    Particularly useful for large domains with varying terrain characteristics.
    """

    def __init__(self, taudem_executor: Any, config: Dict, logger: Any, interim_dir: Path):
        """
        Initialize multi-scale method.

        Args:
            taudem_executor: TauDEMExecutor instance for running commands
            config: Configuration dictionary
            logger: Logger instance
            interim_dir: Directory for interim TauDEM files
        """
        self.taudem = taudem_executor
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.interim_dir = interim_dir
        self.taudem_dir = taudem_executor.taudem_dir

    def run(self, dem_path: Path, pour_point_path: Path, mpi_prefix: str) -> None:
        """
        Run multi-scale hierarchical stream identification.

        Args:
            dem_path: Path to the DEM file
            pour_point_path: Path to the pour point shapefile
            mpi_prefix: MPI command prefix
        """
        max_distance = self._get_config_value(
            lambda: self.config.domain.delineation.move_outlets_max_distance,
            dict_key='MOVE_OUTLETS_MAX_DISTANCE',
        )

        # Get multi-scale parameters
        thresholds = self._get_config_value(lambda: self.config.domain.delineation.multi_scale_thresholds, default=[10000, 5000, 2500, 1000, 500], dict_key='MULTI_SCALE_THRESHOLDS')
        if isinstance(thresholds, str):
            thresholds = [int(x.strip()) for x in thresholds.split(',')]

        self.logger.info(f"Delineating streams at {len(thresholds)} scales: {thresholds}")

        # Common preprocessing
        steps = [
            f"{mpi_prefix}{self.taudem_dir}/gridnet -p {self.interim_dir}/elv-fdir.tif -plen {self.interim_dir}/elv-plen.tif -tlen {self.interim_dir}/elv-tlen.tif -gord {self.interim_dir}/elv-gord.tif",
        ]

        for step in steps:
            self.taudem.run_command(step)

        # Delineate at each threshold scale
        src_files = []
        for i, threshold in enumerate(thresholds):
            self.logger.info(f"Processing scale {i+1}/{len(thresholds)} with threshold {threshold}")

            src_file = self.interim_dir / f"elv-src-scale{i}.tif"
            src_files.append(src_file)

            # Create stream source at this threshold
            cmd = f"{mpi_prefix}{self.taudem_dir}/threshold -ssa {self.interim_dir}/elv-ad8.tif -src {src_file} -thresh {threshold}"
            self.taudem.run_command(cmd)

        # Combine all scales
        self.logger.info("Combining multi-scale stream sources")
        self._combine_multi_scale_sources(src_files)

        # Continue with standard streamnet workflow
        steps = [
            f"{mpi_prefix}{self.taudem_dir}/moveoutletstostreams -p {self.interim_dir}/elv-fdir.tif -src {self.interim_dir}/elv-src.tif -o {pour_point_path} -om {self.interim_dir}/gauges.shp -md {max_distance}",
            f"{mpi_prefix}{self.taudem_dir}/streamnet -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -ord {self.interim_dir}/elv-ord.tif -tree {self.interim_dir}/basin-tree.dat -coord {self.interim_dir}/basin-coord.dat -net {self.interim_dir}/basin-streams.shp -o {self.interim_dir}/gauges.shp -w {self.interim_dir}/elv-watersheds.tif"
        ]

        for step in steps:
            self.taudem.run_command(step)
            self.logger.info("Completed multi-scale method step")

        self.logger.info("Multi-scale hierarchical stream identification completed")
        self.logger.info(f"Combined {len(thresholds)} scales to create variable-density stream network")

    def _combine_multi_scale_sources(self, src_files: List[Path]):
        """
        Combine multiple stream source grids from different threshold scales.

        Args:
            src_files: List of paths to stream source raster files
        """
        try:
            import rasterio

            # Read all source files
            src_arrays = []
            profile = None

            for src_file in src_files:
                with rasterio.open(src_file) as src:
                    if profile is None:
                        profile = src.profile
                    data = src.read(1)
                    src_arrays.append(data)

            # Combine: any cell marked as stream in any scale becomes a stream
            combined = np.zeros_like(src_arrays[0])
            for arr in src_arrays:
                combined = np.logical_or(combined, arr > 0)

            # Convert back to integer
            if profile is not None:
                combined = combined.astype(profile['dtype'])
            else:
                combined = combined.astype(np.int32)

            # Write combined source grid
            output_path = self.interim_dir / "elv-src.tif"
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(combined, 1)

            self.logger.info(f"Combined {len(src_files)} scales into unified stream source grid")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error combining multi-scale sources: {str(e)}", exc_info=True)
            # Fallback: use the finest scale (last threshold)
            self.logger.warning("Using finest scale as fallback")
            import shutil
            shutil.copy(src_files[-1], self.interim_dir / "elv-src.tif")
