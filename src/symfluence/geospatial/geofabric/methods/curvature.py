# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Curvature-based (Peuker-Douglas) stream delineation method.

Uses terrain curvature to identify stream initiation points.
Based on Peuker and Douglas (1975) algorithm.

Extracted from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.mixins import ConfigMixin


class CurvatureMethod(ConfigMixin):
    """
    Curvature-based stream identification using Peuker-Douglas algorithm.

    This uses Peuker–Douglas to find upwardly curved cells, then
    accumulates only those cells and thresholds the result to define streams.
    More physically-based than simple thresholding.
    """

    def __init__(self, taudem_executor: Any, config: Dict, logger: Any, interim_dir: Path):
        """
        Initialize curvature method.

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
        Run curvature-based stream identification.

        Args:
            dem_path: Path to the DEM file
            pour_point_path: Path to the pour point shapefile
            mpi_prefix: MPI command prefix
        """
        max_distance = self._get_config_value(
            lambda: self.config.domain.delineation.move_outlets_max_distance,
            dict_key='MOVE_OUTLETS_MAX_DISTANCE',
        )
        min_source_threshold = self._get_config_value(
            lambda: self.config.domain.delineation.min_source_threshold,
            dict_key='MIN_SOURCE_THRESHOLD',
        )

        steps = [
            # D-Infinity slope
            f"{mpi_prefix}{self.taudem_dir}/dinfflowdir "
            f"-fel {self.interim_dir}/elv-fel.tif "
            f"-ang {self.interim_dir}/elv-ang.tif "
            f"-slp {self.interim_dir}/elv-slp.tif",

            # Peuker–Douglas: curvature-based proto-stream skeleton
            f"{mpi_prefix}{self.taudem_dir}/peukerdouglas "
            f"-fel {self.interim_dir}/elv-fel.tif "
            f"-ss {self.interim_dir}/elv-ss_pd.tif",

            # D8 contributing area of ONLY the PD skeleton cells
            f"{mpi_prefix}{self.taudem_dir}/aread8 "
            f"-p {self.interim_dir}/elv-fdir.tif "
            f"-wg {self.interim_dir}/elv-ss_pd.tif "
            f"-ad8 {self.interim_dir}/elv-ad8_pd.tif -nc",

            # Threshold PD-weighted contributing area
            f"{mpi_prefix}{self.taudem_dir}/threshold "
            f"-ssa {self.interim_dir}/elv-ad8_pd.tif "
            f"-src {self.interim_dir}/elv-src.tif "
            f"-thresh {min_source_threshold}",

            # Snap pour points to streams
            f"{mpi_prefix}{self.taudem_dir}/moveoutletstostreams "
            f"-p {self.interim_dir}/elv-fdir.tif "
            f"-ad8 {self.interim_dir}/elv-ad8.tif "
            f"-src {self.interim_dir}/elv-src.tif "
            f"-o {pour_point_path} "
            f"-om {self.interim_dir}/gauges.shp "
            f"-md {max_distance}",

            # Stream network + watersheds
            f"{mpi_prefix}{self.taudem_dir}/streamnet "
            f"-fel {self.interim_dir}/elv-fel.tif "
            f"-p {self.interim_dir}/elv-fdir.tif "
            f"-ad8 {self.interim_dir}/elv-ad8.tif "
            f"-src {self.interim_dir}/elv-src.tif "
            f"-ord {self.interim_dir}/elv-order.tif "
            f"-tree {self.interim_dir}/elv-tree.dat "
            f"-coord {self.interim_dir}/elv-coord.dat "
            f"-net {self.interim_dir}/elv-net.shp "
            f"-w {self.interim_dir}/elv-watersheds.tif "
            f"-o {self.interim_dir}/gauges.shp",
        ]

        for step in steps:
            self.taudem.run_command(step)
            self.logger.info("Completed curvature method step")

        self.logger.info("Curvature-based stream identification completed")
