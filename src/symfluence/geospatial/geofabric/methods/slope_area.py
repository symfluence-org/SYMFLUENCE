# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Slope-area based stream delineation method.

Uses the relationship between local slope and contributing area to identify
stream initiation points. Based on the geomorphological principle that
streams form where: S = k * A^(-θ).

Extracted from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Any, Dict

from symfluence.core.mixins import ConfigMixin


class SlopeAreaMethod(ConfigMixin):
    """
    Slope-area based stream identification.

    This method uses the relationship between local slope and contributing area
    to identify stream initiation points. More physically-based than simple
    thresholding and can adapt to varying terrain characteristics.
    """

    def __init__(self, taudem_executor: Any, config: Dict, logger: Any, interim_dir: Path):
        """
        Initialize slope-area method.

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
        Run slope-area based stream identification.

        Args:
            dem_path: Path to the DEM file
            pour_point_path: Path to the pour point shapefile
            mpi_prefix: MPI command prefix
        """
        max_distance = self._get_config_value(lambda: self.config.domain.delineation.move_outlets_max_distance, default=200, dict_key='MOVE_OUTLETS_MAX_DISTANCE')

        # Get slope-area parameters from config
        slope_area_threshold = self._get_config_value(lambda: self.config.domain.delineation.slope_area_threshold, default=100.0, dict_key='SLOPE_AREA_THRESHOLD')
        slope_exponent = self._get_config_value(lambda: self.config.domain.delineation.slope_area_exponent, default=2.0, dict_key='SLOPE_AREA_EXPONENT')
        area_exponent = self._get_config_value(lambda: self.config.domain.delineation.area_exponent, default=1.0, dict_key='AREA_EXPONENT')

        steps = [
            # D-infinity flow direction and slope
            f"{mpi_prefix}{self.taudem_dir}/dinfflowdir -fel {self.interim_dir}/elv-fel.tif -ang {self.interim_dir}/elv-ang.tif -slp {self.interim_dir}/elv-slp.tif",

            # D-infinity contributing area
            f"{mpi_prefix}{self.taudem_dir}/areadinf -ang {self.interim_dir}/elv-ang.tif -sca {self.interim_dir}/elv-sca.tif -nc",

            # Calculate slope-area product (S^m * A^n)
            f"{mpi_prefix}{self.taudem_dir}/slopearea -slp {self.interim_dir}/elv-slp.tif -sca {self.interim_dir}/elv-sca.tif -sa {self.interim_dir}/elv-sa.tif -par {slope_exponent} {area_exponent}",

            # Threshold the slope-area grid
            f"{mpi_prefix}{self.taudem_dir}/threshold -ssa {self.interim_dir}/elv-sa.tif -src {self.interim_dir}/elv-src.tif -thresh {slope_area_threshold}",

            # D8 contributing area weighted by slope-area sources
            f"{mpi_prefix}{self.taudem_dir}/aread8 -p {self.interim_dir}/elv-fdir.tif "
            f"-wg {self.interim_dir}/elv-src.tif "
            f"-ad8 {self.interim_dir}/elv-ad8_sa.tif -nc",

            # Move outlets to stream network
            f"{mpi_prefix}{self.taudem_dir}/moveoutletstostreams -p {self.interim_dir}/elv-fdir.tif -src {self.interim_dir}/elv-src.tif -o {pour_point_path} -om {self.interim_dir}/gauges.shp -md {max_distance}",

            # Generate final stream network and watersheds
            f"{mpi_prefix}{self.taudem_dir}/streamnet -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -ord {self.interim_dir}/elv-ord.tif -tree {self.interim_dir}/basin-tree.dat -coord {self.interim_dir}/basin-coord.dat -net {self.interim_dir}/basin-streams.shp -o {self.interim_dir}/gauges.shp -w {self.interim_dir}/elv-watersheds.tif"
        ]

        for step in steps:
            self.taudem.run_command(step)
            self.logger.info("Completed slope-area method step")

        self.logger.info("Slope-area based stream identification completed")
        self.logger.info(f"Used slope^{slope_exponent} * area^{area_exponent} >= {slope_area_threshold} criterion")
