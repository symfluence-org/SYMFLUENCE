# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODFLOW 6 Preprocessor

Generates MODFLOW 6 text input files. ``MODFLOW_GRID_MODE: lumped`` (default)
writes a single-cell groundwater model with the catchment's area;
``basin`` writes a real-basin grid driven by land-model recharge zones
(see ``basin_model.py`` and ``land_adapters.py``).

Lumped-mode input files:

Input files generated:
    mfsim.nam   — simulation name file
    gwf.nam     — GWF model name file
    gwf.tdis    — temporal discretization
    gwf.dis     — spatial discretization (1×1×1)
    gwf.ic      — initial conditions
    gwf.npf     — node property flow (hydraulic conductivity)
    gwf.sto     — storage (Sy, Ss)
    gwf.rch     — recharge package
    gwf.drn     — drain package (baseflow outlet)
    gwf.oc      — output control
    gwf.ims     — iterative model solution
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from symfluence.core.exceptions import ConfigurationError, FileOperationError
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("MODFLOW")
class MODFLOWPreProcessor(BaseModelPreProcessor):
    """Generates MODFLOW 6 input files for lumped groundwater simulation."""

    MODEL_NAME = "MODFLOW"

    # Keeps its own defensive __init__ (tolerates dict / MagicMock coupled-test
    # configs) rather than calling super().__init__, whose strict typed-config
    # access doesn't fit how the coupled groundwater models are wired.
    def __init__(self, config, logger, **kwargs):
        self.config = config
        self.logger = logger

        try:
            self.domain_name = config.domain.name or 'unknown'
        except (AttributeError, TypeError):
            self.domain_name = 'unknown'
        try:
            self.experiment_id = config.domain.experiment_id or 'default'
        except (AttributeError, TypeError):
            self.experiment_id = 'default'

        # PROJECT_DIR is a derived path — check dict sources first, then compute
        _dict = config.to_dict(flatten=True) if hasattr(config, 'to_dict') else (config if isinstance(config, dict) else {})
        project_dir = (_dict or {}).get('PROJECT_DIR')
        if not project_dir or project_dir == '.':
            try:
                data_dir = str(config.system.data_dir)
            except (AttributeError, TypeError):
                data_dir = '.'
            project_dir = str(Path(data_dir) / f"domain_{self.domain_name}")
        self.project_dir = Path(project_dir)

    def _get_modflow_cfg(self, key, default=None):
        """Get MODFLOW-specific config value from typed config."""
        try:
            mf_cfg = self.config.model.modflow
            if mf_cfg:
                attr = key.lower()
                if hasattr(mf_cfg, attr):
                    val = getattr(mf_cfg, attr)
                    if val is not None:
                        return val
        except (AttributeError, TypeError):
            pass
        return default

    def _get_catchment_area_m2(self) -> float:
        """Get catchment area in m2.

        Resolution order:
        1. Explicit CATCHMENT_AREA config key (km2)
        2. config.domain.catchment_area
        3. Auto-detect from watershed shapefile (UTM projection)
        4. Fallback default (27 km2)
        """
        area_km2 = getattr(self.config.domain, 'catchment_area', None)
        if area_km2 is None:
            # Try to compute from watershed shapefile
            area_km2 = self._detect_catchment_area_km2()
        if area_km2 is None:
            self.logger.warning("CATCHMENT_AREA not set and no shapefile found, using default 27 km2")
            area_km2 = 27.0
        return float(area_km2) * 1e6

    def _detect_catchment_area_km2(self):
        """Detect catchment area from the watershed shapefile."""
        try:
            import geopandas as gpd

            domain_name = self.domain_name
            basin_path = (
                self.project_dir / 'shapefiles' / 'river_basins'
                / f"{domain_name}_riverBasins_lumped.shp"
            )
            if not basin_path.exists():
                return None

            gdf = gpd.read_file(str(basin_path))
            # Project to a metric CRS for accurate area calculation
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            area_km2 = gdf_proj.geometry.area.sum() / 1e6
            self.logger.info(f"Auto-detected catchment area from shapefile: {area_km2:.1f} km2")
            return area_km2
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(f"Could not auto-detect catchment area: {e}", exc_info=True)
            return None

    def _get_time_info(self):
        """Get simulation time range and number of stress periods."""
        import pandas as pd

        start = self.config.domain.time_start or '2000-01-01'
        end = self.config.domain.time_end or '2001-01-01'

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        n_days = (end_dt - start_dt).days
        if n_days <= 0:
            n_days = 365

        sp_length = float(self._get_modflow_cfg('STRESS_PERIOD_LENGTH', 1.0))
        n_periods = max(1, int(math.ceil(n_days / sp_length)))

        return n_periods, sp_length, n_days

    def run_preprocessing(self):
        """Generate all MODFLOW 6 input files."""
        settings_dir = self.project_dir / "settings" / "MODFLOW"
        settings_dir.mkdir(parents=True, exist_ok=True)

        if self._get_modflow_cfg('GRID_MODE', 'lumped') == 'basin':
            self._run_basin_preprocessing(settings_dir)
            return

        self.logger.info(f"Generating MODFLOW 6 input files in {settings_dir}")

        # Get parameters
        area_m2 = self._get_catchment_area_m2()
        cell_size = float(self._get_modflow_cfg('CELL_SIZE', math.sqrt(area_m2)))
        nlay = int(self._get_modflow_cfg('NLAY', 1))
        nrow = int(self._get_modflow_cfg('NROW', 1))
        ncol = int(self._get_modflow_cfg('NCOL', 1))
        k = float(self._get_modflow_cfg('K', 5.0))
        sy = float(self._get_modflow_cfg('SY', 0.05))
        ss = float(self._get_modflow_cfg('SS', 1e-5))
        top = float(self._get_modflow_cfg('TOP', 1500.0))
        bot = float(self._get_modflow_cfg('BOT', 1400.0))
        strt = self._get_modflow_cfg('STRT')
        if strt is None:
            strt = (top + bot) / 2.0
        else:
            strt = float(strt)
        drain_elev = self._get_modflow_cfg('DRAIN_ELEVATION')
        if drain_elev is None:
            drain_elev = (top + bot) / 2.0
        else:
            drain_elev = float(drain_elev)
        # Scale drain conductance with catchment area to target a ~100-day
        # aquifer time constant (τ = SY × A / C) with the default SY.
        drain_cond_default = max(50.0, area_m2 * 0.0005)
        drain_cond = float(self._get_modflow_cfg('DRAIN_CONDUCTANCE', drain_cond_default))
        nstp = int(self._get_modflow_cfg('NSTP', 1))

        n_periods, sp_length, n_days = self._get_time_info()

        self.logger.info(
            f"MODFLOW grid: {nlay}×{nrow}×{ncol}, cell_size={cell_size:.0f}m, "
            f"K={k}, Sy={sy}, top={top}, bot={bot}"
        )
        self.logger.info(f"Stress periods: {n_periods} × {sp_length} days")

        # Write all files
        self._write_mfsim_nam(settings_dir, n_periods, sp_length, nstp)
        self._write_gwf_nam(settings_dir)
        self._write_tdis(settings_dir, n_periods, sp_length, nstp)
        self._write_dis(settings_dir, nlay, nrow, ncol, cell_size, top, bot)
        self._write_ic(settings_dir, strt)
        self._write_npf(settings_dir, k)
        self._write_sto(settings_dir, sy, ss)
        self._write_rch(settings_dir)
        self._write_drn(settings_dir, drain_elev, drain_cond)
        self._write_oc(settings_dir)
        self._write_ims(settings_dir)

        self.logger.info(f"MODFLOW 6 input files generated in {settings_dir}")

    # ------------------------------------------------------------------
    # Real-basin grid (MODFLOW_GRID_MODE: basin)
    # ------------------------------------------------------------------
    def _resolve_input(self, configured, subdir: str, pattern: str, what: str) -> Path:
        """Configured path, or the single match of ``pattern`` under ``subdir``."""
        if configured:
            p = Path(configured)
            if not p.exists():
                raise FileOperationError(f'MODFLOW {what} not found: {p}')
            return p
        hits = sorted((self.project_dir / subdir).glob(pattern))
        if len(hits) != 1:
            raise ConfigurationError(f'MODFLOW {what}: expected one {pattern} under {subdir}, found {len(hits)}; '
                                     'set it explicitly')
        return hits[0]

    def _run_basin_preprocessing(self, settings_dir: Path) -> None:
        """Grid from subbasins/network/DEM, recharge zones from the land model."""
        import geopandas as gpd

        from symfluence.models.gsflow.basin_geometry import build_basin_geometry

        from .basin_model import grid_from_basin_geometry, params_from_config, write_simulation, zone_cell_weights
        from .land_adapters import get_land_adapter

        cfg = self.config.model.modflow
        sub = gpd.read_file(self._resolve_input(cfg.subbasin_shapefile, 'shapefiles/river_basins', '*.shp',
                                                'subbasins'))
        net = gpd.read_file(self._resolve_input(cfg.river_network_shapefile, 'shapefiles/river_network', '*.shp',
                                                'river network'))
        dem = self._resolve_input(cfg.dem_path, 'data/attributes/elevation/dem', '*_elv.tif', 'DEM')
        crs = cfg.grid_crs
        if crs is None:
            c = sub.to_crs(4326).geometry.union_all().centroid
            crs = (32600 if c.y >= 0 else 32700) + int((c.x + 180) / 6) + 1
        cell = float(cfg.cell_size or 2000.0)
        geo = build_basin_geometry(sub, net, dem, crs, cell_size=cell, channel_depth=cfg.channel_depth)
        grid = grid_from_basin_geometry(geo)
        if cfg.lake_shapefile:
            if cfg.lake_stage is None:
                raise ConfigurationError('MODFLOW_LAKE_SHAPEFILE needs MODFLOW_LAKE_STAGE')
            grid.set_lake(gpd.read_file(cfg.lake_shapefile), cfg.lake_stage)
        grid.reduced_zones = [int(z) for z in (cfg.reduced_recharge_zones or [])]

        land = str(cfg.coupling_source or '').upper()
        if land in ('', 'NONE', 'DEFAULT'):
            raise ConfigurationError('MODFLOW basin mode needs MODFLOW_COUPLING_SOURCE (the land model)')
        adapter = get_land_adapter(land, {'zone_shapefile': cfg.land_zone_shapefile,
                                          'zone_id_field': cfg.land_zone_id_field,
                                          'outlet_id': cfg.land_outlet_id,
                                          'basin_shapefile': self._resolve_input(cfg.subbasin_shapefile,
                                                                                 'shapefiles/river_basins',
                                                                                 '*.shp', 'subbasins')})
        zones = adapter.recharge_zones(self.project_dir / 'settings' / land)
        weights = zone_cell_weights(grid, zones, dict(zip(zones.zone_id, zones.area_m2)))
        _, _, n_days = self._get_time_info()
        write_simulation(settings_dir, grid, weights, n_days, params_from_config(cfg), save_heads=cfg.save_heads)
        self.logger.info(
            f"MODFLOW basin grid: {grid.nrow}x{grid.ncol}, {int(grid.active.sum())} active {cell:g} m cells, "
            f"{len(grid.river)} river cells; {land} recharge zones {sorted(zones.zone_id)} "
            f"({len(weights)} zone-cell links) in {settings_dir}")
        for note in grid.notes:
            self.logger.debug(f"MODFLOW geometry: {note}")

    def _write_mfsim_nam(self, d: Path, n_periods, sp_length, nstp):
        """Write simulation name file."""
        (d / "mfsim.nam").write_text(
            "BEGIN OPTIONS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN TIMING\n"
            "  TDIS6 gwf.tdis\n"
            "END TIMING\n"
            "\n"
            "BEGIN MODELS\n"
            "  GWF6 gwf.nam gwf\n"
            "END MODELS\n"
            "\n"
            "BEGIN EXCHANGES\n"
            "END EXCHANGES\n"
            "\n"
            "BEGIN SOLUTIONGROUP 1\n"
            "  IMS6 gwf.ims gwf\n"
            "END SOLUTIONGROUP 1\n"
        )

    def _write_gwf_nam(self, d: Path):
        """Write GWF model name file."""
        (d / "gwf.nam").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN PACKAGES\n"
            "  DIS6 gwf.dis\n"
            "  IC6 gwf.ic\n"
            "  NPF6 gwf.npf\n"
            "  STO6 gwf.sto\n"
            "  RCH6 gwf.rch\n"
            "  DRN6 gwf.drn\n"
            "  OC6 gwf.oc\n"
            "END PACKAGES\n"
        )

    def _write_tdis(self, d: Path, n_periods, sp_length, nstp):
        """Write temporal discretization file."""
        lines = [
            "BEGIN OPTIONS\n",
            "  TIME_UNITS DAYS\n",
            "END OPTIONS\n",
            "\n",
            "BEGIN DIMENSIONS\n",
            f"  NPER {n_periods}\n",
            "END DIMENSIONS\n",
            "\n",
            "BEGIN PERIODDATA\n",
        ]
        for _ in range(n_periods):
            lines.append(f"  {sp_length} {nstp} 1.0\n")
        lines.append("END PERIODDATA\n")
        (d / "gwf.tdis").write_text("".join(lines))

    def _write_dis(self, d: Path, nlay, nrow, ncol, cell_size, top, bot):
        """Write spatial discretization file."""
        delr = cell_size  # row width
        delc = cell_size  # column width
        (d / "gwf.dis").write_text(
            "BEGIN OPTIONS\n"
            "  LENGTH_UNITS METERS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN DIMENSIONS\n"
            f"  NLAY {nlay}\n"
            f"  NROW {nrow}\n"
            f"  NCOL {ncol}\n"
            "END DIMENSIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  DELR\n"
            f"    CONSTANT {delr}\n"
            "  DELC\n"
            f"    CONSTANT {delc}\n"
            "  TOP\n"
            f"    CONSTANT {top}\n"
            "  BOTM\n"
            f"    CONSTANT {bot}\n"
            "END GRIDDATA\n"
        )

    def _write_ic(self, d: Path, strt):
        """Write initial conditions file."""
        (d / "gwf.ic").write_text(
            "BEGIN OPTIONS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  STRT\n"
            f"    CONSTANT {strt}\n"
            "END GRIDDATA\n"
        )

    def _write_npf(self, d: Path, k):
        """Write node property flow (hydraulic conductivity) file."""
        (d / "gwf.npf").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_SPECIFIC_DISCHARGE\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  ICELLTYPE\n"
            "    CONSTANT 1\n"
            "  K\n"
            f"    CONSTANT {k}\n"
            "END GRIDDATA\n"
        )

    def _write_sto(self, d: Path, sy, ss):
        """Write storage file."""
        (d / "gwf.sto").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  ICONVERT\n"
            "    CONSTANT 1\n"
            "  SS\n"
            f"    CONSTANT {ss}\n"
            "  SY\n"
            f"    CONSTANT {sy}\n"
            "END GRIDDATA\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  TRANSIENT\n"
            "END PERIOD 1\n"
        )

    def _write_rch(self, d: Path):
        """Write recharge package file.

        Uses time-series format: a separate recharge.ts file will be
        written by the coupler at runtime with actual recharge data.
        For standalone runs, a constant recharge is applied.
        """
        (d / "gwf.rch").write_text(
            "BEGIN OPTIONS\n"
            "  READASARRAYS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  RECHARGE\n"
            "    CONSTANT 0.001\n"
            "END PERIOD 1\n"
        )
        # Write a placeholder recharge time-array-series file.
        # The coupler overwrites gwf.rch with TAS6 reference for coupled runs.
        (d / "recharge.ts").write_text(
            "BEGIN ATTRIBUTES\n"
            "  NAME rch_array\n"
            "  METHOD LINEAR\n"
            "END ATTRIBUTES\n"
            "\n"
            "BEGIN TIME 0.0\n"
            "  CONSTANT 0.001\n"
            "END TIME 0.0\n"
        )

    def _write_drn(self, d: Path, drain_elev, drain_cond):
        """Write drain package file."""
        (d / "gwf.drn").write_text(
            "BEGIN OPTIONS\n"
            "  PRINT_INPUT\n"
            "  PRINT_FLOWS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN DIMENSIONS\n"
            "  MAXBOUND 1\n"
            "END DIMENSIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            f"  1 1 1 {drain_elev} {drain_cond}\n"
            "END PERIOD 1\n"
        )

    def _write_oc(self, d: Path):
        """Write output control file."""
        (d / "gwf.oc").write_text(
            "BEGIN OPTIONS\n"
            "  HEAD FILEOUT gwf.hds\n"
            "  BUDGET FILEOUT gwf.bud\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  SAVE HEAD ALL\n"
            "  SAVE BUDGET ALL\n"
            "END PERIOD 1\n"
        )

    def _write_ims(self, d: Path):
        """Write iterative model solution file."""
        (d / "gwf.ims").write_text(
            "BEGIN OPTIONS\n"
            "  PRINT_OPTION SUMMARY\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN NONLINEAR\n"
            "  OUTER_DVCLOSE 1.0e-6\n"
            "  OUTER_MAXIMUM 100\n"
            "END NONLINEAR\n"
            "\n"
            "BEGIN LINEAR\n"
            "  INNER_DVCLOSE 1.0e-9\n"
            "  INNER_MAXIMUM 300\n"
            "  INNER_RCLOSE 1.0e-3 STRICT\n"
            "END LINEAR\n"
        )
