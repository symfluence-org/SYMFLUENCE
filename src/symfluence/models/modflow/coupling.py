# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA → MODFLOW Coupling

Extracts recharge from SUMMA output (scalarSoilDrainage), converts
to MODFLOW 6 recharge time-series format, and combines MODFLOW
baseflow with SUMMA surface runoff for downstream routing.

Unit conversions:
    SUMMA scalarSoilDrainage: m/s → m/d (× 86400)
    SUMMA scalarSurfaceRunoff: m/s → m3/s (× area_m2)
    MODFLOW drain discharge: m3/d → m3/s (÷ 86400)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_tdis(settings_dir: Path) -> tuple:
    """``(nper, perlen of the first period)`` from ``gwf.tdis``; ``(0, None)`` if absent."""
    tdis = Path(settings_dir) / "gwf.tdis"
    if not tdis.exists():
        return 0, None
    nper, perlen, in_pd = 0, None, False
    for line in tdis.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "NPER" and len(parts) > 1:
            nper = int(parts[1])
        elif key == "BEGIN" and len(parts) > 1 and parts[1].upper() == "PERIODDATA":
            in_pd = True
        elif key == "END":
            in_pd = False
        elif in_pd and perlen is None:
            perlen = float(parts[0])
    return nper, perlen


class SUMMAToMODFLOWCoupler:
    """Couples SUMMA land surface output to MODFLOW 6 groundwater model.

    Workflow:
        1. Extract scalarSoilDrainage from SUMMA output → recharge
        2. Write MODFLOW 6 time-series recharge file
        3. After MODFLOW runs, extract drain discharge (baseflow)
        4. Combine surface runoff + baseflow → total streamflow
    """

    # SUMMA m/s → m/d: (1 m/s) × (86400 s/d) = 86400 m/d
    M_S_TO_M_D = 86400.0

    def __init__(self, config_dict: dict, logger_instance=None):
        self.config_dict = config_dict
        self.logger = logger_instance or logger

    def _open_summa_variable(self, summa_output_dir: Path, variable: str):
        """Open the SUMMA NetCDF file containing the requested variable.

        Opens each file individually to avoid NaN-padding issues that
        occur when open_mfdataset merges files with different time
        dimensions (e.g. hourly timestep file + daily output file).

        Returns:
            (xr.Dataset, list[Path]) — opened dataset and files used
        """
        import xarray as xr

        summa_output_dir = Path(summa_output_dir)
        nc_files = sorted(summa_output_dir.glob("*_output_*.nc"))
        if not nc_files:
            nc_files = sorted(summa_output_dir.glob("*.nc"))

        if not nc_files:
            raise FileNotFoundError(
                f"No SUMMA output NetCDF files found in {summa_output_dir}"
            )

        # Find files that contain the variable and open only those
        matching_files = []
        for nc_file in nc_files:
            ds_check = xr.open_dataset(nc_file)
            has_var = variable in ds_check
            if has_var:
                time_dim = ds_check.sizes.get('time', -1)
                self.logger.debug(
                    f"  {nc_file.name}: has '{variable}', time={time_dim}"
                )
            ds_check.close()
            if has_var:
                matching_files.append(nc_file)

        if not matching_files:
            # Gather all available variables for error message
            all_vars: set[str] = set()
            for nc_file in nc_files:
                ds_check = xr.open_dataset(nc_file)
                all_vars.update(str(var_name) for var_name in ds_check.data_vars)
                ds_check.close()
            raise ValueError(
                f"Variable '{variable}' not found in SUMMA output. "
                f"Available: {sorted(all_vars)}"
            )

        if len(matching_files) == 1:
            ds = xr.open_dataset(matching_files[0])
        else:
            ds = xr.open_mfdataset(matching_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')

        self.logger.debug(
            f"Reading {variable} from {len(matching_files)} file(s)"
        )
        return ds, matching_files

    def extract_recharge_from_summa(
        self,
        summa_output_dir: Path,
        variable: str = 'scalarSoilDrainage',
    ) -> pd.Series:
        """Extract recharge time series from SUMMA NetCDF output.

        Args:
            summa_output_dir: Path to SUMMA output directory
            variable: SUMMA variable name for soil drainage

        Returns:
            Recharge time series in m/d with datetime index
        """
        ds, _ = self._open_summa_variable(summa_output_dir, variable)

        data = ds[variable]

        # Squeeze non-time dimensions (single HRU)
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values

        ds.close()

        # Convert m/s → m/d
        recharge_m_d = values * self.M_S_TO_M_D

        series = pd.Series(recharge_m_d, index=times, name='recharge_m_d')

        # Resample to daily if sub-daily (preserves NaN correctly)
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()
                self.logger.debug("Resampled sub-daily SUMMA output to daily")

        # Replace NaN and ensure non-negative after resampling
        series = series.fillna(0.0).clip(lower=0.0)

        self.logger.debug(
            f"Extracted recharge: {len(series)} timesteps, "
            f"mean={series.mean():.6f} m/d, total={series.sum():.3f} m"
        )

        return series

    @staticmethod
    def aggregate_to_stress_periods(
        daily: np.ndarray,
        stress_period_length: float = 1.0,
        n_stress_periods: int = 0,
    ) -> np.ndarray:
        """Mean of a daily series over each stress period.

        Day ``i`` covers model time ``[i, i + 1)``; period ``k`` covers
        ``[k * L, (k + 1) * L)`` and gets the time-weighted mean of the days it
        overlaps (fractional lengths split days). A period only partly covered
        by the series averages the covered part; periods wholly past its end
        repeat the last period's value (as MODFLOW does for omitted blocks).
        With ``L == 1`` the daily values are returned unchanged.
        """
        vals = np.nan_to_num(np.asarray(daily, dtype=float).ravel(), nan=0.0)
        sp = float(stress_period_length)
        if sp <= 0:
            raise ValueError(f"stress_period_length must be > 0, got {sp}")
        if vals.size == 0:
            return vals
        n = int(n_stress_periods) if n_stress_periods and n_stress_periods > 0 \
            else int(np.ceil(vals.size / sp - 1e-9))
        if sp == 1.0:
            out = vals[:n].copy()
        else:
            days = float(vals.size)
            cum = np.concatenate([[0.0], np.cumsum(vals)])     # integral at whole days
            edges = np.minimum(np.arange(n + 1) * sp, days)
            integral = np.diff(np.interp(edges, np.arange(cum.size, dtype=float), cum))
            covered = np.diff(edges)
            out = np.divide(integral, covered, out=np.full(n, np.nan), where=covered > 1e-12)
            out = out[~np.isnan(out)]
        if out.size < n:
            out = np.concatenate([out, np.full(n - out.size, out[-1])])
        return out

    def write_modflow_recharge_rch(
        self,
        recharge_series: pd.Series,
        rch_path: Path,
        n_stress_periods: int = 0,
        stress_period_length: float = 1.0,
    ) -> Path:
        """Write MODFLOW 6 RCH package with per-period recharge.

        Uses READASARRAYS CONSTANT format — one PERIOD block per stress
        period. The daily recharge is averaged over each stress period of
        ``stress_period_length`` days (see :meth:`aggregate_to_stress_periods`).

        Args:
            recharge_series: Daily recharge in m/d, starting on the first model day
            rch_path: Path to write the gwf.rch file
            n_stress_periods: Total stress periods in TDIS (0 = as many as the
                series covers). Periods beyond the series reuse its last day.
            stress_period_length: Stress-period length in days (TDIS PERLEN)

        Returns:
            Path to written file
        """
        rch_path = Path(rch_path)

        lines = [
            "BEGIN OPTIONS",
            "  READASARRAYS",
            "END OPTIONS",
            "",
        ]

        values = self.aggregate_to_stress_periods(
            np.asarray(recharge_series, dtype=float), stress_period_length, n_stress_periods)
        for i, val in enumerate(values):
            lines.append(f"BEGIN PERIOD {i + 1}")
            lines.append("  RECHARGE")
            lines.append(f"    CONSTANT {val:.8e}")
            lines.append(f"END PERIOD {i + 1}")
            lines.append("")

        rch_path.write_text("\n".join(lines))

        self.logger.debug(
            f"Wrote MODFLOW per-period recharge: {len(values)} periods of "
            f"{stress_period_length} d from {len(recharge_series)} days to {rch_path}"
        )

        return rch_path

    def extract_surface_runoff(
        self,
        summa_output_dir: Path,
        variable: str = 'scalarSurfaceRunoff',
    ) -> pd.Series:
        """Extract surface runoff from SUMMA output.

        Args:
            summa_output_dir: Path to SUMMA output directory
            variable: SUMMA variable name for surface runoff

        Returns:
            Surface runoff in m/s with datetime index
        """
        ds, _ = self._open_summa_variable(summa_output_dir, variable)

        data = ds[variable]
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values
        ds.close()

        series = pd.Series(values, index=times, name='surface_runoff_m_s')

        # Resample to daily if sub-daily
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()

        series = series.fillna(0.0)
        return series

    def extract_fast_runoff(
        self,
        summa_output_dir: Path,
        total_var: str = 'averageRoutedRunoff',
        drainage_var: str = 'scalarSoilDrainage',
    ) -> pd.Series:
        """Extract fast runoff = total routed runoff minus soil drainage.

        In the coupled SUMMA-MODFLOW setup, soil drainage feeds MODFLOW as
        recharge.  The remaining SUMMA runoff (surface + lateral subsurface)
        is the fast-flow component that bypasses the groundwater model.

        Args:
            summa_output_dir: Path to SUMMA output directory
            total_var: SUMMA total runoff variable (m/s)
            drainage_var: SUMMA soil drainage variable (m/s)

        Returns:
            Fast runoff in m/s (daily) with datetime index
        """
        ds_total, _ = self._open_summa_variable(summa_output_dir, total_var)
        total = ds_total[total_var]
        for dim in list(total.dims):
            if dim != 'time' and total.sizes[dim] == 1:
                total = total.squeeze(dim)
        times = pd.to_datetime(ds_total['time'].values)
        total_series = pd.Series(total.values, index=times)
        ds_total.close()

        ds_drain, _ = self._open_summa_variable(summa_output_dir, drainage_var)
        drain = ds_drain[drainage_var]
        for dim in list(drain.dims):
            if dim != 'time' and drain.sizes[dim] == 1:
                drain = drain.squeeze(dim)
        drain_series = pd.Series(drain.values, index=pd.to_datetime(ds_drain['time'].values))
        ds_drain.close()

        # Resample both to daily
        for s in [total_series, drain_series]:
            if len(s) > 1:
                dt = (s.index[1] - s.index[0]).total_seconds()
                if dt < 86400:
                    total_series = total_series.resample('D').mean()
                    drain_series = drain_series.resample('D').mean()
                    break

        fast = (total_series - drain_series).clip(lower=0.0).fillna(0.0)
        fast.name = 'fast_runoff_m_s'

        self.logger.debug(
            f"Fast runoff: mean={fast.mean():.6e} m/s, "
            f"total routed: {total_series.mean():.6e}, "
            f"drainage: {drain_series.mean():.6e}"
        )
        return fast

    def combine_flows(
        self,
        surface_runoff_kg_m2_s: pd.Series,
        drain_discharge_m3_d: pd.Series,
        catchment_area_m2: float,
    ) -> pd.Series:
        """Combine surface runoff and MODFLOW baseflow into total streamflow.

        Args:
            surface_runoff_kg_m2_s: SUMMA surface runoff (kg/m2/s)
            drain_discharge_m3_d: MODFLOW drain discharge (m3/d)
            catchment_area_m2: Catchment area in m2

        Returns:
            Combined streamflow in m3/s
        """
        # Convert surface runoff: m/s → m3/s
        # (m/s) × area_m2 = m3/s
        surface_m3s = surface_runoff_kg_m2_s * catchment_area_m2

        # Convert drain discharge: m3/d → m3/s
        baseflow_m3s = drain_discharge_m3_d / 86400.0

        # Normalize both indices to date-only for alignment
        surface_m3s.index = surface_m3s.index.normalize()
        baseflow_m3s.index = baseflow_m3s.index.normalize()

        # Align indices
        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        if len(common_idx) == 0:
            self.logger.warning(
                "No overlapping dates between surface runoff and baseflow. "
                "Returning surface runoff only."
            )
            return surface_m3s.rename('total_streamflow_m3s')

        surface_aligned = surface_m3s.loc[common_idx]
        baseflow_aligned = baseflow_m3s.loc[common_idx]

        total = surface_aligned + baseflow_aligned
        total.name = 'total_streamflow_m3s'

        self.logger.debug(
            f"Combined flows: surface={surface_aligned.mean():.4f} m3/s, "
            f"baseflow={baseflow_aligned.mean():.4f} m3/s, "
            f"total={total.mean():.4f} m3/s"
        )

        return total
