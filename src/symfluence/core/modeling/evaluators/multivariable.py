# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-agnostic multi-variable evaluators.

These evaluators read model output only through the standard output-variable
adapters (:mod:`symfluence.core.modeling.output_variables`), so every model
with an adapter is scored the same way:

``GLACIER_MB``
    Period-mean glacier mass-balance rates (geodetic, e.g. Hugonnet et al.
    2021, or annual/seasonal glaciological balances) against observations,
    scored by bias of the period-mean rate (with observation uncertainty when
    given).
``BASIN_SWE``, ``BASIN_SCA``, ``BASIN_ET``
    Basin SWE (mm), snow-covered fraction (0–1) and ET (mm/d) against an
    observed series, scored with the bias-sensitive metric set of the base
    evaluator (KGE and its r / alpha / beta components, NSE, PBIAS, RMSE...).

The SUMMA-oriented ``SNOW`` / ``ET`` / ``TWS`` evaluators are unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import (
    ET,
    GLACIER_MASS_BALANCE,
    GLACIER_STORAGE,
    SNOW_COVERED_FRACTION,
    SWE,
    OutputVariableAdapter,
    UnsupportedVariableError,
    get_output_adapter,
)
from symfluence.core.registries import R

from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

DAYS_PER_YEAR = 365.25

# Column-name candidates (case-insensitive) for glacier mass-balance observations.
_START_COLS = ('start_date', 'period_start', 'date_start', 'start', 't0')
_END_COLS = ('end_date', 'period_end', 'date_end', 'end', 't1')
_PERIOD_COLS = ('period',)
_YEAR_COLS = ('year', 'balance_year', 'hydrological_year')
_RATE_COLS = ('dmdtda', 'mb_rate', 'mass_balance_rate', 'annual_net_mb', 'mass_balance', 'mb', 'balance')
_SIGMA_COLS = ('err_dmdtda', 'mb_rate_err', 'mb_err', 'uncertainty', 'sigma', 'error', 'err')
_AREA_COLS = ('area_km2', 'g_area_dyn', 'glacier_area_km2')


def _pick(columns, candidates) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def load_mass_balance_observations(path: Path, units: str = 'm_we_per_yr') -> pd.DataFrame:
    """Observed glacier mass-balance periods.

    Accepts CSV (any delimiter) or NetCDF (converted with ``to_dataframe``)
    with either ``start``/``end`` date columns, a Hugonnet-style ``period``
    column (``YYYY-MM-DD_YYYY-MM-DD``), or a balance ``year`` (index or column;
    the period is 30 Sep of the previous year to 30 Sep of that year, as in the
    LamaH-Ice ``glacier_timeseries.csv``). Rate column: ``dmdtda``,
    ``mb_rate``, ``annual_net_MB``...; optional uncertainty (``err_dmdtda``,
    ``uncertainty``...) and glacier area in km² (``area_km2``, ``g_area_dyn``).

    Returns:
        DataFrame with ``start``, ``end`` (Timestamps), ``rate`` and ``sigma``
        (mm w.e./yr, sigma NaN when absent) and ``area_km2`` (NaN when absent).
    """
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f'Glacier mass-balance observations not found: {path}')
    if path.suffix in ('.nc', '.nc4'):
        import xarray as xr
        with xr.open_dataset(path) as ds:
            frame = ds.to_dataframe().reset_index()
    else:
        frame = pd.read_csv(path, sep=None, engine='python')
    first = frame.columns[0]
    has_period = (_pick(frame.columns, _YEAR_COLS) is not None or _pick(frame.columns, _PERIOD_COLS) is not None
                  or (_pick(frame.columns, _START_COLS) is not None and _pick(frame.columns, _END_COLS) is not None))
    if not has_period:
        # Balance years as the (possibly unnamed) first column, e.g. LamaH-Ice glacier_timeseries.csv
        years = pd.to_numeric(frame[first], errors='coerce')
        if years.notna().all() and years.between(1800, 2300).all() and (years % 1 == 0).all():
            frame = frame.rename(columns={first: 'year'})

    rate_col = _pick(frame.columns, _RATE_COLS)
    if rate_col is None:
        raise EvaluationError(f'No mass-balance rate column in {path}; columns: {list(frame.columns)}')
    start_col, end_col = _pick(frame.columns, _START_COLS), _pick(frame.columns, _END_COLS)
    period_col, year_col = _pick(frame.columns, _PERIOD_COLS), _pick(frame.columns, _YEAR_COLS)
    if start_col is not None and end_col is not None:
        start = pd.to_datetime(frame[start_col])
        end = pd.to_datetime(frame[end_col])
    elif period_col is not None:
        parts = frame[period_col].astype(str).str.split('_', n=1, expand=True)
        start, end = pd.to_datetime(parts[0]), pd.to_datetime(parts[1])
    elif year_col is not None:
        years = pd.to_numeric(frame[year_col], errors='coerce').astype('Int64')
        start = pd.to_datetime({'year': years - 1, 'month': 9, 'day': 30})
        end = pd.to_datetime({'year': years, 'month': 9, 'day': 30})
    else:
        raise EvaluationError(f'No period (start/end, period or year) columns in {path}')

    scale = 1000.0 if units == 'm_we_per_yr' else 1.0
    sigma_col, area_col = _pick(frame.columns, _SIGMA_COLS), _pick(frame.columns, _AREA_COLS)
    out = pd.DataFrame({
        'start': pd.DatetimeIndex(start),
        'end': pd.DatetimeIndex(end),
        'rate': pd.to_numeric(frame[rate_col], errors='coerce').to_numpy() * scale,
        'sigma': (pd.to_numeric(frame[sigma_col], errors='coerce').to_numpy() * scale
                  if sigma_col is not None else np.nan),
        'area_km2': (pd.to_numeric(frame[area_col], errors='coerce').to_numpy()
                     if area_col is not None else np.nan),
    })
    out = out[np.isfinite(out['rate']) & out['start'].notna() & out['end'].notna()]
    return out[out['end'] > out['start']].reset_index(drop=True)


def _state_at(series: pd.Series, when: pd.Timestamp, tolerance: pd.Timedelta) -> Optional[float]:
    """Last finite value stamped at or before ``when``, if within ``tolerance``."""
    s = series.dropna()
    s = s[s.index <= when]
    if s.empty or when - s.index[-1] > tolerance:
        return None
    return float(s.iloc[-1])


def glacier_mass_balance_metrics(sim_rate: np.ndarray, obs_rate: np.ndarray, sigma: np.ndarray,
                                 default_scale: float) -> Dict[str, float]:
    """Bias-based scores of period-mean rates (all mm w.e./yr except GMB_SCORE / GMB_N)."""
    sim_rate, obs_rate = np.asarray(sim_rate, float), np.asarray(obs_rate, float)
    sigma = np.asarray(sigma, float)
    diff = sim_rate - obs_rate
    bias = float(np.mean(diff))
    obs_mean = float(np.mean(obs_rate))
    scale = np.where(np.isfinite(sigma) & (sigma > 0), sigma, default_scale)
    z = diff / scale
    metrics = {
        'GMB_BIAS': bias,
        'GMB_ABS_BIAS': abs(bias),
        'GMB_REL_BIAS': bias / abs(obs_mean) if obs_mean != 0 else np.nan,
        'GMB_RMSE': float(np.sqrt(np.mean(diff ** 2))),
        'GMB_SCORE': float(1.0 - np.sqrt(np.mean(z ** 2))),
        'GMB_SIM_MEAN': float(np.mean(sim_rate)),
        'GMB_OBS_MEAN': obs_mean,
        'GMB_N': float(len(diff)),
    }
    if np.isfinite(sigma).any():
        metrics['GMB_WITHIN_UNCERTAINTY'] = float(np.mean(np.abs(diff) <= np.where(np.isfinite(sigma), sigma, 0)))
    return metrics


class AdapterVariableEvaluator(ModelEvaluator):
    """Evaluator reading one standard variable through the model's output adapter."""

    VARIABLE: str = ''
    #: attribute on ``config.evaluation.multivariable`` / flat key holding the obs path
    OBS_FIELD: str = ''
    OBS_KEY: str = ''

    def __init__(self, config: 'SymfluenceConfig', project_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None, target: Optional[str] = None, **kwargs):
        super().__init__(config, project_dir, logger, target=target)
        self.model_name = self._adapter_model_name()
        self.adapter: Optional[OutputVariableAdapter] = get_output_adapter(
            self.model_name or '', self.project_dir, config)

    # -- model / adapter ------------------------------------------------
    def _adapter_model_name(self) -> Optional[str]:
        override = self._get_config_value(
            lambda: self.config.evaluation.multivariable.adapter_model,
            dict_key='OUTPUT_ADAPTER_MODEL')
        if override:
            return str(override).strip().upper()
        models = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            default='', dict_key='HYDROLOGICAL_MODEL')
        if isinstance(models, (list, tuple)):
            names = [str(m).strip().upper() for m in models if str(m).strip()]
        else:
            names = [m.strip().upper() for m in str(models or '').split(',') if m.strip()]
        return names[0] if len(names) == 1 else None

    def _require_adapter(self) -> OutputVariableAdapter:
        if self.adapter is None:
            raise UnsupportedVariableError(
                f"No output adapter for model {self.model_name!r}; set OUTPUT_ADAPTER_MODEL "
                "when HYDROLOGICAL_MODEL lists several models")
        return self.adapter

    def _sim_dir(self, sim_files: List[Path]) -> Path:
        path = Path(sim_files[0])
        return path if path.is_dir() else path.parent

    # -- ModelEvaluator hooks -------------------------------------------
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """The output directory itself: adapters locate their own files."""
        if self.adapter is None or not Path(sim_dir).is_dir():
            return []
        return [Path(sim_dir)]

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        return self._require_adapter().extract(self.VARIABLE, self._sim_dir(sim_files))

    def get_observed_data_path(self) -> Path:
        configured = self._get_config_value(
            lambda: getattr(self.config.evaluation.multivariable, self.OBS_FIELD),
            dict_key=self.OBS_KEY)
        if not configured:
            raise EvaluationError(f'{self.OBS_KEY} is not set')
        path = Path(str(configured)).expanduser()
        return path if path.is_absolute() else self.project_dir / path

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        names = {str(c).lower(): c for c in columns}
        for candidate in self.OBS_COLUMNS:
            if candidate in names:
                return names[candidate]
        for c in columns:
            if not any(t in str(c).lower() for t in ('date', 'time', 'flag', 'qc', 'unc', 'err')):
                return c
        return None

    OBS_COLUMNS: Tuple[str, ...] = ('value', 'obs', 'observed')

    def _load_observed_data_from_netcdf(self, nc_path: Path) -> Optional[pd.Series]:
        """Plain (ungrouped) NetCDF: first data variable, averaged over non-time dims."""
        import xarray as xr
        try:
            with xr.open_dataset(nc_path) as ds:
                name = next(iter(ds.data_vars))
                da = ds[name]
                dims = [d for d in da.dims if d != 'time']
                return (da.mean(dims) if dims else da).to_series().dropna()
        except (OSError, ValueError, StopIteration, KeyError) as e:
            self.logger.error(f'Could not read observations {nc_path}: {e}')
            return None

    def needs_routing(self) -> bool:
        return False


@R.evaluators.add('BASIN_SWE')
class BasinSWEEvaluator(AdapterVariableEvaluator):
    """Basin SWE (mm) from any model with an output adapter."""

    VARIABLE = SWE
    OBS_FIELD = 'basin_swe_obs_path'
    OBS_KEY = 'BASIN_SWE_OBS_PATH'
    OBS_COLUMNS = ('swe', 'swe_mm', 'value', 'obs')


@R.evaluators.add('BASIN_SCA')
class BasinSCAEvaluator(AdapterVariableEvaluator):
    """Snow-covered fraction (0-1) from any model with an output adapter.

    Observations given in percent (maximum > 1.5) are divided by 100.
    """

    VARIABLE = SNOW_COVERED_FRACTION
    OBS_FIELD = 'basin_sca_obs_path'
    OBS_KEY = 'BASIN_SCA_OBS_PATH'
    OBS_COLUMNS = ('sca', 'fsca', 'snow_cover', 'snow_covered_fraction', 'fsca_outside_glaciers', 'value')

    def _load_observed_data(self) -> Optional[pd.Series]:
        obs = super()._load_observed_data()
        if obs is not None and len(obs) and float(pd.to_numeric(obs, errors='coerce').max()) > 1.5:
            obs = pd.to_numeric(obs, errors='coerce') / 100.0
        return obs


@R.evaluators.add('BASIN_ET')
class BasinETEvaluator(AdapterVariableEvaluator):
    """Basin ET (mm/d) from any model with an output adapter."""

    VARIABLE = ET
    OBS_FIELD = 'basin_et_obs_path'
    OBS_KEY = 'BASIN_ET_OBS_PATH'
    OBS_COLUMNS = ('et', 'et_mm_day', 'aet', 'value', 'obs')

    @property
    def variable_type(self) -> str:
        return 'flux'

    @property
    def values_are_rates(self) -> bool:
        return True


@R.evaluators.add('GLACIER_MB')
class GlacierMassBalanceEvaluator(AdapterVariableEvaluator):
    """Period-mean glacier mass-balance rates against observations.

    For each observed period ``[start, end]`` the simulated specific rate is
    ``(S(end) - S(start)) / years``, using the latest simulated state at or
    before each boundary (within GLACIER_MB_ENDPOINT_TOLERANCE_DAYS). When the
    observation row gives a glacier area and the model reports glacier storage,
    the storage change is divided by that observed area (as the model-specific
    LamaH-Ice constraints do); otherwise the model's own cumulative specific
    balance is differenced. Only periods inside the calibration (or
    evaluation) period are scored.
    """

    VARIABLE = GLACIER_MASS_BALANCE
    OBS_FIELD = 'glacier_mb_obs_path'
    OBS_KEY = 'GLACIER_MB_OBS_PATH'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_units = self._get_config_value(
            lambda: self.config.evaluation.multivariable.glacier_mb_obs_units,
            dict_key='GLACIER_MB_OBS_UNITS')
        self.default_scale = float(self._get_config_value(
            lambda: self.config.evaluation.multivariable.glacier_mb_scale,
            dict_key='GLACIER_MB_SCALE'))
        self.tolerance = pd.Timedelta(days=float(self._get_config_value(
            lambda: self.config.evaluation.multivariable.glacier_mb_endpoint_tolerance_days,
            dict_key='GLACIER_MB_ENDPOINT_TOLERANCE_DAYS')))

    def load_observations(self) -> pd.DataFrame:
        return load_mass_balance_observations(self.get_observed_data_path(), self.obs_units)

    def _simulated_rates(self, sim: Any, obs: pd.DataFrame) -> pd.DataFrame:
        """Observation rows with a ``sim`` column (mm w.e./yr); unmatched rows dropped."""
        cumulative: Optional[pd.Series] = None
        storage: Optional[pd.Series] = None
        if isinstance(sim, (str, Path)):
            adapter = self._require_adapter()
            sim_dir = Path(sim)
            if obs['area_km2'].notna().any() and adapter.supports(GLACIER_STORAGE):
                storage = adapter.extract(GLACIER_STORAGE, sim_dir)
            if adapter.supports(GLACIER_MASS_BALANCE):
                cumulative = adapter.extract(GLACIER_MASS_BALANCE, sim_dir)
        else:
            cumulative = pd.Series(sim)
            cumulative.index = pd.DatetimeIndex(pd.to_datetime(cumulative.index))
        if cumulative is None and storage is None:
            raise UnsupportedVariableError('Model provides no glacier mass balance or storage')

        rows = []
        for row in obs.itertuples(index=False):
            years = (row.end - row.start).days / DAYS_PER_YEAR
            value = None
            if storage is not None and np.isfinite(row.area_km2) and row.area_km2 > 0:
                left = _state_at(storage, row.start, self.tolerance)
                right = _state_at(storage, row.end, self.tolerance)
                if left is not None and right is not None:
                    value = (right - left) / (row.area_km2 * 1e6) * 1000.0 / years
            if value is None and cumulative is not None:
                left = _state_at(cumulative, row.start, self.tolerance)
                right = _state_at(cumulative, row.end, self.tolerance)
                if left is not None and right is not None:
                    value = (right - left) / years
            if value is not None and np.isfinite(value):
                rows.append(dict(start=row.start, end=row.end, rate=row.rate, sigma=row.sigma, sim=value))
        return pd.DataFrame(rows, columns=['start', 'end', 'rate', 'sigma', 'sim'])

    def _period_metrics(self, rates: pd.DataFrame, period: Tuple, prefix: str) -> Dict[str, float]:
        if period[0] is not None and period[1] is not None:
            end_limit = pd.Timestamp(period[1]) + pd.Timedelta(days=1)
            inside = (rates['start'] >= pd.Timestamp(period[0]) - self.tolerance) & (rates['end'] <= end_limit)
            rates = rates[inside]
        if rates.empty:
            return {}
        metrics = glacier_mass_balance_metrics(
            rates['sim'].to_numpy(), rates['rate'].to_numpy(), rates['sigma'].to_numpy(),
            self.default_scale)
        return {f'{prefix}_{k}' if prefix else k: v for k, v in metrics.items()}

    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None,
                          mizuroute_dir: Optional[Path] = None,
                          calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """Period-mean mass-balance scores (GMB_*), or None when nothing can be scored.

        ``sim`` is the model output directory (preferred: allows the
        observed-area operator) or a cumulative specific balance series (mm w.e.).
        ``obs`` may be a pre-loaded frame as returned by
        :func:`load_mass_balance_observations`.
        """
        try:
            observed = obs if isinstance(obs, pd.DataFrame) else self.load_observations()
            rates = self._simulated_rates(sim, observed)
        except (EvaluationError, OSError, ValueError, KeyError) as e:
            self.logger.error(f'Glacier mass-balance evaluation failed: {e}')
            return None
        if rates.empty:
            self.logger.error('No observed glacier mass-balance period has simulated endpoints')
            return None
        metrics: Dict[str, float] = {}
        has_calibration = bool(self.calibration_period[0] and self.calibration_period[1])
        if has_calibration:
            calib = self._period_metrics(rates, self.calibration_period, 'Calib')
            metrics.update(calib)
            for k, v in calib.items():
                metrics.setdefault(k.replace('Calib_', ''), v)
        if not calibration_only and self.evaluation_period[0] and self.evaluation_period[1]:
            metrics.update(self._period_metrics(rates, self.evaluation_period, 'Eval'))
        if not has_calibration and not metrics:
            # No configured periods: score every observed period.
            metrics.update(self._period_metrics(rates, (None, None), ''))
        if not metrics:
            self.logger.error('No observed glacier mass-balance period lies inside the scored period')
        return metrics or None

    def period_table(self, sim: Any) -> pd.DataFrame:
        """Simulated vs observed rate for every matched period (diagnostics)."""
        return self._simulated_rates(sim, self.load_observations())


GLACIER_MB_METRIC_PREFIX = 'GMB_'


def requested_glacier_mb_metrics(config: Any) -> List[str]:
    """GMB_* names used as objective (OPTIMIZATION_METRIC, COMPOSITE_METRIC, NSGA-II, MOEA/D)."""
    from symfluence.core.mixins.config import resolve_config_value

    names: List[str] = []
    for accessor, key in (
        (lambda: config.optimization.metric, 'OPTIMIZATION_METRIC'),
        (lambda: config.optimization.nsga2.primary_metric, 'NSGA2_PRIMARY_METRIC'),
        (lambda: config.optimization.nsga2.secondary_metric, 'NSGA2_SECONDARY_METRIC'),
        (lambda: config.optimization.moead.primary_metric, 'MOEAD_PRIMARY_METRIC'),
        (lambda: config.optimization.moead.secondary_metric, 'MOEAD_SECONDARY_METRIC'),
    ):
        try:
            value = resolve_config_value(config, accessor, dict_key=key)
        except (AttributeError, KeyError, TypeError):
            value = None
        if value:
            names.append(str(value))
    try:
        composite = resolve_config_value(
            config, lambda: config.optimization.composite_metric, dict_key='COMPOSITE_METRIC')
    except (AttributeError, KeyError, TypeError):
        composite = None
    if isinstance(composite, dict):
        names.extend(str(k) for k in composite)
    out: List[str] = []
    for name in names:
        upper = name.strip().upper()
        if upper.startswith(GLACIER_MB_METRIC_PREFIX) and upper not in out:
            out.append(upper)
    return out


def glacier_mb_metrics_for_run(config: Any, output_dirs: List[Any], project_dir: Optional[Path],
                               logger: logging.Logger) -> Dict[str, float]:
    """GLACIER_MB metrics for one calibration run, for workers that score GMB_* objectives.

    Tries each candidate output directory; when none yields metrics every
    requested GMB_* name is set to NaN, so the run scores as failed instead of
    silently falling back to another metric.
    """
    wanted = requested_glacier_mb_metrics(config)
    if not wanted:
        return {}
    evaluator = GlacierMassBalanceEvaluator(config, project_dir, logger)
    for directory in output_dirs:
        if directory is None or not Path(directory).is_dir():
            continue
        metrics = evaluator.calculate_metrics(Path(directory))
        if metrics:
            return metrics
    logger.warning(f'Could not compute {wanted} for this run; scoring it as failed')
    return {name: float('nan') for name in wanted}

__all__ = [
    'AdapterVariableEvaluator',
    'BasinETEvaluator',
    'BasinSCAEvaluator',
    'BasinSWEEvaluator',
    'GlacierMassBalanceEvaluator',
    'glacier_mass_balance_metrics',
    'glacier_mb_metrics_for_run',
    'load_mass_balance_observations',
    'requested_glacier_mb_metrics',
]
