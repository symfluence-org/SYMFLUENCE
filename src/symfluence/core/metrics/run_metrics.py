# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-run performance summary: ``results/<experiment_id>/metrics.json``.

Every run that produces streamflow writes one machine-readable answer to
"how good is my model?", next to the time series in ``results/``:

* after ``postprocess_results`` (stage ``"postprocess"``): the simulated
  discharge columns of ``results/<exp>_results.csv`` scored against the
  observed streamflow, and
* after the calibration final evaluation (stage ``"calibration"``): the
  scores of the best parameter set, taken from the optimizer's
  ``*_final_evaluation.json``.

The latest stage wins: after a calibrated run ``metrics.json`` describes the
calibrated model. A ``metrics.csv`` flavour (one row per model and period)
is written alongside.

Schema (``schema_version`` 1)::

    {
      "schema_version": 1,
      "stage": "postprocess" | "calibration",
      "generated_at": "<ISO 8601>",
      "domain": "<DOMAIN_NAME>",
      "experiment_id": "<EXPERIMENT_ID>",
      "target": {"variable": "streamflow", "station_id": str|null,
                 "observations_path": str|null, "observations_found": bool},
      "periods": {"spinup": [start, end]|null, "calibration": [...]|null,
                  "evaluation": [...]|null, "test": [...]|null},
      "resolution": "daily" | "native" | null,
      "models": {
        "<MODEL>": {
          "source": "<where the simulation came from>",
          "metrics": {"<period>": {"KGE", "r", "alpha", "beta", "NSE",
                                   "logNSE", "PBIAS", "RMSE", "n_obs",
                                   "start", "end"}},
          "benchmarks": {"<period>": {"<benchmark>": {"KGE", "KGE_skill"}}}
        }
      },
      "warnings": ["<message>", ...],
      "provenance": {"symfluence_version", "results_csv", "final_evaluation",
                     "run_manifest", "run_id"}
    }

``<period>`` is one of ``calibration``, ``evaluation``, ``test`` (only when a
period is configured) and ``full`` (the whole overlapping record after
spin-up; postprocess stage only). Missing values are ``null``. Benchmarks
(mean flow, day-of-year climatology, persistence; fitted on the calibration
period) appear when ``symfluence.core.metrics.benchmarks`` is available and a
calibration period is configured; ``TEST_PERIOD`` is honoured when the config
schema has it. Keys are only ever added within a schema version.

Pure computation plus JSON/CSV I/O; logging is left to callers.
"""
from __future__ import annotations

import importlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.metrics.metrics_core import kge, log_nse, nse, pbias, rmse
from symfluence.core.modeling.observation_paths import (
    first_existing_path,
    streamflow_observation_candidates,
)
from symfluence.core.modeling.utilities.periods import parse_period

__all__ = [
    'SCHEMA_VERSION',
    'METRICS_FILENAME',
    'METRICS_CSV_FILENAME',
    'SUMMARY_FIGURE_FILENAME',
    'METRIC_KEYS',
    'SCORED_PERIODS',
    'MIN_POINTS',
    'run_results_dir',
    'configured_periods',
    'expected_observation_path',
    'load_observed_streamflow',
    'align_series',
    'score',
    'build_postprocess_metrics',
    'build_calibration_metrics',
    'metrics_from_final_evaluation',
    'write_metrics',
    'load_metrics',
    'summary_rows',
]

SCHEMA_VERSION = 1
METRICS_FILENAME = 'metrics.json'
METRICS_CSV_FILENAME = 'metrics.csv'
SUMMARY_FIGURE_FILENAME = 'summary.png'

#: Per-period metric keys, in display order.
METRIC_KEYS: Tuple[str, ...] = (
    'KGE', 'r', 'alpha', 'beta', 'NSE', 'logNSE', 'PBIAS', 'RMSE', 'n_obs',
)
#: Configurable scored periods, in display order (``full`` is added last).
SCORED_PERIODS: Tuple[str, ...] = ('calibration', 'evaluation', 'test')
#: Fewer aligned points than this and a period is not scored.
MIN_POINTS = 10

# Observation column heuristics, as the streamflow evaluator uses.
_FLOW_TERMS = ('flow', 'discharge', 'q_', 'streamflow')
_TIME_TERMS = ('datetime', 'date', 'time')

Period = Tuple[pd.Timestamp, pd.Timestamp]


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

def run_results_dir(project_dir: Path, experiment_id: str) -> Path:
    """``<project_dir>/results/<experiment_id>`` -- where metrics.json lives."""
    return Path(project_dir) / 'results' / str(experiment_id)


def _safe_period(value: Any) -> Optional[Period]:
    try:
        parsed = parse_period(value)
    except (ValueError, TypeError):
        return None
    if parsed is None or pd.isna(parsed[0]) or pd.isna(parsed[1]):
        return None
    return parsed


def configured_periods(config: Any) -> Dict[str, Optional[Period]]:
    """Calibration / evaluation / test / spin-up periods from a typed config.

    ``TEST_PERIOD`` is read only when the schema declares it, so this works
    with and without split-sample TEST_PERIOD support.
    """
    domain = getattr(config, 'domain', None)
    return {
        'spinup': _safe_period(getattr(domain, 'spinup_period', None)),
        'calibration': _safe_period(getattr(domain, 'calibration_period', None)),
        'evaluation': _safe_period(getattr(domain, 'evaluation_period', None)),
        'test': _safe_period(getattr(domain, 'test_period', None)),
    }


def _spinup_years(config: Any) -> int:
    try:
        return int(getattr(getattr(config, 'evaluation', None), 'spinup_years', 0) or 0)
    except (TypeError, ValueError):
        return 0


def expected_observation_path(config: Any, project_dir: Path) -> Path:
    """The streamflow observation file a run is scored against.

    ``OBSERVATIONS_PATH`` when set, else the first existing default candidate
    (``observations/streamflow/preprocessed/<domain>_streamflow_processed.csv``
    when none exists -- that is the path named in "missing" warnings).
    """
    configured = getattr(getattr(config, 'paths', None), 'observations_path', None)
    if configured and str(configured).strip().lower() not in ('default', 'none', ''):
        return Path(str(configured))
    domain_name = str(getattr(getattr(config, 'domain', None), 'name', '') or '')
    return first_existing_path(streamflow_observation_candidates(Path(project_dir), domain_name))


def load_observed_streamflow(path: Path) -> Optional[pd.Series]:
    """Read a streamflow observation CSV into a datetime-indexed Series.

    Returns None when the file has no recognisable time or flow column.
    Raises ``OSError`` / ``ValueError`` for unreadable files.
    """
    df = pd.read_csv(path)
    if df.empty:
        return None
    time_col = next((c for c in df.columns if any(t in str(c).lower() for t in _TIME_TERMS)), None)
    if time_col is None:
        time_col = df.columns[0]
    flow_cols = [c for c in df.columns
                 if c != time_col and (any(t in str(c).lower() for t in _FLOW_TERMS)
                                       or str(c).lower() == 'q')]
    if not flow_cols:
        numeric = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) != 1:
            return None
        flow_cols = numeric
    index = pd.to_datetime(df[time_col], errors='coerce')
    series = pd.Series(pd.to_numeric(df[flow_cols[0]], errors='coerce').values,
                       index=pd.DatetimeIndex(index), name='observed')
    series = series[series.index.notna()].sort_index()
    return series


# ---------------------------------------------------------------------------
# Alignment and scoring
# ---------------------------------------------------------------------------

def _clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep='first')]
    return s.dropna().sort_index()


def _median_step(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if len(index) < 2:
        return None
    diffs = np.diff(index.values).astype('timedelta64[s]').astype(float)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return pd.Timedelta(seconds=float(np.median(diffs)))


def align_series(obs: pd.Series, sim: pd.Series) -> Tuple[pd.DataFrame, str]:
    """Pair observations and simulation on a common time axis.

    Both sub-daily: matched at their native (hour-rounded) stamps. Otherwise
    both are averaged to daily means. Returns a frame with ``obs`` and
    ``sim`` columns (NaN pairs dropped) and the resolution used.
    """
    o, s = _clean(obs), _clean(sim)
    day = pd.Timedelta(days=1)
    o_step, s_step = _median_step(o.index), _median_step(s.index)
    if o_step is not None and s_step is not None and o_step < day and s_step < day:
        o.index = o.index.round('h')
        s.index = s.index.round('h')
        o = o.groupby(level=0).mean()
        s = s.groupby(level=0).mean()
        resolution = 'native'
    else:
        o = o.resample('D').mean()
        s = s.resample('D').mean()
        resolution = 'daily'
    frame = pd.concat({'obs': o, 'sim': s}, axis=1, join='inner').dropna()
    return frame, resolution


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def score(obs: Any, sim: Any) -> Dict[str, Optional[float]]:
    """KGE (+ r, alpha, beta), NSE, logNSE, PBIAS, RMSE and n_obs."""
    o = np.asarray(obs, dtype=float)
    s = np.asarray(sim, dtype=float)
    valid = np.isfinite(o) & np.isfinite(s)
    o, s = o[valid], s[valid]
    n = int(o.size)
    if n < MIN_POINTS:
        out: Dict[str, Optional[float]] = {k: None for k in METRIC_KEYS}
        out['n_obs'] = n
        return out
    comps = kge(o, s, return_components=True)
    assert isinstance(comps, dict)
    return {
        'KGE': _num(comps['KGE']),
        'r': _num(comps['r']),
        'alpha': _num(comps['alpha']),
        'beta': _num(comps['beta']),
        'NSE': _num(nse(o, s)),
        'logNSE': _num(log_nse(o, s)),
        'PBIAS': _num(pbias(o, s)),
        'RMSE': _num(rmse(o, s)),
        'n_obs': n,
    }


def _in_period(index: pd.DatetimeIndex, period: Period) -> np.ndarray:
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if end == end.normalize():
        return np.asarray((index >= start) & (index < end + pd.Timedelta(days=1)))
    return np.asarray((index >= start) & (index <= end))


def _spinup_cutoff(periods: Mapping[str, Optional[Period]], spinup_years: int,
                   sim_start: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """First timestamp after spin-up: SPINUP_PERIOD end and/or N years from sim start."""
    cutoffs: List[pd.Timestamp] = []
    spinup = periods.get('spinup')
    if spinup is not None:
        end = pd.Timestamp(spinup[1])
        cutoffs.append(end + pd.Timedelta(days=1) if end == end.normalize() else end)
    if spinup_years > 0 and sim_start is not None:
        cutoffs.append(pd.Timestamp(sim_start) + pd.DateOffset(years=spinup_years))
    return max(cutoffs) if cutoffs else None


def _benchmark_module() -> Any:
    """``symfluence.core.metrics.benchmarks`` when this build has it, else None."""
    try:
        return importlib.import_module('symfluence.core.metrics.benchmarks')
    except ImportError:
        return None


def _benchmark_scores(frame: pd.DataFrame, periods: Mapping[str, Optional[Period]],
                      period_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """KGE of each benchmark and the model's KGE skill over it, per period."""
    bm = _benchmark_module()
    calibration = periods.get('calibration')
    if bm is None or calibration is None or frame.empty:
        return {}
    series = bm.benchmark_series(frame['obs'], calibration)
    if not series:
        return {}
    out: Dict[str, Any] = {}
    index = pd.DatetimeIndex(frame.index)
    for name in SCORED_PERIODS:
        period = periods.get(name)
        model_kge = period_metrics.get(name, {}).get('KGE')
        if period is None or model_kge is None:
            continue
        mask = _in_period(index, period)
        obs_p = frame['obs'][mask]
        if len(obs_p) < MIN_POINTS:
            continue
        scores: Dict[str, Any] = {}
        for bm_name, bm_series in series.items():
            pred = bm_series.reindex(obs_p.index)
            if bm_name == 'mean_flow':
                mean_obs = float(obs_p.mean())
                beta = float(pred.mean()) / mean_obs if mean_obs else float('nan')
                bm_kge = _num(bm.constant_benchmark_kge(beta))
            else:
                ok = pred.notna().values
                bm_kge = (_num(kge(obs_p.values[ok], pred.values[ok]))
                          if ok.sum() >= MIN_POINTS else None)
            scores[bm_name] = {
                'KGE': bm_kge,
                'KGE_skill': _num(bm.skill_score(model_kge, bm_kge)) if bm_kge is not None else None,
            }
        if scores:
            out[name] = scores
    return out


def _period_json(period: Optional[Period]) -> Optional[List[str]]:
    if period is None:
        return None
    return [pd.Timestamp(period[0]).isoformat(), pd.Timestamp(period[1]).isoformat()]


def _version() -> str:
    try:
        from symfluence.symfluence_version import __version__
        return str(__version__)
    except ImportError:
        return '0+unknown'


def _base_document(stage: str, config: Any, periods: Mapping[str, Optional[Period]],
                   obs_path: Optional[Path], obs_found: bool) -> Dict[str, Any]:
    domain = getattr(config, 'domain', None)
    evaluation = getattr(config, 'evaluation', None)
    optimization = getattr(config, 'optimization', None)
    station = getattr(getattr(evaluation, 'streamflow', None), 'station_id', None)
    variable = 'streamflow'
    if stage == 'calibration':
        variable = str(getattr(optimization, 'target', None) or 'streamflow')
    return {
        'schema_version': SCHEMA_VERSION,
        'stage': stage,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'domain': str(getattr(domain, 'name', '') or ''),
        'experiment_id': str(getattr(domain, 'experiment_id', '') or ''),
        'target': {
            'variable': variable,
            'station_id': None if station in (None, '') else str(station),
            'observations_path': str(obs_path) if obs_path is not None else None,
            'observations_found': bool(obs_found),
        },
        'periods': {name: _period_json(periods.get(name))
                    for name in ('spinup', *SCORED_PERIODS)},
        'resolution': None,
        'models': {},
        'warnings': [],
        'provenance': {
            'symfluence_version': _version(),
            'results_csv': None,
            'final_evaluation': None,
            'run_manifest': None,
            'run_id': None,
        },
    }


def _model_name(column: str) -> str:
    return re.sub(r'_discharge(_cms)?$', '', str(column))


def build_postprocess_metrics(
    config: Any,
    project_dir: Path,
    results_csv: Path,
) -> Dict[str, Any]:
    """Score every ``*discharge*`` column of the results CSV against observations.

    Never raises for missing/unaligned data: problems become entries in the
    document's ``warnings`` list (the caller logs them at WARNING level).
    """
    periods = configured_periods(config)
    obs_path: Optional[Path] = None
    try:
        obs_path = expected_observation_path(config, project_dir)
    except ValueError:
        obs_path = None
    doc = _base_document('postprocess', config, periods, obs_path,
                         obs_found=bool(obs_path is not None and obs_path.exists()))
    doc['provenance']['results_csv'] = str(results_csv)
    warnings: List[str] = doc['warnings']

    if not Path(results_csv).exists():
        warnings.append(f"No simulation results found at {results_csv}; nothing was scored.")
        return doc
    results = pd.read_csv(results_csv, index_col=0, parse_dates=True)
    sim_cols = [c for c in results.columns if 'discharge' in str(c).lower()]
    obs_cols = [c for c in results.columns
                if c not in sim_cols and ('obs' in str(c).lower())]
    if not sim_cols:
        warnings.append(f"{results_csv} has no simulated discharge columns; nothing was scored.")
        return doc

    obs: Optional[pd.Series] = None
    if obs_cols:
        obs = results[obs_cols[0]]
        doc['target']['observations_found'] = True
        doc['target']['observations_path'] = f"{results_csv}:{obs_cols[0]}"
    elif obs_path is not None and obs_path.exists():
        if obs_path.suffix.lower() != '.csv':
            warnings.append(
                f"Observations at {obs_path} are not a CSV; set OBSERVATIONS_PATH to a "
                "streamflow CSV to get run metrics."
            )
        else:
            try:
                obs = load_observed_streamflow(obs_path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                warnings.append(f"Could not read observed streamflow {obs_path}: {exc}")
            else:
                if obs is None or obs.dropna().empty:
                    warnings.append(
                        f"Observed streamflow file {obs_path} has no usable time/flow columns."
                    )
                    obs = None
    else:
        warnings.append(
            f"Observed streamflow not found (expected {obs_path}); simulation was not "
            "scored. Run the process_observed_data step or set OBSERVATIONS_PATH."
        )

    for col in sim_cols:
        model = _model_name(col)
        entry: Dict[str, Any] = {'source': f"{results_csv}:{col}", 'metrics': {}, 'benchmarks': {}}
        doc['models'][model] = entry
        if obs is None:
            continue
        frame, resolution = align_series(obs, results[col])
        doc['resolution'] = resolution
        sim_clean = _clean(results[col])
        cutoff = _spinup_cutoff(periods, _spinup_years(config),
                                sim_clean.index[0] if len(sim_clean) else None)
        if cutoff is not None:
            frame = frame[frame.index >= cutoff]
        if frame.empty:
            warnings.append(
                f"{model}: simulation and observations ({obs_path}) do not overlap"
                + (" after spin-up" if cutoff is not None else "") + "; not scored."
            )
            continue
        index = pd.DatetimeIndex(frame.index)
        for name in SCORED_PERIODS:
            period = periods.get(name)
            if period is None:
                continue
            sub = frame[_in_period(index, period)]
            metrics = score(sub['obs'], sub['sim'])
            metrics['start'] = sub.index[0].isoformat() if len(sub) else None
            metrics['end'] = sub.index[-1].isoformat() if len(sub) else None
            entry['metrics'][name] = metrics
            if metrics['n_obs'] < MIN_POINTS:
                warnings.append(
                    f"{model}: only {metrics['n_obs']} aligned observation(s) in the "
                    f"{name} period {_period_json(period)}; not scored."
                )
        full = score(frame['obs'], frame['sim'])
        full['start'] = index[0].isoformat()
        full['end'] = index[-1].isoformat()
        entry['metrics']['full'] = full
        if full['n_obs'] < MIN_POINTS:
            warnings.append(f"{model}: only {full['n_obs']} aligned observation(s); not scored.")
        entry['benchmarks'] = _benchmark_scores(frame, periods, entry['metrics'])
    return doc


# ---------------------------------------------------------------------------
# Calibration final evaluation
# ---------------------------------------------------------------------------

_BM_KGE = re.compile(r'^KGE_benchmark_(.+)$')
_BM_SKILL = re.compile(r'^KGE_skill_(.+)$')


def metrics_from_final_evaluation(final_eval: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map a ``*_final_evaluation.json`` payload onto (metrics, benchmarks) per period.

    Reads ``calibration_metrics``, ``evaluation_metrics`` and (when present)
    ``test_metrics``. Keys ``KGE_benchmark_<name>`` / ``KGE_skill_<name>``
    become benchmark entries.
    """
    metrics: Dict[str, Any] = {}
    benchmarks: Dict[str, Any] = {}
    for name, key in (('calibration', 'calibration_metrics'),
                      ('evaluation', 'evaluation_metrics'),
                      ('test', 'test_metrics')):
        raw = final_eval.get(key) or {}
        if not isinstance(raw, Mapping) or not raw:
            continue
        period = {k: _num(raw.get(k)) for k in METRIC_KEYS}
        if period['r'] is None:
            period['r'] = _num(raw.get('correlation'))
        n = raw.get('n_obs')
        period['n_obs'] = int(n) if _num(n) is not None else None
        period['start'] = None
        period['end'] = None
        metrics[name] = period
        bms: Dict[str, Dict[str, Optional[float]]] = {}
        for k, v in raw.items():
            m = _BM_KGE.match(str(k))
            if m:
                bms.setdefault(m.group(1), {'KGE': None, 'KGE_skill': None})['KGE'] = _num(v)
            m = _BM_SKILL.match(str(k))
            if m:
                bms.setdefault(m.group(1), {'KGE': None, 'KGE_skill': None})['KGE_skill'] = _num(v)
        if bms:
            benchmarks[name] = bms
    return metrics, benchmarks


def build_calibration_metrics(
    config: Any,
    project_dir: Path,
    final_evaluations: Iterable[Tuple[str, Path]],
) -> Dict[str, Any]:
    """metrics.json document from ``(model, final_evaluation.json)`` pairs."""
    periods = configured_periods(config)
    try:
        obs_path: Optional[Path] = expected_observation_path(config, project_dir)
    except ValueError:
        obs_path = None
    doc = _base_document('calibration', config, periods, obs_path,
                         obs_found=bool(obs_path is not None and obs_path.exists()))
    sources: List[str] = []
    for model, path in final_evaluations:
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            doc['warnings'].append(f"{model}: could not read final evaluation {path}: {exc}")
            continue
        metrics, benchmarks = metrics_from_final_evaluation(payload)
        if not metrics:
            doc['warnings'].append(f"{model}: final evaluation {path} holds no metrics.")
        warning = payload.get('period_warning')
        if warning:
            doc['warnings'].append(f"{model}: {warning}")
        doc['models'][model] = {
            'source': str(path),
            'algorithm': payload.get('algorithm'),
            'metrics': metrics,
            'benchmarks': benchmarks,
        }
        sources.append(str(path))
    doc['provenance']['final_evaluation'] = sources[0] if len(sources) == 1 else (sources or None)
    return doc


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def summary_rows(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a metrics document to one row per (model, period)."""
    rows: List[Dict[str, Any]] = []
    order = (*SCORED_PERIODS, 'full')
    for model, entry in (doc.get('models') or {}).items():
        metrics = entry.get('metrics') or {}
        for period in order:
            if period in metrics:
                row: Dict[str, Any] = {'model': model, 'period': period}
                row.update({k: metrics[period].get(k) for k in METRIC_KEYS})
                rows.append(row)
    return rows


def write_metrics(doc: Mapping[str, Any], out_dir: Path) -> Path:
    """Write ``metrics.json`` (and ``metrics.csv``) into ``out_dir``; return the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / METRICS_FILENAME
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False, default=str) + '\n',
                   encoding='utf-8')
    tmp.replace(path)
    columns = ['model', 'period', *METRIC_KEYS]
    pd.DataFrame(summary_rows(doc), columns=columns).to_csv(
        out_dir / METRICS_CSV_FILENAME, index=False
    )
    return path


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Read a metrics.json; None when absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
