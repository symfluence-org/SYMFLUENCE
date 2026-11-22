# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Multi-model ensemble combination and fair comparison.

Takes the simulated discharge of N members (any models, run by SYMFLUENCE or
not) and an observed series, and produces:

* ``equal_weight`` -- the plain member mean;
* ``skill_weighted`` -- a weighted mean whose weights are fitted on the
  **calibration period only** (KGE- or inverse-MSE-based, see
  :func:`fit_weights`), so evaluation/test scores of the combination are
  genuinely out of sample;
* ``median`` (optional) -- the member median;
* ``vote_weighted`` -- a weighted mean with fixed, user-given integer
  ``votes`` per member (only when any member declares votes).

Every member and combination is scored on the calibration, evaluation and
(when configured) test periods, and a *comparison conditions* table records
what each member was given: forcing, forcing correction, discretisation,
calibration algorithm, evaluations used, number of parameters and seed --
read from the member's ``*_calibration_conditions.json``,
``resolved_config.yaml``/``run_manifest.json``, ``forcing_correction.json``
and ``metrics.json`` when present, else ``"unknown"``.

Entry points: :func:`run_ensemble` (Python) and ``symfluence ensemble``
(CLI, :class:`symfluence.cli.commands.EnsembleCommands`).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from symfluence.core.metrics.metrics_core import kge, nse, pbias, rmse

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
WEIGHTING_METHODS = ("kge", "inverse_mse")
PERIOD_NAMES = ("calibration", "evaluation", "test")
_SIM_COLUMN_HINTS = ("q_sim", "qsim", "sim", "simulated", "discharge", "streamflow", "flow", "q")
_OBS_COLUMN_HINTS = ("q_obs", "qobs", "obs", "observed", "discharge_cms", "discharge", "streamflow",
                     "flow", "q")

PeriodLike = Union[str, Sequence[Any], None]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def parse_period(value: PeriodLike) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """``"2000-01-01, 2009-12-31"`` or ``[start, end]`` -> (start, end); None if unset."""
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip() or value.strip().lower() in ("none", "null"):
            return None
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"A period needs a start and an end, got {value!r}")
    start, end = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
    if end < start:
        raise ValueError(f"Period end before start: {value!r}")
    # A date-only end means the whole end day.
    if end == end.normalize():
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return start, end


def _pick_column(df: pd.DataFrame, column: Optional[str], hints: Sequence[str], what: str) -> pd.Series:
    if column:
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not in {what} ({list(df.columns)})")
        return df[column]
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 1:
        return numeric.iloc[:, 0]
    lowered = {str(c).lower(): c for c in numeric.columns}
    for hint in hints:
        if hint in lowered:
            return numeric[lowered[hint]]
    raise ValueError(
        f"Cannot tell which column of {what} is discharge ({list(numeric.columns)}); "
        f"name it with 'column:'")


def read_series(path: Union[str, Path], column: Optional[str] = None,
                variable: Optional[str] = None, select: Optional[Mapping[str, Any]] = None,
                hints: Sequence[str] = _SIM_COLUMN_HINTS) -> pd.Series:
    """Read one discharge time series from CSV or NetCDF.

    CSV: first column is the datetime index; ``column`` names the discharge
    column (auto-detected when there is a single numeric column or one with a
    conventional name). NetCDF: ``variable`` is required, ``select`` picks
    one element of any non-time dimension (e.g. ``{gru: 0}``).
    """
    path = Path(path)
    if path.suffix.lower() in (".nc", ".nc4", ".netcdf"):
        import xarray as xr

        if not variable:
            raise ValueError(f"'variable' is required for NetCDF member {path}")
        with xr.open_dataset(path) as ds:
            da = ds[variable]
            if select:
                da = da.isel({k: int(v) for k, v in select.items()})
            da = da.squeeze(drop=True)
            if da.ndim != 1:
                raise ValueError(f"{path}:{variable} is not one series after 'select' (dims {da.dims})")
            series = da.to_series()
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        series = _pick_column(df, column, hints, str(path))
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.DatetimeIndex(series.index)
    if series.index.tz is not None:
        series.index = series.index.tz_convert(None)
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series.name = None
    return series


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh) if path.suffix.lower() == ".json" else yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _flat_config(path: Path) -> Dict[str, Any]:
    """Flat UPPER_CASE config keys from a resolved, flat or nested config file."""
    raw = _load_yaml_or_json(path)
    if any(isinstance(k, str) and k.isupper() for k in raw):
        return raw
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.core.exceptions import SYMFLUENCEError

    try:
        return SymfluenceConfig.from_file(path, use_env=False, validate=False).to_dict(flatten=True)
    except (SYMFLUENCEError, OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.debug(f"Could not resolve config {path}: {e}")
        return raw


def _first(paths: Sequence[Path]) -> Optional[Path]:
    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _member_files(results_dir: Path) -> Dict[str, Optional[Path]]:
    """Locate a member's provenance files in its results dir (one level deep)."""
    def find(pattern: str) -> Optional[Path]:
        hits = [p for p in results_dir.glob(pattern) if ".prev-" not in p.name]
        hits += [p for p in results_dir.glob(f"*/{pattern}") if ".prev-" not in p.name]
        return _first(hits)

    return {
        "conditions": find("*calibration_conditions.json"),
        "resolved_config": find("resolved_config.yaml"),
        "manifest": find("run_manifest.json"),
        "forcing_correction": find("forcing_correction.json"),
        "metrics": find("metrics.json"),
        "final_evaluation": find("*_final_evaluation.json"),
    }


def _format_correction(fc: Mapping[str, Any]) -> str:
    mode = fc.get("mode") or "none"
    applied = fc.get("applied") or {}
    if not applied:
        return str(mode)
    vals = ", ".join(f"{k}={float(v):.3g}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in applied.items())
    return f"{mode} ({vals})"


def member_conditions(name: str, results_dir: Optional[Path] = None,
                      config_path: Optional[Path] = None,
                      overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The comparison conditions of one member, ``"unknown"`` where not recorded."""
    cond: Dict[str, Any] = {
        "member": name, "model": UNKNOWN, "forcing_dataset": UNKNOWN,
        "forcing_correction": UNKNOWN, "discretisation": UNKNOWN, "n_hrus": UNKNOWN,
        "calibration_algorithm": UNKNOWN, "evaluations_used": UNKNOWN, "n_params": UNKNOWN,
        "seed": UNKNOWN, "budget_mode": UNKNOWN, "config_sha256": UNKNOWN,
    }
    files: Dict[str, Optional[Path]] = _member_files(results_dir) if results_dir and results_dir.is_dir() else {}
    # A *member's* config file, read as plain YAML for reporting: it is not
    # the running config, so there is no typed object to read it through.
    member_settings: Dict[str, Any] = {}
    cfg_file = config_path or files.get("resolved_config")
    if cfg_file and Path(cfg_file).is_file():
        member_settings = _flat_config(Path(cfg_file))
    if member_settings:
        model = member_settings.get("HYDROLOGICAL_MODEL")
        if model:
            cond["model"] = ",".join(model) if isinstance(model, list) else str(model)
        if member_settings.get("FORCING_DATASET"):
            cond["forcing_dataset"] = str(member_settings["FORCING_DATASET"])
        disc = [str(member_settings[k]) for k in ("DOMAIN_DEFINITION_METHOD", "SUB_GRID_DISCRETIZATION") if member_settings.get(k)]
        if disc:
            cond["discretisation"] = " / ".join(disc)
        if member_settings.get("ITERATIVE_OPTIMIZATION_ALGORITHM"):
            cond["calibration_algorithm"] = str(member_settings["ITERATIVE_OPTIMIZATION_ALGORITHM"])
        if member_settings.get("RANDOM_SEED") is not None:
            cond["seed"] = member_settings["RANDOM_SEED"]
        if member_settings.get("CALIBRATION_BUDGET_MODE"):
            cond["budget_mode"] = member_settings["CALIBRATION_BUDGET_MODE"]
        # Shared FORCING_CORRECTION section (when the config has one).
        if member_settings.get("FORCING_CORRECTION_MODE"):
            cond["forcing_correction"] = _format_correction({
                "mode": member_settings["FORCING_CORRECTION_MODE"],
                "applied": {k.replace("FORCING_CORRECTION_", "").lower(): member_settings[k]
                            for k in ("FORCING_CORRECTION_PRECIP_MULTIPLIER",
                                      "FORCING_CORRECTION_SNOW_MULTIPLIER",
                                      "FORCING_CORRECTION_TEMP_OFFSET")
                            if member_settings.get(k) is not None},
            })
    if files.get("manifest"):
        man = _load_yaml_or_json(files["manifest"])  # type: ignore[arg-type]
        sha = (man.get("resolved_config") or {}).get("sha256")
        if sha:
            cond["config_sha256"] = sha
        if man.get("calibration"):
            last = man["calibration"][-1]
            cond["seed"] = last.get("random_seed", cond["seed"])
    if files.get("conditions"):
        cc = _load_yaml_or_json(files["conditions"])  # type: ignore[arg-type]
        for src, dst in (("model", "model"), ("algorithm", "calibration_algorithm"),
                         ("evaluations_used", "evaluations_used"), ("n_params", "n_params"),
                         ("random_seed", "seed"), ("resolved_config_sha256", "config_sha256")):
            if cc.get(src) is not None:
                cond[dst] = cc[src]
        if isinstance(cc.get("budget"), dict) and cc["budget"].get("mode"):
            cond["budget_mode"] = cc["budget"]["mode"]
        if cc.get("random_seed_source") == "drawn":
            cond["seed"] = f"{cc['random_seed']} (drawn)"
    if files.get("forcing_correction"):
        cond["forcing_correction"] = _format_correction(
            _load_yaml_or_json(files["forcing_correction"]))  # type: ignore[arg-type]
    if files.get("metrics"):
        # Per-run metrics.json (written by the run itself): take any condition
        # fields it carries; metric values are reported, not re-used.
        mj = _load_yaml_or_json(files["metrics"])  # type: ignore[arg-type]
        for key in ("n_hrus", "n_params", "evaluations_used", "forcing_dataset", "model"):
            if mj.get(key) is not None and cond.get(key) == UNKNOWN:
                cond[key] = mj[key]
    for key, value in (overrides or {}).items():
        cond[key] = value
    return cond


@dataclass
class EnsembleInputs:
    """Members, observations and periods for :func:`run_ensemble`."""

    members: Dict[str, pd.Series]
    observed: pd.Series
    periods: Dict[str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]]
    votes: Dict[str, float] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)


def _periods_from_config(config_path: Path) -> Dict[str, PeriodLike]:
    member_settings = _flat_config(config_path)
    return {
        "calibration": member_settings.get("CALIBRATION_PERIOD"),
        "evaluation": member_settings.get("EVALUATION_PERIOD"),
        "test": member_settings.get("TEST_PERIOD"),
    }


def load_manifest(path: Union[str, Path], config_path: Optional[Path] = None) -> Tuple[EnsembleInputs, Dict[str, Any]]:
    """Read an ensemble manifest YAML.

    Format (paths relative to the manifest)::

        observations: {file: obs.csv, column: discharge_cms}   # or just a path
        periods:                       # or omit and pass a SYMFLUENCE config
          calibration: 2000-01-01, 2009-12-31
          evaluation: 2010-01-01, 2015-12-31
          test: 2016-01-01, 2020-12-31  # optional
        options: {weighting: kge, median: true, resample: D}
        members:
          - name: GR4J
            simulated: gr/streamflow.csv     # CSV or NetCDF (+ variable/select)
            column: q_sim                    # optional
            results_dir: gr/optimization/GR/dds_run1   # optional, for conditions
            config: gr/config.yaml           # optional
            votes: 2                         # optional fixed vote weight
            conditions: {n_hrus: 1}          # optional explicit values
    """
    path = Path(path)
    base = path.parent
    data = _load_yaml_or_json(path)

    def rel(p: Any) -> Optional[Path]:
        if p in (None, ""):
            return None
        p = Path(str(p)).expanduser()
        return p if p.is_absolute() else base / p

    obs_spec = data.get("observations")
    if obs_spec is None:
        raise ValueError(f"{path}: 'observations' is required")
    if not isinstance(obs_spec, dict):
        obs_spec = {"file": obs_spec}
    observed = read_series(rel(obs_spec["file"]), column=obs_spec.get("column"),  # type: ignore[arg-type]
                           variable=obs_spec.get("variable"), select=obs_spec.get("select"),
                           hints=_OBS_COLUMN_HINTS)

    raw_periods: Dict[str, PeriodLike] = {}
    cfg_for_periods = config_path or rel(data.get("config"))
    if cfg_for_periods:
        raw_periods.update(_periods_from_config(cfg_for_periods))
    for k, v in (data.get("periods") or {}).items():
        raw_periods[k] = v
    periods = {k: parse_period(raw_periods.get(k)) for k in PERIOD_NAMES}

    members: Dict[str, pd.Series] = {}
    votes: Dict[str, float] = {}
    conditions: List[Dict[str, Any]] = []
    for entry in data.get("members") or []:
        name = str(entry["name"])
        if name in members:
            raise ValueError(f"Duplicate member name {name!r}")
        results_dir = rel(entry.get("results_dir"))
        sim_path = rel(entry.get("simulated"))
        if sim_path is None:
            raise ValueError(f"Member {name!r}: 'simulated' (discharge file) is required")
        members[name] = read_series(sim_path, column=entry.get("column"),
                                    variable=entry.get("variable"), select=entry.get("select"))
        if entry.get("votes") is not None:
            votes[name] = float(entry["votes"])
        conditions.append(member_conditions(name, results_dir, rel(entry.get("config")),
                                            entry.get("conditions")))
    if len(members) < 2:
        raise ValueError(f"{path}: an ensemble needs at least two members")
    return EnsembleInputs(members, observed, periods, votes, conditions), dict(data.get("options") or {})


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def _in_period(index: pd.DatetimeIndex, period: Optional[Tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    if period is None:
        return np.zeros(len(index), dtype=bool)
    return np.asarray((index >= period[0]) & (index <= period[1]))


def align(members: Mapping[str, pd.Series], observed: pd.Series,
          resample: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Members as columns on their common time steps, and obs on the same index."""
    sims = pd.DataFrame(dict(members))
    obs = observed.copy()
    if resample:
        sims = sims.resample(resample).mean()
        obs = obs.resample(resample).mean()
    sims = sims.dropna(how="any")
    obs = obs.reindex(sims.index)
    return sims, obs


def fit_weights(sims: pd.DataFrame, observed: pd.Series,
                calibration: Tuple[pd.Timestamp, pd.Timestamp],
                method: str = "kge") -> pd.Series:
    """Skill weights fitted on the calibration period only.

    ``kge``: ``w_i ∝ max(KGE_i, 0)`` -- members no better than KGE 0 get
    no weight. ``inverse_mse``: ``w_i ∝ 1 / MSE_i``. Weights sum to 1; if no
    member has positive skill they fall back to equal weights. Only time
    steps inside *calibration* with an observation are used, so the
    evaluation and test periods never influence the weights.
    """
    if method not in WEIGHTING_METHODS:
        raise ValueError(f"Unknown weighting {method!r}; expected one of {WEIGHTING_METHODS}")
    mask = _in_period(sims.index, calibration) & observed.notna().to_numpy()
    if mask.sum() < 2:
        raise ValueError("No overlapping observations in the calibration period to fit weights")
    obs = observed.to_numpy()[mask]
    raw = {}
    for name in sims.columns:
        sim = sims[name].to_numpy()[mask]
        if method == "kge":
            score = kge(obs, sim)
            raw[name] = max(float(score), 0.0) if np.isfinite(score) else 0.0
        else:
            mse = float(np.mean((obs - sim) ** 2))
            raw[name] = 1.0 / mse if mse > 0 else np.inf
    w = pd.Series(raw, dtype=float)
    if np.isinf(w).any():  # a perfect member takes all the weight
        w = np.isinf(w).astype(float)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / total


def combine(sims: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Weighted mean of the member columns (weights renormalised to 1)."""
    w = pd.Series(weights, dtype=float).reindex(sims.columns).fillna(0.0)
    if w.sum() <= 0:
        raise ValueError("All combination weights are zero")
    return sims.mul(w / w.sum(), axis=1).sum(axis=1)


def score(sim: pd.Series, observed: pd.Series,
          period: Optional[Tuple[pd.Timestamp, pd.Timestamp]]) -> Dict[str, float]:
    """KGE, NSE, RMSE and PBIAS of *sim* over *period* (NaN when unscorable)."""
    mask = _in_period(sim.index, period) & observed.notna().to_numpy() & sim.notna().to_numpy()
    out = {"KGE": np.nan, "NSE": np.nan, "RMSE": np.nan, "PBIAS": np.nan, "n": int(mask.sum())}
    if mask.sum() < 2:
        return out
    o, s = observed.to_numpy()[mask], sim.to_numpy()[mask]
    out.update(KGE=float(kge(o, s)), NSE=float(nse(o, s)), RMSE=float(rmse(o, s)),
               PBIAS=float(pbias(o, s)))
    return out


@dataclass
class EnsembleResult:
    """Output of :func:`run_ensemble`."""

    series: pd.DataFrame            # members + combinations (columns)
    scores: pd.DataFrame            # rows: series, columns: period x metric
    weights: Dict[str, Dict[str, float]]
    conditions: pd.DataFrame
    files: Dict[str, str] = field(default_factory=dict)


def run_ensemble(
    members: Optional[Mapping[str, pd.Series]] = None,
    observed: Optional[pd.Series] = None,
    periods: Optional[Mapping[str, PeriodLike]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    weighting: str = "kge",
    median: bool = False,
    votes: Optional[Mapping[str, float]] = None,
    conditions: Optional[Sequence[Mapping[str, Any]]] = None,
    resample: Optional[str] = None,
    manifest: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    figure: bool = True,
) -> EnsembleResult:
    """Combine member discharge series and score members and combinations.

    Pass either ``members``/``observed``/``periods`` directly or a
    ``manifest`` YAML (see :func:`load_manifest`); ``config_path`` (a
    SYMFLUENCE config) supplies CALIBRATION/EVALUATION/TEST_PERIOD when the
    manifest has no ``periods``. Writes ``ensemble_scores.csv``,
    ``ensemble_comparison.json``, ``comparison_conditions.csv``,
    ``ensemble_series.csv`` and ``ensemble_comparison.png`` into
    ``output_dir`` when given.
    """
    if manifest is not None:
        inputs, options = load_manifest(manifest, Path(config_path) if config_path else None)
        members, observed = inputs.members, inputs.observed
        parsed = dict(inputs.periods)
        votes = votes or inputs.votes
        conditions = conditions or inputs.conditions
        weighting = options.get("weighting", weighting)
        median = bool(options.get("median", median))
        resample = options.get("resample", resample)
    else:
        if members is None or observed is None:
            raise ValueError("Pass members and observed, or a manifest")
        raw = dict(periods or {})
        if config_path and not raw:
            raw = _periods_from_config(Path(config_path))
        parsed = {k: parse_period(raw.get(k)) for k in PERIOD_NAMES}
    if len(members) < 2:
        raise ValueError("An ensemble needs at least two members")
    if parsed.get("calibration") is None:
        raise ValueError("A calibration period is required to fit the skill weights")

    sims, obs = align(members, observed, resample=resample)
    if sims.empty:
        raise ValueError("Members have no time steps in common")

    names = list(sims.columns)
    weights: Dict[str, Dict[str, float]] = {"equal_weight": {n: 1.0 / len(names) for n in names}}
    skill = fit_weights(sims, obs, parsed["calibration"], method=weighting)  # type: ignore[arg-type]
    weights["skill_weighted"] = {n: float(skill[n]) for n in names}
    combos = {
        "equal_weight": sims.mean(axis=1),
        "skill_weighted": combine(sims, weights["skill_weighted"]),
    }
    if median:
        combos["median"] = sims.median(axis=1)
    if votes:
        v = {n: float(votes.get(n, 0.0)) for n in names}
        if sum(v.values()) > 0:
            weights["vote_weighted"] = {n: x / sum(v.values()) for n, x in v.items()}
            combos["vote_weighted"] = combine(sims, v)

    series = pd.concat([sims, pd.DataFrame(combos)], axis=1)
    rows = {}
    for col in series.columns:
        row: Dict[str, float] = {}
        for pname in PERIOD_NAMES:
            if parsed.get(pname) is None:
                continue
            for metric, value in score(series[col], obs, parsed[pname]).items():
                row[f"{pname}_{metric}"] = value
        rows[col] = row
    scores = pd.DataFrame.from_dict(rows, orient="index")
    scores.index.name = "series"
    scores.insert(0, "kind", ["member" if c in names else "combination" for c in scores.index])

    cond_rows = list(conditions or [])
    known = {c.get("member") for c in cond_rows}
    cond_rows += [member_conditions(n) for n in names if n not in known]
    cond_df = pd.DataFrame(cond_rows).set_index("member").reindex(names)

    result = EnsembleResult(series=series, scores=scores, weights=weights, conditions=cond_df)
    if output_dir is not None:
        result.files = _write_outputs(result, obs, parsed, Path(output_dir), weighting, figure)
    return result


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    return value


def _write_outputs(result: EnsembleResult, obs: pd.Series,
                   periods: Mapping[str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]],
                   out: Path, weighting: str, figure: bool) -> Dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "scores": out / "ensemble_scores.csv",
        "conditions": out / "comparison_conditions.csv",
        "series": out / "ensemble_series.csv",
        "json": out / "ensemble_comparison.json",
    }
    result.scores.to_csv(files["scores"])
    result.conditions.to_csv(files["conditions"])
    series = result.series.copy()
    series.insert(0, "observed", obs)
    series.to_csv(files["series"], index_label="time")
    payload = {
        "periods": {k: None if v is None else [str(v[0]), str(v[1])] for k, v in periods.items()},
        "weighting": weighting,
        "weights_fitted_on": "calibration",
        "weights": result.weights,
        "scores": {s: {k: _jsonable(v) for k, v in row.items()}
                   for s, row in result.scores.to_dict(orient="index").items()},
        "conditions": {m: {k: _jsonable(v) for k, v in row.items()}
                       for m, row in result.conditions.to_dict(orient="index").items()},
    }
    with open(files["json"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    if figure:
        fig_path = _plot(result, out / "ensemble_comparison.png")
        if fig_path is not None:
            files["figure"] = fig_path
    return {k: str(v) for k, v in files.items()}


# Categorical slots 1-3 of the default data-viz palette, one per period.
_PERIOD_COLOURS = {"calibration": "#2a78d6", "evaluation": "#eb6834", "test": "#1baf7a"}


def _plot(result: EnsembleResult, path: Path) -> Optional[Path]:
    """Grouped horizontal KGE bars: one row per member/combination, one bar per period."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    periods = [p for p in PERIOD_NAMES if f"{p}_KGE" in result.scores.columns]
    if not periods:
        return None
    labels = list(result.scores.index)
    n = len(labels)
    height = 0.8 / len(periods)
    fig, ax = plt.subplots(figsize=(7, 0.45 * n * len(periods) + 1.5))
    y = np.arange(n)
    for i, p in enumerate(periods):
        vals = result.scores[f"{p}_KGE"].to_numpy(dtype=float)
        ax.barh(y + (i - (len(periods) - 1) / 2) * height, np.nan_to_num(vals, nan=0.0),
                height=height * 0.9, color=_PERIOD_COLOURS[p], label=p)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab} *" if result.scores.loc[lab, "kind"] == "combination" else lab
                        for lab in labels])
    ax.invert_yaxis()
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_xlabel("KGE")
    ax.set_title("Members and combinations (* = combination; weights fitted on calibration)",
                 fontsize=9, pad=22)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(periods), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
