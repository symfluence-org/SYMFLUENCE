# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Uncertainty post-processing for sampling algorithms (DREAM, GLUE).

After a DREAM or GLUE run this:

1. persists the posterior / behavioural parameter samples with their
   objective values and weights to ``<slug>_uq_samples.nc``;
2. re-runs up to ``UQ_PREDICTIVE_RUNS`` of those parameter sets over the full
   simulation period and stores the simulated series in the same file;
3. writes weighted predictive quantiles (``UQ_QUANTILES``) of those runs to
   ``<slug>_predictive_quantiles.csv`` -- for a formal-likelihood DREAM run
   both the parameter-only band and the total band (parameter + Box-Cox AR(1)
   residual error, ``UQ_RESIDUAL_DRAWS`` realisations per set);
4. scores each band on the calibration and evaluation periods separately
   (coverage and width of the 90% band, CRPS) in
   ``<slug>_predictive_scores.json``.

Weights: DREAM posterior draws are equally weighted (MCMC samples already
follow the posterior); GLUE behavioural runs carry their normalised GLUE
likelihoods (Beven & Binley, 1992).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.metrics.formal_likelihood import (
    boxcox_ar1_error_parameters,
    sample_boxcox_ar1_predictive,
)
from symfluence.core.metrics.probabilistic import predictive_scores, weighted_quantiles
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics
from symfluence.core.modeling.utilities.periods import parse_period

if TYPE_CHECKING:
    from .base_model_optimizer import BaseModelOptimizer

__all__ = ["UncertaintyPostprocessor", "extract_uq_samples"]

#: Band scored for coverage / width (the 90% central interval).
SCORED_BAND = (0.05, 0.95)


def extract_uq_samples(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalised samples, objective values and weights from an algorithm result.

    Returns None when the result carries no posterior / behavioural samples.
    """
    post = result.get('posterior_samples')
    if post is not None and np.size(post) > 0:
        samples = np.atleast_2d(np.asarray(post, dtype=float))
        n = samples.shape[0]
        n_chains = int(result.get('n_chains') or 1)
        return {
            'kind': 'posterior',
            'samples': samples,
            'objective': np.asarray(result.get('posterior_fitness', np.full(n, np.nan)), dtype=float),
            'log_likelihood': np.asarray(
                result.get('posterior_log_likelihood', np.full(n, np.nan)), dtype=float),
            'weights': np.full(n, 1.0 / n),
            'chain': np.arange(n) % n_chains,
            'draw': np.arange(n) // n_chains,
            'formal': result.get('likelihood') == 'formal',
        }
    behav = result.get('behavioral_samples')
    if behav is not None and np.size(behav) > 0:
        samples = np.atleast_2d(np.asarray(behav, dtype=float))
        n = samples.shape[0]
        weights = np.asarray(result.get('behavioral_likelihoods', np.full(n, 1.0 / n)), dtype=float)
        return {
            'kind': 'behavioral',
            'samples': samples,
            'objective': np.asarray(result.get('behavioral_fitness', np.full(n, np.nan)), dtype=float),
            'weights': weights / weights.sum(),
            'formal': False,
        }
    return None


class UncertaintyPostprocessor:
    """Persist samples and build/score predictive intervals for one run."""

    def __init__(self, optimizer: 'BaseModelOptimizer', logger: Optional[logging.Logger] = None):
        self.opt = optimizer
        self.logger = logger or optimizer.logger

    # ------------------------------------------------------------------ config
    def _cfg(self, accessor, key):
        return self.opt._get_config_value(accessor, dict_key=key)

    @property
    def save_samples(self) -> bool:
        return bool(self._cfg(lambda: self.opt.config.optimization.uncertainty.save_samples,
                              'UQ_SAVE_SAMPLES'))

    @property
    def predictive_runs(self) -> int:
        return int(self._cfg(lambda: self.opt.config.optimization.uncertainty.predictive_runs,
                             'UQ_PREDICTIVE_RUNS') or 0)

    @property
    def quantiles(self) -> List[float]:
        qs = self._cfg(lambda: self.opt.config.optimization.uncertainty.quantiles, 'UQ_QUANTILES')
        return sorted(float(q) for q in (qs or [0.05, 0.25, 0.5, 0.75, 0.95]))

    @property
    def residual_draws(self) -> int:
        return int(self._cfg(lambda: self.opt.config.optimization.uncertainty.residual_draws,
                             'UQ_RESIDUAL_DRAWS') or 0)

    # -------------------------------------------------------------------- main
    def run(self, result: Dict[str, Any], file_slug: str) -> Dict[str, Path]:
        """Run every enabled step; returns the files written. Never raises."""
        written: Dict[str, Path] = {}
        uq = extract_uq_samples(result)
        if uq is None:
            return written
        try:
            names = list(self.opt.param_manager.all_param_names)
            physical = self._physical_values(uq['samples'], names)
            n_runs = min(self.predictive_runs, uq['samples'].shape[0])
            members, sims = (np.array([], dtype=int), None)
            if n_runs > 0:
                members = self._select_members(uq, n_runs)
                sims = self._simulate_members(uq['samples'], members)
            if self.save_samples:
                written['samples'] = self._write_samples(
                    uq, names, physical, members, sims, file_slug)
            if sims is not None and sims[0] is not None:
                written.update(self._write_predictive(uq, members, sims, file_slug))
        except Exception as e:  # noqa: BLE001 — post-processing must not fail a finished calibration
            self.logger.error(f"Uncertainty post-processing failed: {e}", exc_info=True)
        return written

    # ----------------------------------------------------------------- helpers
    def _physical_values(self, samples: np.ndarray, names: List[str]) -> np.ndarray:
        """Physical parameter values; multi-valued parameters report their mean."""
        out = np.full(samples.shape, np.nan)
        for i, row in enumerate(samples):
            params = self.opt.param_manager.denormalize_parameters(row)
            for j, name in enumerate(names):
                value = np.asarray(params.get(name, np.nan), dtype=float)
                out[i, j] = float(np.nanmean(value)) if value.size else np.nan
        return out

    def _rng(self) -> np.random.Generator:
        seed = getattr(self.opt, 'random_seed', None)
        try:
            return np.random.default_rng(None if seed in (None, 'None') else int(seed))
        except (TypeError, ValueError):
            return np.random.default_rng()

    def _select_members(self, uq: Dict[str, Any], n_runs: int) -> np.ndarray:
        """Random subset (without replacement) of the samples to re-run."""
        n = uq['samples'].shape[0]
        if n_runs >= n:
            return np.arange(n)
        return np.sort(self._rng().choice(n, size=n_runs, replace=False))

    def _simulate_members(
        self, samples: np.ndarray, members: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex], Optional[pd.Series], np.ndarray]:
        """Full-period runs; returns (sims (m, T), time, obs, ok mask)."""
        series: List[Optional[pd.Series]] = []
        obs: Optional[pd.Series] = None
        out_dir = self.opt.results_dir / 'final_evaluation'
        for k, idx in enumerate(members):
            self.logger.info(
                f"UQ predictive run {k + 1}/{len(members)} (sample {int(idx)})")
            params = self.opt.param_manager.denormalize_parameters(samples[idx])
            sim = None
            try:
                # Clear the previous member's output so a failed run can never
                # be read back as this member's simulation.
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                if self.opt.run_final_evaluation(params) is not None:
                    sim, obs_k = self._read_series(out_dir)
                    obs = obs if obs is not None else obs_k
            except Exception as e:  # noqa: BLE001 — one failed member must not stop the rest
                self.logger.warning(f"UQ predictive run for sample {int(idx)} failed: {e}")
            series.append(sim)
        ok = np.array([s is not None and len(s) > 0 for s in series], dtype=bool)
        if not ok.any():
            self.logger.warning(
                "No UQ predictive run produced a readable simulation; "
                "predictive intervals are skipped.")
            return None, None, None, ok
        index = pd.DatetimeIndex(sorted(set().union(*[s.index for s, good in zip(series, ok) if good])))
        sims = np.full((len(series), len(index)), np.nan)
        for i, s in enumerate(series):
            if s is not None and len(s):
                sims[i] = s.reindex(index).to_numpy(dtype=float)
        if obs is not None:
            obs = obs.reindex(index)
        return sims, index, obs, ok

    def _read_series(self, out_dir: Path) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Daily simulated and observed series from a final-evaluation directory."""
        target = self.opt.calibration_target
        for attr in ('get_simulation_files', 'extract_simulated_data', '_load_observed_data'):
            if not hasattr(target, attr):
                raise AttributeError(
                    f"calibration target {type(target).__name__} has no {attr}()")
        files = target.get_simulation_files(out_dir)
        if not files:
            return None, None
        sim = target.extract_simulated_data(files)
        obs = target._load_observed_data()
        daily = self.opt._final_eval_daily_series
        sim = daily(sim) if sim is not None and len(sim) else None
        obs = daily(obs) if obs is not None and len(obs) else None
        return sim, obs

    def _periods(self) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
        out: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
        for label, accessor, key in (
            ('Calib', lambda: self.opt.config.domain.calibration_period, 'CALIBRATION_PERIOD'),
            ('Eval', lambda: self.opt.config.domain.evaluation_period, 'EVALUATION_PERIOD'),
        ):
            value = self._cfg(accessor, key)
            if not value:
                continue
            try:
                period = parse_period(str(value), allow_extra=True)
            except (ValueError, TypeError):
                period = None
            if period is None:
                self.logger.warning(f"Cannot parse {key}={value!r}; period not scored")
                continue
            out[label] = (pd.Timestamp(period[0]), pd.Timestamp(period[1]))
        return out

    # ------------------------------------------------------------------ output
    def _write_samples(self, uq, names, physical, members, sims, file_slug) -> Path:
        import xarray as xr

        n = uq['samples'].shape[0]
        data_vars: Dict[str, Any] = {
            'normalized': (('sample', 'parameter'), uq['samples']),
            'value': (('sample', 'parameter'), physical),
            'objective': (('sample',), uq['objective']),
            'weight': (('sample',), uq['weights']),
        }
        for extra in ('log_likelihood', 'chain', 'draw'):
            if extra in uq:
                data_vars[extra] = (('sample',), np.asarray(uq[extra]))
        coords: Dict[str, Any] = {'sample': np.arange(n), 'parameter': names}
        if sims is not None and sims[0] is not None:
            sim_arr, index, obs, ok = sims
            data_vars['member_sample'] = (('member',), members[ok])
            data_vars['simulated'] = (('member', 'time'), sim_arr[ok])
            if obs is not None:
                data_vars['observed'] = (('time',), obs.to_numpy(dtype=float))
            coords['time'] = index
        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs.update({
            'kind': uq['kind'],
            'algorithm': file_slug,
            'objective_metric': str(getattr(self.opt, 'target_metric', '')),
            'description': (
                'DREAM post-burn-in posterior draws (equal weights)'
                if uq['kind'] == 'posterior'
                else 'GLUE behavioural samples with normalised GLUE likelihood weights'
            ),
        })
        path = self.opt.results_dir / f"{file_slug}_uq_samples.nc"
        ds.to_netcdf(path)
        self.logger.info(f"Saved {n} {uq['kind']} samples to {path}")
        return path

    def _write_predictive(self, uq, members, sims, file_slug) -> Dict[str, Path]:
        sim_arr, index, obs, ok = sims
        ensemble = sim_arr[ok]
        weights = uq['weights'][members[ok]]
        bands: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {
            'parameter': (ensemble, weights),
        }
        if uq.get('formal') and self.residual_draws > 0 and obs is not None:
            total = self._total_ensemble(ensemble, obs)
            if total is not None:
                bands['total'] = (total, None)

        qs = self.quantiles
        table = pd.DataFrame(index=index)
        table.index.name = 'time'
        table['observed'] = obs.to_numpy(dtype=float) if obs is not None else np.nan
        for band, (ens, w) in bands.items():
            values = weighted_quantiles(ens, qs, w)
            for q, row in zip(qs, values):
                table[f'{band}_q{q:g}'] = row
        q_path = self.opt.results_dir / f"{file_slug}_predictive_quantiles.csv"
        table.to_csv(q_path)

        scores: Dict[str, Any] = {}
        if obs is not None:
            obs_arr = obs.to_numpy(dtype=float)
            for band, (ens, w) in bands.items():
                scores[band] = {}
                for label, (start, end) in self._periods().items():
                    mask = np.ones(len(index), dtype=bool)
                    if start is not None:
                        mask &= index >= start
                    if end is not None:
                        mask &= index <= end
                    if mask.any():
                        scores[band][label] = predictive_scores(
                            obs_arr[mask], ens[:, mask], w, band=SCORED_BAND)
        s_path = self.opt.results_dir / f"{file_slug}_predictive_scores.json"
        with open(s_path, 'w', encoding='utf-8') as f:
            json.dump({
                'kind': uq['kind'],
                'n_members': int(ok.sum()),
                'band': list(SCORED_BAND),
                'scores': scores,
            }, f, indent=2, default=float)
        for band, periods in scores.items():
            for label, sc in periods.items():
                self.logger.info(
                    f"Predictive ({band}) {label}: coverage90={sc['coverage_90']:.3f}, "
                    f"width90={sc['width_90']:.4g}, CRPS={sc['crps']:.4g}")
        return {'quantiles': q_path, 'scores': s_path}

    def _total_ensemble(self, ensemble: np.ndarray, obs: pd.Series) -> Optional[np.ndarray]:
        """Parameter ensemble plus Box-Cox AR(1) residual realisations."""
        settings = StreamflowMetrics.from_config(self.opt.config).likelihood
        periods = self._periods()
        calib = periods.get('Calib')
        idx = obs.index
        cal_mask = np.ones(len(idx), dtype=bool)
        if calib is not None:
            cal_mask = (idx >= calib[0]) & (idx <= calib[1])
        obs_arr = obs.to_numpy(dtype=float)
        rng = self._rng()
        draws = []
        for sim in ensemble:
            try:
                phi, sigma, eps = boxcox_ar1_error_parameters(
                    obs_arr[cal_mask], sim[cal_mask], settings)
            except ValueError:
                continue
            draws.append(sample_boxcox_ar1_predictive(
                sim, settings.lam, phi, sigma, eps, self.residual_draws, rng))
        if not draws:
            return None
        return np.concatenate(draws, axis=0)
