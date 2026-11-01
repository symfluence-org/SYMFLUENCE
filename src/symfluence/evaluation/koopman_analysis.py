# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Koopman operator analysis for multi-model hydrological ensembles.

Approximates the Koopman operator of a watershed system using Extended Dynamic
Mode Decomposition (EDMD) with structurally diverse model outputs as lifting
functions (dictionary). Extracts dominant dynamical modes, timescales, and
mode loadings to reveal how different models capture different aspects of
watershed dynamics.

The dictionary uses only model ensemble outputs (no observed streamflow),
ensuring a fair comparison with the ensemble mean. Observed streamflow is
predicted via Ridge regression from the Koopman eigenspace, giving the
Koopman predictor the same information as the ensemble mean.

Architecture:
    This module follows the same pattern as SensitivityAnalyzer and Benchmarker:
    - Standalone class with __init__(config, logger, reporting_manager)
    - Main entry point: run_koopman_analysis() -> Dict[str, Any]
    - Registered with the unified registry for discovery by AnalysisManager

Registration:
    @R.koopman_analyzers.add('DEFAULT')
    class KoopmanAnalyzer: ...

Example:
    >>> analyzer = KoopmanAnalyzer(config, logger)
    >>> results = analyzer.run_koopman_analysis(ensemble_df, obs_streamflow)
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.mixins import ConfigMixin
from symfluence.core.registries import R


@R.koopman_analyzers.add('DEFAULT')
class KoopmanAnalyzer(ConfigMixin):
    """Koopman operator analysis of multi-model hydrological ensembles.

    Uses Extended Dynamic Mode Decomposition (EDMD) to approximate the
    Koopman operator of a watershed, with model ensemble outputs serving
    as the dictionary (lifting functions). Extracts eigenvalues, modes,
    timescales, and mode loadings for physical interpretation.

    The dictionary contains only model outputs (no observed streamflow),
    ensuring a fair comparison with the ensemble mean. Observed streamflow
    is predicted via Ridge regression from the Koopman eigenspace.

    Attributes:
        config: SymfluenceConfig or dict
        logger: Logger instance
        reporting_manager: Optional ReportingManager
        hankel_d: Hankel delay depth for model outputs (1 = no embedding)
        svd_threshold: Cumulative energy threshold for SVD rank selection
        dmd_method: 'standard' or 'fbdmd'
        output_dir: Directory for saving results
    """

    def __init__(
        self,
        config: Any,
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        hankel_d: int = 1,
        svd_threshold: float = 0.99,
        dmd_method: str = "fbdmd",
    ):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.reporting_manager = reporting_manager
        self.hankel_d = hankel_d
        self.svd_threshold = svd_threshold
        self.dmd_method = dmd_method

        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir,
            default=tempfile.gettempdir(),
            dict_key="SYMFLUENCE_DATA_DIR",
        ))
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default="domain",
            dict_key="DOMAIN_NAME",
        )
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_dir = self.project_dir / "reporting" / "koopman_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core DMD algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def delay_embed(series: np.ndarray, n_lags: int) -> np.ndarray:
        """Build Hankel (delay-embedding) matrix from 1-D series."""
        T = len(series)
        indices = np.arange(n_lags)[None, :] + np.arange(T - n_lags + 1)[:, None]
        return series[indices]

    @staticmethod
    def _select_rank(sigma: np.ndarray, threshold: float) -> int:
        """Select SVD truncation rank by cumulative energy threshold."""
        energy = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
        return int(np.searchsorted(energy, threshold) + 1)

    def dmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Exact Dynamic Mode Decomposition.

        Parameters
        ----------
        X : (n, m) snapshot matrix
        Y : (n, m) shifted snapshots
        rank : explicit rank override (None = auto from SVD threshold)

        Returns
        -------
        dict with K_tilde, eigenvalues, modes, amplitudes, U, Sigma, rank
        """
        U, sigma, Vh = np.linalg.svd(X, full_matrices=False)
        r = rank if rank is not None else self._select_rank(sigma, self.svd_threshold)
        r = min(r, len(sigma))

        Ur = U[:, :r]
        Sr = sigma[:r]
        Vr = Vh[:r, :].conj().T
        Sr_inv = np.diag(1.0 / Sr)

        K_tilde = Ur.conj().T @ Y @ Vr @ Sr_inv
        eigenvalues, W = np.linalg.eig(K_tilde)
        modes = Y @ Vr @ Sr_inv @ W
        amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]

        return {
            "K_tilde": K_tilde,
            "eigenvalues": eigenvalues,
            "modes": modes,
            "amplitudes": amplitudes,
            "U": Ur,
            "Sigma": Sr,
            "rank": r,
        }

    def fbdmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forward-Backward DMD (fbDMD).

        Averages the forward and backward DMD operators to cancel first-order
        noise bias on eigenvalues. Standard DMD biases eigenvalues toward the
        unit circle because noise in X inflates the pseudoinverse.

        Reference: Dawson et al. (2016)
        """
        U, sigma, Vh = np.linalg.svd(X, full_matrices=False)
        r = rank if rank is not None else self._select_rank(sigma, self.svd_threshold)
        r = min(r, len(sigma))

        Ur = U[:, :r]
        Sr = sigma[:r]
        Vr = Vh[:r, :].conj().T
        Sr_inv = np.diag(1.0 / Sr)

        X_r = Ur.conj().T @ X
        Y_r = Ur.conj().T @ Y

        K_f = Y_r @ Vr @ Sr_inv

        U_yr, s_yr, Vh_yr = np.linalg.svd(Y_r, full_matrices=False)
        tol = max(Y_r.shape) * s_yr[0] * np.finfo(float).eps
        r_yr = np.sum(s_yr > tol)
        s_yr_inv = np.zeros_like(s_yr)
        s_yr_inv[:r_yr] = 1.0 / s_yr[:r_yr]
        Y_r_pinv = Vh_yr.conj().T @ np.diag(s_yr_inv) @ U_yr.conj().T
        K_b = X_r @ Y_r_pinv

        K_tilde = 0.5 * (K_f + np.linalg.inv(K_b))

        eigenvalues, W = np.linalg.eig(K_tilde)
        modes = Y @ Vr @ Sr_inv @ W
        amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]

        return {
            "K_tilde": K_tilde,
            "eigenvalues": eigenvalues,
            "modes": modes,
            "amplitudes": amplitudes,
            "U": Ur,
            "Sigma": Sr,
            "rank": r,
        }

    def _run_dmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Dispatch to the configured DMD method."""
        if self.dmd_method == "fbdmd":
            return self.fbdmd(X, Y, rank=rank)
        return self.dmd(X, Y, rank=rank)

    # ------------------------------------------------------------------
    # Dictionary construction (models-only — fair comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def build_models_only_dictionary(
        ensemble: np.ndarray,
        model_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build dictionary from model outputs only (no observed streamflow).

        This gives the Koopman operator the same information as the ensemble
        mean: model outputs at time t, with no access to observed streamflow.
        """
        return ensemble.copy(), list(model_names)

    def build_hankel_dictionary(
        self,
        ensemble: np.ndarray,
        model_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build Hankel (time-delay embedded) dictionary from model outputs.

        Each model column is delay-embedded with hankel_d lags, giving the
        Koopman operator memory of recent model behavior without requiring
        observed streamflow.
        """
        d = self.hankel_d
        if d <= 1:
            return self.build_models_only_dictionary(ensemble, model_names)

        T, N = ensemble.shape
        embedded_cols = []
        feature_names = []

        for j in range(N):
            H = self.delay_embed(ensemble[:, j], d)
            H = H[:, ::-1]  # col 0 = t-0, col 1 = t-1, ...
            embedded_cols.append(H)
            for lag in range(d):
                feature_names.append(f"{model_names[j]}(t-{lag})")

        dictionary = np.hstack(embedded_cols)
        return dictionary, feature_names

    def build_dictionary(
        self,
        ensemble: np.ndarray,
        model_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build the EDMD dictionary using the configured method.

        Uses models-only or Hankel-embedded model outputs depending on
        hankel_d. No observed streamflow is included in the dictionary.
        """
        if self.hankel_d > 1:
            return self.build_hankel_dictionary(ensemble, model_names)
        return self.build_models_only_dictionary(ensemble, model_names)

    # ------------------------------------------------------------------
    # Koopman eigenspace regression (fair prediction of Q_obs)
    # ------------------------------------------------------------------

    @staticmethod
    def _koopman_project(
        dmd_result: Dict, dictionary: np.ndarray
    ) -> np.ndarray:
        """Project dictionary into Koopman eigen-coordinates via DMD modes.

        Complex Koopman coordinates are split into [Re(z), Im(z)] to
        produce real-valued features for regression.

        Returns (T, 2r) real-valued Koopman coordinates.
        """
        Phi = dmd_result["modes"]
        Z_complex = np.linalg.lstsq(Phi, dictionary.T, rcond=None)[0].T
        return np.column_stack([Z_complex.real, Z_complex.imag])

    @staticmethod
    def _fit_ridge(Z_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit Ridge regression with leave-one-out cross-validation for alpha."""
        y_mean = y.mean()
        y_c = y - y_mean
        Z = Z_aug[:, :-1]
        Z_mean = Z.mean(axis=0)
        Z_c = Z - Z_mean

        U, s, Vt = np.linalg.svd(Z_c, full_matrices=False)
        s2 = s ** 2

        alphas = np.logspace(-2, 6, 50)
        best_alpha = 1.0
        best_cv_error = np.inf

        for alpha in alphas:
            d = s / (s2 + alpha)
            w_ridge = Vt.T @ (d * (U.T @ y_c))
            y_hat = Z_c @ w_ridge

            h_diag = (U ** 2) @ (s2 / (s2 + alpha))
            residuals = y_c - y_hat
            loo_residuals = residuals / (1.0 - h_diag)
            cv_error = np.mean(loo_residuals ** 2)

            if cv_error < best_cv_error:
                best_cv_error = cv_error
                best_alpha = alpha

        d = s / (s2 + best_alpha)
        w_features = Vt.T @ (d * (U.T @ y_c))
        intercept = y_mean - Z_mean @ w_features

        return np.append(w_features, intercept)

    def koopman_predict_obs(
        self,
        dmd_result: Dict,
        dictionary: np.ndarray,
        obs_Q: np.ndarray,
    ) -> np.ndarray:
        """Learn regression weights: Q_obs = w^T z + b from Koopman coordinates.

        Parameters
        ----------
        dmd_result : DMD result dict
        dictionary : (T, N) normalized model-only dictionary (training data)
        obs_Q : (T,) observed streamflow for the training period

        Returns
        -------
        weights : regression weights [w; bias]
        """
        Z = self._koopman_project(dmd_result, dictionary)
        Z_aug = np.column_stack([Z, np.ones(Z.shape[0])])
        return self._fit_ridge(Z_aug, obs_Q)

    @staticmethod
    def koopman_apply_regression(
        dmd_result: Dict,
        dictionary: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Apply pre-fit regression weights to predict Q_obs."""
        Phi = dmd_result["modes"]
        Z_complex = np.linalg.lstsq(Phi, dictionary.T, rcond=None)[0].T
        Z = np.column_stack([Z_complex.real, Z_complex.imag])
        Z_aug = np.column_stack([Z, np.ones(Z.shape[0])])
        return Z_aug @ weights

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def eigenvalues_to_continuous(lam: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Discrete eigenvalues -> continuous-time omega = log(lambda)/dt."""
        lam_safe = np.where(np.abs(lam) > 1e-15, lam, 1e-15)
        return np.log(lam_safe) / dt

    @staticmethod
    def extract_timescales(omega: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract decay_days and period_days from continuous eigenvalues."""
        with np.errstate(divide="ignore", invalid="ignore"):
            decay_days = np.where(
                np.abs(omega.real) > 1e-15, -1.0 / omega.real, np.inf
            )
            period_days = np.where(
                np.abs(omega.imag) > 1e-10, 2 * np.pi / np.abs(omega.imag), np.inf
            )
        return {"decay_days": decay_days, "period_days": period_days}

    @staticmethod
    def mode_importance(
        eigenvalues: np.ndarray, amplitudes: np.ndarray, n_steps: int
    ) -> np.ndarray:
        """Rank modes by time-averaged amplitude x persistence."""
        abs_lam = np.abs(eigenvalues)
        abs_b = np.abs(amplitudes)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_exp = np.minimum(
                abs_lam ** np.minimum(
                    n_steps, 700 / np.maximum(np.log(abs_lam + 1e-15), 1e-15)
                ),
                1e100,
            )
            geo_sum = np.where(
                np.abs(abs_lam - 1.0) > 1e-10,
                np.abs(1.0 - safe_exp) / np.abs(1.0 - abs_lam),
                float(n_steps),
            )
        return abs_b * geo_sum / n_steps

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_kge(sim: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
        """KGE, NSE, r, alpha, beta from aligned arrays."""
        mask = np.isfinite(sim) & np.isfinite(obs)
        s, o = sim[mask], obs[mask]
        if len(s) < 10:
            return {"KGE": np.nan, "NSE": np.nan, "r": np.nan,
                    "alpha": np.nan, "beta": np.nan}
        r = np.corrcoef(s, o)[0, 1]
        alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
        beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        ss_res = np.sum((o - s) ** 2)
        ss_tot = np.sum((o - np.mean(o)) ** 2)
        nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"KGE": kge, "NSE": nse, "r": r, "alpha": alpha, "beta": beta}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_koopman_analysis(
        self,
        ensemble_df: pd.DataFrame,
        obs_streamflow: pd.Series,
        train_end: str = "2007-12-31",
        eval_start: str = "2008-01-01",
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the complete Koopman operator analysis pipeline.

        Uses a models-only dictionary (no observed streamflow) to ensure
        fair comparison with the ensemble mean. Predicts Q_obs via Ridge
        regression from the Koopman eigenspace.

        Parameters
        ----------
        ensemble_df : (T, N) DataFrame of model outputs (m^3/s), aligned daily
        obs_streamflow : (T,) Series of observed streamflow (m^3/s)
        train_end : last training date
        eval_start : first evaluation date
        rank : explicit DMD rank (None = auto)

        Returns
        -------
        dict with eigenvalues, timescales, metrics, mode_loadings, etc.
        """
        self.logger.info("Starting Koopman operator analysis")

        model_names = list(ensemble_df.columns)
        n_models = len(model_names)

        # Build models-only dictionary (no observed streamflow)
        dictionary, feature_names = self.build_dictionary(
            ensemble_df.values, model_names
        )
        if self.hankel_d > 1:
            dates = obs_streamflow.index[self.hankel_d - 1:]
        else:
            dates = obs_streamflow.index

        self.logger.info(
            f"Dictionary: {dictionary.shape} "
            f"({n_models} models, hankel_d={self.hankel_d}, "
            f"dmd={self.dmd_method})"
        )

        # Split train/eval
        train_mask = dates <= pd.Timestamp(train_end)
        eval_mask = dates >= pd.Timestamp(eval_start)
        train_raw = dictionary[train_mask]
        eval_raw = dictionary[eval_mask]

        obs_train = obs_streamflow.loc[dates[train_mask]].values
        obs_eval = obs_streamflow.loc[dates[eval_mask]].values

        # Normalize from training data only
        means = train_raw.mean(axis=0)
        stds = train_raw.std(axis=0)
        stds[stds < 1e-12] = 1.0
        train = (train_raw - means) / stds
        eval_ = (eval_raw - means) / stds

        # DMD on training data
        X_train = train[:-1, :].T
        Y_train = train[1:, :].T
        result = self._run_dmd(X_train, Y_train, rank=rank)
        self.logger.info(f"DMD rank: {result['rank']}")

        # Eigenstructure analysis
        omega = self.eigenvalues_to_continuous(result["eigenvalues"])
        timescales = self.extract_timescales(omega)
        importance = self.mode_importance(
            result["eigenvalues"], result["amplitudes"], train.shape[0]
        )

        # Mode loadings
        abs_modes = np.abs(result["modes"])
        col_max = abs_modes.max(axis=0)
        col_max[col_max < 1e-15] = 1.0
        loadings = abs_modes / col_max

        # Koopman eigenspace regression: predict Q_obs from model outputs
        weights = self.koopman_predict_obs(result, train, obs_train)
        Q_koopman_eval = self.koopman_apply_regression(result, eval_, weights)

        eval_metrics = self.compute_kge(Q_koopman_eval, obs_eval)

        # Ensemble mean for comparison (denormalized)
        ens_eval_denorm = eval_ * stds + means
        ens_mean_eval = ens_eval_denorm.mean(axis=1)
        ens_metrics = self.compute_kge(ens_mean_eval, obs_eval)

        self.logger.info(
            f"Koopman eval KGE={eval_metrics['KGE']:.3f}, "
            f"NSE={eval_metrics['NSE']:.3f}"
        )
        self.logger.info(
            f"Ensemble mean eval KGE={ens_metrics['KGE']:.3f}, "
            f"NSE={ens_metrics['NSE']:.3f}"
        )

        # Assemble results
        results = {
            "eigenvalues": result["eigenvalues"],
            "modes": result["modes"],
            "amplitudes": result["amplitudes"],
            "rank": result["rank"],
            "singular_values": result["Sigma"],
            "decay_days": timescales["decay_days"],
            "period_days": timescales["period_days"],
            "importance": importance,
            "loadings": pd.DataFrame(
                loadings, index=feature_names,
                columns=[f"Mode_{j+1}" for j in range(result["rank"])]
            ),
            "feature_names": feature_names,
            "eval_metrics": eval_metrics,
            "ens_mean_metrics": ens_metrics,
            "normalization": {"means": means, "stds": stds},
            "regression_weights": weights,
            "dmd_method": self.dmd_method,
            "hankel_d": self.hankel_d,
        }

        # Save to disk
        self._save_results(results)

        # Visualize
        if self.reporting_manager:
            try:
                self.reporting_manager.visualize_koopman_analysis(results, self.output_dir)
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"Koopman visualization failed: {e}")

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Persist results to disk."""
        np.savez(
            self.output_dir / "koopman_results.npz",
            eigenvalues=results["eigenvalues"],
            modes=results["modes"],
            amplitudes=results["amplitudes"],
            rank=results["rank"],
            singular_values=results["singular_values"],
            decay_days=results["decay_days"],
            period_days=results["period_days"],
            importance=results["importance"],
            regression_weights=results["regression_weights"],
            dmd_method=results["dmd_method"],
            hankel_d=results["hankel_d"],
        )
        results["loadings"].to_csv(self.output_dir / "mode_loadings.csv")

        mode_table = pd.DataFrame({
            "eigenvalue_abs": np.abs(results["eigenvalues"]),
            "decay_days": results["decay_days"],
            "period_days": results["period_days"],
            "importance": results["importance"],
        })
        mode_table.to_csv(self.output_dir / "mode_summary.csv", index=False)
        self.logger.info(f"Koopman results saved to {self.output_dir}")
