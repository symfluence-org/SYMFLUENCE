#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Log-Likelihood Functions for Bayesian Calibration

Provides Gaussian log-likelihood computation with support for observation
uncertainty (e.g., FLUXNET _uc measurement errors) and model structural error.
Designed for use with MCMC algorithms (DREAM) and other Bayesian methods
(ABC, GLUE) where a proper likelihood function is needed.

Motivated by CalLMIP Phase 1 protocol requirements (MacBean & Deepak, 2025):
    "Daily measurement errors are also provided [...] based on FLUXNET processing
    (Pastorello et al., 2020). These measurement errors need to be combined with
    modelling errors for the observation error covariance matrix."

References:
    Pastorello, G., et al. (2020). The FLUXNET2015 dataset and the ONEFlux
    processing pipeline. Scientific Data, 7(1), 225.

    Vrugt, J.A. (2016). Markov chain Monte Carlo simulation using the DREAM
    software package. Environ. Model. Softw., 75, 273-316.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def gaussian_log_likelihood(
    obs: np.ndarray,
    sim: np.ndarray,
    obs_uncertainty: Optional[np.ndarray] = None,
    model_error: Optional[Union[float, np.ndarray]] = None,
    model_error_fraction: Optional[float] = None,
    min_total_error: float = 1e-6,
) -> float:
    """
    Compute Gaussian log-likelihood with combined observation and model error.

    The log-likelihood is:
        log L = -0.5 * sum[ (obs_i - sim_i)^2 / sigma_total_i^2
                           + log(2*pi*sigma_total_i^2) ]

    where sigma_total_i^2 = sigma_obs_i^2 + sigma_model_i^2

    The normalization term (log(2*pi*sigma_total^2)) is included so that
    the likelihood correctly penalizes overly large error estimates when
    model error is treated as a calibration parameter.

    Args:
        obs: Observed values (1D array)
        sim: Simulated values (1D array, same length as obs)
        obs_uncertainty: Per-timestep observation standard deviations (sigma_obs).
            If None, observation error is assumed zero (all uncertainty from model).
        model_error: Model structural error standard deviation. Can be:
            - A scalar (constant model error)
            - A 1D array (time-varying model error, same length as obs)
            - None (use model_error_fraction or zero)
        model_error_fraction: If provided, model error is computed as
            model_error_fraction * |sim_i| for each timestep. Ignored if
            model_error is explicitly provided.
        min_total_error: Floor on total error variance to avoid division by zero.
            Default 1e-6.

    Returns:
        Log-likelihood value (negative; closer to 0 = better fit)

    Example:
        >>> obs = np.array([1.0, 2.0, 3.0])
        >>> sim = np.array([1.1, 1.9, 3.2])
        >>> obs_uc = np.array([0.1, 0.15, 0.12])
        >>> ll = gaussian_log_likelihood(obs, sim, obs_uncertainty=obs_uc,
        ...                              model_error_fraction=0.1)
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)

    if obs.shape != sim.shape:
        raise ValueError(
            f"obs and sim must have same shape, got {obs.shape} and {sim.shape}"
        )

    # Remove NaN pairs
    valid = np.isfinite(obs) & np.isfinite(sim)
    if obs_uncertainty is not None:
        obs_uncertainty = np.asarray(obs_uncertainty, dtype=np.float64)
        valid &= np.isfinite(obs_uncertainty)

    n_valid = valid.sum()
    if n_valid == 0:
        logger.warning("No valid data points for likelihood computation")
        return -np.inf

    obs_v = obs[valid]
    sim_v = sim[valid]

    # Build total error variance: sigma_total^2 = sigma_obs^2 + sigma_model^2
    sigma_obs_sq = np.zeros(n_valid)
    if obs_uncertainty is not None:
        uc_v = obs_uncertainty[valid]
        sigma_obs_sq = uc_v ** 2

    sigma_model_sq = np.zeros(n_valid)
    if model_error is not None:
        me = np.asarray(model_error, dtype=np.float64)
        if me.ndim == 0:
            sigma_model_sq = np.full(n_valid, me ** 2)
        else:
            sigma_model_sq = me[valid] ** 2
    elif model_error_fraction is not None:
        sigma_model_sq = (model_error_fraction * np.abs(sim_v)) ** 2

    sigma_total_sq = sigma_obs_sq + sigma_model_sq
    sigma_total_sq = np.maximum(sigma_total_sq, min_total_error)

    # Gaussian log-likelihood (with normalization)
    residuals_sq = (obs_v - sim_v) ** 2
    log_lik = -0.5 * np.sum(
        residuals_sq / sigma_total_sq + np.log(2 * np.pi * sigma_total_sq)
    )

    return float(log_lik)


def heteroscedastic_gaussian_log_likelihood(
    obs: np.ndarray,
    sim: np.ndarray,
    obs_uncertainty: Optional[np.ndarray] = None,
    model_error_base: float = 0.0,
    model_error_fraction: float = 0.0,
    min_total_error: float = 1e-6,
) -> float:
    """
    Gaussian log-likelihood with heteroscedastic (magnitude-dependent) model error.

    Model error is: sigma_model_i = model_error_base + model_error_fraction * |sim_i|

    This is common in hydrology where model errors tend to scale with the
    magnitude of the simulated variable (e.g., higher flows have larger errors).

    Both model_error_base and model_error_fraction can be treated as calibration
    parameters to be estimated alongside physical model parameters.

    Args:
        obs: Observed values
        sim: Simulated values
        obs_uncertainty: Per-timestep observation standard deviations
        model_error_base: Additive (constant) component of model error
        model_error_fraction: Multiplicative component of model error
        min_total_error: Floor on total error variance

    Returns:
        Log-likelihood value
    """
    sim_arr = np.asarray(sim, dtype=np.float64)
    sigma_model = model_error_base + model_error_fraction * np.abs(sim_arr)
    return gaussian_log_likelihood(
        obs, sim, obs_uncertainty=obs_uncertainty,
        model_error=sigma_model, min_total_error=min_total_error
    )


def multivariate_log_likelihood(
    variables: dict[str, dict],
    assume_independent: bool = True,
) -> float:
    """
    Compute joint log-likelihood across multiple observed variables.

    For CalLMIP, this combines likelihoods for NEE, Qle, and Qh:
        log L_total = log L_NEE + log L_Qle + log L_Qh

    This assumes independence between variables (diagonal observation error
    covariance matrix), which is a standard simplification.

    Args:
        variables: Dictionary mapping variable names to their data dicts.
            Each value should contain:
                - 'obs': observed values (np.ndarray)
                - 'sim': simulated values (np.ndarray)
                - 'obs_uncertainty': observation uncertainty (np.ndarray, optional)
                - 'model_error': model error (float or np.ndarray, optional)
                - 'model_error_fraction': fractional model error (float, optional)
        assume_independent: If True (default), sum individual log-likelihoods.
            Full covariance not yet supported.

    Returns:
        Joint log-likelihood value

    Example:
        >>> variables = {
        ...     'NEE': {'obs': nee_obs, 'sim': nee_sim,
        ...             'obs_uncertainty': nee_uc, 'model_error_fraction': 0.1},
        ...     'Qle': {'obs': qle_obs, 'sim': qle_sim,
        ...             'obs_uncertainty': qle_uc, 'model_error_fraction': 0.15},
        ... }
        >>> ll = multivariate_log_likelihood(variables)
    """
    if not assume_independent:
        raise NotImplementedError(
            "Full cross-variable covariance not yet implemented. "
            "Use assume_independent=True for diagonal approximation."
        )

    total_ll = 0.0
    for var_name, var_data in variables.items():
        ll = gaussian_log_likelihood(
            obs=var_data['obs'],
            sim=var_data['sim'],
            obs_uncertainty=var_data.get('obs_uncertainty'),
            model_error=var_data.get('model_error'),
            model_error_fraction=var_data.get('model_error_fraction'),
        )
        logger.debug(f"Log-likelihood for {var_name}: {ll:.2f}")
        total_ll += ll

    return total_ll


def load_fluxnet_uncertainties(
    nc_path: str,
    variable_name: str,
    time_slice: Optional[tuple] = None,
) -> Optional[pd.Series]:
    """
    Load measurement uncertainty from a FLUXNET-format NetCDF file.

    CalLMIP/FLUXNET convention: uncertainty for variable X is stored as X_uc
    (e.g., NEE_uc, Qle_uc, Qh_uc). Values represent standard deviations of
    daily measurement errors from FLUXNET processing (Pastorello et al., 2020).

    Args:
        nc_path: Path to the flux observation NetCDF file.
        variable_name: Base variable name (e.g., 'NEE', 'Qle', 'Qh').
            The function will look for '{variable_name}_uc'.
        time_slice: Optional (start, end) timestamps to subset.

    Returns:
        pandas Series of uncertainty values indexed by time, or None if not found.
    """
    from pathlib import Path

    import xarray as xr

    nc_path = Path(nc_path)
    if not nc_path.exists():
        logger.warning(f"Flux observation file not found: {nc_path}")
        return None

    uc_var_name = f"{variable_name}_uc"

    try:
        with xr.open_dataset(nc_path) as ds:
            if uc_var_name not in ds.data_vars:
                # Try common alternatives
                alternatives = [
                    f"{variable_name}_UC",
                    f"{variable_name}_uncertainty",
                    f"{variable_name}_RANDUNC",
                    f"{variable_name}_JOINTUNC",
                ]
                found = None
                for alt in alternatives:
                    if alt in ds.data_vars:
                        found = alt
                        break

                if found is None:
                    logger.info(
                        f"No uncertainty variable found for {variable_name} "
                        f"in {nc_path}. Tried: {uc_var_name}, {alternatives}"
                    )
                    return None
                uc_var_name = found

            da = ds[uc_var_name]

            # Collapse spatial dims if present
            spatial_dims = [d for d in da.dims if d != 'time']
            for dim in spatial_dims:
                if da.sizes[dim] == 1:
                    da = da.isel({dim: 0})
                else:
                    da = da.mean(dim=dim)

            series = da.to_series()

            if time_slice is not None:
                start, end = time_slice
                series = series.loc[start:end]

            logger.debug(
                f"Loaded {len(series)} uncertainty values for {variable_name} "
                f"from {uc_var_name} (mean={series.mean():.4g})"
            )
            return series

    except Exception as e:  # noqa: BLE001 — optional feature, must not break evaluation
        logger.error(f"Error loading uncertainties from {nc_path}: {e}")
        return None
