# CalLMIP Phase 1 — SUMMA with SYMFLUENCE

Configuration and workflow for the [Calibrated Land Model Intercomparison Project (CalLMIP)](https://callmip-org.github.io) using SUMMA through SYMFLUENCE.

## Overview

CalLMIP Phase 1 requires site-level calibration of land surface models against flux tower observations (NEE, Qle, Qh) with posterior uncertainty quantification.

This example uses **DREAM** (MCMC) for Bayesian calibration with Gaussian log-likelihood that incorporates FLUXNET measurement uncertainties.

## Site: DK-Sor (Phase 1a Test)

- **Location:** Sorø, Denmark (55.49°N, 11.64°E)
- **Vegetation:** Deciduous Broadleaf Forest (beech)
- **Record:** 18 years in PLUMBER2
- **Variables:** NEE, Qle (latent heat), Qh (sensible heat) with daily measurement uncertainties

## Quick Start

### 1. Download CalLMIP Data

From [modelevaluation.org](https://modelevaluation.org) (CalLMIP workspace) and [GitHub](https://github.com/callmip-org/Phase1/tree/main/Data/Phase1a-test):
- Meteorological forcing: `DK-Sor_forcing.nc` (PLUMBER2 format)
- Flux observations: `DK-Sor_flux_obs.nc` (NEE, Qle, Qh + `_uc` uncertainties)
- CO2 / N deposition data

### 2. Set Up SUMMA for DK-Sor

```bash
# Initialize domain (creates directory structure, converts forcing)
symfluence workflow run --config config_callmip_DK-Sor.yaml --steps setup
```

Update paths in the config YAML to point to your downloaded data.

### 3. Prior Simulation (default parameters)

```bash
symfluence workflow run --config config_callmip_DK-Sor.yaml --steps run
```

Save output as: `SUMMA.<version>_Expt1_DK-Sor_Cal_Prior.nc`

### 4. Run DREAM Calibration

```bash
symfluence workflow run --config config_callmip_DK-Sor.yaml --steps optimize
```

This runs DREAM MCMC with ~38,000 model evaluations (2000 iterations x 19 chains). Expect 1-3 days on 4 cores depending on SUMMA speed.

### 5. Posterior Simulation + Uncertainty

After calibration, run with optimized parameters and generate ensemble:

```bash
# Single best-fit posterior run
symfluence workflow run --config config_callmip_DK-Sor.yaml --steps evaluate

# Posterior ensemble for uncertainty bounds (draws from DREAM posterior)
python -m symfluence.tools.posterior_ensemble \
  --results-path <path_to_optimization_results.json> \
  --config config_callmip_DK-Sor.yaml \
  --n-samples 100
```

Save output as: `SUMMA.<version>_Expt1_DK-Sor_Cal_Posterior.nc`

## CalLMIP Protocol Compliance

| Protocol Requirement | Status | Notes |
|---|---|---|
| Experiment 1: NEE + LE + H | Partial | Qle/Qh via ET evaluator; NEE requires SUMMA carbon module |
| Measurement uncertainty (_uc) | Implemented | Gaussian log-likelihood with FLUXNET `_uc` errors |
| Model error estimation | Implemented | Configurable base + fractional model error |
| Posterior uncertainties | Implemented | DREAM provides posterior distributions + ensemble |
| Daily output frequency | Supported | SUMMA daily output via outputControl.txt |
| ALMA variable naming | Manual | Post-process SUMMA output to ALMA conventions |
| Spin-up protocol | Supported | Cycling forcing at 1850 CO2 (285 ppm) |
| Temporal validation (last year) | Configured | EVALUATION_PERIOD set to last year |

## Architecture: How the Likelihood Works

```
CalLMIP flux data (NetCDF)
  ├── NEE     + NEE_uc    (measurement uncertainty)
  ├── Qle     + Qle_uc
  └── Qh      + Qh_uc

DREAM proposes parameter set
  └── SUMMA runs → simulated NEE, Qle, Qh

Gaussian Log-Likelihood:
  σ_total² = σ_obs² (from _uc) + σ_model² (base + fraction×|sim|)
  log L = -0.5 × Σ [(obs-sim)²/σ_total² + log(2π×σ_total²)]

Metropolis-Hastings acceptance:
  Accept if log L_new > log L_old (or probabilistically)

After burn-in → posterior parameter samples → ensemble runs → uncertainty
```

## Key Files

| File | Description |
|---|---|
| `config_callmip_DK-Sor.yaml` | Main configuration for DK-Sor calibration |
| `src/symfluence/evaluation/likelihood.py` | Gaussian log-likelihood with uncertainty |
| `src/symfluence/optimization/optimizers/algorithms/dream.py` | DREAM MCMC algorithm |
| `src/symfluence/models/summa/calibration/targets.py` | SUMMA calibration targets |
| `src/symfluence/evaluation/evaluators/et.py` | ET/energy flux evaluator |

## What Still Needs Work

### For full CalLMIP compliance:

1. **NEE evaluator**: SUMMA does not include a carbon cycle. Options:
   - Use SUMMA coupled with an external carbon model
   - Implement a simplified NEE = GPP - Reco scheme
   - Discuss with CalLMIP steering committee about scope

2. **Multi-variable composite likelihood**: The config currently calibrates against
   a single target (Qle). For simultaneous NEE+Qle+Qh, extend to use
   `multivariate_log_likelihood()` from `likelihood.py` which sums individual
   log-likelihoods across variables.

3. **PLUMBER2 forcing conversion**: Convert PLUMBER2 NetCDF format to SUMMA
   forcing format (variable naming, units, time conventions).

4. **ALMA output post-processing**: Convert SUMMA output variable names to
   standard ALMA conventions for upload to modelevaluation.org.

5. **CO2 / N deposition transient**: CalLMIP spin-up requires historical CO2
   (285 ppm in 1850) and N deposition (0.79 kg N/ha/yr) during transient period.

6. **Posterior ensemble tool**: The `posterior_ensemble` script for generating
   ensemble runs from DREAM posterior samples needs to be finalized.

## Timeline

- **Phase 1a (test):** By Feb 27, 2026 — Prior + posterior at DK-Sor
- **Phase 1b (full):** By Jul 31, 2026 — Multi-site calibration (22+ sites)
- **Global runs:** By Oct 30, 2026 — Global simulations with calibrated parameters
