.. _uncertainty_sensitivity:

Uncertainty and Sensitivity
===========================

This page covers the parts of SYMFLUENCE that quantify uncertainty rather
than find a single best parameter set: Bayesian calibration with DREAM, GLUE,
the predictive intervals both produce, and global sensitivity analysis.

.. contents::
   :local:
   :depth: 2

Formal likelihood for streamflow
--------------------------------

An efficiency score such as KGE is not a likelihood. Using
``exp(KGE / T)`` as the target density of an MCMC sampler gives a
"posterior" whose width is set by the arbitrary temperature ``T``, not by the
data. SYMFLUENCE therefore provides a residual error model whose density is a
proper likelihood ``p(Q_obs | θ)``, the metric ``LOGLIK_BC_AR1``
(Bates & Campbell, 2001; a simplified Schoups & Vrugt, 2010):

* **Heteroscedastic errors** — observed and simulated flows are Box-Cox
  transformed, ``z = ((q + ε)^λ − 1) / λ`` (``log(q + ε)`` for λ = 0), so a
  fixed error in z corresponds to an error that grows with flow. ``ε`` keeps
  zero flows finite.
* **Autocorrelated errors** — residuals ``e_t = z(obs_t) − z(sim_t)`` follow a
  stationary AR(1) process ``e_t = φ e_{t−1} + a_t``, ``a_t ~ N(0, σ²)``.
* The density is evaluated in flow space (the Box-Cox Jacobian is included).

Error-model parameters:

``σ``
   Fixed with ``LIKELIHOOD_SIGMA``; by default it is **integrated out
   analytically** under the Jeffreys prior ``p(σ) ∝ 1/σ``, which needs no
   extra sampled dimension.
``φ``
   Fixed with ``LIKELIHOOD_AR1_PHI``; by default the conditional
   least-squares estimate for each simulation (a profile likelihood in φ).

Simplifications relative to the full generalised likelihood of Schoups & Vrugt
(2010): Gaussian rather than skew-exponential-power innovations, no residual
bias term, and constant variance in Box-Cox space. Gaps in the observed record
are treated as contiguous by the AR(1) recursion.

.. code-block:: yaml

   LIKELIHOOD_BOX_COX_LAMBDA: 0.3   # 0 = log; larger = less heteroscedastic
   LIKELIHOOD_AR1_PHI: null         # null = estimated per simulation
   LIKELIHOOD_SIGMA: null           # null = integrated out (Jeffreys prior)
   LIKELIHOOD_EPSILON: null         # null = 1% of mean observed flow

``LOGLIK_BC_AR1`` can be used as ``OPTIMIZATION_METRIC`` with any algorithm
(a maximum-likelihood calibration), but it is mainly meant for DREAM.

DREAM
-----

DREAM (Vrugt et al., 2009; Vrugt, 2016) runs parallel Markov chains with
differential-evolution proposals. With ``DREAM_LIKELIHOOD: formal`` (the
default) it samples the posterior under ``LOGLIK_BC_AR1`` with a uniform prior
on the parameter bounds:

.. code-block:: yaml

   ITERATIVE_OPTIMIZATION_ALGORITHM: DREAM
   DREAM_LIKELIHOOD: formal          # default
   NUMBER_OF_ITERATIONS: 2000        # generations; the first 20% are burn-in
   POPULATION_SIZE: 20               # chains (at least 2d + 1 are used)

When DREAM is selected the optimizer **replaces OPTIMIZATION_METRIC (and
clears COMPOSITE_METRIC) with LOGLIK_BC_AR1** and logs that it did so.
``DREAM_TEMPERATURE`` should stay at 1 — any other value tempers the
posterior. Failed model runs are rejected (zero likelihood). The formal
likelihood is defined for streamflow; for other ``OPTIMIZATION_TARGET`` values
DREAM keeps the configured metric and warns that the result is not a Bayesian
posterior.

``DREAM_LIKELIHOOD: informal`` restores the old behaviour (acceptance on
``exp(OPTIMIZATION_METRIC / DREAM_TEMPERATURE)``) with a warning; it is an
exploratory sampler, not Bayesian inference.

Convergence is reported as the Gelman-Rubin R-hat every 50 generations after
burn-in (target < 1.1).

GLUE
----

GLUE (Beven & Binley, 1992) samples the parameter space (LHS by default),
keeps the "behavioural" runs whose score (``OPTIMIZATION_METRIC``, in
maximisation convention) is at least ``GLUE_THRESHOLD`` and weights them by
``(score − threshold)^GLUE_SHAPING_FACTOR``. The default threshold is 0.0,
which for KGE is only slightly better than the mean-flow benchmark (KGE ≈
−0.41); choose a threshold that reflects what you consider acceptable.

GLUE weights are informal likelihoods: the resulting bounds are conditional
on the threshold and shaping factor and have no probabilistic coverage
guarantee.

Predictive intervals and their scores
-------------------------------------

After a DREAM or GLUE run the calibration step:

1. writes ``<algorithm>_uq_samples.nc`` (in the calibration results
   directory): the post-burn-in posterior draws (with chain and draw indices
   and log-likelihood) or the behavioural samples (with GLUE weights), in
   normalized and physical units;
2. re-runs up to ``UQ_PREDICTIVE_RUNS`` randomly chosen samples over the full
   simulation period and stores their simulated series in the same file;
3. writes ``<algorithm>_predictive_quantiles.csv`` with the observed series
   and the weighted quantiles ``UQ_QUANTILES`` of those runs:

   * ``parameter_q*`` — parameter uncertainty only (DREAM draws equally
     weighted, GLUE runs likelihood-weighted);
   * ``total_q*`` — formal DREAM only: each run plus ``UQ_RESIDUAL_DRAWS``
     realisations of its Box-Cox AR(1) error model (φ and σ estimated on the
     calibration period), i.e. the full predictive distribution of observed
     flow;

4. scores each band separately on ``CALIBRATION_PERIOD`` and
   ``EVALUATION_PERIOD`` in ``<algorithm>_predictive_scores.json``:
   coverage of the 90% band (nominal 0.90), its average width, and the
   continuous ranked probability score (CRPS) of the ensemble.

The re-runs happen before the usual final evaluation, so the model is left
with the best parameter set applied.

.. code-block:: yaml

   UQ_SAVE_SAMPLES: true
   UQ_PREDICTIVE_RUNS: 50            # 0 disables the re-runs
   UQ_QUANTILES: [0.05, 0.25, 0.5, 0.75, 0.95]
   UQ_RESIDUAL_DRAWS: 20

Parameter-only bands usually cover far fewer than 90% of observations: they
omit input, structural and observation error. Judge a band by coverage *and*
width (a band can be wide enough to cover everything and still be useless);
CRPS rewards both.

Global sensitivity analysis
---------------------------

The ``run_sensitivity_analysis`` workflow step runs a **dedicated sampling
design** through the model's calibration workers, so each design point is a
real model run on the calibration period, scored with the calibration
objective (``OPTIMIZATION_METRIC``, maximisation convention). Designs are
generated and analysed with SALib (``pip install "symfluence[sensitivity]"``).

``morris`` (default)
   Elementary effects (Morris, 1991) on ``SENSITIVITY_MORRIS_TRAJECTORIES``
   trajectories over a ``SENSITIVITY_MORRIS_LEVELS``-level grid:
   ``r (d + 1)`` runs. Reports μ* (overall importance), μ and σ (non-linearity
   or interactions). A screening method.
``sobol``
   First-order (S1) and total-order (ST) Sobol indices from a Saltelli design
   (Saltelli et al., 2010): ``N (d + 2)`` runs for base sample size
   ``SENSITIVITY_SOBOL_N``. ``ST − S1`` measures interactions.

.. code-block:: yaml

   SENSITIVITY_METHODS: [morris, sobol]
   SENSITIVITY_MORRIS_TRAJECTORIES: 10
   SENSITIVITY_MORRIS_LEVELS: 4
   SENSITIVITY_SOBOL_N: 256          # power of 2
   SENSITIVITY_MAX_EVALUATIONS: 5000 # refuse larger designs

Outputs (in ``reporting/sensitivity_analysis/<EXPERIMENT_ID>/``):
``morris_sensitivity.csv``, ``sobol_sensitivity.csv`` (with bootstrap
confidence intervals) and ``all_sensitivity_results.csv``. Failed design runs
are assigned the worst successful response and counted in the log; the
indices are unreliable if many runs fail.

The optimiser's trajectory is *not* a sensitivity design: its samples
cluster near the optimum. Earlier versions estimated Sobol and RBD-FAST
indices from it through an additive surrogate that could not represent
interactions (it always gave ST = S1); those methods, and VISCOUS on the
trajectory, were removed. With ``SENSITIVITY_TRAJECTORY_CORRELATION: true``
(default) the step still writes ``trajectory_correlation.csv``, the absolute
Spearman correlation between each parameter and the score along the
calibration trajectory — a description of where the optimiser went, not a
sensitivity index. Set ``SENSITIVITY_METHODS: []`` to write only that.

References
----------

* Bates, B.C. and Campbell, E.P. (2001). A Markov chain Monte Carlo scheme for
  parameter estimation and inference in conceptual rainfall-runoff modeling.
  *Water Resources Research*, 37(4), 937–947.
* Beven, K. and Binley, A. (1992). The future of distributed models: model
  calibration and uncertainty prediction. *Hydrological Processes*, 6(3),
  279–298.
* Gneiting, T. and Raftery, A.E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
* Morris, M.D. (1991). Factorial sampling plans for preliminary computational
  experiments. *Technometrics*, 33(2), 161–174.
* Saltelli, A. et al. (2010). Variance based sensitivity analysis of model
  output. *Computer Physics Communications*, 181(2), 259–270.
* Schoups, G. and Vrugt, J.A. (2010). A formal likelihood function for
  parameter and predictive inference of hydrologic models with correlated,
  heteroscedastic, and non-Gaussian errors. *Water Resources Research*, 46,
  W10531.
* Vrugt, J.A. (2016). Markov chain Monte Carlo simulation using the DREAM
  software package. *Environmental Modelling & Software*, 75, 273–316.
* Yilmaz, K.K., Gupta, H.V. and Wagener, T. (2008). A process-based diagnostic
  approach to model evaluation. *Water Resources Research*, 44, W09417.
