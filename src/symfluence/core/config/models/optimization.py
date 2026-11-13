# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Optimization configuration models.

Contains configuration classes for calibration algorithms:
PSOConfig, DEConfig, DDSConfig, SCEUAConfig, NSGA2Config,
EmulationConfig, and the parent OptimizationConfig.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG

# Supported optimization algorithms
OptimizationAlgorithmType = Literal[
    'PSO', 'DE', 'DDS', 'ASYNC-DDS', 'SCE-UA', 'NSGA-II',
    'ADAM', 'LBFGS', 'CMA-ES', 'CMAES', 'DREAM', 'GLUE',
    'BASIN-HOPPING', 'BASINHOPPING', 'BH',
    'NELDER-MEAD', 'NELDERMEAD', 'NM', 'SIMPLEX', 'GA',
    'BAYESIAN-OPT', 'BAYESIAN_OPT', 'BAYESIAN', 'BO',
    'MOEAD', 'MOEA-D', 'MOEA_D',
    'SIMULATED-ANNEALING', 'SIMULATED_ANNEALING', 'SA', 'ANNEALING',
    'ABC', 'ABC-SMC', 'ABC_SMC', 'APPROXIMATE-BAYESIAN'
]

# Supported optimization metrics (uppercase for case-insensitive validation)
OptimizationMetricType = Literal[
    'KGE', 'KGEP', 'NSE', 'RMSE', 'MAE', 'PBIAS', 'R2', 'CORRELATION',
    'COMPOSITE',
    # Transformed-flow variants (transformation applied to obs/sim before scoring)
    'KGE_LOG', 'KGE_INV', 'KGE_SQRT', 'KGE_BOX_COX',
    'NSE_LOG', 'NSE_SQRT', 'RMSE_LOG'
]

# Supported sampling methods
SamplingMethodType = Literal['lhs', 'random', 'sobol', 'halton']


class PSOConfig(BaseModel):
    """Particle Swarm Optimization algorithm settings"""
    model_config = FROZEN_CONFIG

    swrmsize: int = Field(default=20, alias='SWRMSIZE', ge=2, le=10000)
    cognitive_param: float = Field(default=1.5, alias='PSO_COGNITIVE_PARAM', ge=0, le=4.0)
    social_param: float = Field(default=1.5, alias='PSO_SOCIAL_PARAM', ge=0, le=4.0)
    inertia_weight: float = Field(default=0.7, alias='PSO_INERTIA_WEIGHT', ge=0, le=1.0)
    inertia_reduction_rate: float = Field(default=0.99, alias='PSO_INERTIA_REDUCTION_RATE', ge=0, le=1.0)
    inertia_schedule: str = Field(default='LINEAR', alias='INERTIA_SCHEDULE')


class DEConfig(BaseModel):
    """Differential Evolution algorithm settings"""
    model_config = FROZEN_CONFIG

    scaling_factor: float = Field(default=0.5, alias='DE_SCALING_FACTOR', ge=0, le=2.0)
    crossover_rate: float = Field(default=0.9, alias='DE_CROSSOVER_RATE', ge=0, le=1.0)


class DDSConfig(BaseModel):
    """Dynamically Dimensioned Search algorithm settings"""
    model_config = FROZEN_CONFIG

    r: float = Field(default=0.2, alias='DDS_R', gt=0, le=1.0)
    async_pool_size: int = Field(default=10, alias='ASYNC_DDS_POOL_SIZE', ge=1)
    async_batch_size: int = Field(default=10, alias='ASYNC_DDS_BATCH_SIZE', ge=1)
    max_stagnation_batches: int = Field(default=10, alias='MAX_STAGNATION_BATCHES', ge=1)
    stagnation_threshold: int = Field(
        default=100, alias='DDS_STAGNATION_THRESHOLD', ge=1,
        description='Iterations without improvement before DDS stops early'
    )


class SCEUAConfig(BaseModel):
    """Shuffled Complex Evolution - University of Arizona algorithm settings"""
    model_config = FROZEN_CONFIG

    number_of_complexes: int = Field(default=2, alias='NUMBER_OF_COMPLEXES')
    points_per_subcomplex: int = Field(default=5, alias='POINTS_PER_SUBCOMPLEX')
    number_of_evolution_steps: Optional[int] = Field(default=None, alias='NUMBER_OF_EVOLUTION_STEPS')
    evolution_stagnation: int = Field(default=5, alias='EVOLUTION_STAGNATION')
    percent_change_threshold: float = Field(default=0.01, alias='PERCENT_CHANGE_THRESHOLD')


class NSGA2Config(BaseModel):
    """Non-dominated Sorting Genetic Algorithm II settings"""
    model_config = FROZEN_CONFIG

    multi_target: bool = Field(default=False, alias='NSGA2_MULTI_TARGET')
    primary_target: str = Field(default='streamflow', alias='NSGA2_PRIMARY_TARGET')
    secondary_target: str = Field(default='gw_depth', alias='NSGA2_SECONDARY_TARGET')
    primary_metric: str = Field(default='KGE', alias='NSGA2_PRIMARY_METRIC')
    secondary_metric: str = Field(default='KGE', alias='NSGA2_SECONDARY_METRIC')
    crossover_rate: float = Field(default=0.9, alias='NSGA2_CROSSOVER_RATE', ge=0, le=1.0)
    mutation_rate: float = Field(default=0.1, alias='NSGA2_MUTATION_RATE', ge=0, le=1.0)
    eta_c: int = Field(default=20, alias='NSGA2_ETA_C', ge=1)
    eta_m: int = Field(default=20, alias='NSGA2_ETA_M', ge=1)


class MOEADConfig(BaseModel):
    """MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) settings.

    MOEA/D shares multi-target configuration with NSGA2 but uses decomposition-based
    approach. These fields mirror the NSGA2 multi-target settings with MOEAD-prefixed
    aliases for configuration convenience.
    """
    model_config = FROZEN_CONFIG

    multi_target: bool = Field(default=False, alias='MOEAD_MULTI_TARGET')
    primary_target: str = Field(default='streamflow', alias='MOEAD_PRIMARY_TARGET')
    secondary_target: str = Field(default='gw_depth', alias='MOEAD_SECONDARY_TARGET')
    primary_metric: str = Field(default='KGE', alias='MOEAD_PRIMARY_METRIC')
    secondary_metric: str = Field(default='KGE', alias='MOEAD_SECONDARY_METRIC')
    neighbors: int = Field(default=20, alias='MOEAD_NEIGHBORS', ge=1)
    crossover_rate: float = Field(default=1.0, alias='MOEAD_CR', ge=0, le=1.0)
    scaling_factor: float = Field(default=0.5, alias='MOEAD_F', ge=0, le=2.0)
    mutation_rate: float = Field(default=0.1, alias='MOEAD_MUTATION', ge=0, le=1.0)
    max_replacements: int = Field(default=2, alias='MOEAD_NR', ge=1)
    decomposition: str = Field(default='tchebycheff', alias='MOEAD_DECOMPOSITION')


class ABCConfig(BaseModel):
    """Approximate Bayesian Computation (ABC-SMC) algorithm settings.

    ABC-SMC uses Sequential Monte Carlo to efficiently sample from the
    approximate posterior distribution when the likelihood is intractable.

    Reference:
        Beaumont, M.A., Cornuet, J.M., Marin, J.M., and Robert, C.P. (2009).
        Adaptive approximate Bayesian computation. Biometrika, 96(4), 983-990.
    """
    model_config = FROZEN_CONFIG

    # Population settings
    n_particles: int = Field(
        default=100,
        alias='ABC_PARTICLES',
        ge=10,
        le=10000,
        description="Number of particles in the population. More particles = better posterior "
                    "approximation but higher computational cost. Recommend 100-500 for hydrology."
    )

    # Tolerance schedule
    n_generations: int = Field(
        default=20,
        alias='ABC_GENERATIONS',
        ge=3,
        le=100,
        description="Maximum number of SMC generations. Each generation reduces the tolerance."
    )
    initial_tolerance: float = Field(
        default=0.5,
        alias='ABC_INITIAL_TOLERANCE',
        gt=0,
        le=10.0,
        description="Initial acceptance tolerance (distance threshold). For KGE, this is (1-KGE), "
                    "so 0.5 means accepting KGE >= 0.5."
    )
    final_tolerance: float = Field(
        default=0.05,
        alias='ABC_FINAL_TOLERANCE',
        gt=0,
        le=1.0,
        description="Target final tolerance. For KGE, 0.05 means targeting KGE >= 0.95."
    )
    tolerance_quantile: float = Field(
        default=0.75,
        alias='ABC_TOLERANCE_QUANTILE',
        gt=0.1,
        le=0.95,
        description="Quantile of accepted distances for adaptive tolerance schedule. "
                    "Higher values (0.7-0.9) give more gradual tolerance decrease."
    )
    tolerance_decay: float = Field(
        default=0.9,
        alias='ABC_TOLERANCE_DECAY',
        gt=0.5,
        le=0.99,
        description="Maximum tolerance reduction per generation (geometric decay factor). "
                    "Values closer to 1.0 give slower, more stable convergence."
    )

    # Perturbation kernel settings
    perturbation_scale: float = Field(
        default=2.0,
        alias='ABC_PERTURBATION_SCALE',
        gt=0,
        le=10.0,
        description="Scaling factor for perturbation kernel bandwidth. Higher = more exploration."
    )
    kernel_type: Literal['gaussian', 'uniform', 'component_wise'] = Field(
        default='component_wise',
        alias='ABC_KERNEL_TYPE',
        description="Type of perturbation kernel: 'gaussian' (full covariance), "
                    "'uniform' (box kernel), 'component_wise' (independent Gaussians)."
    )
    use_olcm: bool = Field(
        default=True,
        alias='ABC_USE_OLCM',
        description="Use Optimal Local Covariance Matrix adaptation for perturbation kernel."
    )

    # Acceptance settings
    min_acceptance_rate: float = Field(
        default=0.05,
        alias='ABC_MIN_ACCEPTANCE_RATE',
        gt=0.001,
        le=0.5,
        description="Minimum acceptance rate before stopping. Lower allows more generations."
    )
    min_ess_ratio: float = Field(
        default=0.5,
        alias='ABC_MIN_ESS_RATIO',
        gt=0.1,
        le=1.0,
        description="Minimum effective sample size ratio before resampling."
    )

    # Convergence settings
    convergence_threshold: float = Field(
        default=0.001,
        alias='ABC_CONVERGENCE_THRESHOLD',
        gt=0,
        le=0.1,
        description="Stop if tolerance improvement < this fraction."
    )
    min_generations: int = Field(
        default=5,
        alias='ABC_MIN_GENERATIONS',
        ge=1,
        description="Minimum generations to run before checking convergence."
    )


class CMAESConfig(BaseModel):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) settings"""
    model_config = FROZEN_CONFIG

    initial_sigma: float = Field(default=0.3, alias='CMAES_INITIAL_SIGMA', gt=0, le=1.0)


class DREAMConfig(BaseModel):
    """DREAM (DiffeRential Evolution Adaptive Metropolis) MCMC settings"""
    model_config = FROZEN_CONFIG

    de_pairs: int = Field(default=3, alias='DREAM_PAIRS', ge=1)
    crossover_probability: float = Field(default=0.9, alias='DREAM_CR', ge=0, le=1.0)
    epsilon_std: float = Field(default=1e-3, alias='DREAM_EPS', gt=0)
    temperature: float = Field(default=1.0, alias='DREAM_TEMPERATURE', gt=0)
    outlier_threshold: float = Field(default=2.0, alias='DREAM_OUTLIER_THRESHOLD', gt=0)


class GAConfig(BaseModel):
    """Genetic Algorithm settings"""
    model_config = FROZEN_CONFIG

    crossover_rate: float = Field(default=0.9, alias='GA_CROSSOVER_RATE', ge=0, le=1.0)
    mutation_rate: float = Field(default=0.1, alias='GA_MUTATION_RATE', ge=0, le=1.0)
    mutation_scale: float = Field(default=0.1, alias='GA_MUTATION_SCALE', gt=0)
    tournament_size: int = Field(default=3, alias='GA_TOURNAMENT_SIZE', ge=2)
    elitism_count: int = Field(default=2, alias='GA_ELITISM_COUNT', ge=0)


class NelderMeadConfig(BaseModel):
    """Nelder-Mead Simplex algorithm settings"""
    model_config = FROZEN_CONFIG

    alpha: float = Field(default=1.0, alias='NM_ALPHA', gt=0)
    gamma: float = Field(default=2.0, alias='NM_GAMMA', gt=1.0)
    rho: float = Field(default=0.5, alias='NM_RHO', gt=0, lt=1.0)
    sigma: float = Field(default=0.5, alias='NM_SIGMA', gt=0, lt=1.0)
    simplex_size: float = Field(default=0.1, alias='NM_SIMPLEX_SIZE', gt=0, le=1.0)
    x_tol: float = Field(default=1e-6, alias='NM_X_TOL', gt=0)
    f_tol: float = Field(default=1e-6, alias='NM_F_TOL', gt=0)
    adaptive: bool = Field(default=True, alias='NM_ADAPTIVE')


class GLUEConfig(BaseModel):
    """GLUE (Generalized Likelihood Uncertainty Estimation) settings"""
    model_config = FROZEN_CONFIG

    threshold: float = Field(default=0.0, alias='GLUE_THRESHOLD')
    shaping_factor: float = Field(default=1.0, alias='GLUE_SHAPING_FACTOR', gt=0)
    sampling: str = Field(default='lhs', alias='GLUE_SAMPLING')


class BasinHoppingConfig(BaseModel):
    """Basin Hopping global optimization settings"""
    model_config = FROZEN_CONFIG

    step_size: float = Field(default=0.5, alias='BH_STEP_SIZE', gt=0, le=1.0)
    temperature: float = Field(default=1.0, alias='BH_TEMPERATURE', gt=0)
    local_steps: int = Field(default=50, alias='BH_LOCAL_STEPS', ge=1)
    local_method: str = Field(default='nelder_mead', alias='BH_LOCAL_METHOD')
    target_accept: float = Field(default=0.5, alias='BH_TARGET_ACCEPT', gt=0, le=1.0)
    adapt_interval: int = Field(default=10, alias='BH_ADAPT_INTERVAL', ge=1)


class BOConfig(BaseModel):
    """Bayesian Optimization settings"""
    model_config = FROZEN_CONFIG

    initial_samples_factor: int = Field(default=2, alias='BO_INITIAL_SAMPLES_FACTOR', ge=1)
    acquisition: str = Field(default='ei', alias='BO_ACQUISITION')
    xi: float = Field(default=0.01, alias='BO_XI', ge=0)
    kappa: float = Field(default=2.576, alias='BO_KAPPA', gt=0)
    restarts: int = Field(default=10, alias='BO_RESTARTS', ge=1)


class SAConfig(BaseModel):
    """Simulated Annealing settings"""
    model_config = FROZEN_CONFIG

    initial_temp: float = Field(default=1.0, alias='SA_INITIAL_TEMP', gt=0)
    final_temp: float = Field(default=1e-6, alias='SA_FINAL_TEMP', gt=0)
    cooling_schedule: str = Field(default='exponential', alias='SA_COOLING_SCHEDULE')
    cooling_rate: float = Field(default=0.95, alias='SA_COOLING_RATE', gt=0, lt=1.0)
    step_size: float = Field(default=0.1, alias='SA_STEP_SIZE', gt=0)
    steps_per_temp: int = Field(default=10, alias='SA_STEPS_PER_TEMP', ge=1)
    adaptive_step: bool = Field(default=True, alias='SA_ADAPTIVE_STEP')


class AdamConfig(BaseModel):
    """Adam optimizer settings"""
    model_config = FROZEN_CONFIG

    lr: float = Field(default=0.01, alias='ADAM_LR', gt=0)
    beta1: float = Field(default=0.9, alias='ADAM_BETA1', ge=0, lt=1.0)
    beta2: float = Field(default=0.999, alias='ADAM_BETA2', ge=0, lt=1.0)
    eps: float = Field(default=1e-8, alias='ADAM_EPS', gt=0)
    steps: Optional[int] = Field(
        default=None, alias='ADAM_STEPS', ge=1,
        description='Number of Adam optimization steps'
    )


class LBFGSConfig(BaseModel):
    """L-BFGS quasi-Newton optimizer settings"""
    model_config = FROZEN_CONFIG

    lr: float = Field(default=0.1, alias='LBFGS_LR', gt=0)
    history_size: int = Field(default=10, alias='LBFGS_HISTORY_SIZE', ge=1)
    c1: float = Field(default=1e-4, alias='LBFGS_C1', gt=0, lt=1.0)
    c2: float = Field(default=0.9, alias='LBFGS_C2', gt=0, lt=1.0)
    steps: Optional[int] = Field(
        default=None, alias='LBFGS_STEPS', ge=1,
        description='Number of L-BFGS optimization steps (falls back to NUMBER_OF_ITERATIONS)'
    )


class EmulationConfig(BaseModel):
    """Model emulation settings"""
    model_config = FROZEN_CONFIG

    num_samples: int = Field(default=100, alias='EMULATION_NUM_SAMPLES', ge=1)
    seed: int = Field(default=22, alias='EMULATION_SEED')
    sampling_method: SamplingMethodType = Field(default='lhs', alias='EMULATION_SAMPLING_METHOD')
    parallel_ensemble: bool = Field(default=False, alias='EMULATION_PARALLEL_ENSEMBLE')
    max_parallel_jobs: int = Field(default=100, alias='EMULATION_MAX_PARALLEL_JOBS', ge=1)
    skip_mizuroute: bool = Field(default=False, alias='EMULATION_SKIP_MIZUROUTE')
    use_attributes: bool = Field(default=False, alias='EMULATION_USE_ATTRIBUTES')
    max_iterations: int = Field(default=3, alias='EMULATION_MAX_ITERATIONS', ge=1)


class MultiGaugeConfig(BaseModel):
    """Multi-gauge calibration settings.

    All fields default to None so the flat view stays unchanged when unset
    and the calibration workers' own fallback defaults keep applying
    (e.g. aggregation 'mean', min_gauges 5).
    """
    model_config = FROZEN_CONFIG

    calibration: Optional[bool] = Field(
        default=None, alias='MULTI_GAUGE_CALIBRATION',
        description='Calibrate against multiple observation gauges simultaneously'
    )
    obs_dir: Optional[str] = Field(
        default=None, alias='MULTI_GAUGE_OBS_DIR',
        description='Directory containing per-gauge observation files'
    )
    gauge_ids: Optional[Union[List[Union[int, str]], int, str]] = Field(
        default=None, alias='MULTI_GAUGE_IDS',
        description='Explicit gauge IDs to calibrate against (default: all found)'
    )
    exclude_ids: Optional[Union[List[Union[int, str]], int, str]] = Field(
        default=None, alias='MULTI_GAUGE_EXCLUDE_IDS',
        description='Gauge IDs to exclude from multi-gauge calibration'
    )
    aggregation: Optional[str] = Field(
        default='mean', alias='MULTI_GAUGE_AGGREGATION',
        description="Aggregation of per-gauge scores: 'mean', 'median' or 'weighted'"
    )
    min_gauges: Optional[int] = Field(
        default=5, alias='MULTI_GAUGE_MIN_GAUGES',
        description='Minimum number of usable gauges required'
    )
    max_distance: Optional[float] = Field(
        default=None, alias='MULTI_GAUGE_MAX_DISTANCE',
        description='Maximum gauge-to-subbasin mapping distance'
    )
    min_obs_cv: Optional[float] = Field(
        default=None, alias='MULTI_GAUGE_MIN_OBS_CV',
        description='Minimum coefficient of variation of observations'
    )
    min_specific_q: Optional[float] = Field(
        default=None, alias='MULTI_GAUGE_MIN_SPECIFIC_Q',
        description='Minimum specific discharge for a gauge to be used'
    )
    min_overlap_days: Optional[int] = Field(
        default=10, alias='MULTI_GAUGE_MIN_OVERLAP_DAYS',
        description='Minimum days of obs/sim overlap for a gauge to be used'
    )
    kge_floor: Optional[float] = Field(
        default=None, alias='MULTI_GAUGE_KGE_FLOOR',
        description='Floor applied to per-gauge KGE before aggregation'
    )
    gauge_segment_mapping: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, alias='GAUGE_SEGMENT_MAPPING',
        description='Gauge-ID to river-segment mapping: a CSV path (auto-generated if absent) '
                    'or an explicit {gauge_id: segment_id} dict'
    )


class OptimizationConfig(BaseModel):
    """Calibration and optimization configuration"""
    model_config = FROZEN_CONFIG

    # General optimization settings
    methods: Union[List[str], str] = Field(default_factory=list, alias='OPTIMIZATION_METHODS')
    target: str = Field(default='streamflow', alias='OPTIMIZATION_TARGET')
    calibration_variable: str = Field(default='streamflow', alias='CALIBRATION_VARIABLE')
    calibration_timestep: str = Field(default='daily', alias='CALIBRATION_TIMESTEP')
    algorithm: OptimizationAlgorithmType = Field(default='PSO', alias='ITERATIVE_OPTIMIZATION_ALGORITHM')
    metric: OptimizationMetricType = Field(default='KGE', alias='OPTIMIZATION_METRIC')
    box_cox_lambda: float = Field(
        default=0.2, alias='BOX_COX_LAMBDA', ge=0.0, le=1.0,
        description="Lambda for the Box-Cox flow transformation used by *_BOX_COX "
                    "metrics (0 = log transform, smaller values emphasize low flows more)"
    )
    iterations: int = Field(default=1000, alias='NUMBER_OF_ITERATIONS', ge=1)
    population_size: int = Field(default=50, alias='POPULATION_SIZE', ge=2, le=10000)
    final_evaluation_numerical_method: str = Field(default='ida', alias='FINAL_EVALUATION_NUMERICAL_METHOD')
    cleanup_parallel_dirs: bool = Field(default=True, alias='CLEANUP_PARALLEL_DIRS')
    checkpoint_interval: int = Field(default=10, alias='CHECKPOINT_INTERVAL', ge=1)
    initial_guess: Optional[Dict[str, Any]] = Field(
        default=None,
        alias='INITIAL_GUESS',
        description="Calibration seed parameters. If set, these take precedence over "
                    "warm-start files and model defaults."
    )
    skip_warm_start: Optional[bool] = Field(
        default=True,
        alias='SKIP_WARM_START',
        description='Skip warm-starting from previous best parameters. Unset defaults to '
                    'true (no warm start), so calibrations are reproducible regardless of '
                    'what previous runs exist on the machine; set false to opt in.'
    )
    parameter_bounds: Optional[Dict[str, Any]] = Field(
        default=None,
        alias='PARAMETER_BOUNDS',
        description='Generic parameter bounds override, e.g. {param: [min, max]}'
    )
    initial_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        alias='INITIAL_PARAMETERS',
        description='Initial parameter values used by parameter managers'
    )
    likelihood_function: Optional[str] = Field(
        default=None,
        alias='LIKELIHOOD_FUNCTION',
        description="Likelihood for uncertainty-aware metrics (e.g. 'gaussian')"
    )
    model_error_base: Optional[float] = Field(
        default=0.0,
        alias='MODEL_ERROR_BASE',
        description='Additive model error term for likelihood evaluation'
    )
    model_error_fraction: Optional[float] = Field(
        default=0.0,
        alias='MODEL_ERROR_FRACTION',
        description='Model error as a fraction of simulated value'
    )
    transfer_function_type: Optional[str] = Field(
        default='linear',
        alias='TRANSFER_FUNCTION_TYPE',
        description="Regionalization transfer function form (e.g. 'linear')"
    )
    transfer_function_b_bounds: Optional[List[float]] = Field(
        default=None,
        alias='TRANSFER_FUNCTION_B_BOUNDS',
        description='Bounds for the transfer-function slope coefficient b'
    )
    transfer_function_log_attrs: Optional[Union[List[str], str]] = Field(
        default=None,
        alias='TRANSFER_FUNCTION_LOG_ATTRS',
        description='Attributes to log-transform before the transfer function'
    )

    # Gradient-based optimization settings (Adam, L-BFGS)
    gradient_mode: Literal['auto', 'native', 'finite_difference'] = Field(
        default='auto',
        alias='GRADIENT_MODE',
        description="Gradient computation method for Adam/L-BFGS: "
                    "'auto' uses native gradients if available (e.g., JAX autodiff for HBV), "
                    "'native' requires native gradients (error if unavailable), "
                    "'finite_difference' always uses FD (useful for comparison)"
    )
    gradient_epsilon: float = Field(
        default=1e-4,
        alias='GRADIENT_EPSILON',
        gt=0,
        le=0.1,
        description="Perturbation size for finite-difference gradient approximation"
    )
    gradient_clip_value: float = Field(
        default=1.0,
        alias='GRADIENT_CLIP_VALUE',
        gt=0,
        description="Maximum gradient L2 norm (prevents exploding gradients)"
    )

    @field_validator('algorithm', mode='before')
    @classmethod
    def normalize_algorithm(cls, v):
        """Normalize algorithm name to uppercase for case-insensitive matching"""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator('metric', mode='before')
    @classmethod
    def normalize_metric(cls, v):
        """Normalize metric name to uppercase for case-insensitive matching"""
        if isinstance(v, str):
            return v.upper()
        return v

    # Composite metric settings (for OPTIMIZATION_METRIC: COMPOSITE)
    composite_metric: Optional[Dict[str, float]] = Field(
        default=None,
        alias='COMPOSITE_METRIC',
        description="Weighted combination of metrics for composite optimization. "
                    "Example: {'KGE': 0.6, 'KGE_LOG': 0.4}"
    )

    # Multivariate objective settings
    objective_weights: Optional[Dict[str, float]] = Field(
        default=None,
        alias='OBJECTIVE_WEIGHTS',
        description="Variable-specific weights for multivariate objective. "
                    "Example: {'STREAMFLOW': 0.7, 'SWE': 0.2, 'ET': 0.1}"
    )
    objective_metrics: Optional[Dict[str, str]] = Field(
        default=None,
        alias='OBJECTIVE_METRICS',
        description="Primary metric per variable for multivariate objective. "
                    "Example: {'STREAMFLOW': 'kge', 'SWE': 'nse'}"
    )

    # Worker retry settings
    worker_max_retries: int = Field(
        default=3,
        alias='WORKER_MAX_RETRIES',
        ge=0,
        description="Maximum number of retry attempts for worker model evaluations"
    )
    worker_base_delay: float = Field(
        default=0.5,
        alias='WORKER_BASE_DELAY',
        gt=0,
        description="Base delay for exponential backoff between retries (seconds)"
    )
    worker_max_delay: float = Field(
        default=30.0,
        alias='WORKER_MAX_DELAY',
        gt=0,
        description="Maximum delay between retries (seconds)"
    )
    worker_jitter: float = Field(
        default=0.1,
        alias='WORKER_JITTER',
        ge=0,
        le=1.0,
        description="Jitter factor for randomizing retry delays (0-1)"
    )

    # Error logging and debugging options
    params_keep_trials: bool = Field(
        default=False,
        alias='PARAMS_KEEP_TRIALS',
        description="Convenience flag: enables ERROR_LOGGING_MODE='failures' to save "
                    "parameter files and logs from failed runs for debugging"
    )
    error_logging_mode: str = Field(
        default='none',
        alias='ERROR_LOGGING_MODE',
        description="Error artifact capture mode: 'none' (disabled), 'failures' "
                    "(save artifacts from failed runs), 'all' (save all runs)"
    )
    stop_on_model_failure: bool = Field(
        default=False,
        alias='STOP_ON_MODEL_FAILURE',
        description="Stop optimization immediately when a model run fails"
    )
    error_log_dir: str = Field(
        default='error_logs',
        alias='ERROR_LOG_DIR',
        description="Subdirectory name for error artifacts within the output directory"
    )

    # Algorithm-specific settings
    pso: Optional[PSOConfig] = Field(default_factory=PSOConfig)
    de: Optional[DEConfig] = Field(default_factory=DEConfig)
    dds: Optional[DDSConfig] = Field(default_factory=DDSConfig)
    sce_ua: Optional[SCEUAConfig] = Field(default_factory=SCEUAConfig)
    nsga2: Optional[NSGA2Config] = Field(default_factory=NSGA2Config)
    moead: Optional[MOEADConfig] = Field(default_factory=MOEADConfig)
    abc: Optional[ABCConfig] = Field(default_factory=ABCConfig)
    cmaes: Optional[CMAESConfig] = Field(default_factory=CMAESConfig)
    dream: Optional[DREAMConfig] = Field(default_factory=DREAMConfig)
    ga: Optional[GAConfig] = Field(default_factory=GAConfig)
    nelder_mead: Optional[NelderMeadConfig] = Field(default_factory=NelderMeadConfig)
    glue: Optional[GLUEConfig] = Field(default_factory=GLUEConfig)
    basin_hopping: Optional[BasinHoppingConfig] = Field(default_factory=BasinHoppingConfig)
    bayesian_opt: Optional[BOConfig] = Field(default_factory=BOConfig)
    sa: Optional[SAConfig] = Field(default_factory=SAConfig)
    adam: Optional[AdamConfig] = Field(default_factory=AdamConfig)
    lbfgs: Optional[LBFGSConfig] = Field(default_factory=LBFGSConfig)
    emulation: Optional[EmulationConfig] = Field(default_factory=EmulationConfig)
    multi_gauge: Optional[MultiGaugeConfig] = Field(default_factory=MultiGaugeConfig)

    @field_validator('methods', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return []
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator('methods')
    @classmethod
    def validate_methods(cls, v):
        """Reject unknown OPTIMIZATION_METHODS (e.g. 'iteraton' silently disabled calibration)."""
        from .choices import OPTIMIZATION_METHODS, unknown_value_message

        methods = [v] if isinstance(v, str) else list(v or [])
        normalized = [str(m).strip().lower() for m in methods if str(m).strip()]
        for method in normalized:
            if method not in OPTIMIZATION_METHODS:
                raise ValueError(unknown_value_message('OPTIMIZATION_METHODS', method, OPTIMIZATION_METHODS))
        return normalized
