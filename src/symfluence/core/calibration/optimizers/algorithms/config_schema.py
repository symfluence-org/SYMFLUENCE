#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Algorithm Configuration Schema

Defines hyperparameter defaults, valid ranges, and academic references for all
optimization algorithms. This module serves as the central documentation for
magic numbers and algorithm-specific constants.

References:
    - CMA-ES: Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review"
    - NSGA-II: Deb, K. et al. (2002). "A fast and elitist multiobjective GA: NSGA-II"
    - DREAM: Vrugt, J.A. et al. (2009). "Accelerating MCMC simulation by DE"
    - PSO: Kennedy, J. & Eberhart, R. (1995). "Particle swarm optimization"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# =============================================================================
# CMA-ES Configuration
# Reference: Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review"
#            In Towards a New Evolutionary Computation, pp. 75-102.
# =============================================================================

@dataclass
class CMAESDefaults:
    """
    Default hyperparameters for CMA-ES algorithm.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a stochastic,
    derivative-free optimization algorithm. These defaults follow Hansen (2006)
    recommendations for robust performance across problem types.

    Reference:
        Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review.
        Hansen, N. and Ostermeier, A. (2001). Completely Derandomized
        Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2).
    """

    # Population size heuristic
    # λ = 4 + floor(3 * ln(n)) is the recommended default population size
    # where n is the problem dimension. This provides good exploration-exploitation
    # balance across problem sizes. (Hansen 2006, Section 3.1)
    MIN_POPULATION: int = 4
    POPULATION_LOG_FACTOR: float = 3.0

    # Initial step size (sigma)
    # σ₀ = 0.3 covers approximately 1/3 of the normalized [0,1] search space,
    # providing a good starting point for exploration without being too aggressive.
    # (Hansen 2006, Section 3.2 - "Initial Step Size")
    INITIAL_SIGMA: float = 0.3

    # Sigma bounds
    # σ_min = 1e-10: Prevents numerical underflow and signals convergence
    # σ_max = 1.0: Prevents step sizes larger than the search space
    # (Hansen 2006, Section 4.4 - "Termination Criteria")
    SIGMA_MIN: float = 1e-10
    SIGMA_MAX: float = 1.0

    # Convergence threshold
    # The run stops when σ <= CONVERGENCE_THRESHOLD. It must not be below
    # SIGMA_MIN: σ is clamped to SIGMA_MIN every generation, so a smaller
    # threshold (it used to be 1e-12) can never be reached and the stop was
    # dead code. At σ = 1e-10 in the normalized [0,1] space the algorithm has
    # effectively converged. (Hansen 2006, Section 4.4)
    CONVERGENCE_THRESHOLD: float = SIGMA_MIN

    # Eigenvalue floor for numerical stability
    # Prevents negative eigenvalues from numerical errors in covariance matrix
    EIGENVALUE_FLOOR: float = 1e-20

    @staticmethod
    def compute_population_size(n_params: int) -> int:
        """
        Compute recommended population size for given problem dimension.

        Formula: λ = 4 + floor(3 * ln(n))

        This heuristic from Hansen (2006) balances exploration (larger λ) with
        computational cost (smaller λ). For small n, λ ≈ 4-6; for n=100, λ ≈ 18.

        Args:
            n_params: Number of parameters (problem dimension)

        Returns:
            Recommended population size λ
        """
        return CMAESDefaults.MIN_POPULATION + int(
            CMAESDefaults.POPULATION_LOG_FACTOR * np.log(n_params)
        )

    @staticmethod
    def compute_strategy_parameters(n_params: int, mu: int, mu_eff: float) -> Dict[str, float]:
        """
        Compute CMA-ES strategy parameters following Hansen (2006).

        These parameters control the adaptation rates of the step size and
        covariance matrix. The formulas are derived from theoretical analysis
        and empirical tuning.

        Args:
            n_params: Number of parameters (dimension n)
            mu: Number of selected parents (μ)
            mu_eff: Variance-effective selection mass (μ_eff)

        Returns:
            Dictionary containing:
            - c_sigma: Step-size adaptation learning rate
            - d_sigma: Step-size damping factor
            - c_c: Covariance matrix evolution path learning rate
            - c_1: Rank-one update learning rate
            - c_mu: Rank-μ update learning rate
            - chi_n: Expected length of N(0,I) distributed random vector

        Reference:
            Hansen (2006), Table 1: Default parameter settings
        """
        n = n_params

        # Step-size control parameters (Section 3.3)
        # c_sigma controls how fast the step-size evolution path is updated
        # Typical values: 0.2 - 0.5 depending on μ_eff and n
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)

        # d_sigma is the damping factor for step-size adaptation
        # Larger d_sigma = slower adaptation, more stable but less responsive
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation parameters (Section 3.4)
        # c_c controls the evolution path for rank-one update
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)

        # c_1 is the learning rate for rank-one update (captures correlations)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)

        # c_mu is the learning rate for rank-μ update (uses all μ best samples)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Expected length of N(0,I) random vector (used for step-size control)
        # chi_n ≈ √n * (1 - 1/(4n) + 1/(21n²)) is an approximation of E[||N(0,I)||]
        # This normalizes the step-size adaptation to be dimension-independent
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        return {
            'c_sigma': c_sigma,
            'd_sigma': d_sigma,
            'c_c': c_c,
            'c_1': c_1,
            'c_mu': c_mu,
            'chi_n': chi_n
        }

    @staticmethod
    def validate_sigma(sigma: float) -> Tuple[bool, str]:
        """
        Validate step size value.

        Args:
            sigma: Step size to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if sigma < CMAESDefaults.SIGMA_MIN:
            return False, f"Sigma {sigma} below minimum {CMAESDefaults.SIGMA_MIN}"
        if sigma > CMAESDefaults.SIGMA_MAX:
            return False, f"Sigma {sigma} above maximum {CMAESDefaults.SIGMA_MAX}"
        return True, ""


# =============================================================================
# NSGA-II Configuration
# Reference: Deb, K., et al. (2002). "A fast and elitist multiobjective
#            genetic algorithm: NSGA-II". IEEE Trans. Evol. Comput., 6(2).
# =============================================================================

@dataclass
class NSGA2Defaults:
    """
    Default hyperparameters for NSGA-II algorithm.

    NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective
    evolutionary algorithm. These defaults follow Deb et al. (2002).

    Reference:
        Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II.
        IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
    """

    # Crossover rate (probability of applying crossover to a pair)
    # 0.9 is the recommended value from Deb (2002), providing high recombination
    # while preserving some parents unchanged. (Section IV-A)
    CROSSOVER_RATE: float = 0.9

    # Mutation rate (probability of mutating each gene)
    # 1/n is often recommended, but 0.5 provides more exploration in bounded spaces
    # This higher rate compensates for the polynomial mutation's local nature.
    MUTATION_RATE: float = 0.5

    # SBX crossover distribution index (η_c)
    # Higher values (15-20) produce children closer to parents (exploitation)
    # Lower values (2-5) produce more diverse children (exploration)
    # 15 is a common default balancing both. (Deb 2002, Section III-B)
    ETA_C: float = 15.0

    # Polynomial mutation distribution index (η_m)
    # Higher values produce smaller mutations (local search)
    # Lower values produce larger mutations (global search)
    # 10-20 is typical; 10 provides moderate perturbation. (Deb 2002, Section III-C)
    ETA_M: float = 10.0

    # SBX crossover: probability of swapping genes between parents
    # 0.5 gives equal probability of child inheriting from either parent
    # (Deb & Agrawal, 1995 - original SBX paper)
    SBX_SWAP_PROBABILITY: float = 0.5

    # Minimum difference between parents for SBX crossover
    # Prevents numerical issues when parents are nearly identical
    SBX_EPSILON: float = 1e-9

    @staticmethod
    def validate_eta(eta: float, name: str) -> Tuple[bool, str]:
        """
        Validate distribution index value.

        Args:
            eta: Distribution index to validate
            name: Parameter name for error message

        Returns:
            Tuple of (is_valid, error_message)
        """
        if eta < 0:
            return False, f"{name} must be non-negative, got {eta}"
        if eta > 100:
            return False, f"{name} unusually high ({eta}), typical range is 2-20"
        return True, ""

    @staticmethod
    def validate_rate(rate: float, name: str) -> Tuple[bool, str]:
        """
        Validate probability rate value.

        Args:
            rate: Rate to validate (should be in [0, 1])
            name: Parameter name for error message

        Returns:
            Tuple of (is_valid, error_message)
        """
        if rate < 0 or rate > 1:
            return False, f"{name} must be in [0, 1], got {rate}"
        return True, ""


# =============================================================================
# DREAM Configuration
# Reference: Vrugt, J.A. et al. (2009). "Accelerating Markov chain Monte Carlo
#            simulation by differential evolution with self-adaptive randomized
#            subspace sampling". Int. J. Nonlinear Sci. Numer. Simul., 10(3).
# =============================================================================

@dataclass
class DREAMDefaults:
    """
    Default hyperparameters for DREAM algorithm.

    DREAM (DiffeRential Evolution Adaptive Metropolis) is a Markov Chain Monte
    Carlo algorithm that uses differential evolution for proposal generation.
    These defaults follow Vrugt et al. (2009, 2016).

    References:
        Vrugt, J.A., et al. (2009). Accelerating Markov chain Monte Carlo
        simulation by differential evolution with self-adaptive randomized
        subspace sampling. Int. J. Nonlinear Sci. Numer. Simul., 10(3), 273-290.

        Vrugt, J.A. (2016). Markov chain Monte Carlo simulation using the DREAM
        software package. Environ. Model. Softw., 75, 273-316.
    """

    # Minimum number of chains
    # DREAM needs at least 2*d+1 chains for good mixing, where d is dimension.
    # This ensures sufficient diversity for the differential evolution proposals.
    # (Vrugt 2009, Section 2.2 - "Number of Chains")
    MIN_CHAINS_FACTOR: int = 2
    MIN_CHAINS_OFFSET: int = 1

    # Number of chain pairs for DE proposal (δ)
    # Using 3 pairs provides robust proposal generation.
    # More pairs = more complex proposals, but 3 is sufficient for most problems.
    # (Vrugt 2009, Section 2.1 - "Differential Evolution")
    DE_PAIRS: int = 3

    # Crossover probability for subspace sampling (CR)
    # 0.9 means 90% of dimensions are updated in each proposal.
    # High CR improves mixing in low dimensions; lower CR helps high dimensions.
    # (Vrugt 2009, Section 2.3 - "Snooker Update")
    CROSSOVER_PROBABILITY: float = 0.9

    # Small random noise for ergodicity (ε)
    # Added to proposals to ensure detailed balance and ergodicity.
    # 1e-3 gives ~0.1% noise in normalized [0,1] space.
    # (Vrugt 2009, Equation 4)
    EPSILON_STD: float = 1e-3

    # Temperature for likelihood (T)
    # T=1.0 is standard MCMC; lower T makes acceptance more greedy,
    # higher T increases exploration. (Vrugt 2016, Section 2.4)
    TEMPERATURE: float = 1.0

    # Outlier detection threshold (IQR multiplier)
    # Chains with log-likelihood below Q1 - threshold*IQR are considered outliers.
    # 2.0 is a moderate threshold. (Vrugt 2009, Section 2.4)
    OUTLIER_THRESHOLD: float = 2.0

    # Burn-in fraction
    # First 20% of iterations are discarded as burn-in for posterior estimation.
    # (Vrugt 2016, Section 3.1 - "Burn-in Period")
    BURN_IN_FRACTION: float = 0.2

    # Mode jump probability
    # 10% of proposals use γ=1 instead of optimal γ for potential mode jumping.
    # This helps escape local modes in multimodal distributions.
    # (Vrugt 2009, Section 2.2 - "Jump Rate")
    MODE_JUMP_PROBABILITY: float = 0.1

    # Gelman-Rubin convergence threshold (R-hat)
    # R-hat < 1.1 indicates approximate convergence of chains.
    # (Gelman & Rubin, 1992; Vrugt 2016, Section 3.2)
    RHAT_THRESHOLD: float = 1.1

    @staticmethod
    def compute_min_chains(n_params: int) -> int:
        """
        Compute minimum number of chains for good mixing.

        Formula: N = 2*d + 1

        This ensures the differential evolution has sufficient chain diversity
        to generate meaningful proposals in d-dimensional space.

        Args:
            n_params: Number of parameters (dimension d)

        Returns:
            Minimum recommended number of chains
        """
        return (DREAMDefaults.MIN_CHAINS_FACTOR * n_params +
                DREAMDefaults.MIN_CHAINS_OFFSET)

    @staticmethod
    def compute_optimal_gamma(n_pairs: int, d_star: int) -> float:
        """
        Compute optimal jump rate scaling factor.

        Formula: γ = 2.38 / √(2 * δ * d*)

        where δ is the number of chain pairs and d* is the effective dimension
        (typically CR * n_params). This formula maximizes the expected squared
        jumping distance. (Vrugt 2009, Equation 5)

        Args:
            n_pairs: Number of chain pairs (δ)
            d_star: Effective dimension (typically CR * n_params)

        Returns:
            Optimal gamma value
        """
        return 2.38 / np.sqrt(2 * n_pairs * max(1, d_star))

    @staticmethod
    def validate_chains(n_chains: int, n_params: int) -> Tuple[bool, str]:
        """
        Validate number of chains.

        Args:
            n_chains: Number of chains to use
            n_params: Problem dimension

        Returns:
            Tuple of (is_valid, warning_message)
        """
        min_chains = DREAMDefaults.compute_min_chains(n_params)
        if n_chains < min_chains:
            return False, (f"Using {n_chains} chains for {n_params} parameters. "
                          f"Recommended minimum: {min_chains}")
        return True, ""


# =============================================================================
# PSO Configuration
# Reference: Kennedy, J. & Eberhart, R. (1995). "Particle swarm optimization".
#            Proc. IEEE Int. Conf. Neural Networks, pp. 1942-1948.
#            Shi, Y. & Eberhart, R. (1998). "A modified particle swarm optimizer".
#            Proc. IEEE Int. Conf. Evol. Comput., pp. 69-73.
# =============================================================================

@dataclass
class PSODefaults:
    """
    Default hyperparameters for PSO algorithm.

    PSO (Particle Swarm Optimization) simulates social behavior for optimization.
    These defaults follow Kennedy & Eberhart (1995) and Shi & Eberhart (1998).

    References:
        Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization.
        Proceedings of IEEE International Conference on Neural Networks.

        Shi, Y. and Eberhart, R. (1998). A modified particle swarm optimizer.
        IEEE International Conference on Evolutionary Computation.
    """

    # Inertia weight (w)
    # Controls the influence of previous velocity on current movement.
    # w=0.7 provides good balance between exploration and exploitation.
    # Lower values (0.4) favor exploitation; higher (0.9) favor exploration.
    # (Shi & Eberhart 1998, "Inertia Weight Approach")
    INERTIA: float = 0.7

    # Cognitive coefficient (c1)
    # Controls attraction to personal best position.
    # c1=1.5 gives moderate self-confidence. Typical range: 1.0-2.0.
    # (Kennedy & Eberhart 1995, Equation 1)
    COGNITIVE: float = 1.5

    # Social coefficient (c2)
    # Controls attraction to global best position.
    # c2=1.5 gives moderate social influence. Typical range: 1.0-2.0.
    # Often c1=c2 for balanced behavior. (Kennedy & Eberhart 1995)
    SOCIAL: float = 1.5

    # Maximum velocity (v_max)
    # Limits velocity to prevent particles from overshooting.
    # v_max=0.2 limits movement to 20% of search space per iteration.
    # Prevents oscillation and improves convergence stability.
    # (Kennedy & Eberhart 1995, Section "Vmax")
    V_MAX: float = 0.2

    # Constriction coefficient alternative parameters
    # Some implementations use constriction coefficient χ instead of inertia.
    # φ = c1 + c2, χ = 2κ / |2 - φ - √(φ² - 4φ)|, where κ ∈ [0, 1]
    # (Clerc & Kennedy, 2002)
    USE_CONSTRICTION: bool = False
    CONSTRICTION_KAPPA: float = 1.0

    @staticmethod
    def validate_coefficients(w: float, c1: float, c2: float) -> Tuple[bool, str]:
        """
        Validate PSO coefficients for convergence.

        For guaranteed convergence, the following should hold:
        - w < 1 (inertia allows deceleration)
        - c1 + c2 < 4 * (1 + w) (prevents oscillation)

        Args:
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient

        Returns:
            Tuple of (is_valid, warning_message)
        """
        warnings = []

        if w >= 1.0:
            warnings.append(f"Inertia w={w} >= 1 may cause divergence")

        if c1 + c2 >= 4 * (1 + w):
            warnings.append(f"c1+c2={c1+c2} may cause oscillation "
                          f"(should be < {4*(1+w):.2f})")

        if c1 < 0 or c2 < 0:
            warnings.append("Coefficients should be non-negative")

        if warnings:
            return False, "; ".join(warnings)
        return True, ""

    @staticmethod
    def compute_constriction_coefficient(c1: float, c2: float,
                                         kappa: float = 1.0) -> float:
        """
        Compute constriction coefficient from cognitive/social coefficients.

        Formula: χ = 2κ / |2 - φ - √(φ² - 4φ)|
        where φ = c1 + c2 and κ ∈ [0, 1]

        Reference: Clerc, M. and Kennedy, J. (2002). The particle swarm -
        explosion, stability, and convergence in a multidimensional complex space.

        Args:
            c1: Cognitive coefficient
            c2: Social coefficient
            kappa: Constriction parameter (default 1.0)

        Returns:
            Constriction coefficient χ
        """
        phi = c1 + c2
        if phi <= 4:
            # Constriction not needed for small phi
            return 1.0
        return 2 * kappa / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))


# =============================================================================
# Utility Functions
# =============================================================================

def get_algorithm_defaults(algorithm: str) -> Any:
    """
    Get default configuration class for an algorithm.

    Args:
        algorithm: Algorithm name ('cmaes', 'nsga2', 'dream', 'pso')

    Returns:
        Default configuration dataclass

    Raises:
        ValueError: If algorithm name is not recognized
    """
    defaults_map = {
        'cmaes': CMAESDefaults,
        'cma-es': CMAESDefaults,
        'nsga2': NSGA2Defaults,
        'nsga-ii': NSGA2Defaults,
        'dream': DREAMDefaults,
        'pso': PSODefaults,
        'ga': GADefaults,
        'de': DEDefaults,
        'nelder-mead': NelderMeadDefaults,
        'nelder_mead': NelderMeadDefaults,
        'glue': GLUEDefaults,
        'basin-hopping': BasinHoppingDefaults,
        'basin_hopping': BasinHoppingDefaults,
        'bayesian-opt': BODefaults,
        'bayesian_optimization': BODefaults,
        'moead': MOEADDefaults,
        'moea/d': MOEADDefaults,
        'sa': SADefaults,
        'simulated-annealing': SADefaults,
        'simulated_annealing': SADefaults,
        'lbfgs': LBFGSDefaults,
        'l-bfgs': LBFGSDefaults,
        'adam': AdamDefaults,
    }

    algorithm_lower = algorithm.lower()
    if algorithm_lower not in defaults_map:
        raise ValueError(f"Unknown algorithm '{algorithm}'. "
                        f"Available: {list(defaults_map.keys())}")

    return defaults_map[algorithm_lower]


def validate_hyperparameters(algorithm: str, params: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate hyperparameters for an algorithm.

    Args:
        algorithm: Algorithm name
        params: Dictionary of hyperparameter values

    Returns:
        Dictionary of validation warnings (empty if all valid)
    """
    warnings = {}
    defaults = get_algorithm_defaults(algorithm)

    if algorithm.lower() in ('cmaes', 'cma-es'):
        if 'sigma' in params:
            valid, msg = defaults.validate_sigma(params['sigma'])
            if not valid:
                warnings['sigma'] = msg

    elif algorithm.lower() in ('nsga2', 'nsga-ii'):
        if 'eta_c' in params:
            valid, msg = defaults.validate_eta(params['eta_c'], 'eta_c')
            if not valid:
                warnings['eta_c'] = msg
        if 'eta_m' in params:
            valid, msg = defaults.validate_eta(params['eta_m'], 'eta_m')
            if not valid:
                warnings['eta_m'] = msg

    elif algorithm.lower() == 'dream':
        if 'n_chains' in params and 'n_params' in params:
            valid, msg = defaults.validate_chains(
                params['n_chains'], params['n_params']
            )
            if not valid:
                warnings['n_chains'] = msg

    elif algorithm.lower() == 'pso':
        w = params.get('inertia', defaults.INERTIA)
        c1 = params.get('cognitive', defaults.COGNITIVE)
        c2 = params.get('social', defaults.SOCIAL)
        valid, msg = defaults.validate_coefficients(w, c1, c2)
        if not valid:
            warnings['coefficients'] = msg

    return warnings


# =============================================================================
# GA (Genetic Algorithm) Configuration
# Reference: Holland, J.H. (1975). Adaptation in Natural and Artificial Systems.
#            University of Michigan Press.
# =============================================================================

@dataclass
class GADefaults:
    """
    Default hyperparameters for GA (Genetic Algorithm).

    GA is a population-based evolutionary algorithm that uses selection,
    crossover, and mutation to evolve solutions.

    Reference:
        Holland, J.H. (1975). Adaptation in Natural and Artificial Systems.
        University of Michigan Press.
    """

    # Crossover rate (probability of applying crossover to a pair)
    # 0.9 is a common default providing high recombination.
    CROSSOVER_RATE: float = 0.9

    # Mutation rate (probability of mutating each gene)
    # 0.1 provides moderate mutation for exploration.
    MUTATION_RATE: float = 0.1

    # Mutation scale (magnitude of mutations)
    # 0.1 gives moderate perturbation in normalized [0,1] space.
    MUTATION_SCALE: float = 0.1

    # Tournament size for selection
    # 3 is a common default balancing selection pressure and diversity.
    TOURNAMENT_SIZE: int = 3

    # Number of elite individuals preserved each generation
    # 2 ensures the best solutions are never lost.
    ELITISM_COUNT: int = 2

    @staticmethod
    def validate_rates(crossover_rate: float, mutation_rate: float) -> Tuple[bool, str]:
        """
        Validate GA rates.

        Args:
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability

        Returns:
            Tuple of (is_valid, warning_message)
        """
        warnings = []

        if crossover_rate < 0 or crossover_rate > 1:
            warnings.append(f"Crossover rate {crossover_rate} should be in [0, 1]")

        if mutation_rate < 0 or mutation_rate > 1:
            warnings.append(f"Mutation rate {mutation_rate} should be in [0, 1]")

        if crossover_rate < 0.5:
            warnings.append(f"Low crossover rate {crossover_rate} may reduce exploration")

        if mutation_rate > 0.5:
            warnings.append(f"High mutation rate {mutation_rate} may cause random search")

        if warnings:
            return False, "; ".join(warnings)
        return True, ""


# =============================================================================
# DE (Differential Evolution) Configuration
# Reference: Storn, R. and Price, K. (1997). "Differential Evolution - A Simple
#            and Efficient Heuristic for Global Optimization over Continuous
#            Spaces". Journal of Global Optimization, 11(4), 341-359.
# =============================================================================

@dataclass
class DEDefaults:
    """
    Default hyperparameters for DE (Differential Evolution).

    DE is a population-based evolutionary algorithm that uses vector differences
    for mutation. Effective for continuous optimization problems.

    Reference:
        Storn, R. and Price, K. (1997). Differential Evolution - A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces.
        Journal of Global Optimization, 11(4), 341-359.
    """

    # Differential weight (F)
    # Controls the amplification of differential variation.
    # F=0.8 is a robust default. Typical range: 0.4-1.0.
    # (Storn & Price 1997, Section 3.1)
    F: float = 0.8

    # Crossover probability (CR)
    # Probability of inheriting from mutant vector vs parent.
    # CR=0.9 provides high recombination. Typical range: 0.1-1.0.
    # (Storn & Price 1997, Section 3.2)
    CR: float = 0.9

    @staticmethod
    def validate_parameters(F: float, CR: float) -> Tuple[bool, str]:
        """
        Validate DE parameters.

        Args:
            F: Differential weight
            CR: Crossover probability

        Returns:
            Tuple of (is_valid, warning_message)
        """
        warnings = []

        if F < 0 or F > 2:
            warnings.append(f"F={F} outside typical range [0, 2]")

        if CR < 0 or CR > 1:
            warnings.append(f"CR={CR} should be in [0, 1]")

        if warnings:
            return False, "; ".join(warnings)
        return True, ""


# =============================================================================
# Nelder-Mead Configuration
# Reference: Nelder, J.A. and Mead, R. (1965). "A Simplex Method for Function
#            Minimization". The Computer Journal, 7(4), 308-313.
#            Gao, F. and Han, L. (2012). "Implementing the Nelder-Mead simplex
#            algorithm with adaptive parameters".
# =============================================================================

@dataclass
class NelderMeadDefaults:
    """
    Default hyperparameters for Nelder-Mead Simplex algorithm.

    A derivative-free optimization method using simplex transformations.

    References:
        Nelder, J.A. and Mead, R. (1965). A Simplex Method for Function
        Minimization. The Computer Journal, 7(4), 308-313.

        Gao, F. and Han, L. (2012). Implementing the Nelder-Mead simplex
        algorithm with adaptive parameters. Computational Optimization
        and Applications, 51(1), 259-277.
    """

    # Reflection coefficient (alpha)
    # Controls how far the reflected point is from the centroid.
    # Standard value is 1.0. (Nelder & Mead 1965)
    ALPHA: float = 1.0

    # Expansion coefficient (gamma)
    # Controls how far to expand when reflection is successful.
    # Standard value is 2.0. (Nelder & Mead 1965)
    GAMMA: float = 2.0

    # Contraction coefficient (rho)
    # Controls how far to contract toward centroid.
    # Standard value is 0.5. (Nelder & Mead 1965)
    RHO: float = 0.5

    # Shrinkage coefficient (sigma)
    # Controls how much to shrink the simplex.
    # Standard value is 0.5. (Nelder & Mead 1965)
    SIGMA: float = 0.5

    # Initial simplex size
    # Size of initial simplex in normalized [0,1] space.
    SIMPLEX_SIZE: float = 0.1

    # Convergence tolerance for simplex size
    X_TOL: float = 1e-6

    # Convergence tolerance for function values
    F_TOL: float = 1e-6

    # Use adaptive parameters for high dimensions (Gao & Han 2012)
    ADAPTIVE: bool = True

    @staticmethod
    def compute_adaptive_parameters(n_params: int) -> Dict[str, float]:
        """
        Compute adaptive parameters for high-dimensional problems.

        Uses formulas from Gao & Han (2012) for dimensions > 2.

        Args:
            n_params: Number of parameters

        Returns:
            Dictionary with adaptive alpha, gamma, rho, sigma
        """
        if n_params <= 2:
            return {
                'alpha': NelderMeadDefaults.ALPHA,
                'gamma': NelderMeadDefaults.GAMMA,
                'rho': NelderMeadDefaults.RHO,
                'sigma': NelderMeadDefaults.SIGMA
            }

        return {
            'alpha': 1.0,
            'gamma': 1.0 + 2.0 / n_params,
            'rho': 0.75 - 0.5 / n_params,
            'sigma': 1.0 - 1.0 / n_params
        }

    @staticmethod
    def validate_parameters(alpha: float, gamma: float, rho: float,
                           sigma: float) -> Tuple[bool, str]:
        """
        Validate Nelder-Mead parameters.

        Args:
            alpha: Reflection coefficient
            gamma: Expansion coefficient
            rho: Contraction coefficient
            sigma: Shrinkage coefficient

        Returns:
            Tuple of (is_valid, warning_message)
        """
        warnings = []

        if alpha <= 0:
            warnings.append(f"Alpha {alpha} should be positive")

        if gamma <= 1:
            warnings.append(f"Gamma {gamma} should be > 1")

        if rho <= 0 or rho >= 1:
            warnings.append(f"Rho {rho} should be in (0, 1)")

        if sigma <= 0 or sigma >= 1:
            warnings.append(f"Sigma {sigma} should be in (0, 1)")

        if warnings:
            return False, "; ".join(warnings)
        return True, ""


# =============================================================================
# GLUE Configuration
# Reference: Beven, K. and Binley, A. (1992). GLUE.
# =============================================================================

@dataclass
class GLUEDefaults:
    """Default hyperparameters for GLUE."""
    THRESHOLD: float = 0.0
    SHAPING_FACTOR: float = 1.0
    SAMPLING: str = 'lhs'

    @staticmethod
    def validate_threshold(threshold: float, metric_range: Tuple[float, float] = (-1, 1)) -> Tuple[bool, str]:
        """Validate GLUE threshold."""
        if threshold < metric_range[0] or threshold > metric_range[1]:
            return False, f"Threshold {threshold} outside expected range {metric_range}"
        return True, ""


# =============================================================================
# Basin Hopping Configuration
# Reference: Wales, D.J. and Doye, J.P.K. (1997). Basin-Hopping.
# =============================================================================

@dataclass
class BasinHoppingDefaults:
    """Default hyperparameters for Basin Hopping."""
    STEP_SIZE: float = 0.5
    TEMPERATURE: float = 1.0
    LOCAL_STEPS: int = 50
    LOCAL_METHOD: str = 'nelder_mead'
    TARGET_ACCEPT: float = 0.5
    ADAPT_INTERVAL: int = 10

    @staticmethod
    def validate_parameters(step_size: float, temperature: float) -> Tuple[bool, str]:
        """Validate Basin Hopping parameters."""
        if step_size <= 0:
            return False, f"Step size must be positive, got {step_size}"
        if temperature <= 0:
            return False, f"Temperature must be positive, got {temperature}"
        return True, ""


# =============================================================================
# Bayesian Optimization Configuration
# Reference: Snoek, J. et al. (2012). Practical Bayesian Optimization.
# =============================================================================

@dataclass
class BODefaults:
    """Default hyperparameters for Bayesian Optimization."""
    INITIAL_SAMPLES_FACTOR: int = 2  # n_initial = max(5, factor * n_params)
    ACQUISITION: str = 'ei'  # 'ei', 'ucb', 'pi'
    XI: float = 0.01  # Exploration parameter for EI/PI
    KAPPA: float = 2.576  # UCB parameter (97.5th percentile)
    RESTARTS: int = 10

    @staticmethod
    def validate_acquisition(acquisition: str) -> Tuple[bool, str]:
        """Validate acquisition function."""
        valid = ['ei', 'ucb', 'pi']
        if acquisition.lower() not in valid:
            return False, f"Acquisition must be one of {valid}, got {acquisition}"
        return True, ""


# =============================================================================
# MOEA/D Configuration
# Reference: Zhang, Q. and Li, H. (2007). MOEA/D.
# =============================================================================

@dataclass
class MOEADDefaults:
    """Default hyperparameters for MOEA/D."""
    NEIGHBORS: int = 20
    CR: float = 1.0  # Crossover rate
    F: float = 0.5   # DE scaling factor
    MUTATION: float = 0.1
    NR: int = 2      # Max replacements per offspring (Zhang & Li 2007)
    DECOMPOSITION: str = 'tchebycheff'

    @staticmethod
    def validate_decomposition(decomposition: str) -> Tuple[bool, str]:
        """Validate decomposition method."""
        valid = ['tchebycheff', 'weighted_sum', 'pbi']
        if decomposition.lower() not in valid:
            return False, f"Decomposition must be one of {valid}, got {decomposition}"
        return True, ""


# =============================================================================
# Simulated Annealing Configuration
# Reference: Kirkpatrick, S. et al. (1983). Optimization by Simulated Annealing.
# =============================================================================

@dataclass
class SADefaults:
    """Default hyperparameters for Simulated Annealing."""
    INITIAL_TEMP: float = 1.0
    FINAL_TEMP: float = 1e-6
    COOLING_SCHEDULE: str = 'exponential'
    COOLING_RATE: float = 0.95
    STEP_SIZE: float = 0.1
    STEPS_PER_TEMP: int = 10
    ADAPTIVE_STEP: bool = True

    @staticmethod
    def validate_temperatures(initial: float, final: float) -> Tuple[bool, str]:
        """Validate SA temperatures."""
        if initial <= 0:
            return False, f"Initial temp must be positive, got {initial}"
        if final <= 0:
            return False, f"Final temp must be positive, got {final}"
        if final >= initial:
            return False, "Final temp must be < initial temp"
        return True, ""


# =============================================================================
# L-BFGS Configuration
# Reference: Nocedal, J. (1980). Updating quasi-Newton matrices.
# =============================================================================

@dataclass
class LBFGSDefaults:
    """Default hyperparameters for L-BFGS."""
    LR: float = 0.1
    HISTORY_SIZE: int = 10
    C1: float = 1e-4  # Armijo condition
    C2: float = 0.9   # Wolfe condition
    GRADIENT_EPSILON: float = 1e-4
    GRADIENT_CLIP_VALUE: float = 1.0

    @staticmethod
    def validate_wolfe(c1: float, c2: float) -> Tuple[bool, str]:
        """Validate Wolfe condition parameters."""
        if not 0 < c1 < c2 < 1:
            return False, f"Must have 0 < c1 < c2 < 1, got c1={c1}, c2={c2}"
        return True, ""


# =============================================================================
# Adam Configuration
# Reference: Kingma, D.P. and Ba, J. (2015). Adam.
# =============================================================================

@dataclass
class AdamDefaults:
    """Default hyperparameters for Adam optimizer."""
    LR: float = 0.01
    BETA1: float = 0.9
    BETA2: float = 0.999
    EPS: float = 1e-8
    GRADIENT_EPSILON: float = 1e-4
    GRADIENT_CLIP_VALUE: float = 1.0

    @staticmethod
    def validate_betas(beta1: float, beta2: float) -> Tuple[bool, str]:
        """Validate Adam beta parameters."""
        if not 0 <= beta1 < 1:
            return False, f"Beta1 must be in [0,1), got {beta1}"
        if not 0 <= beta2 < 1:
            return False, f"Beta2 must be in [0,1), got {beta2}"
        return True, ""


__all__ = [
    'CMAESDefaults',
    'NSGA2Defaults',
    'DREAMDefaults',
    'PSODefaults',
    'GADefaults',
    'DEDefaults',
    'NelderMeadDefaults',
    'GLUEDefaults',
    'BasinHoppingDefaults',
    'BODefaults',
    'MOEADDefaults',
    'SADefaults',
    'LBFGSDefaults',
    'AdamDefaults',
    'get_algorithm_defaults',
    'validate_hyperparameters',
]
