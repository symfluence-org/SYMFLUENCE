---
name: add-optimizer
description: >-
  Add a new optimization/search algorithm to SYMFLUENCE's calibration loop —
  subclass OptimizationAlgorithm, implement the callback-driven optimize(), and
  register it in ALGORITHM_REGISTRY. Covers the algorithm↔worker contract,
  normalized-parameter conventions, the evaluate/record/update callbacks, and
  gradient support.
when_to_use:
  - Implementing a new search algorithm (a new metaheuristic, gradient method, or sampler)
  - Understanding how an optimizer drives the BaseWorker apply→run→metrics loop
  - Wiring a new algorithm so `OPTIMIZATION_ALGORITHM` can select it
---

# Adding a SYMFLUENCE Optimization Algorithm

Algorithms are the *search strategy* of calibration. They are decoupled from the
model: an algorithm proposes normalized parameter vectors and receives scores
through callbacks — it never touches model files itself (that's the worker; see
the debug-calibration and add-model-handler skills). Paths relative to
`src/symfluence/optimization/`.

## 1. The mechanism

Algorithms live in `core/calibration/optimizers/algorithms/<name>.py`, each a subclass of
`OptimizationAlgorithm` (`core/calibration/optimizers/algorithms/base_algorithm.py`). They are
registered in a **plain dict** — `ALGORITHM_REGISTRY` in
`core/calibration/optimizers/algorithms/__init__.py` (this is NOT the unified `R.*` facade; it's a
local name→class map) — and instantiated by `get_algorithm(name, config, logger)`
(case-insensitive; `-`/space are normalized to `_`).

Selection: the config key **`OPTIMIZATION_ALGORITHM`** (`config.optimization.algorithm`)
picks the algorithm; `base_model_optimizer.py` calls `get_algorithm(...)` and
drives it. (Run `symfluence list optimizers` for the currently registered set.)

**To add one you must both** (a) create the algorithm class and (b) add an import
+ a `ALGORITHM_REGISTRY` entry in `algorithms/__init__.py`. A file alone won't be
found.

## 2. The contract

Subclass `OptimizationAlgorithm(ConfigMixin, ABC)`. The base `__init__(config,
logger)` coerces the config and pre-reads the common knobs, so call `super().__init__`:

- `self.max_iterations` ← `NUMBER_OF_ITERATIONS` (`optimization.iterations`, default 100)
- `self.population_size` ← `POPULATION_SIZE` (`optimization.population_size`, default 30)
- `self.target_metric` ← `OPTIMIZATION_METRIC` (`optimization.metric`, default `KGE`)
- `self.penalty_score` ← `ModelDefaults.PENALTY_SCORE` (use for failed evaluations)
- `self._get_config_value(lambda: self.config.x.y, default=, dict_key=)` for any
  algorithm-specific hyperparameters (read your own keys here).

Two abstract members:

```python
@property
def name(self) -> str: ...        # e.g. "DDS", "PSO"

def optimize(
    self,
    n_params: int,
    evaluate_solution: Callable[[np.ndarray, int], float],   # one x∈[0,1]^n, iter → score
    evaluate_population: Callable[[np.ndarray, int], np.ndarray],  # batch (parallel workers)
    denormalize_params: Callable[[np.ndarray], Dict],        # only if you need real values
    record_iteration: Callable,                              # log per-iteration result
    update_best: Callable,                                   # report a new incumbent
    log_progress: Callable,
    evaluate_population_objectives: Optional[Callable] = None,  # multi-objective
    compute_gradient: Optional[NativeGradientCallback] = None,  # native ∇ (loss, grad)
    gradient_mode: str = 'auto',                             # 'auto'|'native'|'finite_difference'
    **kwargs,
) -> Dict[str, Any]: ...           # return best params/score + history
```

Conventions that matter:
- **Parameters are normalized to `[0, 1]^n`.** Search in that space; the worker
  denormalizes (log/linear) when applying. Use `denormalize_params` only if your
  algorithm needs the physical values.
- **Higher score is better.** The worker already transforms the objective to a
  maximization convention, so maximize. A failed evaluation comes back as
  `self.penalty_score` — handle it (don't let NaN poison the search).
- **Evaluate via the callbacks, never by running the model yourself.** Use
  `evaluate_population` for parallel workers (one call → many scores); fall back
  to `evaluate_solution` for serial steps.
- Stay in bounds with the base helpers `self._clip_to_bounds(x)` /
  `self._reflect_at_bounds(x)`. Gradient algorithms can use the base's gradient
  wrappers (native callback vs finite-difference) gated by `gradient_mode`.

## 3. Minimal template

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""MyOpt — <one-line description of the search strategy>."""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from .base_algorithm import OptimizationAlgorithm


class MyOptAlgorithm(OptimizationAlgorithm):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.my_step = self._get_config_value(
            lambda: self.config.optimization.my_step, default=0.2, dict_key='MYOPT_STEP')

    @property
    def name(self) -> str:
        return "MyOpt"

    def optimize(self, n_params, evaluate_solution, evaluate_population,
                 denormalize_params, record_iteration, update_best, log_progress,
                 evaluate_population_objectives=None, compute_gradient=None,
                 gradient_mode='auto', **kwargs) -> Dict[str, Any]:
        x = np.random.default_rng().random(n_params)          # vary by run/seed
        best_x, best_f = x, evaluate_solution(x, 0)
        update_best(best_x, best_f)
        for it in range(1, self.max_iterations):
            cand = self._clip_to_bounds(x + self.my_step * (np.random.random(n_params) - 0.5))
            f = evaluate_solution(cand, it)
            if f > best_f:
                best_x, best_f, x = cand, f, cand
                update_best(best_x, best_f)
            record_iteration(it, best_f)
            log_progress(it, self.max_iterations, best_f)
        return {"best_params": best_x, "best_score": best_f}
```

Mirror the richest in-tree exemplar for real structure: **`algorithms/dds.py`**
(serial + population), `pso.py` (population), `adam.py`/`lbfgs.py` (gradient via
`compute_gradient`), `nsga2.py`/`moead.py` (multi-objective via
`evaluate_population_objectives`).

## 4. Step-by-step

1. Copy the closest exemplar in `core/calibration/optimizers/algorithms/` (DDS for sampling/greedy,
   PSO for population, Adam/L-BFGS for gradient, NSGA-II for multi-objective).
2. Implement `name` + `optimize`; read hyperparameters via `self._get_config_value`.
3. **Register it:** add `from .myopt import MyOptAlgorithm` and a
   `ALGORITHM_REGISTRY` entry (plus any aliases) in `algorithms/__init__.py`.
4. (If it has tunable hyperparameters) add defaults/validation in
   `algorithms/config_schema.py` and a typed field in
   `core/config/models/optimization.py`.
5. Select it: set `OPTIMIZATION_ALGORITHM: MyOpt` in the config.
6. Verify: `python -c "from symfluence.core.calibration.optimizers.algorithms import get_algorithm; print(get_algorithm('myopt', {}, __import__('logging').getLogger()).name)"`.
7. Smoke-test a short calibration on a tiny domain; confirm the score moves and
   `record_iteration`/`update_best` fire. Then `ruff` + `mypy`.

## 5. Key file reference

| Concern | File |
|---------|------|
| Algorithm base class | `core/calibration/optimizers/algorithms/base_algorithm.py` (`OptimizationAlgorithm`) |
| Registry + `get_algorithm` (MUST edit) | `core/calibration/optimizers/algorithms/__init__.py` (`ALGORITHM_REGISTRY`) |
| Hyperparameter defaults/validation | `core/calibration/optimizers/algorithms/config_schema.py` |
| Algorithm config field/selector | `core/config/models/optimization.py` (`algorithm`, `OPTIMIZATION_ALGORITHM`) |
| Driver that calls the algorithm | `optimizers/base_model_optimizer.py` (`get_algorithm`, callback wiring) |
| Exemplars | `algorithms/{dds,pso,de,sce_ua,nsga2,adam,lbfgs}.py` |
| Worker side of the loop (scores) | `workers/base_worker.py` — see the debug-calibration skill |
