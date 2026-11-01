---
name: debug-calibration
description: >-
  Diagnose SYMFLUENCE calibration/optimization that misbehaves — flat or stuck
  scores, workers crashing, params not reaching the model, NaN metrics, parallel
  process-dir cross-talk, or regionalization not varying spatially. Fault-tree
  for the DDS/PSO/SCE-UA/DE loop and the BaseWorker apply→run→metrics cycle.
when_to_use:
  - A calibration "runs but doesn't improve", or the score is flat/stuck
  - A calibration worker errors, crashes, or returns no score / NaN metrics
  - Params don't change the output, or regionalized params come out uniform
---

# Debugging SYMFLUENCE Calibration

The calibration loop is: optimizer proposes normalized params → worker applies
them to model files → runs the model → reads output → computes an objective →
optimizer uses the score. A failure can hide in any link. This skill is the
fault-tree. Paths relative to `src/symfluence/`.

## 1. The loop, end to end (know the moving parts)

```
optimizer (algorithms/{dds,pso,sce_ua,de}.py)
  proposes x ∈ [0,1]^n
   → parameter_manager.denormalize_parameters(x)            → {param: real value}
   → WorkerTask(params, settings_dir, output_dir, proc_id, iteration, sim_dir)
   → BaseWorker.evaluate(task):                              # workers/base_worker.py
        apply_parameters(params, settings_dir)  → write trial params to model files
        run_model(config, settings_dir, output_dir)         → execute model subprocess
        calculate_metrics(output_dir, config)    → {KGE, NSE, ...}
        → score (transformed to MAXIMIZATION) → WorkerResult
   → optimizer.record_iteration / update_best
```

Workers run in isolated dirs (`mixins/parallel/directory_manager.py`):
`process_{id}/settings/{model}/` (par files) and
`process_{id}/simulations/{exp}/{model}/` (outputs). Each process gets a copy of
base settings.

## 2. Triage by symptom

### A. Score is flat / never improves (most common)
The optimizer is exploring but the objective never changes → almost always the
params aren't reaching the model, or the metric is computed on the wrong data.
Check in this order:

1. **Do params actually change model files?** Add logging in the model's
   `apply_parameters` / `parameter_manager.update_model_files`. Inspect a
   `process_N/settings/{model}/` par file across two iterations — is it different?
   If not: `update_model_files` is writing the wrong file/key, or writing to the
   base settings dir instead of the process dir (`settings_dir` from the task).
2. **Does the model actually re-read those files each run?** Some models cache or
   need a specific file regenerated. Confirm `run_model` points at the process
   dir, not the global `settings/{model}/`.
3. **Is the metric reading the simulation that just ran?** The calibration target
   (`get_simulation_files`) must locate *this process's* output in
   `process_N/simulations/...`. If it globs the global sim dir, every iteration
   scores the same stale file → flat score.
4. **Are parameter bounds degenerate?** In `_load_parameter_bounds`, `min == max`
   (or near it) pins the param. Print the bounds dict.
5. **Is normalization collapsing the range?** A `'transform': 'log'` bound with a
   non-positive `min` produces NaN/constant. Verify
   `denormalize_parameters` round-trips: `denorm(norm(x)) ≈ x`.

### B. Workers fail / crash / return no score
- `BaseWorker.evaluate` wraps each attempt with retry/backoff
  (`WORKER_MAX_RETRIES` default 3, `WORKER_BASE_DELAY`). Repeated failures →
  `WorkerResult.error` is set and score is None. Grep the run log for the error
  string.
- **Model subprocess nonzero exit:** check the model's own log in
  `process_N/simulations/{exp}/{model}/`. Reproduce by running that model command
  manually in the process dir.
- **Missing input in process dir:** base-settings copy
  (`directory_manager.copy_base_settings`) didn't include a needed file →
  `run_model` can't start. Compare a process dir against the working single-run
  `settings/{model}/`.
- **Silent model failure (exit 0 but no/garbage output):** the metric step then
  produces NaN. Treat like §C.

### C. Metrics are NaN / nonsensical
- **No overlap** between simulated and observed periods → KGE undefined. Check
  the obs file date range vs `EXPERIMENT_TIME_START/END` and any spinup.
- **Unit mismatch** (mm/day vs cms) in the postprocessor/target — scores look
  wildly off but not NaN. Verify `streamflow_unit` and area conversion
  (`get_catchment_area_km2`).
- **Wrong variable or reach** extracted: target's `extract_simulated_data` reads
  the wrong column/var, or routed models read the wrong `SIM_REACH_ID`.
- **All-NaN simulation:** model ran but produced fill values → back to §B silent
  failure.

### D. Parallel-specific
- Cross-talk between processes = a worker reading/writing a shared path instead
  of its `process_N` path. Every file op in the worker must derive from
  `task.settings_dir` / `task.output_dir` / `task.sim_dir`, never a global.
- Fewer processes improving than expected → some process dirs missing files;
  inspect each `process_N`.

### E. Regionalization: params don't vary spatially
- `TransferFunctionRegionalization` computes `param = a + b * attribute_norm`
  (`optimization/regionalization/strategies.py`). If `b` isn't being calibrated
  (`calibrate_b: False`) or the attribute is constant across units, params are
  uniform. Check the model's param config (e.g.
  `models/hype/calibration/hype_regionalization.py` `HYPE_*_PARAM_CONFIG`).
- Attribute normalization producing all-same values (e.g. a missing attribute
  column defaulting to a constant) → no spatial signal. Verify the attributes
  DataFrame has real per-unit variation.

## 3. Fast diagnostic moves

- **Single-iteration trace:** run the optimizer for 1–2 iterations with verbose
  logging; confirm a par file changes between them and the score changes with it.
- **Manual worker call:** construct a `WorkerTask` with two distinct param sets
  and call `worker.evaluate(task)` directly — isolates the loop from the
  optimizer.
- **Inspect a process dir mid-run:** `process_N/settings/{model}/` (did params
  land?) and `process_N/simulations/{exp}/{model}/` (did the model write output?).
- **Round-trip the param manager:**
  `pm.denormalize_parameters(pm.normalize_parameters(p))` should ≈ `p`.
- **Check the objective transform:** all metrics are converted to a maximization
  convention in `BaseWorker` (`MetricTransformer`); a "decreasing" KGE bug is
  often a sign-convention confusion — confirm what `score` represents.

## 4. Where each thing lives

| Concern | File |
|---------|------|
| Worker base, `evaluate`, retry, score transform | `optimization/workers/base_worker.py` |
| `WorkerTask` / `WorkerResult` dataclasses | `optimization/workers/base_worker.py` |
| Parameter normalize/denormalize/bounds | `optimization/core/base_parameter_manager.py` |
| Optimizer algorithms (loop, callbacks) | `core/calibration/optimizers/algorithms/{dds,pso,sce_ua,de,nsga2,moead}.py` |
| Process-dir creation & settings copy | `core/calibration/mixins/parallel/directory_manager.py` |
| Calibration targets (sim file location, extraction) | `optimization/calibration_targets/`, `models/<m>/calibration/targets.py` |
| Regionalization strategies | `optimization/regionalization/strategies.py` |
| Model-specific worker / param manager | `models/<model>/calibration/{worker,parameter_manager}.py` |
| Optimizer/worker registration | `model_manifest(worker=, parameter_manager=)` |
| Optimization algorithms | `ALGORITHM_REGISTRY` in `core/calibration/optimizers/algorithms/__init__.py` |

## 5. Config keys that affect calibration behavior

`CALIBRATION_METRIC` (default `KGE`), `COMPOSITE_METRIC` (weighted multi-metric),
`WORKER_MAX_RETRIES`, `WORKER_BASE_DELAY`, the optimizer's iteration/population
settings, multi-gauge keys (`MULTI_GAUGE_CALIBRATION`, `MULTI_GAUGE_AGGREGATION`,
`MULTI_GAUGE_KGE_FLOOR`, `MULTI_GAUGE_MIN_OVERLAP_DAYS`), and `SIM_REACH_ID` for
routed-output targets. Bounds and which params are calibrated come from the
model's `parameter_manager._load_parameter_bounds` (often overridable via config).

## 6. Known model-specific gotchas

These are documented in project memory; check there too:
- **HYPE:** structural ceiling around KGE ~0.13 on some domains — a flat score
  near that may be the model, not a bug.
- **FUSE:** `run_pre` mode (the `run_def` path has been broken); watch for silent
  failures; `para_def.nc` requirements.
- **NGEN:** SIGSEGV retry logic; UDUNITS2 path issues; NOAH→CFE coupling.
