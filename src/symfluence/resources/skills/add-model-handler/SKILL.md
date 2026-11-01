---
name: add-model-handler
description: >-
  Add or modify a SYMFLUENCE hydrological model — runner, preprocessor,
  postprocessor, config manager, and calibration wiring — via model_manifest()
  and the unified R.* registries. Covers the models/ taxonomy (process-based,
  conceptual, ML, LSM, framework, routing) and the settings/forcing/simulations
  path conventions.
when_to_use:
  - Adding a new model (SUMMA/HYPE/FUSE/GR/NGEN style) or wiring one into calibration
  - Modifying a model's runner / preprocessor / postprocessor / config manager
  - Understanding how a model executes end to end in the pipeline
---

# Adding & Understanding SYMFLUENCE Models

SYMFLUENCE orchestrates 40+ models through one pipeline: preprocess →
run → postprocess → (calibrate). This skill is the map of `models/` plus the
recipe for adding one. Paths are relative to `src/symfluence/` unless noted.

## 1. The registration mechanism (read this first)

**Canonical pattern: `model_manifest()`** — called from the package's `register()`
function (see §2). Defined in `core/registry.py` (`model_manifest`). One call
wires every component into the unified `R.*` registries:

```python
# models/<name>/__init__.py
from symfluence.core.registry import model_manifest
from .config import MyModelConfigAdapter
from .extractor import MyModelResultExtractor

model_manifest(
    "MYMODEL",                       # normalized to UPPERCASE key
    runner=MyModelRunner,            # the runner class (or pass a dotted path)
    runner_method="run",             # method on runner to call; default "run"
    preprocessor=MyModelPreProcessor,
    postprocessor=MyModelPostProcessor,
    config_adapter=MyModelConfigAdapter,
    result_extractor=MyModelResultExtractor,
    plotter=MyModelPlotter,                       # optional
    worker=MyModelWorker,                         # optional, for calibration
    parameter_manager=MyModelParameterManager,    # optional, for calibration
    build_instructions_module="symfluence.models.mymodel.build_instructions",  # optional, lazy str
    forcing_adapter_module="symfluence.models.mymodel.forcing_adapter",        # optional, lazy str
    init_preset_module="symfluence.models.mymodel.init_preset",                # optional, lazy str
)
```

Full keyword list (all optional except `model_name`): `preprocessor`, `runner`,
`runner_method`, `postprocessor`, `visualizer`, `config_adapter`,
`config_schema`, `config_defaults`, `config_transformers`, `config_validator`,
`result_extractor`, `optimizer`, `worker`, `parameter_manager`,
`decision_analyzer`, `sensitivity_analyzer`, `koopman_analyzer`, `plotter`,
`forcing_adapter`, `forcing_adapter_module`, `init_preset_module`,
`build_instructions_module`, `aliases`, `self_training`.

`aliases=("HEC-HMS",)` declares alternate spellings that resolve to the
canonical name across every registry — hyphenated forms, short names, a legacy
key — so the mapping lives with the package that owns the name rather than in a
core table. `self_training=True` marks a model whose "calibration" is internal
training during the run step (gradient descent) rather than an external
DDS/PSO search: it registers no optimizer or worker, and the calibration and
sensitivity-analysis paths skip it instead of reporting a failure.

The three `*_module` keywords declare *where* a capability lives instead of
importing it: the dotted path is recorded in the registry and imported only
when a consumer asks for it. Use them for anything that registers through a
decorator in a side module (forcing adapters, `symfluence init` presets, build
instructions) — the framework never scans a source tree for those files, so a
capability that is not declared is not discovered, for in-tree and external
plugin packages alike.

**REMOVED legacy pattern (pre-1.0 cleanup — these decorators no longer exist):**
`@ModelRegistry.register_runner/...`, `@OptimizerRegistry.register_worker/...`,
and the per-subsystem registration shims (`@AcquisitionRegistry.register`,
`@EvaluationRegistry.register`, `@ObjectiveRegistry.register`,
`@ForcingAdapterRegistry.register_adapter`, `@AnalysisRegistry.register_*`,
`@PresetRegistry.register_preset`, `@DelineationRegistry.register`,
`BMIRegistry().register`). The registry classes survive as lookup facades
only. If you see the decorators in old branches or examples, translate to
`model_manifest()` or `R.<registry>.add()`. **Pass everything through
`model_manifest()`** (`runner=`, `worker=`, `parameter_manager=`); use
`R.<registry>.add(...)` directly only for one-offs.

**Lower-level direct API** (what `model_manifest` calls underneath):
`R.runners.add("MYMODEL", MyRunner, runner_method="run")`,
`R.runners.add_lazy("MYMODEL", "dotted.path.to.Runner")` (import on first use),
`R.runners.alias("ALT_NAME", "MYMODEL")`. Same `.add/.add_lazy/.alias` on every
`R.*` registry (`registries.py`).

## 2. How models get loaded (so registration runs)

Both in-tree and external models register through the **`symfluence.plugins`
entry-point group** — there is no hard-coded model list. Each model package
exposes a top-level `register()` that calls `model_manifest(...)` / `R.*.add(...)`,
and an entry point points at it:

- **In-tree:** `pyproject.toml` declares, under
  `[project.entry-points."symfluence.plugins"]`,
  `symfluence_<name> = "symfluence.models.<name>:register"`.
- **External plugin:** the plugin's own `pyproject.toml` declares the same group,
  e.g. `mymodel = "mypkg:register"`. (jFUSE, cFUSE register this way.)

`core/_bootstrap.py` (`_discover_plugins`) loads every entry point and calls its
`register()` once, from `symfluence/__init__.py`. **To add a built-in model you
must both expose `register()` in the package and add its `symfluence_<name>`
entry point to `pyproject.toml`** — otherwise registration silently never runs.

## 3. Model taxonomy — pick the closest exemplar

(Run `symfluence list models` for the models registered right now; the table below
is the categories, not an exhaustive list.)

| Category | Execution | Exemplars | Notes |
|----------|-----------|-----------|-------|
| Process-based executable | subprocess (Fortran/C binary) | `summa`, `fuse`, `hype`, `mesh`, `vic`, `swat`, `prms` | `_build_command()` returns argv; needs `build_instructions` |
| Conceptual / lumped | in-process (Python/R) | `gr` (airGR via rpy2), `crhm`, `mhm` | no `_build_command`; direct invocation |
| ML / data-driven | in-process (PyTorch) | `lstm`, `gnn` | `model.py` defines net; lazy-imports torch via `__getattr__` |
| Land-surface / ESM | subprocess + CIME build | `clm`, `noahmp`, `clmparflow` | params via NetCDF not text files |
| Framework / coupling | BMI orchestration | `ngen`, `troute` | heavy config generation (realization JSON) |
| Routing | subprocess, takes runoff input | `mizuroute`, `troute` | coupled after a hydrologic model |
| Groundwater | subprocess | `modflow`, `parflow` | heavy mesh/grid preprocessing |

Flagship reference: **`models/summa/`**. Simpler reference: **`models/gr/`**.
Routing reference: **`models/mizuroute/`**.

## 4. Runner contract

`BaseModelRunner` — `core/modeling/base/base_runner.py`. Mixes in `ModelComponentMixin`,
`PathResolverMixin`, `ShapefileAccessMixin`, `SubprocessExecutionMixin`,
`SlurmExecutionMixin`. Set the class var `MODEL_NAME = "MYMODEL"`.

Constructor: `__init__(self, config, logger, reporting_manager=None)` (config is
coerced from dict to `SymfluenceConfig`).

`run()` dispatch order (`base_runner.py`): (1) legacy method dispatch if the
registry records `runner_method != 'run'` (e.g. `run_summa`); (2) template
execution — if `_build_run_command()` returns argv, it's executed via
`execute_subprocess`, outputs verified, `output_dir` returned; (3) else
`NotImplementedError`.

**For a subprocess model, the minimum is `_build_run_command()` (or
`_build_command()` on the template base).** Common override hooks:
`_setup_model_specific_paths()`, `_get_output_dir()` (default
`project_dir/simulations/{experiment_id}/{model_name}`), `_get_expected_outputs()`,
`_prepare_run()`, `_validate_required_config()`, `_get_environment()`.

**Convenience base: `UnifiedModelRunner`** (`core/modeling/templates/model_template.py`)
combines `BaseModelRunner` + spatial orchestration. SUMMA/FUSE subclass it.
Minimal:
```python
from symfluence.core.modeling.templates import UnifiedModelRunner

class MyModelRunner(UnifiedModelRunner):
    MODEL_NAME = "MYMODEL"
    def _build_command(self):
        return [str(self.model_exe), "-m", str(self.file_manager)]
```
In-process models (GR/LSTM) subclass `BaseModelRunner` directly, skip
`_build_command`, and execute in `run()`.

## 5. Preprocessor contract

`BaseModelPreProcessor` — `core/modeling/base/base_preprocessor.py`. Consumes the
**model-ready store** (`data/model_ready/forcings`, `.../attributes`) and the
discretized domain (HRU/GRU shapefiles), and writes model-native inputs.

Set `MODEL_NAME`. Implement `run_preprocessing() -> bool` — typically just
`return self.run_preprocessing_template()`. The template runs these hooks in
order: `_pre_setup()` → `create_directories()` → `copy_base_settings()` →
`_prepare_forcing()` → `_create_model_configs()` → `_post_setup()`.

Standard attributes it sets up:
- `setup_dir = project_dir/settings/{MODEL_NAME}/`
- `forcing_dir = project_forcing_dir/{MODEL_NAME}_input/`
- `forcing_basin_path = project_dir/data/model_ready/forcings`
- `shapefile_path`, `intersect_path` for forcing↔catchment intersection.

Base settings templates live in `resources/base_settings/{MODEL}/` and are
copied into `setup_dir` by `copy_base_settings()`. Real preprocessors delegate
detail to manager classes (e.g. SUMMA → `SummaForcingProcessor`,
`SummaConfigManager`, `SummaAttributesManager`).

## 6. Config manager

`models/<model>/config_manager.py` generates model-native config files
(SUMMA `fileManager.txt`/`attributes.nc`/`coldState.nc`/`trialParams.txt`;
HYPE `info.txt`/`filedir.txt`/`par.txt`/`GeoData.txt`) from the SYMFLUENCE YAML
config + spatial/temporal params. Called from the preprocessor's
`_create_model_configs()`. No fixed base class — it's a plain helper, usually
mixing in `PathResolverMixin`.

## 7. Postprocessor contract

`BaseModelPostProcessor` — `core/modeling/base/base_postprocessor.py`. Converts model
output → standardized streamflow for evaluation. Abstract: `_get_model_name()`
and `extract_streamflow() -> Optional[Path]`. Writes to
`project_dir/results/{experiment_id}_results.csv` and a CF NetCDF in
`data/model_output/`.

**Most models should subclass `StandardModelPostProcessor`**
(`core/modeling/base/standard_postprocessor.py`) and just set class attributes:
```python
class MyModelPostProcessor(StandardModelPostProcessor):
    model_name = "MYMODEL"
    output_file_pattern = "{domain}_{experiment}_output.nc"
    streamflow_variable = "discharge"
    streamflow_unit = "mm_per_day"   # or "cms"
    netcdf_selections = {"hru": 0}
```
Use `RoutedModelPostProcessor` if the model's streamflow comes from mizuRoute
(`IRFroutedRunoff`, reach selection via `SIM_REACH_ID`). Helpers:
`convert_mm_per_day_to_cms`, `get_catchment_area_km2`, `read_netcdf_streamflow`,
`save_streamflow_to_results`.

## 8. Calibration integration (optional but common)

To make a model calibratable, provide a **worker** and a **parameter manager**
(pass via `model_manifest(worker=..., parameter_manager=...)`), and one or more
**calibration targets**.

**Worker** — `BaseWorker` (`optimization/workers/base_worker.py`). Implement:
- `apply_parameters(params, settings_dir, **kw) -> bool` — write trial params to
  model files.
- `run_model(config, settings_dir, output_dir, **kw) -> bool` — execute the model.
- `calculate_metrics(output_dir, config, **kw) -> Dict[str, Any]` — read output,
  compute KGE/NSE/etc.
The base `evaluate(task: WorkerTask) -> WorkerResult` orchestrates the three with
retry/backoff and score transformation (objective → maximization). `WorkerTask`
carries `params` (normalized [0,1]), `settings_dir`, `output_dir`, `sim_dir`,
`proc_id`, `iteration`.

**Parameter manager** — `BaseParameterManager`
(`optimization/core/base_parameter_manager.py`). Implement `_get_parameter_names`,
`_load_parameter_bounds` (`{name: {'min','max','transform':'log'|'linear'}}`),
`update_model_files(params)`, `get_initial_parameters`. Base provides
`normalize_parameters` / `denormalize_parameters` (handles log-space) and
`validate_parameters`.

**Calibration target** — subclass the relevant evaluator
(`StreamflowEvaluator`, `ETEvaluator`, `SnowEvaluator`, …) in
`models/<model>/calibration/targets.py`; implement `get_simulation_files` and
`extract_simulated_data`. Register with the composite-key decorator on the target
class: `@R.calibration_targets.add('MYMODEL_STREAMFLOW')` (e.g. `GR_STREAMFLOW`,
`SUMMA_SNOW`).

**Optimizer loop:** algorithms in `core/calibration/optimizers/algorithms/`
(`dds.py`, `pso.py`, `sce_ua.py`, `de.py`, …) call back into the worker each
iteration: denormalize → `apply_parameters` → `run_model` → `calculate_metrics`.
Workers run in `process_N/` dirs created by
`core/calibration/mixins/parallel/directory_manager.py`:
`process_{id}/settings/{model}/` and `process_{id}/simulations/{exp}/{model}/`.

**Regionalization** (spatially-varying params from attributes):
`optimization/regionalization/strategies.py` — transfer function
`param = a + b * attribute_norm` (`TransferFunctionRegionalization`). A model
supplies its attribute→param map, e.g.
`models/hype/calibration/hype_regionalization.py` (`HYPE_LU_PARAM_CONFIG`,
`HYPE_SOIL_PARAM_CONFIG`).

## 9. Path & naming conventions

```
settings/{MODEL}/                         model config files (per-domain)
forcing/{MODEL}_input/                    preprocessed model-native forcing
simulations/{experiment_id}/{MODEL}/      model run outputs
results/{experiment_id}_results.csv       standardized streamflow
resources/base_settings/{MODEL}/          template config files (in package)
optimization/.../process_N/{settings,simulations}/{MODEL}/   calibration workers
```
Classes: `{Model}Runner`, `{Model}PreProcessor`, `{Model}PostProcessor`,
`{Model}ConfigAdapter`, `{Model}ResultExtractor`, `{Model}Worker`,
`{Model}ParameterManager`. Class var `MODEL_NAME = "MYMODEL"` on runner &
preprocessor. SPDX header on every file. Lazy-import heavy deps (torch, rpy2,
xarray) inside methods. Line length 120, Python 3.11+.

## 10. Step-by-step: add a new model

1. Pick the taxonomy category (§3) and copy the closest exemplar dir.
2. Create `models/<name>/` with: `__init__.py` (calls `model_manifest`),
   `config.py` (`ConfigAdapter`), `runner.py`, `preprocessor.py`,
   `postprocessor.py`, `extractor.py`. Add `config_manager.py`,
   `build_instructions.py`, `calibration/` as the category needs (§ taxonomy).
3. Implement the runner (`_build_command` for subprocess models; in-process
   `run()` otherwise), preprocessor (`run_preprocessing` →
   `run_preprocessing_template` + hooks), postprocessor (subclass
   `StandardModelPostProcessor`).
4. Add base-settings templates under `resources/base_settings/<MODEL>/`.
5. **Expose `register()` in the package and add a `symfluence_<name>` entry point
   to `pyproject.toml`** (§2) — for an external plugin, declare it in the plugin's
   own `pyproject.toml`. Without both, nothing loads.
6. For calibration: add worker + parameter_manager (via `model_manifest`) and a
   calibration target.
7. Verify registration:
   `python -c "import symfluence; from symfluence.core.registries import R; print('MYMODEL' in [k.upper() for k in R.runners.keys()])"`
   (adjust to the registry listing API in `core/registry.py`).
8. Smoke-test: a tiny domain through `model_specific_preprocessing` → `run_model`
   → `evaluate_model`. Then `ruff check src/symfluence/` and `mypy`.

## 11. Key file reference

| Concern | File |
|---------|------|
| `model_manifest()` (canonical registration) | `core/registry.py` (`model_manifest`) |
| Unified registries (`R.runners` etc.) | `core/registries.py` (`R`), `core/registry.py` (`Registry`) |
| Model entry points (MUST add) | `pyproject.toml` `[project.entry-points."symfluence.plugins"]` |
| Plugin entry-point discovery | `core/_bootstrap.py` (`_discover_plugins`) |
| Runner base | `core/modeling/base/base_runner.py` |
| Runner template (combined) | `core/modeling/templates/model_template.py` (`UnifiedModelRunner`) |
| Preprocessor base | `core/modeling/base/base_preprocessor.py` |
| Postprocessor base / standard | `core/modeling/base/base_postprocessor.py`, `standard_postprocessor.py` |
| Calibration worker base | `optimization/workers/base_worker.py` (`BaseWorker`, `WorkerTask`) |
| Parameter manager base | `optimization/core/base_parameter_manager.py` |
| Optimizer algorithms | `core/calibration/optimizers/algorithms/{dds,pso,sce_ua,de}.py` |
| Worker process dirs | `core/calibration/mixins/parallel/directory_manager.py` |
| Regionalization | `optimization/regionalization/strategies.py` |
| Calibration targets | `optimization/calibration_targets/`, `models/<m>/calibration/targets.py` |
| Flagship exemplar | `models/summa/` |
| Simple / routing exemplars | `models/gr/`, `models/mizuroute/` |
