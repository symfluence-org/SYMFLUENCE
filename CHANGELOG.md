# Changelog

All notable changes to SYMFLUENCE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **GR forcing corrections and lumped glacier melt (opt-in)**: calibratable `PCORR` (precipitation
  multiplier) and `TCORR` (additive temperature offset) for GR4J + CemaNeige; `GR_SNOW_LAYERS` sets the
  CemaNeige layer count. `GR_GLACIER_ENABLED` runs CemaNeige and GR4J separately and adds degree-day ice
  melt from the outline's share of each elevation layer (`GR_GLACIER_OUTLINE`), with the ice factor
  calibrated (`ICE_DDF`) or solved each run to an observed mean mass balance (`GR_GLACIER_MB_TARGET`),
  optionally scaled to observed glacier area per water year (`GR_GLACIER_AREA_FILE`). Defaults leave
  existing GR runs bit-identical.
- **Wflow distributed glacier mode (opt-in)**: `WFLOW_SPATIAL_MODE: distributed` builds a DEM-derived
  Wflow SBM grid (`WFLOW_GRID_RESOLUTION`) with native river routing, an explicit gauge outlet
  (`WFLOW_OUTLET_MODE: gauge`), per-cell lapsed forcing and finite glacier ice from mapped outlines and
  thickness (`WFLOW_GLACIER_OUTLINE_FILE`, `WFLOW_GLACIER_THICKNESS_DIR`). Run gates check native land,
  river and subsurface balances. Optional calibration-period snow-cover and glacier mass-balance scores
  (`WFLOW_CRYOSPHERE_CONSTRAINTS`) for composite objectives. New `PCORR` precipitation multiplier applied
  through Wflow's native forcing `scale`; calibratable `cf_soil`; per-parameter bound and start overrides
  (`WFLOW_PARAMETER_BOUNDS`, `WFLOW_INITIAL_PARAMETERS`); fixed seasonal temperature offsets
  (`WFLOW_TEMPERATURE_OFFSETS`, DJF/MAM/JJA/SON; Oudin PET recomputed) and frozen-soil infiltration
  (`WFLOW_FROZEN_SOIL`).
- **Native Raven HBV-EC adapter (experimental)**: `RAVEN` model with elevation-band HRUs, optional
  glacier HRUs with finite ice storage (patched Raven build pinned in `binary install raven`), run
  gates for mass balance and ice initialisation, DDS calibration, and an optional LamaH-Ice-style glacier
  mass-balance score (`RAVEN_GLACIER_CONSTRAINTS`) for composite objectives. Calibratable forcing
  corrections `PRECIP_CORR` (rain and snow), `SNOW_CORR` (snow only) and `TEMP_CORR` (additive K), and
  per-parameter bound overrides via `RAVEN_PARAM_BOUNDS`. See `docs/source/models/model_raven.rst`.
- **mHM distributed forcing and glacier options (opt-in)**: with `MHM_DISTRIBUTED`, the grid is now
  conditioned on the catchment polygon; temperature can be lapsed to each cell (`MHM_TEMP_LAPSE_RATE`);
  PET is `hamon` or `oudin` (`MHM_PET_METHOD`). Glacier outlines (`MHM_GLACIER_OUTLINE`) mark sealed
  glacier cells that receive external degree-day ice melt. Calibration pseudo-parameters
  `precipCorrectionFactor` and `glacierIceDDF` adjust the precipitation input; alternatively, the ice
  factor can be solved each evaluation to match an observed calibration-period glacier mass balance
  (`MHM_GLACIER_MB_TARGET`), optionally over the outline's area (`MHM_GLACIER_AREA_FROM_OUTLINE`).
  Configurable model step `MHM_TIMESTEP`. The `degreeDayFactor_pervious` calibration range widens from
  0–2 to 0–5 mm/°C/day; the upstream range was binding at Tungnaá.
- **`symfluence binary install mhm --patched`** pins upstream mHM `fd474cc3` and applies a fix for
  non-hourly time steps (see `models/mhm/patches/README.md`); `MHM_TIMESTEP` other than 1 now
  requires it.

### Changed
- **Core metrics facade exports repaired**: `StreamflowMetrics` and
  `MetricTransformer`, already part of the documented metrics contract, are
  now available directly from `symfluence.core.metrics` as intended.
- **HydroBM remains a default dependency; SALib is now optional**:
  Sobol and RBD-FAST support is installed with `symfluence[sensitivity]`.
  Importing the evaluation/analysis manager remains safe without SALib, while
  invoking either SALib-backed method raises an actionable install message.
- **Models contract advanced to 0.4.0**: adds `parallel_calibration` to the
  per-model declarations a registered `ModelConfigSchema` carries (the new
  `ParallelCalibrationConfig`: the settings-file dialect — quoted values vs
  tab-separated directives — and the mizuRoute control values a model needs
  when calibration runs in parallel). This is deliberately a sibling of
  `runoff`, not more fields on it: for models whose calibration path converts
  output the two describe genuinely different files (FUSE's converter writes
  `proc_NN_<exp>_timestep.nc` from its `runs_def.nc`), and HYPE needs the
  settings half while having no runoff declaration at all. The last per-model
  branching in `core/calibration/mixins/parallel/config_updater.py` is gone,
  enforced by an AST guard against model-name comparisons. Equivalence was
  proven differentially against the previous implementation across 288
  scenarios with zero byte-level mismatches.
- **Models contract advanced to 0.3.0**: the additive model surface now
  includes the per-model declarations a registered `ModelConfigSchema`
  carries for core to read instead of tabulating per model — `runoff`
  (the new `RunoffConfig`: the runoff artifact mizuRoute/tRoute consume,
  with its directory, filename pattern, variable, units, timestep, spatial
  dimension/id names, control-file label and source-name aliases),
  `spatial_mode_key`, and `routing_integration_key`. `RoutingDecider` and
  `runoff_loader` now read those declarations, so a model package — in-tree
  or external — becomes routable by registering a schema and nothing else.
  No served value changed: where the schema and the deleted tables
  disagreed, the table's value was preserved verbatim, since that is what
  routing consumes today.
- **Models and calibration contracts advanced to 0.2.0**: the additive model
  surface now includes canonical forcing-artifact selection and model-output
  location, canonical model-ready forcing and attribute reads/CF conventions,
  model-facing NetCDF encoding, shared time-window/dataset-alignment helpers, plus the
  model-neutral CFIF variable schema and shared forcing variable
  standardization/unit conversion, evaluator registry resolution, and the
  structure-ensemble analyzer base; optional coupling is reached through a
  model-facing capability facade while the graph engine remains in
  `symfluence.coupling`; generic calibration targets are likewise requested
  through a model-facing core facade while their implementations remain in
  the host evaluation/optimization stack; model plotters now subclass and use
  the core reporting contract while reporting orchestration remains in
  `symfluence.reporting`; generic evaluator bases and observation-path
  conventions now form a core model-facing contract while analysis
  orchestration remains in `symfluence.evaluation`; the calibration surface now
  includes model-agnostic multi-gauge
  metrics and gauge mapping, plus parameter regionalization
  strategies/transfer functions. Old
  `data`, `evaluation`, and `optimization` import paths remain compatibility
  shims for downstream packages.
- **Data handlers are now consumed registry-first everywhere**
  (community-service liftoff prep): the five remaining direct handler
  imports outside `data/` (MODIS-ET and FLUXNET-ET acquisition in the ET
  evaluator; LamaH-ICE streamflow helpers in the FUSE/SUMMA/HYPE workers)
  resolve through `R.observation_handlers`/`R.acquisition_handlers`, which
  now self-seed on first lookup like the other deferred registries. The
  layering guard forbids handler-module imports from outside `data/`.

### Fixed
- **Wflow parallel calibration**: the Wflow worker now provides `evaluate_worker_function`; previously
  every parallel or async Wflow DDS run failed with "Subclasses must implement" and calibration ran
  serially. Final streamflow scoring of distributed Wflow reads the native outlet `output.csv` instead of
  the cryosphere state NetCDF.

### Deprecated
- **Back-compat shim import paths** left by the service-decomposition
  promotions (`symfluence.models.{base,mixins,execution,state,templates,
  utilities,adapters,spatial_modes,model_manager,coupled.calibration.*}`,
  `symfluence.optimization.{optimizers,
  mixins,workers.base_worker,workers.inmemory_worker,workers.summa,
  core.base_parameter_manager,core.parameter_bounds_registry,
  parameter_managers}`, `symfluence.evaluation.metrics*`,
  `symfluence.cli.services.build_snippets*`,
  `symfluence.geospatial.geometry_utils`,
  `symfluence.resources.{get_base_settings_dir,copy_base_settings_to_project}`):
  external packages should migrate
  to the canonical `symfluence.core.*` / `symfluence.project.*` paths. The
  shims will be removed at 2.0. In-tree code no longer uses them
  (guard-enforced).

  Note `symfluence.optimization.core.parameter_bounds_registry` was listed
  here as a shim while actually being a 1185-line stale duplicate that
  predated #368; it is now genuinely a shim, so the entry is true for the
  first time.

### Changed
- **Base-settings resolution promoted to `core/modeling/base_settings.py`**,
  breaking the `resources` <-> `core` cycle: `symfluence.resources` reached into
  `core.registries` to pick a model's settings anchor while
  `core/modeling/base/base_preprocessor.py` imported the resolver straight back.
  `resources` is now data-only (it exposes `get_bundled_base_settings_dir` for
  its OWN bundled fallback and imports nothing from `symfluence.core`), and the
  seven model modules that read settings through `symfluence.resources`
  (clm, fuse, mizuroute, ngen, noahmp, summa, troute) go through the core
  contract instead. `symfluence.resources` is now a forbidden import prefix for
  `models` in the layering guard, so the models distribution depends on
  `symfluence.core` alone. Resolution order (registry first, bundled fallback,
  `FileNotFoundError` otherwise) is unchanged.
- **Forcing-adapter base promoted to `core/modeling/adapters/`** (completes
  the adapter contract tier; the per-model `forcing_adapter` modules keep
  registering through it). In-tree code and tests no longer import any
  back-compat shim path (shims now exist solely for external packages),
  and model test fixtures use the per-package `base_settings` layout.
- **Adapter contract tier promoted to `core/modeling/`** (phase-2 step 0):
  `models/{base,mixins,execution,state,templates,utilities,spatial_modes}`
  (~9k LOC) — the runner/preprocessor/postprocessor/extractor bases, adapter
  mixins, execution and state infrastructure, scaffolding templates, and
  shared adapter utilities every model package subclasses — now live in core
  (the `models` contract family). Historical `symfluence.models.*` paths
  remain as shims (verified to alias the same class objects, exercised by a
  new external-plugin compat test against the installed JAX packages). The
  ConfigKey schema machinery moved to `core/modeling/config_schema.py` with
  per-model definitions staying in `models/config` and registering through
  it; `spatial_orchestrator` resolves mizuRoute/t-route through `R.runners`/
  `R.preprocessors` instead of importing model packages; fews and
  data-assimilation now import model state from core (their models
  allow-list entries removed — both packages are fully models-clean).

### Added
- **Per-family contract versioning** (ADR-0009): `symfluence.core.contracts`
  declares independent semantic versions for the `models`, `calibration`,
  `metrics`, and `geospatial-utils` extensibility surfaces, with
  `contract_version()` / `is_compatible()` / `assert_compatible()` carrying
  the acquisition-backend contract's proven semantics (same major; pre-1.0
  minors additive-only; forward skew declined). The acquisition family keeps
  its existing constant and import path.
- **`register_algorithm()`** — sanctioned registration seam for external
  optimization algorithms (decorator or direct call, aliases, collision
  refusal); externally registered algorithms appear in `list_algorithms()`
  and resolve through `get_algorithm()`.

### Changed
- **Geometry utilities promoted to core** (`core.geometry_utils`, was
  `geospatial.geometry_utils`): the audit of the models→geospatial surface
  found exactly one module — the `GeospatialUtilsMixin` and catchment
  area/centroid helpers model preprocessors build against — so it joins the
  core contract surface (`geospatial-utils` family). Old import path kept as
  a shim; the layering guard now forbids `models/` from importing
  `symfluence.geospatial` at all.
- **Model-specific data decentralized** (service-decomposition prep, tail):
  - Base settings templates moved from `resources/base_settings/` into their
    model packages (`models/<model>/base_settings/`), resolved through the new
    `R.base_settings` registry (package-anchor per model, external plugins use
    the same path); `symfluence.resources.base_settings` remains as fallback
    for framework fixtures (TEST) and unregistered models.
  - `SystemDepsRegistry.register_tool_requirements()`: model packages can
    declare build-time system dependencies (and contribute new dependency
    definitions) at runtime; the central `system_deps.yml` keeps the shared
    toolchain catalogue.
  - Per-model legacy flat-key aliases (`SETTINGS_SUMMA_*`, `SETTINGS_MIZU_*`,
    …, 60 keys across 10 models) moved from `core/config/legacy_aliases.py`
    onto each model's config schema (`LEGACY_FLAT_ALIASES` class attribute),
    collected via `R.config_schemas` at map-generation time; core keeps only
    framework-level aliases.
  - GUI model selectors derive their choices from `R.config_schemas` instead
    of hardcoded lists (new external models appear automatically).
- **Per-model config schemas moved into their model packages**
  (service-decomposition prep, continued): the 31 typed schema classes
  (`SUMMAConfig`, `FUSEConfig`, …) moved from
  `core/config/models/model_configs_*.py` to
  `models/<model>/config_schema.py`. Each reaches the core config system
  through `R.config_schemas` — via the model manifest's config adapter (29
  models) or an explicit `add_lazy` registration (ignacio, wmfire) — the same
  path an external plugin uses (`model_manifest(config_schema=...)`). The
  `IN_TREE_CONFIG_SCHEMAS` bridge is gone; legacy import paths under
  `symfluence.core.config.models` keep working via registry-backed PEP 562
  shims, and a guard test keeps core/config free of model schema classes.
  WMFire's schema is now registered under `WMFIRE` (previously reachable only
  nested inside `RHESSysConfig`).
- **Calibration contract promoted to core** (service-decomposition prep for
  extracting the model suite into its own package). The generic calibration
  engine moved from `optimization/` into `core/calibration/` — optimizer base
  (`BaseModelOptimizer`), the algorithm suite, evaluators, final-evaluation
  pipeline, parallel-execution/results/retry/gradient mixins, worker bases
  (`BaseWorker`, `InMemoryModelWorker`), parameter management
  (`BaseParameterManager`, the parameter bounds registry) — so model adapter
  packages subclass calibration bases from `symfluence.core` only. Performance
  metrics moved from `evaluation/` to `core/metrics/`, and build-environment
  snippet helpers from `cli/services/` to `core/build/` (models no longer
  import the CLI layer). All historical import paths keep working via
  back-compat shims.
- **The models layer is now import-time removable**: `optimization/` no longer
  imports `symfluence.models` at module level (model-specific aggregators —
  `parameter_managers`, `calibration_targets`, the SUMMA optimizer mixin —
  resolve lazily via PEP 562), and the SUMMA calibration worker internals
  moved to their canonical home, `models/summa/calibration/worker_impl/`.
  `scripts/check_core_layering.py` now also enforces these package boundaries
  (no module-level imports of `models` outside `models/`; `models` imports no
  interface layer), with every remaining call-time seam allow-listed with a
  reason.
- **Coupling process adapters resolve models through the registry**:
  `coupling/adapters/process_adapters.py` looks up runners and result
  extractors via `R.runners` / `R.result_extractors` instead of importing
  model modules at call time, closing the last coupling→models seam.

### Fixed
- **A preprocessor returning `False` now fails the workflow step**: most model preprocessors catch
  their own exceptions and return `False`, which `ModelManager.preprocess_models` ignored, so the
  workflow reported success with incomplete model inputs.
- **mHM calibration wrote `maxDegreeDayFactor_*` whenever it set `degreeDayFactor_*`**: the namelist
  regex was unanchored and case-insensitive, so every calibration capped snowmelt at the base factor.
  Parallel mHM workers also wrote to a shared output directory; each now writes to its own.
- **mHM distributed preprocessing** read attributes from the wrong directory and fell back to the DEM
  bounding box, which roughly doubled the basin; catchment properties no longer fall back to a
  placeholder location.
- **Two silent parameter-bound collisions in the central catalogue** (found by
  the new collision guard test): FUSE's snow bounds (`MBASE`, `MFMAX`,
  `MFMIN`) were overridden by Snow-17's — different ranges *and* units
  (mm/°C/6hr vs mm/(°C·day)) — so FUSE calibrated against Snow-17 melt
  bounds; and CFE's deliberately tightened `soil_depth` (1–5 m, anti-segfault)
  was overridden by RHESSys's 2–15 m. Colliding entries are now namespaced
  (`fuse_*`, `rhessys_*`) with prefixes stripped in the per-model getters;
  bare names keep Snow-17/CFE semantics for existing callers.
- **`NgenStreamflowTarget` never registered with `R.calibration_targets`** —
  the only in-tree target class missing its registration decorator; NGEN
  target resolution always fell through to the import fallback.

### Added
- **Parameter-bounds extension seam** — `register_model_bounds()` /
  `get_model_bounds()` / `registered_bound_models()` in
  `core.calibration.parameters`: external model packages register calibration
  bounds (new definitions and/or catalogue-name compositions, with optional
  prefix stripping) without editing core. The centralized catalogue remains
  the single source of truth for shared parameters, per the recorded
  architecture decision (now amended in the module docstring).
- **The SYMFLUENCE agent interface** (`symfluence agent`): hands off to an
  installed coding-agent CLI (Claude Code, Codex, Gemini, ...) primed as the
  SYMFLUENCE agent through four provider-agnostic layers, each wired through
  whatever mechanism the CLI declares in the launcher registry: (1) the
  packaged skills; (2) an identity block with live project context (detected
  configs, domain directories) and operating house rules, delivered via the
  CLI's system-prompt flag or the `AGENTS.md` preamble; (3) a dependency-free
  MCP server (`symfluence agent mcp`); (4) packaged specialist subagents
  (`calibration-debugger`, `platform-scout`).
- **Agent modes** (`symfluence agent model` / `symfluence agent code`): two
  first-class session modes, each a frozen `ModeProfile` (`agent/modes.py`)
  that shapes priming end to end. *Modelling* primes the operational skills
  (`explore-platform`, `run-workflow-locally`, `debug-calibration`), both
  subagents, modelling house rules (never edit platform source; drive
  everything through the workflow/MCP tools; report in hydrological terms),
  and a headless tool allowlist for the native chat. *Coding* keeps the full
  skill set and the host CLI's own permissions. Bare `symfluence agent` opens
  the TUI agent screen; `agent doctor` diagnoses the setup per mode (with
  `--json`); `agent mcp --mode` serves one mode's tool profile; skills
  materialize per-mode and copy whole skill directories.
- **MCP background jobs and platform-inspection tools**: the agent MCP server
  exposes 15 tools. `start_workflow_job` / `get_job_status` / `cancel_job` /
  `list_jobs` run workflow executions as detached background jobs (a stdlib
  wrapper records the exit code durably; records and logs live in the agent
  cache; cancellation signals the process group), so a chat session can start
  an hours-long calibration without blocking the MCP connection.
  `read_run_log`, `list_domains`, `calibration_status`, `get_results_summary`,
  `get_plot_paths`, and `compare_experiments` answer the modelling questions
  (log tails, domain inventory, calibration progress, headline KGE/NSE
  metrics, figures, cross-experiment ranking) by reading the `domain_*` tree
  directly with stdlib CSV parsing. `update_config` is the one writer: a
  line-preserving edit of the user's experiment YAML that must pass typed
  `SymfluenceConfig` validation and backs the original up first.
- **Agent home screen with a suspend round-trip**: a minimal Agent home in the
  TUI (mode `7`) — two mode cards (Model / Code) over two dim context lines
  (detected config with a cycle key, runtime readiness); diagnostics behind a
  `d` details modal. Starting a coding session round-trips: the TUI suspends,
  the primed coding-agent CLI runs full-screen in the same terminal, and the
  home screen returns when it exits. Terminals that cannot suspend fall back
  to the classic `AgentHandoff` exec after the TUI exits (`--direct`, a
  one-shot prompt, no TTY, or a missing TUI extra hand off immediately).
- **Native modelling chat**: `symfluence agent model` in the TUI opens a
  SYMFLUENCE-native chat screen driving headless Claude Code over its
  stream-JSON protocol — one bounded subprocess per turn, resumed via Claude
  Code's own session store (`agent/headless.py`: tolerant NDJSON parser →
  typed events; session ids persisted per project+mode). Assistant prose
  streams into the conversation; tool invocations render as compact
  expandable cards (never raw JSON); a run sidebar polls the domain tree and
  background-job records independently of the agent, so an hours-long
  calibration keeps ticking on screen. `esc` interrupts a turn, `ctrl+e`
  exports the conversation as Markdown, and each turn's footer shows
  duration, cost, and the running session total. Codex/Gemini (or a failed
  stream) fall back to the suspend round-trip with modelling priming.
  Stdlib-only driver; no new dependencies.
- **Interactive permission approvals** in the modelling chat: tools outside
  the modelling allowlist route through a hidden `approve_action`
  permission-prompt bridge (`agent/approvals.py`) — the MCP server blocks,
  the chat pops an allow/deny modal, and no reply is a denial. Outside the
  chat the same tools stay hard-denied. ADR-0004 rewritten for the current
  architecture (human-in-the-loop at the layer that executes actions).
- **MESH multi-GRU preprocessing fixes and elevation-band precip correction**:
  GRU-dependent hydrology parameters (ZSNL/ZPLS/ZPLG/...) expand to one value
  per GRU so multi-GRU domains initialize; new forcing keys
  `MESH_PRECIP_LAPSE_RATE` (orographic gradient for elevation-banded forcing)
  and `MESH_PRECIP_MULTIPLIER` (uniform gauge-undercatch correction), both
  defaulting to no-op; `SKIP_WARM_START` documented as defaulting to true so
  calibrations are reproducible regardless of machine history.

### Changed
- **Logging-protocol overhaul** (ADR-0005 enforcement, from an audit of the
  paper-run log corpus where WARNING+ERROR lines outnumbered INFO 1.85:1):
  - One shared protocol module (`core/logging_utils.py`): canonical file
    format (~46% smaller lines: `TIME LEVEL [logger] message`, filename
    `symfluence_{domain}_{experiment_id}_{ts}.log`), `log_once` de-duplicated
    emission, a single third-party suppression table, and one
    `get_worker_logger` bootstrap replacing four bespoke worker setups.
  - Milestone-level INFO everywhere: per-item/per-chunk loops now emit one
    summary line with per-item detail at DEBUG (an ERA5 96-chunk resume drops
    from ~190 INFO lines to 3); hot-loop conditions that produced tens of
    thousands of repeated WARNING/ERROR lines emit once and demote to DEBUG;
    external-model stdout goes to sidecar files referenced by one log line.
  - Comparable calibration progress: every optimizer emits the fixed schema
    `{ALG} {i}/{max} {unit} ({pct}%) | Best | Improved | Crashes | Elapsed`
    with explicit units (evals/gens/epochs/loops) and `[P##]` worker tags;
    GLUE and NSGA-II now stream progress.
  - Honest run reporting: `✗ Failed` completions on failed steps,
    failure-aware workflow end block, `run_summary.json` schema v2 with
    error/warning totals counted from actual log records, per-step
    status+duration aligned with the run manifest.
  - Global `--quiet/-q` flag (console WARNING+; file log unaffected);
    `--debug` unchanged and dominant.
  - `setup_logging` is idempotent per (domain, experiment id) — re-use no
    longer re-opens a new log file per facade construction.
- **Agent verb consolidation**: `agent launch` is deprecated (alias for
  `agent code`, to be removed after one release); the already-deprecated
  `agent start`/`agent run` and the `agent list`/`agent skills` verbs are
  removed (`agent doctor` covers the last two).

### Fixed
- **GR4J no longer dies with a Windows access violation the moment it runs the
  model**: the GR runner built its R script by interpolating `str(Path)` into R
  string literals, so on Windows R's lexer received
  `read.csv("C:\Users\...\domain_x\input.csv")` — in which `\U` opens a Unicode
  escape and `\d` is not an escape at all. Standalone `Rscript` reports that as
  a parse error; the *embedded* interpreter raises it from inside
  `R_ParseVector` and takes the process down with `STATUS_ACCESS_VIOLATION`
  (0xC0000005, reported as exit 139) with no Python frame to trace — after the
  workflow had already spent ~36 minutes preprocessing. Paths now go through
  `r_environment.r_path()`, which renders them with forward slashes (accepted by
  R on every platform), and every generated script is evaluated through
  `r_environment.run_r_script()`, which refuses source containing an invalid R
  escape so a regression is an actionable `ModelExecutionError` instead of a
  bare crash. The same interpolation affected the distributed GR4J path and the
  GR postprocessor's `load()` of `GR_results.Rdata`.
- **GR4J no longer segfaults on Windows because the embedded R cannot load its
  own packages**: rpy2 registers `R_HOME/bin/x64` only via
  `os.add_dll_directory()`, which R's internal `dyn.load()` does not consult, so
  `stats.dll` could not resolve `Rlapack.dll` and *every* compiled R package —
  airGR included — failed to load, while standalone `Rscript` worked fine. R
  reported this as the misleading `package 'stats' in options("defaultPackages")
  was not found`. The new `models/gr/r_environment.py` puts R's binary directory
  on `PATH` (locating R from `R_HOME`, `PATH`, the registry, or `Program Files`)
  before rpy2 starts the interpreter. The GR runner also no longer answers a
  failed `library(airGR)` with `install.packages()` against CRAN: on an offline
  machine that wedged the interpreter into a SIGSEGV 53 minutes into a run.
  airGR is now verified up front, and an unusable one raises an actionable error
  quoting the embedded R's own `R.home()`, `.libPaths()` and load error.
- **Models that build to `<name>.exe` on Windows are found again**: model
  definitions declare POSIX-style executable names (`gsflow`, `prms`) because
  the build scripts run under MSYS bash, where `[ -f bin/gsflow ]` and
  `cp x bin/gsflow` transparently resolve to `gsflow.exe`. `Path.exists()` does
  not, so a GSFLOW that had compiled and linked perfectly well was reported as
  `Model executable not found`. `BaseModelRunner.get_model_executable` now tries
  the `.exe` variant on Windows — matching the `exists_any` rule the install
  verifier already used — and the error lists every name it searched. GSFLOW's
  calibration worker, which resolves the path by hand, got the same treatment,
  and its runner now passes `exe_name_key='GSFLOW_EXE'` so dict-style configs
  can override the name.
- **Windows model processes no longer deadlock at exit**: the Windows release
  job now rebuilds netCDF-C without the AWS S3 SDK
  (`scripts/build_netcdf_no_s3_mingw.sh`) instead of shipping MSYS2's stock
  package. netCDF's `DLL_PROCESS_DETACH` hook called `Aws::ShutdownAPI`, which
  waits on AWS CRT worker threads that `RtlExitUserProcess` has already
  terminated — so every SUMMA/FUSE/HYPE run left an unkillable process that
  never signalled, kept its output NetCDF handles locked, and broke subsequent
  calibration evaluations. Byte-range, DAP and NCZarr support are unchanged, and
  the build asserts the result is a drop-in (export-table diff plus a check that
  every symbol `libnetcdff-7.dll` imports is still exported). Linux and macOS
  are untouched: the deadlock needs ExitProcess semantics, and Debian's netCDF
  is already built without the AWS SDK.
- **Every packaged Windows tool now ships the DLLs it imports**: the release
  bundler enumerated images with an `*.exe` glob while staging deliberately
  drops the `.exe` suffix from tool names, so every Fortran model was skipped
  and `libnetcdff-7.dll` was never bundled. Images are now detected by content,
  and `scripts/check_windows_dll_closure.sh` walks each packaged binary's import
  table and fails the release if anything is unresolvable — a missing DLL aborts
  a Windows process with `0xC0000135` before `main()` with no output at all,
  which calibration recorded as a `-9999` score rather than a failure (observed:
  seven native HYPE calibrations silently lost).
- **Stale plugins no longer vanish silently**: a plugin whose `register()`
  fails on SYMFLUENCE API drift (e.g. importing a removed alias) now logs a
  WARNING naming the incompatible package instead of disappearing from the
  registry at DEBUG level; a worker-contract test sweep (AST-resolved method
  calls on every in-repo worker class) guards against the API-drift class of
  calibration-voiding AttributeErrors observed in the paper-run logs.
- **Review follow-ups to the agent interface and CLI option handling** (#300,
  #299): `--dry-run binary <tool>` now previews instead of executing; global
  option normalization no longer steals host-CLI flags from agent-session
  pass-through (structure-aware boundary at REMAINDER subcommands, option
  names derived from a single spec); the MCP server survives non-object
  JSON-RPC input and bounds subprocess output via a temp file; the agent
  screen forwards `--` pass-through args and validates the preselected
  runtime; priming failures degrade with visible warnings and the launch card
  only claims layers that actually activated (a SYMFLUENCE-generated
  `AGENTS.md` is refreshed, a user-authored one is reported as blocking
  injection); frontmatter parsing consolidated into
  `resources.parse_frontmatter`.

---

## [0.9.2] - 2026-07-07

Archival release accompanying the SYMFLUENCE paper series
("Out-standing in Every Field", "The Registry as Social Contract",
"From Configuration to Prediction"; Eythorsson et al., 2026, Water
Resources Research, submitted). First release archived on Zenodo via the
GitHub integration.

### Added
- **Citation metadata**: `CITATION.cff` (GitHub "Cite this repository") and
  `.zenodo.json` (Zenodo deposit metadata) with the full author list and
  ORCIDs (#269).
- Guard test validating every shipped paper calibration config against the
  strict config schema.

### Changed
- **Paper supplementary configs trimmed to the final manuscript** (#268):
  removed the four cut experiments (decision ensemble, sensitivity analysis,
  large-sample and large-domain Iceland calibration), reduced the forcing
  ensemble to the four reported products (ERA5, RDRS, AORC, CONUS404), and
  kept only the 130 fixed-seed (42) calibration combinations the paper
  reports; all READMEs updated to the manuscript's section and figure
  numbering.

### Fixed
- Shipped calibration configs for the five JAX-native models (HBV, HEC-HMS,
  SAC-SMA, TOPMODEL, Xinanjiang) failed strict validation due to stale keys
  from older plugin schemas (`initial_params`; HBV also `smoothing`,
  `smoothing_factor`) and could not be loaded; the stale keys are removed
  (85 configs).
- Paper-config seed guard repointed after the #268 trim removed the config
  it loaded (#270).

---

## [0.9.1] - 2026-06-14

> **Note**: Versions 0.8.0–0.9.0 were tagged and published without CHANGELOG
> entries. This section documents the changes since the v0.9.0 release; the
> 0.8.x/0.9.0 history is not reconstructed here.

### Added
- **Canonical forcing contract (CFIF)**: a single canonical forcing reader keyed
  to CF standard names (CFIF) that model preprocessors consume through one path,
  with automatic resampling of the source to each model's cadence. Adopted by
  SWAT, Wflow, IGNACIO, LISFLOOD, CWatM, PCR-GLOBWB, RHESSys, CLM-ParFlow, and mHM.
- **Model-ready data store for attributes and observations**: new attributes
  reader plus observation/streamflow loaders that source from the model-ready
  store. SUMMA soil/land class and HYPE GeoData (soil/land/elevation) now read
  from the store; `StreamflowMetrics` and the remaining calibration loaders are
  routed through it.
- **Windows binary builds** for many models — WATFLOOD (bundled Waterloo CHARM
  prebuilt), ParFlow, CLM-ParFlow, PRMS, VIC, PIHM, mHM, WRF-Hydro, and glacier —
  via MinGW/MSYS2 POSIX shims and CMake/NetCDF wiring.
- **WATFLOOD**: full Unix and Windows source builds; CHARM runs on the datastore;
  parallel-calibration plumbing.
- **GSFLOW**: datastore-driven inputs and a distributed-groundwater option.
- **mHM**: CARRA forcing support and distributed-morphology (MPR) mode.
- WATFLOOD and GSFLOW standalone postprocessors registered.

### Changed
- **Agent interface simplified to a launcher.** `symfluence agent launch` now hands
  off to an installed coding-agent CLI (Claude Code, Codex, Gemini, ...), primed with
  the SYMFLUENCE skills, instead of running a bespoke in-house LLM agent. It detects
  the CLI on `PATH` and uses whichever API key is set (`ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` / `GEMINI_API_KEY`); override with `SYMFLUENCE_AGENT_CLI`, skip
  skill materialization with `SYMFLUENCE_NO_SKILLS`. The skills now ship inside the
  package. *(net ~ −7,000 LOC)*
- **Config schema canonicalization (toward 1.0)**: promoted data-handler,
  HYPE/NGEN/FUSE/GR/mizuRoute/path, multi-gauge, optimizer/evaluator, and
  parameter-bounds flat keys to typed Pydantic fields; `RECOGNIZED_FLAT_KEYS` is
  now closed; the `state` and `data_assimilation` sections are wired into the
  flat-key transform; IGNACIO/GNN/LSTM schemas reconciled with the registered
  ones. Strict config-key validation is enforced and dogfooded on shipped configs.
- Documentation: root `ARCHITECTURE.md` at-a-glance, an ADR practice with the
  initial ADR set, and a threat model (`docs/security.md`).
- Evaluation: unified catchment-area resolution between trial and final-eval
  metrics (resolves the FUSE final-eval KGE divergence).
- Performance: RTI review item 20 hot-spots — rasterio hoist, Julian-date
  vectorization, coastal-partition rewrite — plus a CLI performance gate.
- Core layering restored: no `core` → upper-layer imports (RTI item 19).

### Fixed
- **Data acquisition**: MERIT-Basins region table and dead-host download; Daymet
  gridded OPeNDAP route (descending y, DAP2); HRRR LCC-grid windowing; CERRA
  longwave variable name; NEX-GDDP-CMIP6 synthetic-airpres survival and hardening;
  DEM merge pixel-grid snap; MSWEP remote path and default version; SMHI
  skip-if-raw-exists guard; EM-Earth S3 handler (403 visibility, credentials,
  month window); WSC GeoMet paging corruption and NLDAS-2 v2.0 variable names;
  USGS NWIS gauge-local → UTC timestamps; GRACE atomic `.part` downloads.
- **HYPE**: clone over `https://` with retry, replacing the deprecated
  `git://` SourceForge transport that intermittently broke nightly and release
  builds.
- **CLM** build: NetCDF library symlinks for PIO (Debian multiarch), git
  `safe.directory` re-assert after HOME change, `mpi-serial` pin, and Perl
  `XML::LibXML` install.
- Hardcoded `*3600` hourly-conversion bugs in the Wflow and IGNACIO forcing paths.
- **Windows builds**: WATFLOOD DLL staging, PIHM `timegm` guard, ParFlow
  `mkdir`/path compatibility, PRMS NetCDF path quoting and `gnu17`, VIC version
  macros, mHM NetCDF detection, and WRF-Hydro checkout.
- **Glacier**: earthaccess auth for NSIDC, RGI region-name transform, and
  coldstart fixes.
- **PRMS**: resilience-handler tracebacks (`exc_info=True`) and HRU parameters
  driven from the datastore.
- SWAT preprocessor registered via `core.registries.R` to avoid a circular import.

### Removed
- **Deprecated `register*` decorator registry API** — registration is now
  R-facade only. *(breaking)*
- **Deprecated `*Postprocessor` spelling aliases** — use `*PostProcessor`.
  *(breaking; external plugins must update)*
- **HYDROGEOSPHERE** model stub. *(breaking)*
- The **CalMIP** example stub.
- **DPE** (differentiable PE) and other dead/external config keys; shipped
  configs curated accordingly.
- **The bespoke in-house LLM agent** (`symfluence.agent` internals) and the
  optional `[llm]`/`openai` dependency — replaced by the launcher above.
  `symfluence agent start`/`run` are deprecated aliases for `launch`. *(breaking:
  the agent now requires an external coding-agent CLI)*

### CI / Release
- `release-binaries`: build MODFLOW 6 from source on Linux, build ngen on
  Linux/macOS, skip CLM on Windows (ESMF/CIME on MinGW unsupported), close
  presence-check holes and surface failed optional builds, and don't clobber the
  latest release on a tagless `workflow_dispatch`.
- Strict config-key validation dogfooded on shipped configs (kept out of the
  lint job, which has no `symfluence` install).
- Require the `PostProcessor`-spelling plugin releases.

---

## [0.7.1] - 2026-03-01

### Fixed
- **HPC file-locking workaround**: Added tempfile fallback (attempt 4) to ERA5
  NetCDF writer for parallel filesystems (Lustre/GPFS/BeeGFS) where HDF5
  `fcntl()` locking fails even with `HDF5_USE_FILE_LOCKING=FALSE`. Writes to
  `$TMPDIR` first, then moves the file to the target path.
- **mizuRoute Makefile build**: Collapse multi-line `LIBNETCDF` before appending
  RPATH flags to avoid breaking backslash continuation (caused "missing separator"
  errors on CI).
- **ngen linker (expat)**: Detect system UDUNITS2 via `pkg-config`/`ldconfig`
  (not just explicit env var) so `-lexpat` is added for XML symbol resolution.
- **Wflow calibration**: Fix snow parameters, routing, and unit conversion issues.
  Add Oudin PET, log-transform bounds, and sub-daily resampling support.
- **FUSE preprocessing**: Make streamflow observations optional.
- **MPI launcher**: Add fallback when preferred launcher fails; prefer `mpirun`
  over `srun` for launcher detection.

### Added
- SPDX license headers on all source files.
- Coverage tracking in cross-platform CI workflow.
- Unit tests for NGEN, JFUSE, HYPE model runners and configs.
- Unit tests for core modules (file_utils, validation, path_resolver).
- Unit tests for `_safe_to_netcdf` fallback chain.
- WATFLOOD: Expand calibration to 16 parameters with tests.
- Bow River preset and default `DEM_SOURCE` to Copernicus.
- Config-hash stage invalidation for workflow orchestrator.

---

## [0.7.0] - 2026-02-10

> **Note**: This release jumps from the last tagged release v0.5.2 directly to v0.7.0.
> Versions 0.5.3–0.6.0 were development milestones on the `develop` branch and were
> never tagged or published to `main`. Their changes are included in this release and
> documented below in the [0.6.0] and [0.5.x] sections for historical reference.

> **Breaking Change**: This release refactors the CLI to a subcommand architecture.
> All existing CLI commands will need to be updated.

### Added
- **New CLI Commands**
  - `symfluence gui launch` - Panel-based web GUI for workflow management
  - `symfluence data download|list|info` - Standalone dataset acquisition without a full project
  - `--shapefile` option for `symfluence data download` to derive bounding box from a shapefile

- **Documentation Improvements**
  - New CFuse model guide (experimental differentiable PyTorch-based FUSE)
  - New JFuse model guide (experimental JAX-based FUSE with gradient support)
  - New WM-Fire model guide (wildfire spread simulation with RHESSys)
  - Comprehensive CLI reference documentation with all commands and options
  - Expanded configuration parameter reference
  - Fixed Python API examples in calibration documentation
  - New testing guide for contributors with pytest patterns and CI integration
  - Added agent-assisted contribution workflow documentation

- **Agent Improvements**
  - Enhanced `show_staged_changes` with diff statistics and file summary
  - Improved `generate_pr_description` with auto-detection of modified files
  - Better PR descriptions with context-specific sections and testing checklists
  - Added documentation for fuzzy matching threshold parameter

- **HBV Module Refactoring**
  - New `losses.py` module with differentiable NSE/KGE loss functions and gradient utilities
  - New `parameters.py` module with parameter bounds, defaults, and scaling utilities
  - Lazy imports in `__init__.py` for optional JAX dependency
  - Reduced `model.py` by ~650 lines through modularization

- **HBV Parameter Regionalization**
  - Neural transfer functions for spatially-varying parameter estimation
  - Support for catchment attribute-based parameter prediction

- **HBV Calibration Improvements**
  - Hydrograph signature metrics for multi-objective calibration
  - Enhanced optimizer with adaptive learning rates

- New modular command structure in `src/symfluence/cli/`:
  - `argument_parser.py` - Main parser with subcommand structure
  - `validators.py` - Validation utilities
  - `commands/` directory with category-specific handlers

### Changed
- **Complete CLI Refactor**
  - Replaced flat flag-based interface with modern two-level subcommand architecture
  - New structure: `symfluence <category> <action>` instead of `symfluence --flag`
  - 9 command categories: workflow, project, binary, config, job, example, agent, gui, data
  - Eliminated complex mode detection logic
  - Archived old `cli_argument_manager.py` for reference

- **New CLI Structure**
  ```bash
  # Workflow commands
  symfluence workflow run [--config CONFIG]
  symfluence workflow step STEP_NAME
  symfluence workflow list-steps
  symfluence workflow status

  # Project commands
  symfluence project init [PRESET]
  symfluence project pour-point LAT/LON --domain-name NAME --definition METHOD
  symfluence project list-presets

  # Binary/tool commands
  symfluence binary install [TOOL...]
  symfluence binary validate
  symfluence binary doctor

  # Configuration commands
  symfluence config validate
  symfluence config validate-env
  symfluence config list-templates

  # Job commands
  symfluence job submit [WORKFLOW_CMD] [SLURM_OPTIONS]

  # Example commands
  symfluence example launch EXAMPLE_ID
  symfluence example list

  # Agent commands
  symfluence agent start
  symfluence agent run PROMPT

  # GUI commands
  symfluence gui launch

  # Data commands
  symfluence data download DATASET --bbox W S E N
  symfluence data list
  symfluence data info DATASET
  ```

- **HBV Sub-Daily Parameter Scaling**
  - Implemented exact exponential scaling for recession coefficients (K0, K1, K2)
  - Formula: `k_sub = 1 - (1 - k_daily)^(dt/24)` replaces linear approximation
  - Eliminates ~5-13% error in recession behavior at sub-daily timesteps
  - Flux rate parameters (CFMAX, PERC) continue to use linear scaling
  - Added `FLUX_RATE_PARAMS` and `RECESSION_PARAMS` constants

- **MESH Preprocessing Consolidation**
  - Moved configuration defaults to dedicated `config_defaults.py`
  - Streamlined `config_generator.py` and `meshflow_manager.py`
  - Expanded `parameter_fixer.py` with robust parameter handling
  - Updated NALCMS to CLASS land cover mapping (wetland, snow/ice corrections)
  - Added unit conversion utilities for forcing data
  - Improved logging (replaced debug prints with proper logger calls)

- **FLUXCOM ET Acquisition Rework**
  - Replaced stub ICOS downloader with fully functional `_download_from_icos()` using ICOS Carbon Portal metadata API
  - Smart variable matching (exact name, then substring, then fallback)
  - Automatic unit conversion (W/m2, mm/hr, mm/day)
  - Nearest-neighbor fallback for basins smaller than the grid cell

- **MODIS ET Acquisition Improvements**
  - Spatial subsetting via sinusoidal projection for large tiles with small domains
  - Improved special-value masking (all MODIS QC codes, not just fill value)
  - SDS name matching now tries exact match before substring search

- **Reporting Module Cleanup**
  - Simplified plotter implementations
  - Improved shapefile handling

- **Example Notebooks — Typed Config Migration**
  - Migrated notebooks 02a–03b from legacy `yaml.safe_load` + flat dict pattern to `SymfluenceConfig.from_minimal()` typed config API
  - All 9 example notebooks (01a–04b) now use the same typed, validated, frozen config pattern
  - Removed MAF-specific language from all example notebooks; acquisition comments now reference `DATA_ACCESS: 'cloud'` or `'maf'` config setting
  - Fixed pre-existing bugs: 02a `NameError` on config access, 03b `cf.managers` wrong variable name, 03b config overwrite bug, 03b outdated CLI syntax

### Fixed
- MESH lumped mode routing: switched from `run_def` to `noroute` mode to correctly preserve lower-zone baseflow (`wf_lzs`); `run_def` in MESH 1.5.6 bypasses `STGGW` storage, causing zero baseflow
- Traceback handling in MESH postprocessor
- Added missing `pydantic` runtime dependency to `pyproject.toml`
- Fixed invalid escape sequences in `__init__.py` warning filters for Python 3.12+ compatibility
- CI pipeline now correctly fails on unit/integration test regressions (removed `|| true` from pytest invocations)

### Documentation
- Added comprehensive sub-daily simulation section to HBV model guide
- Documented parameter scaling approaches (exact vs linear)
- Added validation methodology for sub-daily implementations

### Migration Guide

| Old Command (v0.6.x) | New Command (v0.7.0) |
|----------------------|----------------------|
| `symfluence --calibrate_model` | `symfluence workflow step calibrate_model` |
| `symfluence --setup_project --create_pour_point` | `symfluence workflow steps setup_project create_pour_point` |
| `symfluence --get_executables summa` | `symfluence binary install summa` |
| `symfluence --validate_binaries` | `symfluence binary validate` |
| `symfluence --doctor` | `symfluence binary doctor` |
| `symfluence --init fuse-provo --scaffold` | `symfluence project init fuse-provo --scaffold` |
| `symfluence --list_presets` | `symfluence project list-presets` |
| `symfluence --workflow_status` | `symfluence workflow status` |
| `symfluence --list_steps` | `symfluence workflow list-steps` |
| `symfluence --pour_point 51/-115 --domain_def delineate --domain_name Test` | `symfluence project pour-point 51/-115 --domain-name Test --definition delineate` |
| `symfluence --agent` | `symfluence agent start` |
| `symfluence --example_notebook 1a` | `symfluence example launch 1a` |

**Benefits of new CLI:**
- Clearer command organization and discoverability
- Better help messages (`symfluence workflow --help` shows workflow-specific options)
- Easier to extend with new commands
- Industry-standard pattern (like git, docker, kubectl)

---

## [0.6.0] - 2025-12-29

### Added
- **Calibration Observation Data Utilities**
  - `download_smhi_discharge.py` - Download discharge data from Swedish Meteorological Institute
  - `prepare_streamflow_for_calibration.py` - Convert discharge CSV to calibration format
  - `setup_calibration.py` - Automated calibration setup with parameter bounds
  - Calibration demo tests for Elliðaár (Iceland) and Fyris (Sweden) basins

- **CARRA/CERRA Data Processing Improvements**
  - Fixed CARRA longitude normalization in spatial subsetting
  - `FORCING_TIME_STEP_SIZE` configuration support (10800s for CERRA)
  - `FORCING_SHAPE_ID_NAME` configuration support with default 'ID'

### Changed
- CLI orchestrator integration completed with full workflow execution support
- Version bump to 0.6.0 across all version references

### Removed
- **Deprecated CONFLUENCE backward compatibility**
  - Removed `CONFLUENCE.py` wrapper file
  - Removed `./confluence` shell script
  - Removed `CONFLUENCE_DATA_DIR` and `CONFLUENCE_CODE_DIR` configuration support
  - All documentation now uses SYMFLUENCE exclusively

### Fixed
- CARRA spatial subsetting for small basin extents
- EASYMORE remapping failures for CARRA datasets

---

## [0.5.11] - 2025-12-15

### Changed
- Enhanced ngen outlet detection
- Cleaned up technical debt

### Fixed
- Mypy type errors in config property inheritance
- Completed typed config migration for base classes

### Improved
- Centralized evaluation metric logic
- Improved linting and added ruff tests to pyproject.toml

---

## [0.5.3] - 2025-11-15

### Added
- Support for cloud acquisition of:
  - Copernicus DEM
  - MODIS land cover
  - Global USDA-NRCS soil texture class map
  - Forcing datasets: ERA5, NEX-GDDP, CONUS404, AORC
- Agnostic pipeline for cloud-based ERA5 matched workflows
- Full cloud-integrated workflow tested with ERA5
- Made MPI worker log generation optional
- Initial t-route support (in progress)

---

## [0.5.2] - 2025-11-12

### Major: Formal Initial Release with End-to-End CI Validation

**This release marks the first fully reproducible SYMFLUENCE workflow with continuous integration.**

### Added
- **End-to-End CI Pipeline (Example Notebook 2a Equivalent)**
  Integrated a comprehensive GitHub Actions workflow that builds, validates, and runs SYMFLUENCE automatically on every commit to `main`.
  - Compiles all hydrologic model dependencies (TauDEM, mizuRoute, FUSE, NGEN).
  - Validates MPI, NetCDF, GDAL, and HDF5 environments.
  - Executes key steps (`setup_project`, `create_pour_point`, `define_domain`, `discretize_domain`, `model_agnostic_preprocessing`, `run_model`, `calibrate_model`, `run_benchmarking`).
  - Confirms reproducible outputs under `SYMFLUENCE_DATA_DIR/domain_Bow_at_Banff`.
  - Runs both wrapper (`./symfluence`) and direct Python entrypoints equivalently.

### Changed
- Updated `external_tools_config.py` to include automatic path resolution for TauDEM binaries (e.g., `moveoutletstostrms → moveoutletstostreams`).
- Expanded logging and run summaries for CI visibility.
- Protected `main` branch to require successful CI validation before merge.

### Notes
This release formalizes SYMFLUENCE's **reproducibility framework**, guaranteeing that all supported workflows can be rebuilt and validated automatically on clean systems.

---

## [0.5.0] - 2025-01-09

### Major: CONFLUENCE → SYMFLUENCE Rebranding

**This is the rebranding release.** The project is now SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration).

### Added
- Complete rebranding to SYMFLUENCE
- New domain: [symfluence.org](https://symfluence.org)
- PyPI package: `pip install symfluence`
- Backward compatibility for all CONFLUENCE names (with deprecation warnings)

### Changed
- Main script: `CONFLUENCE.py` → `symfluence.py`
- Shell command: `./confluence` → `./symfluence`
- Config parameters: `CONFLUENCE_*` → `SYMFLUENCE_*`
- Repository: `DarriEy/CONFLUENCE` → `symfluence-org/SYMFLUENCE`

### Deprecated
- All CONFLUENCE naming (removed in v0.6.0)

---

## Links

- **PyPI**: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- **Documentation**: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **GitHub**: [github.com/symfluence-org/SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE)
- **Issues**: [github.com/symfluence-org/SYMFLUENCE/issues](https://github.com/symfluence-org/SYMFLUENCE/issues)
