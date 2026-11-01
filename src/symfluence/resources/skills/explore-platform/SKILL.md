---
name: explore-platform
description: >-
  Discover what SYMFLUENCE offers by introspecting the live registry and config
  schema — available models, forcing/observation datasets, optimizers, calibration
  targets, metrics, presets, templates, and config keys. Start here before
  choosing a model/dataset to run, or before extending the platform.
when_to_use:
  - "A user asks \"what models / forcings / optimizers / metrics are available?\""
  - Choosing a HYDROLOGICAL_MODEL, FORCING_DATASET, optimizer, or metric for a config
  - Checking whether a config key exists / finding the right key name
  - Before extending the platform (see what's already registered first)
---

# Exploring the SYMFLUENCE platform

SYMFLUENCE is registry-driven: models, datasets, optimizers, metrics, calibration
targets, presets, and the whole config schema are looked up from a live registry,
not a fixed list. **Always discover capabilities by querying the platform — never
rely on a hardcoded or remembered list, which goes stale.** This skill is the
discovery entry point for both running the platform and extending it.

## 1. The one command: `symfluence list`

```bash
symfluence list                # all catalogs + counts
symfluence list models         # registered hydrological/routing models (HYDROLOGICAL_MODEL)
symfluence list forcings       # forcing & attribute datasets (FORCING_DATASET, attribute sources)
symfluence list observations   # streamflow / in-situ observation networks
symfluence list optimizers     # calibration algorithms (OPTIMIZATION_ALGORITHM)
symfluence list targets        # calibration targets (streamflow, snow, ET, ...)
symfluence list metrics        # objective/evaluation metrics (OPTIMIZATION_METRIC)
symfluence list presets        # project init presets
symfluence list templates      # config templates you can copy
symfluence list steps          # the workflow steps (= symfluence workflow list-steps)
symfluence list config-keys    # every recognized config key (the public contract)
```

Each catalog is read straight from the registry / config schema at runtime, so
what it prints is exactly what this install supports — including any plugins.

## 2. The Python equivalents (when scripting or in a notebook)

```python
from symfluence.core.registries import R
sorted(R.runners.keys())                 # models
sorted(R.acquisition_handlers.keys())    # forcing/attribute datasets
sorted(R.observation_handlers.keys())    # observation networks
sorted(R.calibration_targets.keys())     # calibration targets (e.g. SUMMA_STREAMFLOW)
sorted(R.metrics.keys())                 # metrics

from symfluence.core.calibration.optimizers.algorithms import list_algorithms
list_algorithms()                        # optimizers

from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP
sorted(FLAT_TO_NESTED_MAP)               # every recognized config key
```

`R` exposes ~30 registries (`core/registries.py`) — the same `.keys()` works on
any of them (`preprocessors`, `postprocessors`, `workers`, `parameter_managers`,
`evaluators`, `delineation_strategies`, `bmi_adapters`, …).

## 3. From "what's available" to using it

The catalog names ARE the config values. To run something, set the matching key:

- A model from `list models` → `HYDROLOGICAL_MODEL: <name>`
- A dataset from `list forcings` → `FORCING_DATASET: <name>`
- An optimizer from `list optimizers` → `OPTIMIZATION_ALGORITHM: <name>`
- A metric from `list metrics` → `OPTIMIZATION_METRIC: <name>`
- A key from `list config-keys` → confirm spelling / discover an option

Then author and run the config — see the **run-workflow-locally** skill (CLI,
required keys, stage markers, `config validate`).

## 4. From "it's missing" to extending it

If `symfluence list` doesn't show what you need, add it — the registry is the
extension surface. Pick the matching skill:

- A new dataset → **add-data-handler**
- A new model → **add-model-handler**
- A new optimizer/search algorithm → **add-optimizer**
- A calibration that misbehaves → **debug-calibration**

After registering, re-run `symfluence list <kind>` to confirm the new entry
appears (registration runs at import; if it's absent, the import or entry point
isn't wired — see the relevant skill's troubleshooting).

## 5. Verify before asserting

When a user asks "can SYMFLUENCE do X?", run the matching `symfluence list` (or
`R.*.keys()`) and answer from the result — do not answer from memory. The live
registry is the source of truth.
