# ADR-0011: Unused back-compat shim modules are deleted; plugin-used ones warn and go before 1.0

- **Status:** Accepted
- **Date:** 2026-11-01
- **Related:** ADR-0001, GOVERNANCE.md §4, CHANGELOG "Deprecated"

## Context

The service-decomposition promotions (adapter contract tier into
`core/modeling/`, calibration contract into `core/calibration/`, metrics into
`core/metrics/`, and so on) left one forwarding module at every historical
import path. Each was roughly `from <canonical> import *` plus a
`__getattr__` forwarder. There were about 95 of them under `models/`,
`optimization/`, `evaluation/`, `data/`, `cli/services/`, `geospatial/`, and
`core/config/models/model_configs_*.py`. The CHANGELOG said they would be
removed at 2.0.

ADR-0001 already set the principle: compatibility surfaces should go *before*
1.0 so they never enter the SemVer-protected API. Most of these shims had no
importer at all. In-tree code was already guard-enforced onto canonical paths,
and an inventory of the installed external plugins (jsacsma, jhbv, jhechms,
jsnow17, jtopmodel, jxaj, droute) showed that only 22 shim modules are
imported from outside. None of the shims emitted a `DeprecationWarning`, so
plugin authors got no signal to migrate.

## Decision

1. **Shims nobody imports are deleted now.** 73 shim modules were removed,
   along with the now-empty `models/templates`, `models/coupled`,
   `optimization/mixins`, `optimization/multi_gauge` and
   `optimization/optimizers/algorithms` packages. In-tree code and tests that
   still went through a shim (mostly package `__init__` re-exports using
   relative imports, which the guard test could not see) were repointed to the
   canonical module.
2. **Shims an external plugin imports stay until 1.0, and warn.** Each kept
   shim emits a `DeprecationWarning` on import that names its canonical
   module. Calibration autodiscovery skips the shim packages under
   `symfluence.models` so that a plain `import symfluence` does not fire them.
   Only plugin imports do.
3. **Removal plan.** Each plugin rewrites its imports to the canonical paths
   below and cuts a release. The `jax` extra (and the droute pin) then
   requires those releases, and the remaining shims are deleted in the same
   change, before the 1.0 tag. `tests/unit/core/test_external_plugin_compat.py`
   (no installed plugin dropped during discovery) is the gate for that
   deletion. This supersedes the "removed at 2.0" wording in the CHANGELOG.

### Shims kept, and why

| Shim module | Canonical module | Imported by |
|---|---|---|
| `symfluence.data.utils.netcdf_utils` | `symfluence.core.modeling.netcdf_utils` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.evaluation.metrics` | `symfluence.core.metrics.metrics` | droute, jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.geospatial.geometry_utils` | `symfluence.core.geometry_utils` | droute, jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.adapters` | `symfluence.core.modeling.adapters` | jsacsma, jxaj |
| `symfluence.models.base` | `symfluence.core.modeling.base` | droute, jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.base.base_config` | `symfluence.core.modeling.base.base_config` | jhbv, jhechms, jsacsma, jsnow17, jtopmodel, jxaj |
| `symfluence.models.base.standard_postprocessor` | `symfluence.core.modeling.base.standard_postprocessor` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.execution` | `symfluence.core.modeling.execution` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.mixins` | `symfluence.core.modeling.mixins` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.mixins.pet_calculator` | `symfluence.core.modeling.mixins.pet_calculator` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.spatial_modes` | `symfluence.core.modeling.spatial_modes` | jhbv |
| `symfluence.models.state` | `symfluence.core.modeling.state` | jhbv |
| `symfluence.models.utilities` | `symfluence.core.modeling.utilities` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.models.utilities.base_topology_generator` | `symfluence.core.modeling.utilities.base_topology_generator` | droute |
| `symfluence.models.utilities.runoff_loader` | `symfluence.core.modeling.utilities.runoff_loader` | droute |
| `symfluence.optimization.core.base_parameter_manager` | `symfluence.core.calibration.parameters.base_parameter_manager` | droute, jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.optimization.core.parameter_bounds_registry` | `symfluence.core.calibration.parameters.parameter_bounds_registry` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.optimization.optimizers.base_model_optimizer` | `symfluence.core.calibration.optimizers.base_model_optimizer` | droute, jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.optimization.regionalization.strategies` | `symfluence.core.calibration.regionalization.strategies` | droute |
| `symfluence.optimization.workers.base_worker` | `symfluence.core.calibration.workers.base_worker` | droute, jhbv, jhechms, jsacsma, jtopmodel, jxaj |
| `symfluence.optimization.workers.inmemory_worker` | `symfluence.core.calibration.workers.inmemory_worker` | jhbv, jhechms, jsacsma, jtopmodel, jxaj |

The `symfluence.optimization.optimizers` package itself stays only as the
parent of `base_model_optimizer`. Its re-exports had no importer and were
removed.

Out of scope here: `symfluence.optimization.parameter_managers` (still
imported in-tree by `optimization_manager` to seed discovery), the
`symfluence.resources` base-settings forwarders, and the explicit
"compatibility exports" modules such as `evaluation/likelihood.py`,
`data/observation/paths.py` and `reporting/*`.

## Consequences

- Downstream code that imported a deleted path gets `ModuleNotFoundError`.
  The canonical homes follow the pattern in the table above
  (`models.*` → `core.modeling.*`, `optimization.*` → `core.calibration.*`,
  `evaluation.metric*` → `core.metrics.*`). Pre-1.0 SemVer allows this
  without a deprecation cycle (GOVERNANCE.md §4.2). No installed plugin is
  affected.
- Plugin authors now see a `DeprecationWarning` (under pytest, or with
  `-W default`) that tells them exactly what to rewrite.
- `tests/unit/core/test_no_shim_imports_in_tree.py` now resolves relative
  imports, so a package `__init__` can no longer route through a shim with
  `from .shim import ...`.

## References

- ADR-0001: legacy registry shims are removed before 1.0
- `tests/unit/core/test_external_plugin_compat.py`: plugin registration gate
- `tests/unit/core/test_no_shim_imports_in_tree.py`: in-tree guard
