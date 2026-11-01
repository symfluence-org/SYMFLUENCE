# ADR-0002: Plugins may ship their own typed configuration schema

- **Status:** Accepted
- **Date:** 2026-06-05
- **Related:** [ADR-0000](0000-initial-architecture-decisions.md) (decisions 1 and 4), GOVERNANCE.md §3

## Context

Until mid-2026, runners and post-processors were pluggable but typed
configuration was not. The top-level configuration model named only the models
the framework already knew about, and `R.config_schemas` — where a plugin would
register a typed Pydantic schema — was not read by the core validation
pipeline. A third-party plugin could therefore register a runner but could not
supply a typed config schema that participated in the validated config tree; it
had to fall back on untyped pass-through keys.

That asymmetry contradicted the project's own plugins-first contribution model
(GOVERNANCE.md §3): the documentation promised that a first-class model could
live in an independent package, but a first-class model includes its typed
configuration. The question for 1.0 was whether to open the typed-config path
or to document the closed shape honestly.

## Decision

Typed plugin configuration is **opened before 1.0**. A plugin may register its
own typed configuration schema and have it participate in the validated config
tree on equal footing with in-tree models.

This is already implemented in PR #139. A plugin declares its schema through
`model_manifest(config_schema=...)`; the schema is registered into
`R.config_schemas`, and the core resolution path
(`core/config/config_resolution.get_config_schema`) reads it during
validation. The top-level `ModelConfig` no longer carries a hardcoded
`Optional[*Config]` field per model — model-specific configuration is resolved
generically from the registry and re-flattened on serialization so that
stage-marker hashing stays byte-stable.

Plugins that register only a `config_adapter` (rather than an explicit
`config_schema`) are bridged automatically: `model_manifest()` registers the
adapter's `get_config_schema()` into `R.config_schemas`, and adapters whose
schemas use bare field names plus a `CONFIG_PREFIX` get their flat-key
transformers derived from the prefix (PR #178). The JAX model plugins work
through this path with zero plugin-side changes.

## Consequences

- The plugin contract at 1.0 is: **runners, post-processors, and typed
  configuration schemas are all pluggable.** Plugin documentation should
  describe typed config as supported.
- Adding a new in-tree model no longer requires editing the top-level config
  model — it requires registering a schema, the same path a plugin uses. This
  removes a class of "registered but unvalidated" drift.
- `R.config_schemas` is now a consumed registry. A CI guard
  (`tests/unit/config/test_config_schema_consumed.py`) verifies registered
  schemas are actually read, so the path cannot silently regress to the closed
  shape.

## References

- PR #139 — typed-config plumbing; PR #178 — adapter bridging
- `core/config/config_resolution.py` — `get_config_schema`
  (formerly re-exported at `models/config_resolution.py`; that shim was removed
  under ADR-0011)
- GOVERNANCE.md §3 (Plugins First), §4.1 (configuration schema keys)
