# Architecture Decision Records

This directory records significant architecture and policy decisions for
SYMFLUENCE as short, dated documents. An ADR captures the *context* a decision
was made in, the *decision* itself, and its *consequences*, so that a choice
which is easy to make once and easy to forget later remains legible to future
contributors.

ADRs are immutable once accepted. A decision is changed not by editing its ADR
but by adding a new ADR that supersedes it; the old record stays in place with
its status updated to `Superseded by ADR-XXXX`. This keeps the history of *why*
intact rather than overwriting it.

## When to write one

Write an ADR when a decision (a) affects a stable interface or public contract
(see [GOVERNANCE.md](../../GOVERNANCE.md) §4), (b) sets a project-wide policy a
contributor could otherwise reasonably guess wrong, or (c) resolves a question
that has surfaced more than once. Routine bug fixes and local implementation
choices do not need one.

## Format

Each ADR follows a light [MADR](https://adr.github.io/madr/)-style template:
a status line, **Context**, **Decision**, **Consequences**, and **References**.
Keep them short — an ADR is a record, not a design document.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0000](0000-initial-architecture-decisions.md) | Initial architecture decisions (recorded retroactively) | Accepted |
| [0001](0001-remove-legacy-registry-shims.md) | Legacy registry shim classes are removed before 1.0 | Accepted |
| [0002](0002-plugins-may-ship-typed-config.md) | Plugins may ship their own typed configuration schema | Accepted |
| [0003](0003-config-dict-override-is-supported.md) | `_config_dict_override` is a supported escape hatch | Accepted |
| [0004](0004-bundled-agent-human-in-the-loop.md) | The bundled AI agent does not push without a human | Accepted |
| [0005](0005-logging-level-policy.md) | Logging-level policy: ERROR is the operational ceiling | Accepted |
| [0006](0006-config-unknown-keys-warn-by-default.md) | Unknown config keys: warn by default, strict opt-in | Accepted |
| [0007](0007-gui-single-user-localhost.md) | The web GUI is a single-user localhost tool | Accepted |
| [0008](0008-coverage-gate-raise-and-ratchet.md) | Coverage policy: ratcheted global gate plus core-spine gate | Accepted |
| [0011](0011-retire-unused-back-compat-shims.md) | Unused back-compat shim modules are deleted; plugin-used ones warn and go before 1.0 | Accepted |

ADR-0000 records the baseline decisions the codebase was built on, written
down retroactively when the ADR practice was adopted; ADR-0001 onward record
decisions made (or made explicit) ahead of the 1.0 release.

## Related records

- [config_flat_key_audit.md](config_flat_key_audit.md) — the executed
  2026-06 flat-key audit that closed the recognized-flat-key set (referenced
  by `core/config/legacy_aliases.py` and the config-authority tests). An
  audit snapshot rather than an ADR, kept here with the rest of the decision
  history.
