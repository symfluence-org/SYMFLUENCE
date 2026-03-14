# SYMFLUENCE Governance

This document describes how SYMFLUENCE is governed: who makes decisions, how
those decisions are made, and what commitments the project makes to its
contributors and users. It is intended to be read alongside
[CONTRIBUTING.md](CONTRIBUTING.md), which covers the mechanics of contribution,
and [CLA.md](CLA.md), which covers intellectual property.

---

## 1. Purpose of This Document

SYMFLUENCE's architecture is designed around typed registry contracts that
enable decentralized contribution without central gatekeeping. The value of the
framework lies not in any single model or data handler but in the coherence of
these contracts — the fact that 27 models, 68 data handlers, and 17
optimization algorithms interoperate through shared interface definitions.

This governance document exists to protect that coherence as the contributor
base grows. Architecture can enable coordination. It cannot sustain it. The
norms established here are the social complement to the technical patterns
described in the SYFMLUENCE papers (Eythorsson et al., 2026a,b,c).

---

## 2. Project Leadership

### 2.1 Benevolent Dictator for Now (BDfN)

SYMFLUENCE is currently maintained by a single lead maintainer (**Darri
Eythorsson**), who holds final authority over:

- Releases to `main` and PyPI
- Changes to registry interface definitions (Section 4)
- Additions to the core package (as distinct from plugins)
- Acceptance or rejection of pull requests

This is a pragmatic arrangement reflecting the project's current stage, not a
permanent governance model. The intent is to transition to a broader
maintainership structure as the contributor community develops (Section 6).

### 2.2 Maintainers

Maintainers have commit access to the `develop` branch, can review and approve
pull requests, and can triage issues. Maintainer status is granted by the lead
maintainer based on sustained, high-quality contribution.

Current maintainers:

| Name               | Scope          | GitHub        |
|--------------------|----------------|---------------|
| Darri Eythorsson   | Lead / All     | @DarriEy      |

Maintainers are expected to:

- Review PRs within their scope in a reasonable timeframe
- Follow the interface change process (Section 4)
- Uphold the plugin-first contribution model (Section 3)

### 2.3 Contributors

Anyone who has signed the [CLA](CLA.md) and had a contribution merged is a
recognized contributor. Contributors do not have commit access but participate
through pull requests, issues, and discussions.

---

## 3. Contribution Model: Plugins First

SYMFLUENCE's plugin architecture — the `symfluence.plugins` entry-point system
— exists specifically so that most contributions do not require changes to the
core repository. This is a governance choice, not just a technical one.

### 3.1 What Goes in a Plugin

The default path for contributing new capability is an independent Python
package that registers with SYMFLUENCE at install time. This includes:

- **New model integrations.** Implement the four component interfaces
  (preprocessor, runner, postprocessor, extractor) and register via
  `model_manifest()`. See the six JAX-native packages (jHBV, jSACSMA, etc.)
  as reference implementations.
- **New data handlers.** Implement the acquisition handler interface and
  register via decorator.
- **New optimization algorithms.** Implement the optimizer interface and
  register.
- **New evaluation metrics.** Implement the metric signature and register.

Plugin packages are owned and maintained by their authors. They set their own
release schedules, choose their own licenses (subject to compatibility with
GPL-3.0), and are not subject to core review beyond interface compliance.

### 3.2 What Goes in Core

Changes to the core SYMFLUENCE package are appropriate when they:

- Fix bugs in existing functionality
- Improve performance or reliability of existing components
- Add infrastructure that benefits all plugins (e.g., new base class methods)
- Update documentation
- Modify registry interface definitions (subject to Section 4)

**Pull requests that add new models, data handlers, or algorithms directly to
the core package will be redirected to the plugin path** unless there is a
compelling reason for core inclusion (e.g., the component is needed for the
test suite or serves as a reference implementation).

This is not gatekeeping — it is the mechanism by which the project scales
without concentrating maintenance burden on the core team. A plugin that works
today will continue to work as long as the interface contracts are maintained,
regardless of what happens to the core repository.

### 3.3 Rationale

The plugin-first model serves three purposes:

1. **Reduces core maintenance burden.** Each model integration maintained in
   core is a perpetual maintenance commitment. Plugins distribute this cost.
2. **Prevents institutional capture.** No single institution can reshape the
   core by volume of contribution. The interfaces are the shared property;
   implementations are private.
3. **Makes forking unnecessary.** If the default contribution path is "publish
   your own package," there is no reason to fork the core to add capability.
   Forks that modify interface definitions fragment the ecosystem; plugins
   that conform to them extend it.

---

## 4. Interface Stewardship

The registry interface definitions — the typed contracts that govern how
components interact — are the most valuable and most fragile part of the
project. Their coherence is what converts independent contributions into
collective capability. This section describes how they are managed.

### 4.1 What Constitutes an Interface

The following are considered **stable interfaces** subject to the change
process described below:

- **Registry type contracts.** The abstract base classes that define what a
  preprocessor, runner, postprocessor, extractor, optimizer, metric, or data
  handler must implement. These are the `Base*` classes in the core package.
- **The `model_manifest()` signature.** The entry point for model registration.
- **The `Registry[T]` public API.** The `get()`, `register()`, and `list()`
  methods and their signatures.
- **The CF-Intermediate Format (CFIF).** The variable names, units, and
  conventions of the standardized forcing format.
- **Configuration schema keys.** The top-level YAML configuration sections
  and their required fields.
- **The `symfluence.plugins` entry-point contract.** What a plugin must
  provide and what it receives.

The following are **not** stable interfaces and may change between minor
releases:

- Internal implementation details of managers, orchestrators, and utilities
- Logging format and verbosity behavior
- File system layout of intermediate artifacts
- Visualization and plotting internals

### 4.2 Interface Change Process

Changes to stable interfaces follow a graduated process based on scope:

**Additive changes** (new optional fields, new base class methods with default
implementations, new registry types):

- Require a GitHub issue describing the change and its motivation
- Require review by at least one maintainer
- Must not break existing registered components
- Are released in MINOR versions

**Breaking changes** (renamed methods, removed fields, changed signatures,
modified CFIF conventions):

- Require a GitHub issue with an explicit migration guide
- Require review by the lead maintainer
- Must be preceded by a deprecation period of at least one MINOR release,
  during which the old interface remains functional with warnings
- Must include updates to all in-tree components that use the changed
  interface
- Must include CI tests that verify backward compatibility during the
  deprecation period
- Are released in MAJOR versions (or MINOR versions pre-1.0, per SemVer)

**Emergency changes** (security fixes, critical correctness bugs in interface
definitions):

- May bypass the deprecation period
- Require documentation in CHANGELOG with a clear explanation
- Should be accompanied by a notification to known plugin maintainers

### 4.3 Registry Namespace

Each registry maintains a namespace of string keys. To prevent collisions as
the ecosystem grows:

- Core-provided components use short, uppercase keys (e.g., `SUMMA`, `DDS`,
  `KGE`).
- Plugin-provided components should use a namespaced key format:
  `ORGNAME_COMPONENT` (e.g., `RTI_NGEN_V2`, `ECCC_MESH_SVS`).
- The lead maintainer reserves the right to reject or rename keys that
  conflict with existing registrations.
- A registry of registered keys will be maintained in the documentation to
  prevent silent collisions.

### 4.4 Compatibility Testing

The CI suite includes interface compliance tests that verify:

- All registered components satisfy their declared type contracts
- The plugin entry-point loader successfully discovers and registers
  installed plugins
- CFIF-compliant forcing files are readable by all registered model
  preprocessors

Plugin authors are encouraged to run these tests against their packages.
Instructions are provided in the plugin development guide.

---

## 5. Decision-Making

### 5.1 Routine Decisions

Bug fixes, documentation improvements, and non-interface changes are decided
through normal pull request review. Any maintainer can approve and merge.

### 5.2 Significant Decisions

The following require explicit approval from the lead maintainer:

- Changes to stable interfaces (Section 4)
- Addition of new maintainers
- Changes to this governance document
- Release of new MAJOR versions
- Changes to the licensing structure, including the granting of commercial
  licenses under the dual licensing model described in the CLA (Section 5)

### 5.3 Dispute Resolution

If a contributor disagrees with a maintainer decision:

1. Raise the concern in the relevant GitHub issue or PR discussion.
2. If unresolved, open a dedicated GitHub discussion tagged `governance`.
3. The lead maintainer makes the final decision, with reasoning documented
   publicly in the discussion thread.

The goal is not unanimity but transparency. Decisions should be explained, not
just announced.

---

## 6. Evolution and Succession

### 6.1 Transition Criteria

The current single-maintainer structure is appropriate for a project with a
small contributor base. Transition to a broader governance model should be
considered when:

- Three or more individuals have demonstrated sustained contribution over at
  least six months
- At least two institutions are actively contributing or depending on the
  framework
- The lead maintainer's availability becomes a bottleneck for releases or
  reviews

### 6.2 Possible Future Models

This document does not prescribe a specific future governance structure. The
companion papers discuss several models used by comparable projects (Apache
PMC, Blender Foundation, NumFOCUS fiscal sponsorship, institutional hosting).
The appropriate model will depend on the community that develops.

What this document does prescribe is that any future governance model must:

- Preserve the plugin-first contribution norm (Section 3)
- Maintain the interface change process (Section 4)
- Keep the project openly licensed under GPL-3.0-or-later
- Ensure that no single institution can unilaterally modify stable interfaces

### 6.3 Succession

If the lead maintainer is unable to continue in the role:

- Commit access and PyPI publishing credentials are documented in a sealed
  succession plan held by [to be designated — a trusted colleague or
  institutional contact].
- The designated successor(s) assume maintainer responsibilities.
- This governance document remains in effect unless explicitly amended through
  the process described in Section 5.2.

The project should not depend on any single person. That is, after all, the
argument of the papers.

---

## 7. Code of Conduct

Contributors are expected to engage respectfully and constructively. The
project does not currently adopt a formal code of conduct but reserves the
right to do so as the community grows. In the interim:

- Critique code, not people.
- Assume good faith.
- If a discussion becomes unproductive, step away and return later.

---

## 8. Relationship to Companion Papers

This governance document implements commitments made in the companion paper
series:

- **Paper 1** (Eythorsson et al., 2026a) argued that infrastructure
  maintenance deserves recognition equivalent to scientific contribution.
  This document operationalizes that argument by defining maintainer roles
  and contribution norms.
- **Paper 2** (Eythorsson et al., 2026b), Section 5.3, identified the need
  for stewardship institutions to maintain registry coherence. This document
  is the initial stewardship institution, appropriate to the project's
  current scale.
- **Paper 2**, Section 2.4, described the registry as a social contract.
  Section 4 of this document codifies what that contract requires and how it
  evolves.

---

## Document History

| Version | Date       | Author           | Summary                    |
|---------|------------|------------------|----------------------------|
| 1.0     | 2026-03-XX | Darri Eythorsson | Initial governance document |
