# Contributing to SYMFLUENCE

We welcome all contributions — from bug fixes and documentation improvements to new model integrations and performance optimizations. This guide outlines how to get started and how to collaborate effectively.

---

## 1. Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/SYMFLUENCE.git
   cd SYMFLUENCE
   git remote add upstream https://github.com/symfluence-org/SYMFLUENCE.git
   ```

2. **Set up your environment**
   ```bash
   ./scripts/symfluence-bootstrap --install
   ```
   This will create and manage a `.venv` automatically.
   If you prefer manual setup:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   ./scripts/symfluence-bootstrap --help
   ```

---

## 2. Licensing & CLA

### Open Source License

SYMFLUENCE is licensed under **GPL-3.0-or-later**. By contributing, you agree
that your contributions will be licensed under the same terms. This is a
copyleft license — any derivative work that is distributed must also be released
under GPL-3.0-or-later.

When creating new source files, include the standard SPDX header:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
```

### Contributor License Agreement (CLA)

All external contributors must sign the [Contributor License Agreement](CLA.md)
before a pull request can be merged. The CLA is checked automatically by CI —
unsigned PRs will fail the CLA check.

**Why a CLA?** The CLA grants the SYMFLUENCE Team the rights needed to
maintain the project under GPL-3.0-or-later while preserving the option to offer
the software under additional license terms (e.g., commercial licenses) in the
future. This dual-licensing flexibility is described in
[CLA.md Section 5](CLA.md#5-dual-licensing). Your contributions always remain
available under the open source license.

**How to sign:**

1. Read the [CLA](CLA.md)
2. Add your name and details to the `CLA_SIGNATURES` file via a pull request
3. Once merged, the CLA check will pass on all your future PRs

Repository owners and bots are exempt.

---

## 3. Plugin-First Contributions

If you want to add a **new model integration, data handler, optimization
algorithm, or evaluation metric**, the default path is to publish an independent
Python package that registers with SYMFLUENCE via the `symfluence.plugins`
entry-point system — not a pull request to the core repository.

This is a deliberate governance choice that keeps the core maintainable and
prevents any single institution from reshaping it by volume of contribution.
See [GOVERNANCE.md Section 3](GOVERNANCE.md#3-contribution-model-plugins-first)
for the full rationale.

**What belongs in core:** bug fixes, performance improvements, documentation,
infrastructure that benefits all plugins, and changes to registry interface
definitions (subject to the [interface change process](GOVERNANCE.md#4-interface-stewardship)).

Pull requests that add new models or algorithms directly to the core will be
redirected to the plugin path unless there is a compelling reason for core
inclusion.

---

## 4. Branching Strategy

SYMFLUENCE uses a simple branching model for organized development:

```
main          ← Protected, releases only (v0.6.0, v0.7.0, v1.0.0)
  ↓
develop       ← Active development, merge here first
  ↓
feature/*     ← Individual features
hotfix/*      ← Emergency fixes
```

### Working on Features

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create your feature branch
git checkout -b feature/my-feature

# Work, commit, push
git add .
git commit -m "Add feature description"
git push -u origin feature/my-feature

# Open PR to merge into develop
```

### Creating Releases

```bash
# When develop is ready for release
git checkout main
git merge develop
git tag -a v0.7.0 -m "Release v0.7.0"
git push origin main --tags
```

### Hotfixes

```bash
# For critical production bugs
git checkout -b hotfix/critical-bug main
# Fix and commit
git checkout main
git merge hotfix/critical-bug
git tag -a v0.7.1 -m "Hotfix: critical bug"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge hotfix/critical-bug
git push origin develop
```

---

## 5. Making Changes

### Code Style
- Follow **PEP 8** and use clear, descriptive variable names.
- Include **type hints** and informative **docstrings**.
- Keep functions focused and testable.
- Prefer explicit over implicit; avoid magic numbers.

### Type Checking Notes

SYMFLUENCE uses mypy for static type checking. Some files contain `# type: ignore` comments due to limitations in third-party type stubs:

| Library | Issue | Affected Modules |
|---------|-------|------------------|
| **matplotlib** | Incomplete type stubs for plotting APIs | `reporting/plotters/` |
| **xarray** | Complex generic types not fully captured | `reporting/processors/`, data modules |
| **pandas** | Some DataFrame operations lack precise types | Various data processing |
| **GDAL/rasterio** | C extension bindings have limited type info | Geospatial modules |

These suppressions are intentional and reviewed. When adding new `# type: ignore` comments:
1. Use the most specific form: `# type: ignore[error-code]`
2. Add a brief comment explaining why it's needed
3. Prefer fixing the type issue if possible

### Docstring Conventions
We use NumPy-style docstrings for consistency. Include:
- A one-line summary
- Extended description for complex functions
- Args/Parameters, Returns, Raises sections
- Examples for public APIs

```python
def calculate_runoff(precip: float, area: float) -> float:
    """
    Compute runoff from precipitation rate and catchment area.

    Converts precipitation depth rate to volumetric flow rate using
    standard hydrological unit conversions.

    Args:
        precip: Precipitation rate in mm/hour
        area: Catchment area in km²

    Returns:
        Runoff in m³/s (cubic meters per second)

    Example:
        >>> calculate_runoff(10.0, 100.0)
        277.78
    """
    return (precip / 1000) * area * 1e6 / 3600
```

### Running Tests
Before submitting a PR, ensure tests pass:

```bash
# Run quick tests (recommended before each commit)
pytest tests/ -m "not slow and not requires_cloud" -x

# Run full local test suite
pytest tests/ -m "not requires_cloud"

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests
pytest tests/e2e/ --run-full-examples  # End-to-end tests (slow)
```

See `tests/TESTING.md` for detailed testing documentation.

### Commit Messages
Use concise, descriptive messages:
- `Add FUSE model interface`
- `Fix NetCDF write bug`
- `Update optimization documentation`

---

## 6. Submitting Your Work

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-update
   ```

2. **Commit and push**
   ```bash
   git commit -m "Describe your change"
   git push origin feature/my-update
   ```

3. **Open a Pull Request (PR)**
   Include:
   - **Description:** what and why
   - **Type:** new feature, fix, documentation, etc.
   - **Testing:** how it was verified
   - **Related issues:** e.g., "Closes #42"

Example:
```
## Description
Adds support for SUMMA-MIZU coupled runs.

## Type
- [x] Feature
- [ ] Fix
- [ ] Docs

## Testing
Validated on example domain; all tests pass.

## Related Issues
Closes #117
```

---

## 7. Code Review
All submissions are reviewed by maintainers. Expect constructive feedback — discussions help keep the codebase consistent and maintainable.

Please be responsive and open to suggestions.

---

## 8. Reporting Issues
When reporting, include:
- **Description:** what went wrong
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment:** OS, Python version, SYMFLUENCE commit/branch

Example:
```
## Description
Model setup fails on FIR cluster with NetCDF 4.9.2.

## Steps
1. Run `symfluence workflow step setup_project --config config.yaml`
2. Error during model initialization

## Expected
Setup completes successfully

## Actual
KeyError: 'MESH_PARAM_FILE'

## Environment
OS: Rocky Linux 8
Python: 3.11.7
Branch: main
```

---

## 9. Feature Requests
We value ideas for improvement. When proposing features:
- Describe the functionality and motivation.
- Suggest how it might fit into existing workflows.
- Include example usage if possible.

---

## 10. Contribution Types
We welcome:
- Bug fixes
- Documentation updates
- Example projects or tutorials
- New model or data integrations
- Performance improvements
- Visualization or reporting enhancements

---

## 11. API Stability and Versioning

SYMFLUENCE follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (SemVer):

```
MAJOR.MINOR.PATCH (e.g., 1.2.3)
```

### Version Guarantees

| Version Change | What It Means | Example |
|----------------|---------------|---------|
| **MAJOR** (1.x → 2.x) | Breaking changes to public API | Removing deprecated functions, changing return types |
| **MINOR** (1.1 → 1.2) | New features, backward compatible | Adding new models, new CLI commands |
| **PATCH** (1.1.1 → 1.1.2) | Bug fixes, backward compatible | Fixing calculation errors, typos |

### Public API Definition

The **public API** includes:
- All classes and functions exported in `__all__` from top-level modules
- CLI commands documented in `--help`
- Configuration file format (YAML keys)
- Python API: `SYMFLUENCE`, `SymfluenceConfig`, and exported exceptions

The following are **not** part of the public API:
- Internal modules (prefixed with `_` or in `internal/` directories)
- Undocumented functions or classes
- Debug/logging output format
- Specific error message text

### Pre-1.0 Stability

While SYMFLUENCE is pre-1.0 (currently 0.x.x):
- MINOR versions may include breaking changes (documented in CHANGELOG)
- PATCH versions are always backward compatible
- Deprecation warnings will be issued at least one MINOR version before removal

### Deprecation Policy

1. **Announce**: Deprecated features are marked with `warnings.warn()` and documented in CHANGELOG
2. **Grace period**: Deprecated features remain functional for at least one MINOR release
3. **Remove**: Removal is announced in CHANGELOG with migration guidance

### For Contributors

When making changes:
- **Adding features**: Increment MINOR version
- **Fixing bugs**: Increment PATCH version
- **Breaking changes**: Increment MAJOR version (or MINOR if pre-1.0), document migration path
- Always update `CHANGELOG.md` with your changes

---

## 12. Questions
If you're unsure where to start:
- Open a GitHub discussion or issue
- Review existing docs at [symfluence.readthedocs.io](https://symfluence.readthedocs.io)

Thank you for helping improve SYMFLUENCE.
