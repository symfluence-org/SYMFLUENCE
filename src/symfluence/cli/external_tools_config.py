#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE External Tools Configuration

This module provides build configurations for external tools required by SYMFLUENCE.

Architecture:
    - Infrastructure tools (sundials, taudem, gistool, datatool, ngiab) are defined
      directly in this file and registered via R.build_instructions.add()
    - Model-specific tools (summa, fuse, mizuroute, ... and any external plugin
      model) are owned by the model package.  The package declares its build
      instructions when it registers itself — ``model_manifest(
      build_instructions_module="<pkg>.build_instructions")`` — and plugin
      discovery runs that registration for every installed model at startup.
      This module never enumerates model packages or looks for them on disk.

Public API:
    get_external_tools_definitions() -> Dict[str, Dict[str, Any]]
        Returns all tool definitions (both infrastructure and model-specific).
        This is the primary interface used by BinaryManager.

Tools Defined Here (Infrastructure):
    - SUNDIALS: Differential equation solver library (required by SUMMA)
    - TauDEM: Terrain Analysis Using Digital Elevation Models
    - GIStool: Geospatial data extraction tool
    - Datatool: Meteorological data processing tool
    - NGIAB: NextGen In A Box deployment system
    - Enzyme AD: Automatic differentiation via LLVM (used by cFUSE)
"""
from __future__ import annotations

from typing import Any, Dict

from symfluence.core.build.build_snippets import (
    get_common_build_environment,
)
from symfluence.core.registries import R

from .external_tools_build_commands import (
    ENZYME_BUILD_COMMAND,
    OPENFEWS_BUILD_COMMAND,
    SUNDIALS_BUILD_COMMAND,
    TAUDEM_BUILD_COMMAND,
)
from .services.build_registry import BuildInstructionsRegistry


def _register_sundials(common_env: str) -> None:
    """Register SUNDIALS build instructions."""
    # ================================================================
    # SUNDIALS - Solver Library (Install First - Required by SUMMA)
    # ================================================================
    R.build_instructions.add('sundials', {
        'description': 'SUNDIALS - SUite of Nonlinear and DIfferential/ALgebraic equation Solvers',
        'config_path_key': 'SUNDIALS_INSTALL_PATH',
        'config_exe_key': 'SUNDIALS_DIR',
        'default_path_suffix': 'installs/sundials/install/sundials/',
        'default_exe': 'lib/libsundials_core.a',
        'repository': None,
        'branch': None,
        'install_dir': 'sundials',
        'build_commands': [
            common_env,
            SUNDIALS_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': [
                'install/sundials/lib64/libsundials_core.a',
                'install/sundials/lib/libsundials_core.a',
                'install/sundials/include/sundials/sundials_config.h'
            ],
            'check_type': 'exists_any'
        },
        'order': 1,
        'library_only': True,
    })

def _register_taudem(common_env: str) -> None:
    """Register TauDEM build instructions."""
    # ================================================================
    # TauDEM - Terrain Analysis
    # ================================================================
    R.build_instructions.add('taudem', {
        'description': 'Terrain Analysis Using Digital Elevation Models',
        'config_path_key': 'TAUDEM_INSTALL_PATH',
        'config_exe_key': 'TAUDEM_EXE',
        'default_path_suffix': 'installs/TauDEM/bin',
        'default_exe': 'pitremove',
        'repository': 'https://github.com/dtarb/TauDEM.git',
        # Pin to a tagged release so that upstream HEAD drift cannot break
        # installs. v5.4.0 (Dec 2025) is the most recent tagged release and
        # ships the CMakeLists layout the TAUDEM_BUILD_COMMAND expects.
        'branch': 'v5.4.0',
        'install_dir': 'TauDEM',
        'build_commands': [
            common_env,
            TAUDEM_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/pitremove', 'bin/pitremove.exe'],
            'check_type': 'exists_any'
        },
        'order': 6
    })

def _register_gistool() -> None:
    """Register GIStool build instructions."""
    # ================================================================
    # GIStool - Geospatial Data Extraction
    # ================================================================
    R.build_instructions.add('gistool', {
        'description': 'Geospatial data extraction and processing tool',
        'config_path_key': 'INSTALL_PATH_GISTOOL',
        'config_exe_key': 'EXE_NAME_GISTOOL',
        'default_path_suffix': 'installs/gistool',
        'default_exe': 'extract-gis.sh',
        'repository': 'https://github.com/kasra-keshavarz/gistool.git',
        'branch': None,
        'install_dir': 'gistool',
        'build_commands': [
            r'''
set -e
chmod +x extract-gis.sh
            '''.strip()
        ],
        'verify_install': {
            'file_paths': ['extract-gis.sh'],
            'check_type': 'exists'
        },
        'dependencies': [],
        'test_command': None,
        'order': 7
    })

def _register_datatool() -> None:
    """Register Datatool build instructions."""
    # ================================================================
    # Datatool - Meteorological Data Processing
    # ================================================================
    R.build_instructions.add('datatool', {
        'description': 'Meteorological data extraction and processing tool',
        'config_path_key': 'DATATOOL_PATH',
        'config_exe_key': 'DATATOOL_SCRIPT',
        'default_path_suffix': 'installs/datatool',
        'default_exe': 'extract-dataset.sh',
        'repository': 'https://github.com/kasra-keshavarz/datatool.git',
        'branch': None,
        'install_dir': 'datatool',
        'build_commands': [
            r'''
set -e
chmod +x extract-dataset.sh
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--help',
        'verify_install': {
            'file_paths': ['extract-dataset.sh'],
            'check_type': 'exists'
        },
        'order': 8
    })

def _register_openfews() -> None:
    """Register openFEWS build instructions."""
    # ================================================================
    # openFEWS - Delft-FEWS Flood Early Warning System
    # ================================================================
    R.build_instructions.add('openfews', {
        'description': 'openFEWS - Delft Flood Early Warning System (open-source distribution)',
        'config_path_key': 'OPENFEWS_INSTALL_PATH',
        'config_exe_key': 'OPENFEWS_EXE',
        'default_path_suffix': 'installs/openfews',
        'default_exe': 'bin/fews.sh',
        'repository': None,
        'branch': None,
        'install_dir': 'openfews',
        'build_commands': [
            OPENFEWS_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/fews.sh', 'Modules/symfluence_adapter.xml'],
            'check_type': 'exists'
        },
        'order': 11,
        'optional': True,
    })

def _register_ngiab() -> None:
    """Register NGIAB build instructions."""
    # ================================================================
    # NGIAB - NextGen In A Box
    # ================================================================
    R.build_instructions.add('ngiab', {
        'description': 'NextGen In A Box - Container-based ngen deployment',
        'config_path_key': 'NGIAB_INSTALL_PATH',
        'config_exe_key': 'NGIAB_SCRIPT',
        'default_path_suffix': 'installs/ngiab',
        'default_exe': 'guide.sh',
        'repository': None,
        'branch': 'main',
        'install_dir': 'ngiab',
        'build_commands': [
            r'''
set -e
# Detect HPC vs laptop/workstation and fetch the right NGIAB wrapper repo into ../ngiab
IS_HPC=false
for scheduler in sbatch qsub bsub; do
  if command -v $scheduler >/dev/null 2>&1; then IS_HPC=true; break; fi
done
[ -n "$SLURM_CLUSTER_NAME" ] && IS_HPC=true
[ -n "$PBS_JOBID" ] && IS_HPC=true
[ -n "$SGE_CLUSTER_NAME" ] && IS_HPC=true
[ -d "/scratch" ] && IS_HPC=true

if $IS_HPC; then
  NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
  echo "HPC environment detected; using NGIAB-HPCInfra"
else
  NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
  echo "Non-HPC environment detected; using NGIAB-CloudInfra"
fi

# Clone into a temp directory first, then move contents into the install dir.
# We cannot do `cd .. && rm -rf ngiab` because _build.sh itself lives inside
# the ngiab/ directory.  On HPC parallel filesystems (GPFS, Lustre) removing a
# directory that still has an open file handle (the running script) fails with
# "Directory not empty".
TMPCLONE="$(mktemp -d "${TMPDIR:-/tmp}/ngiab_clone.XXXXXX")"
git clone "$NGIAB_REPO" "$TMPCLONE/ngiab"

# Wipe current contents (except _build.sh which is still running) and copy new
# files in.  Using rsync-style approach: remove old files, copy new ones.
find . -mindepth 1 -maxdepth 1 ! -name '_build.sh' -exec rm -rf {} + 2>/dev/null || true
cp -a "$TMPCLONE/ngiab/." .
rm -rf "$TMPCLONE"

[ -f guide.sh ] && chmod +x guide.sh && bash -n guide.sh || true
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['guide.sh'],
            'check_type': 'exists'
        },
        'order': 10,
    })


def _register_enzyme() -> None:
    """Register Enzyme AD build instructions."""
    # ================================================================
    # Enzyme AD - Automatic Differentiation (used by cFUSE)
    # ================================================================
    R.build_instructions.add('enzyme', {
        'description': 'Enzyme AD - Automatic Differentiation via LLVM',
        'config_path_key': None,
        'config_exe_key': None,
        'default_path_suffix': 'installs/enzyme',
        'default_exe': None,
        'repository': 'https://github.com/EnzymeAD/Enzyme.git',
        'branch': 'main',
        'install_dir': 'enzyme',
        'build_commands': [
            ENZYME_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': [
                'lib/ClangEnzyme.dylib',
                'lib/LLVMEnzyme.so',
            ],
            'check_type': 'exists_any'
        },
        'order': 14,  # Before cfuse (order=15)
        'optional': True,
        'library_only': True,
    })


def _register_infrastructure_tools() -> None:
    """Register infrastructure tool build instructions."""
    common_env = get_common_build_environment()
    _register_sundials(common_env)
    _register_taudem(common_env)
    _register_gistool()
    _register_datatool()
    _register_openfews()
    _register_ngiab()
    _register_enzyme()


def _resolve_registered_build_instructions() -> None:
    """Resolve every registered build-instruction entry, dropping broken ones.

    Model-specific build instructions are *declared*, not discovered: a model
    package registers the dotted path of its build-instructions module when it
    registers itself, via ``model_manifest(build_instructions_module=...)``
    (or ``R.build_instructions.add_lazy(...)``/an eager import in its
    ``register()``).  Plugin discovery calls every ``symfluence.plugins``
    ``register()`` at startup, so by the time the CLI asks for tool
    definitions the registry already holds an entry for every installed model —
    in-tree or external — and there is nothing left to go looking for on disk.

    This replaces a hardcoded list of in-tree ``<model>.build_instructions``
    module paths that were imported by joining a filesystem path relative to
    this file.  That list could not see external plugin packages at all, and it
    hardcoded the location of the models package, which is being extracted into
    its own distribution.

    All this pass does is force each lazy entry to resolve *here*, so that a
    single unresolvable entry is reported and skipped (the behaviour the old
    per-module ``try/except`` gave) rather than propagating out of
    ``BuildInstructionsRegistry.get_all_instructions()`` and taking the whole
    ``symfluence binary`` listing down with it.
    """
    import logging

    logger = logging.getLogger(__name__)

    for tool_name in R.build_instructions.keys():
        try:
            R.build_instructions.get(tool_name)
        except Exception as exc:  # noqa: BLE001 - a broken tool must not hide the rest
            logger.warning(
                "Failed to load build instructions for %s: %s", tool_name, exc
            )
            try:
                R.build_instructions.remove(tool_name)
            except RuntimeError:  # frozen registry — leave the entry in place
                logger.debug("Could not drop unresolvable entry %s", tool_name)


def _recover_build_instructions_from_failed_plugins() -> None:
    """Recover build instructions for packages whose ``register()`` never ran.

    Declaration-based discovery has one gap, and it is the gap that matters
    most: a model package declares ``build_instructions_module`` from inside
    ``register()``, so if the package's ``__init__.py`` raises on import, the
    declaration never happens and the tool disappears from
    ``symfluence binary install`` — precisely when the user most needs it,
    because *building the binary is often what fixes the broken import*.

    Recovery is narrow by construction. It runs only for entry points that
    actually failed during discovery, and for each one it stubs a single
    module: the failing leaf package. Its parents import fine (only the leaf
    ``__init__`` raised), so giving the stub a correct ``__path__`` is enough
    for ``<pkg>.build_instructions`` and any relative import inside it to
    resolve, without executing the broken ``__init__``.

    The predecessor of this function ran the same trick unconditionally for a
    hardcoded list of in-tree models — bypassing every package's ``__init__``
    on every invocation, and unable to see external plugins at all.
    """
    import importlib
    import importlib.util
    import logging
    import sys
    import types
    from pathlib import Path

    from symfluence.core._bootstrap import failed_plugin_entry_points

    logger = logging.getLogger(__name__)

    for name, value in failed_plugin_entry_points():
        module_name = value.split(":", 1)[0]
        target = f"{module_name}.build_instructions"
        if target in sys.modules:
            continue
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        if spec is None or not spec.origin:
            continue

        package_dir = Path(spec.origin).parent
        if not (package_dir / "build_instructions.py").exists():
            continue

        previous = sys.modules.get(module_name)
        stub = types.ModuleType(module_name)
        stub.__path__ = [str(package_dir)]
        stub.__package__ = module_name
        sys.modules[module_name] = stub
        try:
            importlib.import_module(target)
            logger.info(
                "Recovered build instructions for plugin %r whose package failed "
                "to import; its binary can still be built.", name,
            )
        except Exception as exc:  # noqa: BLE001 - recovery is best-effort by definition
            logger.debug("No build-instruction recovery for %r: %s", name, exc)
            sys.modules.pop(target, None)
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous


# Register infrastructure tools on module load
_register_infrastructure_tools()


def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Get all external tool definitions (both infrastructure and model-specific).

    This function maintains backward compatibility with BinaryManager.
    It aggregates:
    1. Infrastructure tools (sundials, taudem, gistool, datatool, ngiab)
    2. Model-specific tools (summa, fuse, mizuroute, etc.)

    Returns:
        Dictionary mapping tool names to their complete configuration including:
        - description: Human-readable description
        - config_path_key: Key in config file for installation path
        - config_exe_key: Key in config file for executable name
        - default_path_suffix: Default relative path for installation
        - default_exe: Default executable/library filename
        - repository: Git repository URL (None for non-git installs)
        - branch: Git branch to checkout (None for default)
        - install_dir: Directory name for installation
        - requires: List of tool dependencies (other tools)
        - build_commands: Shell commands for building
        - dependencies: System dependencies required
        - test_command: Command argument for testing (None to skip)
        - verify_install: Installation verification criteria
        - order: Installation order (lower numbers first)
    """
    # Resolve the model build instructions declared by registered plugins, then
    # make a best-effort recovery for any plugin whose package failed to import
    # (its register() never ran, so it declared nothing).
    # Drain the modules packages declared: importing each fires its
    # @R.build_instructions.add('<tool>') decorator. Declaration-based, so it
    # works identically for in-tree and external packages.
    R.build_instructions.load_modules()
    _recover_build_instructions_from_failed_plugins()
    _resolve_registered_build_instructions()

    # Return all aggregated instructions
    return BuildInstructionsRegistry.get_all_instructions()


if __name__ == "__main__":
    """Test the configuration definitions."""
    tools = get_external_tools_definitions()
    print(f"Loaded {len(tools)} external tool definitions:")
    for name, info in sorted(tools.items(), key=lambda x: x[1].get('order', 99)):
        print(f"   {info.get('order', '?'):2}. {name:12s} - {info['description'][:60]}")
