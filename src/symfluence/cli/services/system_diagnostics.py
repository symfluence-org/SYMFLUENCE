# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
System diagnostics service for SYMFLUENCE.

Provides system health checks, toolchain information, and library detection.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..console import Console
from .base import BaseService


class SystemDiagnostics(BaseService):
    """
    Service for running system diagnostics.

    Handles:
    - Binary status checks
    - Toolchain metadata reading
    - System library detection
    - npm binary detection
    """

    def __init__(
        self,
        external_tools: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the SystemDiagnostics.

        Args:
            external_tools: Dictionary of tool definitions. If None, loads on demand.
            console: Console instance for output.
        """
        super().__init__(console=console)
        self._external_tools = external_tools

    @property
    def external_tools(self) -> Dict[str, Any]:
        """Lazy load external tools definitions."""
        if self._external_tools is None:
            from ..external_tools_config import get_external_tools_definitions
            self._external_tools = get_external_tools_definitions()
        return self._external_tools

    def run_diagnostics(self) -> bool:
        """
        Run system diagnostics: check binaries, toolchain, and system libraries.

        Returns:
            True if diagnostics completed successfully.
        """
        self._console.rule()

        # Check binaries
        self._console.newline()
        self._console.info("Checking binaries...")
        self._console.rule()

        config = self._load_config()
        symfluence_data = str(self._get_data_dir(config))

        npm_bin_dir = self.detect_npm_binaries()

        if npm_bin_dir:
            self._console.info(f"Detected npm-installed binaries: {npm_bin_dir}")
        if symfluence_data:
            self._console.info(f"Checking source installs in: {symfluence_data}")

        found_binaries = 0
        total_binaries = 0

        # Build table rows for binary status
        binary_rows = self._check_binary_status(symfluence_data, npm_bin_dir)
        for row in binary_rows:
            total_binaries += 1
            if "[green]OK[/green]" in row[1]:
                found_binaries += 1

        self._console.table(
            columns=["Tool", "Status", "Location"],
            rows=binary_rows,
            title="Binary Status",
        )

        # Check toolchain metadata
        self._console.newline()
        self._console.info("Toolchain metadata...")
        self._console.rule()

        toolchain_found = self._check_toolchain_metadata(symfluence_data, npm_bin_dir)

        # Check system libraries
        self._console.newline()
        self._console.info("System libraries...")
        self._console.rule()

        lib_rows, found_libs, total_libs = self._check_system_libraries()

        self._console.table(
            columns=["Library", "Status", "Location"],
            rows=lib_rows,
            title="System Libraries",
        )

        # Geospatial Python library compatibility
        self._console.newline()
        self._console.info("Python geospatial libraries...")
        self._console.rule()

        geo_ok = self._check_geospatial_compatibility()

        # Summary
        self._console.newline()
        self._console.rule()
        self._console.info("Summary:")
        self._console.indent(f"Binaries: {found_binaries}/{total_binaries} found")
        tc_status = "[green]Found[/green]" if toolchain_found else "[red]Not found[/red]"
        self._console.indent(f"Toolchain metadata: {tc_status}")
        self._console.indent(f"System libraries: {found_libs}/{total_libs} found")
        geo_status = "[green]OK[/green]" if geo_ok else "[yellow]Issues detected[/yellow]"
        self._console.indent(f"Geospatial libraries: {geo_status}")

        if found_binaries == total_binaries and toolchain_found and found_libs >= 3 and geo_ok:
            self._console.newline()
            self._console.success("System is ready for SYMFLUENCE!")
        elif found_binaries == 0:
            self._console.newline()
            self._console.warning("No binaries found. Install with:")
            self._console.indent("npm install -g symfluence (for pre-built binaries)")
            self._console.indent("symfluence binary install (to build from source)")
        else:
            self._console.newline()
            self._console.warning("Some components missing. Review output above.")

        self._console.rule()
        return True

    def get_tools_info(self) -> bool:
        """
        Display installed tools information from toolchain metadata.

        Returns:
            True if tools info was displayed, False if no metadata found.
        """
        symfluence_data = os.getenv("SYMFLUENCE_DATA")
        npm_bin_dir = self.detect_npm_binaries()

        toolchain_locations = []
        if symfluence_data:
            toolchain_locations.append(
                Path(symfluence_data) / "installs" / "toolchain.json"
            )
        if npm_bin_dir:
            toolchain_locations.append(npm_bin_dir.parent / "toolchain.json")

        toolchain_path = None
        for path in toolchain_locations:
            if path.exists():
                toolchain_path = path
                break

        if not toolchain_path:
            self._console.warning("No binaries installed yet.")
            self._console.newline()
            self._console.info("Model binaries (SUMMA, mizuRoute, FUSE, etc.) must be installed")
            self._console.info("before 'symfluence binary info' can report on them.")
            self._console.newline()
            self._console.indent("Install binaries with one of:")
            self._console.indent("  npm install -g symfluence        # pre-built binaries")
            self._console.indent("  symfluence binary install         # build from source")
            return False

        return self._read_toolchain_metadata(toolchain_path)

    def detect_npm_binaries(self) -> Optional[Path]:
        """
        Detect if SYMFLUENCE binaries are installed via npm.

        Returns:
            Path to npm-installed binaries, or None if not found.
        """
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                npm_root = Path(result.stdout.strip())
                npm_bin_dir = npm_root / "symfluence" / "dist" / "bin"

                if npm_bin_dir.exists() and npm_bin_dir.is_dir():
                    return npm_bin_dir

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
            pass

        return None

    def _check_binary_status(
        self, symfluence_data: str, npm_bin_dir: Optional[Path]
    ) -> List[List[str]]:
        """
        Check status of all binaries.

        Args:
            symfluence_data: Path to SYMFLUENCE data directory.
            npm_bin_dir: Path to npm binary directory if available.

        Returns:
            List of table rows [name, status, location].
        """
        binary_rows = []

        for name, tool_info in self.external_tools.items():
            # Skip hidden tools (e.g. watflood) and library-only tools (e.g. sundials)
            if tool_info.get('hidden', False) or tool_info.get('library_only', False):
                continue

            found = False
            location = None

            # 1. Check in SYMFLUENCE_DATA (installed from source)
            if symfluence_data:
                rel_path_suffix = tool_info.get("default_path_suffix", "")
                exe_name = tool_info.get("default_exe", "")

                full_path = Path(symfluence_data) / rel_path_suffix

                if name in ("taudem",):
                    if full_path.exists() and full_path.is_dir():
                        found = True
                        location = full_path
                elif exe_name:
                    if exe_name.endswith(".so") and sys.platform == "darwin":
                        exe_name_mac = exe_name.replace(".so", ".dylib")
                        candidates = [exe_name, exe_name_mac]
                    else:
                        candidates = [exe_name]

                    for cand in candidates:
                        exe_path = full_path / cand
                        if exe_path.exists():
                            found = True
                            location = exe_path
                            break

                        exe_path_no_ext = full_path / cand.replace(".exe", "")
                        if exe_path_no_ext.exists():
                            found = True
                            location = exe_path_no_ext
                            break

            # 2. Check npm installation as fallback
            if not found and npm_bin_dir:
                npm_path = npm_bin_dir / name
                if npm_path.exists():
                    found = True
                    location = npm_path
                else:
                    exe_name = tool_info.get("default_exe", "")
                    if exe_name:
                        for candidate in [exe_name, exe_name.replace(".exe", "")]:
                            npm_exe_path = npm_bin_dir / candidate
                            if npm_exe_path.exists():
                                found = True
                                location = npm_exe_path
                                break

            # Skip optional tools that aren't installed — avoids a wall of MISSING
            if tool_info.get('optional', False) and not found:
                continue

            status = "[green]OK[/green]" if found else "[red]MISSING[/red]"
            loc_str = str(location) if location else "-"
            binary_rows.append([name, status, loc_str])

        return binary_rows

    def _check_toolchain_metadata(
        self, symfluence_data: str, npm_bin_dir: Optional[Path]
    ) -> bool:
        """
        Check for and display toolchain metadata.

        Args:
            symfluence_data: Path to SYMFLUENCE data directory.
            npm_bin_dir: Path to npm binary directory if available.

        Returns:
            True if toolchain metadata found, False otherwise.
        """
        toolchain_locations = []
        if symfluence_data:
            toolchain_locations.append(
                Path(symfluence_data) / "installs" / "toolchain.json"
            )
        if npm_bin_dir:
            toolchain_locations.append(npm_bin_dir.parent / "toolchain.json")

        for toolchain_path in toolchain_locations:
            if toolchain_path.exists():
                try:
                    with open(toolchain_path, encoding="utf-8") as f:
                        toolchain = json.load(f)

                    platform = toolchain.get("platform", "unknown")
                    build_date = toolchain.get("build_date", "unknown")
                    fortran = toolchain.get("compilers", {}).get("fortran", "unknown")

                    self._console.success(f"Found: {toolchain_path}")
                    self._console.indent(f"Platform: {platform}")
                    self._console.indent(f"Build date: {build_date}")
                    self._console.indent(f"Fortran: {fortran}")
                    return True
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
                    self._console.warning(f"Error reading {toolchain_path}: {e}")

        self._console.warning("No toolchain metadata found — run 'symfluence binary install' first")
        return False

    def _check_geospatial_compatibility(self) -> bool:
        """Check that rasterio, fiona, and GDAL import without compression-library conflicts."""
        all_ok = True

        checks = [
            ("rasterio", "rasterio"),
            ("fiona", "fiona"),
            ("geopandas", "geopandas"),
        ]

        for display_name, module_name in checks:
            try:
                mod = __import__(module_name)
                version = getattr(mod, '__version__', getattr(mod, '__gdal_version__', '?'))
                self._console.indent(f"{display_name} {version}: [green]OK[/green]")
            except ImportError:
                self._console.indent(f"{display_name}: [yellow]not installed[/yellow]")
            except Exception as e:  # noqa: BLE001
                all_ok = False
                err_str = str(e)
                self._console.indent(f"{display_name}: [red]FAILED[/red]")
                if 'ZSTD' in err_str or 'zstd' in err_str:
                    self._console.indent(
                        f"  ZSTD mismatch: {display_name} was built against a "
                        "different ZSTD than the one installed. Reinstall with:"
                    )
                    self._console.indent(f"    pip install --force-reinstall --no-binary :all: {module_name}")
                elif 'BLOSC' in err_str or 'blosc' in err_str:
                    self._console.indent(
                        f"  BLOSC mismatch: {display_name} was built against a "
                        "different BLOSC than the one installed. Reinstall with:"
                    )
                    self._console.indent(f"    pip install --force-reinstall --no-binary :all: {module_name}")
                else:
                    self._console.indent(f"  Error: {err_str}")

        # Check GDAL Python bindings (optional)
        try:
            from osgeo import gdal
            gdal_version = gdal.__version__ if hasattr(gdal, '__version__') else gdal.VersionInfo()
            self._console.indent(f"GDAL bindings {gdal_version}: [green]OK[/green]")
        except ImportError:
            self._console.indent("GDAL bindings: [dim]not installed (optional)[/dim]")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            self._console.indent(f"GDAL bindings: [red]FAILED[/red] — {e}")

        # Cross-check: try a rasterio open to exercise the compression codecs
        if all_ok:
            try:
                import rasterio
                drivers = rasterio.drivers.raster_driver_extensions()
                if 'tif' in drivers:
                    self._console.indent("rasterio GeoTIFF driver: [green]OK[/green]")
                else:
                    self._console.indent("rasterio GeoTIFF driver: [yellow]not available[/yellow]")
                    all_ok = False
            except Exception as e:  # noqa: BLE001
                self._console.indent(f"rasterio driver check: [red]FAILED[/red] — {e}")
                all_ok = False

        if not all_ok:
            self._console.newline()
            self._console.warning(
                "Geospatial library issues detected. This typically happens when "
                "mixing conda and pip installs, or when system GDAL differs from "
                "the version rasterio/fiona were built against."
            )
            self._console.indent("Recommended fix: reinstall the geospatial stack from a single source:")
            self._console.indent("  conda install -c conda-forge rasterio fiona geopandas gdal")
            self._console.indent("  OR: pip install --force-reinstall rasterio fiona")

        return all_ok

    def _check_system_libraries(self) -> tuple:
        """
        Check system library availability using the canonical dependency registry.

        Returns:
            Tuple of (rows, found_count, total_count).
        """
        try:
            from .system_deps import SystemDepsRegistry

            registry = SystemDepsRegistry()
            results = registry.check_all_deps()

            lib_rows = []
            found_libs = 0
            for r in results:
                if r.found:
                    status = "[green]OK[/green]"
                    if not r.version_ok:
                        status = "[yellow]OLD[/yellow]"
                    version_str = f" ({r.version})" if r.version else ""
                    loc_str = f"{r.path}{version_str}" if r.path else "-"
                    found_libs += 1
                else:
                    status = "[red]MISSING[/red]"
                    loc_str = r.install_command or "-"
                lib_rows.append([r.display_name, status, loc_str])

            return lib_rows, found_libs, len(results)

        except (ImportError, AttributeError, OSError, TypeError, ValueError, RuntimeError):
            # Fall back to legacy hardcoded check if registry fails
            return self._check_system_libraries_legacy()

    def _check_system_libraries_legacy(self) -> tuple:
        """Legacy fallback: check a hardcoded set of system tools."""
        system_tools = {
            "nc-config": "NetCDF",
            "nf-config": "NetCDF-Fortran",
            "h5cc": "HDF5",
            "gdal-config": "GDAL",
            "mpirun": "MPI",
        }

        lib_rows = []
        found_libs = 0
        for tool, name in system_tools.items():
            location = shutil.which(tool)
            if location:
                lib_rows.append([name, "[green]OK[/green]", location])
                found_libs += 1
            else:
                lib_rows.append([name, "[red]MISSING[/red]", "-"])

        return lib_rows, found_libs, len(system_tools)

    def _read_toolchain_metadata(self, toolchain_path: Path) -> bool:
        """
        Read and display toolchain metadata from file.

        Args:
            toolchain_path: Path to toolchain.json file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(toolchain_path, encoding="utf-8") as f:
                toolchain = json.load(f)

            self._console.rule()
            self._console.info(f"Platform: {toolchain.get('platform', 'unknown')}")
            self._console.info(f"Build Date: {toolchain.get('build_date', 'unknown')}")
            self._console.info(f"Toolchain file: {toolchain_path}")

            # Compilers
            if "compilers" in toolchain:
                self._console.newline()
                self._console.info("Compilers:")
                self._console.rule()
                compilers = toolchain["compilers"]
                compiler_rows = [
                    [key.capitalize(), value] for key, value in compilers.items()
                ]
                self._console.table(
                    columns=["Compiler", "Version"], rows=compiler_rows
                )

            # Libraries
            if "libraries" in toolchain:
                self._console.newline()
                self._console.info("Libraries:")
                self._console.rule()
                libraries = toolchain["libraries"]
                lib_rows = [[key.capitalize(), value] for key, value in libraries.items()]
                self._console.table(columns=["Library", "Version"], rows=lib_rows)

            # Tools
            if "tools" in toolchain:
                self._console.newline()
                self._console.info("Installed Tools:")
                self._console.rule()
                for tool_name, tool_info in toolchain["tools"].items():
                    self._console.newline()
                    self._console.info(f"  {tool_name.upper()}:")
                    if "commit" in tool_info:
                        commit_short = (
                            tool_info.get("commit", "")[:8]
                            if len(tool_info.get("commit", "")) > 8
                            else tool_info.get("commit", "")
                        )
                        self._console.indent(f"Commit: {commit_short}", level=2)
                    if "branch" in tool_info:
                        self._console.indent(
                            f"Branch: {tool_info.get('branch', '')}", level=2
                        )
                    if "executable" in tool_info:
                        self._console.indent(
                            f"Executable: {tool_info.get('executable', '')}", level=2
                        )

            self._console.newline()
            self._console.rule()
            return True

        except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            self._console.error(f"Error reading toolchain file: {e}")
            return False
