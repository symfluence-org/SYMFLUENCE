#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
SYMFLUENCE Node-Agnostic Local Scratch Manager (Fixed Staging)

This module provides functionality to use local scratch storage on HPC systems
to reduce IOPS on shared filesystems during parallel optimization, with support
for multi-node jobs.

Key improvements in this version:
- Fixed staging to properly find outputs in process-specific directories
- Results staged back mirror the non-scratch structure exactly
- Handles parallel_proc_XX subdirectories correctly
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir


class LocalScratchManager(ConfigMixin):
    """
    Manages local scratch space for optimization on HPC systems with multi-node support.

    This class handles the complete lifecycle of using local scratch:
    - Detection of HPC/SLURM environment
    - Node-aware directory structure creation
    - Data copying to each node's local scratch
    - Path management per node
    - Results staging back to permanent storage from each node
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger,
                 project_dir: Path, algorithm_name: str, mpi_rank: Optional[int] = None):
        """
        Initialize the local scratch manager.

        Args:
            config: SYMFLUENCE configuration dictionary
            logger: Logger instance
            project_dir: Main project directory path (ORIGINAL, not scratch)
            algorithm_name: Name of optimization algorithm (for directory naming)
            mpi_rank: MPI rank of this process (None for serial execution)
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.project_dir = project_dir  # This is the ORIGINAL project dir
        self.algorithm_name = algorithm_name
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name, dict_key='DOMAIN_NAME'
        )
        self.experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default='run_1', dict_key='EXPERIMENT_ID'
        )
        self.mpi_rank = mpi_rank if mpi_rank is not None else 0

        # Get node information
        self.node_name = socket.gethostname()
        self.node_id = self._get_node_id()

        # Check if local scratch should be used
        self.use_scratch = self._should_use_scratch()

        # Initialize paths (will be set during setup)
        self.scratch_root = None
        self.scratch_data_dir = None
        self.scratch_project_dir = None
        self.original_data_dir = None

        # Store original project_dir for staging back
        self.original_project_dir = project_dir

        if self.use_scratch:
            self.logger.info(f"Local scratch mode ENABLED for rank {self.mpi_rank} on node {self.node_name}")
            self._initialize_scratch_paths()
        else:
            self.logger.info(f"Local scratch mode DISABLED for rank {self.mpi_rank} - using standard filesystem")

    def _get_node_id(self) -> str:
        """
        Get a unique identifier for the current node.

        Returns:
            String identifier for the node
        """
        # Try to get SLURM node ID first
        slurm_nodeid = os.environ.get('SLURM_NODEID')
        if slurm_nodeid:
            return f"node{slurm_nodeid}"

        # Otherwise use hostname hash for uniqueness
        node_hash = hashlib.md5(self.node_name.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"node_{node_hash}"

    def _should_use_scratch(self) -> bool:
        """
        Determine if local scratch should be used.

        Returns:
            True if scratch should be used, False otherwise
        """
        # Check config flag
        use_scratch_config = self._get_config_value(lambda: self.config.system.use_local_scratch, default=False, dict_key='USE_LOCAL_SCRATCH')

        if not use_scratch_config:
            return False

        # Check if we're in a SLURM environment
        slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')

        if not slurm_tmpdir:
            self.logger.warning(
                f"Rank {self.mpi_rank}: USE_LOCAL_SCRATCH is True but SLURM_TMPDIR not found. "
                "Falling back to standard filesystem."
            )
            return False

        try:
            Path(slurm_tmpdir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.warning(
                f"Rank {self.mpi_rank}: USE_LOCAL_SCRATCH is True but SLURM_TMPDIR ({slurm_tmpdir}) "
                f"could not be created: {e}. Falling back to standard filesystem."
            )
            return False

        # Multi-node support: Log node configuration but don't disable
        num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', '1')
        try:
            num_nodes = int(num_nodes)
            if num_nodes > 1:
                self.logger.debug(
                    f"Rank {self.mpi_rank}: Job spans {num_nodes} nodes. "
                    f"Using local scratch on node {self.node_name}"
                )
        except ValueError:
            pass

        self.logger.debug(
            f"Rank {self.mpi_rank}: SLURM environment detected: "
            f"TMPDIR={slurm_tmpdir}, JOB_ID={slurm_job_id}, NODE={self.node_name}"
        )
        return True

    def _initialize_scratch_paths(self) -> None:
        """Initialize scratch directory paths with node awareness."""
        self.scratch_root = Path(os.environ.get('SLURM_TMPDIR'))

        # Create scratch data directory with rank-specific subdirectory to avoid conflicts
        # when multiple ranks are on the same node
        self.scratch_data_dir = self.scratch_root / "conf_data" / f"rank_{self.mpi_rank}"
        self.scratch_project_dir = self.scratch_data_dir / f"domain_{self.domain_name}"

        # Store original for reference
        self.original_data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))

        self.logger.debug(f"Rank {self.mpi_rank} on {self.node_name}:")
        self.logger.debug(f"  Scratch root: {self.scratch_root}")
        self.logger.debug(f"  Scratch data dir: {self.scratch_data_dir}")
        self.logger.debug(f"  Scratch project dir: {self.scratch_project_dir}")
        self.logger.debug(f"  Original project dir (for staging): {self.original_project_dir}")

    def _needs_routing(self) -> bool:
        """
        Determine if mizuRoute routing is needed based on configuration.

        Returns:
            True if routing is needed, False otherwise
        """
        # Check routing delineation setting
        routing_delineation = self._get_config_value(lambda: self.config.domain.delineation.routing, default='lumped', dict_key='ROUTING_DELINEATION').lower()

        # Skip routing for lumped or none
        if routing_delineation in ['lumped', 'none', 'off', 'false']:
            return False

        # Check domain definition method
        domain_method = self._get_config_value(lambda: self.config.domain.definition_method, default='lumped', dict_key='DOMAIN_DEFINITION_METHOD').lower()

        # If domain is point or lumped, typically no routing needed
        if domain_method in ['point', 'lumped']:
            # Unless specifically using river_network routing
            if routing_delineation == 'river_network':
                return True
            return False

        # For distributed domains, routing is typically needed
        return True

    def setup_scratch_space(self) -> bool:
        """
        Setup complete scratch space with all necessary data.
        Each MPI rank sets up its own scratch space on its local node.

        This method:
        1. Creates directory structure on local node
        2. Copies settings files to local scratch
        3. Copies forcing data to local scratch
        4. Copies observation data to local scratch
        5. Updates file paths in configuration files

        Returns:
            True if setup successful, False otherwise
        """
        if not self.use_scratch:
            return True

        try:
            self.logger.debug(f"{'='*60}")
            self.logger.debug(f"Rank {self.mpi_rank}: Setting up local scratch on {self.node_name}")
            self.logger.debug(f"{'='*60}")

            # Check if we've already set up scratch for this rank
            # (useful when multiple workers share the same rank on a node)
            setup_marker = self.scratch_project_dir / ".scratch_setup_complete"
            if setup_marker.exists():
                self.logger.debug(
                    f"Rank {self.mpi_rank}: Scratch already set up on {self.node_name}, skipping"
                )
                return True

            # Create directory structure
            self._create_scratch_directories()

            # Copy data to scratch
            self._copy_settings_to_scratch()
            self._copy_forcing_to_scratch()
            self._copy_observations_to_scratch()

            # Only copy mizuRoute settings if routing is needed
            if self._needs_routing():
                self._copy_mizuroute_settings_to_scratch()
            else:
                self.logger.debug(f"Rank {self.mpi_rank}: Routing not needed - skipping mizuRoute settings copy")

            # Update configuration files with scratch paths
            self._update_file_paths_for_scratch()

            # Mark setup as complete
            setup_marker.touch()

            self.logger.info(f"Rank {self.mpi_rank}: Scratch setup completed successfully on {self.node_name}")
            return True

        except (OSError, IOError, PermissionError) as e:
            self.logger.error(f"Rank {self.mpi_rank}: Error setting up scratch space on {self.node_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fall back to standard filesystem
            self.use_scratch = False
            return False

    def _create_scratch_directories(self) -> None:
        """Create necessary directory structure in scratch space."""
        self.logger.debug(f"Rank {self.mpi_rank}: Creating scratch directories on {self.node_name}...")

        # Create main directories
        directories = [
            self.scratch_data_dir,
            self.scratch_project_dir,
            self.scratch_project_dir / "settings" / "SUMMA",
            self.scratch_project_dir / "settings" / "mizuRoute",
            resolve_data_subdir(self.scratch_project_dir, 'forcing') / "SUMMA_input",
            resolve_data_subdir(self.scratch_project_dir, 'observations') / "streamflow" / "preprocessed",
            self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "SUMMA",
            self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "mizuRoute",
            self.scratch_project_dir / "optimization",
            self.scratch_project_dir / f"_workLog_domain_{self.domain_name}",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"  Created: {directory}")

    def _copy_settings_to_scratch(self) -> None:
        """Copy SUMMA settings to scratch."""
        self.logger.debug(f"Rank {self.mpi_rank}: Copying SUMMA settings to scratch on {self.node_name}...")

        source_settings = self.original_project_dir / "settings" / "SUMMA"
        dest_settings = self.scratch_project_dir / "settings" / "SUMMA"

        # Copy all settings files
        self._rsync_directory(source_settings, dest_settings)

    def _copy_forcing_to_scratch(self) -> None:
        """Copy forcing data to scratch."""
        self.logger.debug(f"Rank {self.mpi_rank}: Copying forcing data to scratch on {self.node_name}...")

        source_forcing = resolve_data_subdir(self.original_project_dir, 'forcing') / "SUMMA_input"
        dest_forcing = resolve_data_subdir(self.scratch_project_dir, 'forcing') / "SUMMA_input"

        # Copy forcing data
        self._rsync_directory(source_forcing, dest_forcing)

    def _copy_observations_to_scratch(self) -> None:
        """Copy observation data to scratch."""
        self.logger.debug(f"Rank {self.mpi_rank}: Copying observation data to scratch on {self.node_name}...")

        source_obs = resolve_data_subdir(self.original_project_dir, 'observations') / "streamflow" / "preprocessed"
        dest_obs = resolve_data_subdir(self.scratch_project_dir, 'observations') / "streamflow" / "preprocessed"

        # Only copy if source exists
        if source_obs.exists():
            self._rsync_directory(source_obs, dest_obs)
        else:
            self.logger.warning(f"Rank {self.mpi_rank}: No observation data found to copy")

    def _copy_mizuroute_settings_to_scratch(self) -> None:
        """Copy mizuRoute settings to scratch."""
        self.logger.debug(f"Rank {self.mpi_rank}: Copying mizuRoute settings to scratch on {self.node_name}...")

        source_mizu = self.original_project_dir / "settings" / "mizuRoute"
        dest_mizu = self.scratch_project_dir / "settings" / "mizuRoute"

        # Only copy if source exists
        if source_mizu.exists():
            self._rsync_directory(source_mizu, dest_mizu)
        else:
            self.logger.warning(f"Rank {self.mpi_rank}: No mizuRoute settings found to copy")

    def _rsync_directory(self, source: Path, dest: Path) -> None:
        """
        Use rsync to efficiently copy directory contents.
        Falls back to shutil if rsync is not available.

        Args:
            source: Source directory
            dest: Destination directory
        """
        if not source.exists():
            self.logger.warning(f"Rank {self.mpi_rank}: Source directory does not exist: {source}")
            return

        try:
            # Ensure destination exists
            dest.mkdir(parents=True, exist_ok=True)

            if os.name != 'nt':
                # Use rsync for efficient copying (handles existing files better)
                subprocess.run(
                    ['rsync', '-a', '--delete', f'{source}/', f'{dest}/'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.logger.debug(f"  Copied: {source} -> {dest}")
            else:
                # Windows: no rsync, use shutil directly
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
                self.logger.debug(f"  Copied: {source} -> {dest}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.warning(f"Rank {self.mpi_rank}: rsync failed, falling back to shutil.copytree: {e}")
            # Fallback to shutil if rsync not available
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)

    def _update_file_paths_for_scratch(self) -> None:
        """Update file paths in configuration files to point to scratch."""
        self.logger.debug(f"Rank {self.mpi_rank}: Updating file paths for scratch space...")

        # Update fileManager.txt if it exists
        file_manager = self.scratch_project_dir / "settings" / "SUMMA" / "fileManager.txt"
        if file_manager.exists():
            self._update_file_manager(file_manager)

        # Update mizuRoute control file if it exists
        mizu_control = self.scratch_project_dir / "settings" / "mizuRoute" / "mizuroute.control"
        if mizu_control.exists():
            self._update_mizuroute_control(mizu_control)

    def _update_file_manager(self, file_manager_path: Path) -> None:
        """
        Update paths in SUMMA fileManager.txt.

        Args:
            file_manager_path: Path to fileManager.txt
        """
        self.logger.debug(f"Rank {self.mpi_rank}: Updating fileManager: {file_manager_path}")

        # Read file
        with open(file_manager_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Prepare paths
        forcing_path = str(resolve_data_subdir(self.scratch_project_dir, 'forcing') / "SUMMA_input") + "/"

        # Update lines
        updated_lines = []
        for line in lines:
            if line.strip().startswith('forcingPath'):
                # Replace forcing path
                updated_lines.append(f"forcingPath          '{forcing_path}'\n")
            else:
                updated_lines.append(line)

        # Write back
        with open(file_manager_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

        self.logger.debug(f"  Rank {self.mpi_rank}: fileManager.txt updated")

    def _update_mizuroute_control(self, control_path: Path) -> None:
        """
        Update paths in mizuRoute control file.

        Args:
            control_path: Path to mizuroute.control
        """
        self.logger.debug(f"Rank {self.mpi_rank}: Updating mizuRoute control: {control_path}")

        # This will be updated by the optimizer when it creates process-specific
        # directories, so we just ensure the file exists
        self.logger.debug(f"  Rank {self.mpi_rank}: mizuroute.control will be updated by optimizer")

    def stage_results_back(self) -> None:
        """
        Stage optimization results back to permanent storage.

        FIXED: This now stages results to mirror the exact same structure
        as if scratch was not used. Results are merged from all process-specific
        directories back to the original locations.

        This copies:
        - Simulation results (SUMMA, mizuRoute) from all parallel_proc_XX dirs
        - Optimization output files
        - Log files
        - Best parameter files
        """
        if not self.use_scratch:
            return

        self.logger.debug(f"{'='*60}")
        self.logger.debug(f"Rank {self.mpi_rank}: Staging results back from {self.node_name}")
        self.logger.debug(f"  Source (scratch): {self.scratch_project_dir}")
        self.logger.debug(f"  Destination (original): {self.original_project_dir}")
        self.logger.debug(f"{'='*60}")

        try:
            # Stage simulation results (the main outputs)
            self._stage_simulation_results_fixed()

            # Stage optimization outputs (parameter files, history, etc.)
            self._stage_optimization_results_fixed()

            # Stage work logs if they exist
            self._stage_work_logs_fixed()

            self.logger.info(f"Rank {self.mpi_rank}: Results staged successfully to: {self.original_project_dir}")
            self.logger.debug(f"{'='*60}")

        except (OSError, IOError, PermissionError) as e:
            self.logger.error(f"Rank {self.mpi_rank}: Error staging results back: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Don't raise - we want to keep the results in scratch for manual recovery
            self.logger.error(
                f"Rank {self.mpi_rank}: Results remain in scratch at: "
                f"{self.scratch_project_dir}"
            )

    def _stage_simulation_results_fixed(self) -> None:
        """
        Stage simulation outputs back to permanent storage.

        FIXED: This handles the parallel_proc_XX directory structure properly.
        """
        scratch_sim_base = self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}"
        dest_sim_base = self.original_project_dir / "simulations" / f"run_{self.algorithm_name}"

        if not scratch_sim_base.exists():
            self.logger.warning(f"Rank {self.mpi_rank}: No simulation directory found at {scratch_sim_base}")
            return

        # Ensure destination exists
        dest_sim_base.mkdir(parents=True, exist_ok=True)

        # Find all items in the scratch simulation directory
        items_staged = 0

        for item in scratch_sim_base.iterdir():
            dest_item = dest_sim_base / item.name

            if item.is_dir():
                # Check if this is a parallel_proc_XX directory
                if item.name.startswith("parallel_proc_"):
                    # Stage the entire parallel_proc directory
                    self.logger.debug(f"Rank {self.mpi_rank}: Staging {item.name}")
                    self._rsync_directory(item, dest_item)
                    items_staged += 1

                elif item.name in ["SUMMA", "mizuRoute", "settings", "logs"]:
                    # Stage standard directories
                    self.logger.debug(f"Rank {self.mpi_rank}: Staging {item.name}")
                    self._rsync_directory(item, dest_item)
                    items_staged += 1

                else:
                    # Stage any other directories
                    self.logger.debug(f"Rank {self.mpi_rank}: Staging {item.name}")
                    self._rsync_directory(item, dest_item)
                    items_staged += 1

            elif item.is_file():
                # Copy individual files
                shutil.copy2(item, dest_item)
                items_staged += 1

        self.logger.info(f"Rank {self.mpi_rank}: Staged {items_staged} items from simulations directory")

    def _stage_optimization_results_fixed(self) -> None:
        """
        Stage optimization-specific results (parameter files, history, etc.).

        FIXED: Stages to the original optimization directory structure.
        """
        scratch_opt = self.scratch_project_dir / "optimization"
        dest_opt = self.original_project_dir / "optimization"

        if scratch_opt.exists() and any(scratch_opt.iterdir()):
            self.logger.debug(f"Rank {self.mpi_rank}: Staging optimization results")

            # Create destination
            dest_opt.mkdir(parents=True, exist_ok=True)

            # Stage each experiment directory
            for item in scratch_opt.iterdir():
                dest_item = dest_opt / item.name

                if item.is_dir():
                    self._rsync_directory(item, dest_item)
                elif item.is_file():
                    shutil.copy2(item, dest_item)

            self.logger.debug(f"Rank {self.mpi_rank}: Optimization results staged to {dest_opt}")
        else:
            self.logger.debug(f"Rank {self.mpi_rank}: No optimization results to stage")

    def _stage_work_logs_fixed(self) -> None:
        """
        Stage work logs if they exist.

        FIXED: Stages to the original work log location.
        """
        work_log_dir = self.scratch_project_dir / f"_workLog_domain_{self.domain_name}"
        dest_log = self.original_project_dir / f"_workLog_domain_{self.domain_name}"

        if work_log_dir.exists() and any(work_log_dir.iterdir()):
            self.logger.debug(f"Rank {self.mpi_rank}: Staging work logs")
            self._rsync_directory(work_log_dir, dest_log)
        else:
            self.logger.debug(f"Rank {self.mpi_rank}: No work logs to stage")

    def get_effective_data_dir(self) -> Path:
        """
        Get the effective data directory (scratch or original).

        Returns:
            Path to use for SYMFLUENCE_DATA_DIR
        """
        if self.use_scratch and self.scratch_data_dir:
            return self.scratch_data_dir
        else:
            return self.original_data_dir if self.original_data_dir else Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))

    def get_effective_project_dir(self) -> Path:
        """
        Get the effective project directory (scratch or original).

        Returns:
            Path to use for project operations
        """
        if self.use_scratch and self.scratch_project_dir:
            return self.scratch_project_dir
        else:
            return self.original_project_dir

    def get_original_project_dir(self) -> Path:
        """
        Get the original project directory (for staging back).

        Returns:
            Path to the original (non-scratch) project directory
        """
        return self.original_project_dir

    def cleanup_scratch(self) -> None:
        """
        Clean up scratch space (optional - SLURM usually handles this).

        This is mainly for manual cleanup if needed.
        """
        if not self.use_scratch:
            return

        self.logger.debug(f"Rank {self.mpi_rank}: Scratch space will be automatically cleaned by SLURM")
        # SLURM_TMPDIR is automatically cleaned up by SLURM after job completion
        # So we don't need to do anything here
