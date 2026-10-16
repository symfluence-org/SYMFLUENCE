# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coupled Groundwater Worker

Executes a land surface model + MODFLOW coupled calibration pipeline.
The land surface model is determined by LAND_SURFACE_MODEL config key
and its worker is loaded dynamically from the OptimizerRegistry.

When dCoupler is available, uses CouplingGraphBuilder to construct and
execute the coupling graph. Otherwise falls back to sequential coupling:
1. Run land surface model → extract recharge
2. Write MODFLOW recharge time-series → run MODFLOW
3. Combine surface runoff + drain discharge → total streamflow

Config keys:
    LAND_SURFACE_MODEL: Land surface model name (SUMMA, CLM, MESH, etc.)
    GROUNDWATER_MODEL: Must be MODFLOW
    COUPLING_MODE: 'dcoupler' or 'sequential' (default: auto-detect)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.registries import R

logger = logging.getLogger(__name__)

# Map land surface model names to their recharge output variables
# and the SUMMA-specific variable for extraction
RECHARGE_VARIABLES = {
    'SUMMA': 'scalarSoilDrainage',
    'CLM': 'QCHARGE',
    'MESH': 'DRAINAGE',
}

# Map land surface model names to their surface runoff variables
# (fast-flow component that bypasses the groundwater model)
SURFACE_RUNOFF_VARIABLES = {
    'SUMMA': 'scalarSurfaceRunoff',
    'CLM': 'QOVER',
    'MESH': 'RUNOFF',
}


@R.workers.add('COUPLED_GW')
class CoupledGWWorker(BaseWorker):
    """Worker for coupled land-surface + MODFLOW calibration.

    Dynamically loads the land surface model's worker and orchestrates
    sequential execution with MODFLOW groundwater coupling. Uses dCoupler
    for graph-based coupling when available.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, logger)
        self._land_worker = None
        self._land_model_name = None

    @property
    def land_model_name(self) -> str:
        """Get land surface model name from config."""
        if self._land_model_name is None:
            from symfluence.models.modflow.land_adapters import resolve_land_model
            self._land_model_name = resolve_land_model(self.config or {})
        assert self._land_model_name is not None
        return self._land_model_name

    @property
    def land_worker(self) -> BaseWorker:
        """Lazy-load the land surface model's worker."""
        if self._land_worker is None:
            worker_cls = R.workers.get(self.land_model_name)
            if worker_cls is None:
                raise ValueError(
                    f"No worker registered for land surface model "
                    f"'{self.land_model_name}'"
                )
            self._land_worker = worker_cls(config=self.config, logger=self.logger)
        assert self._land_worker is not None
        return self._land_worker

    def _use_dcoupler(self, config: Dict[str, Any]) -> bool:
        """Determine if dCoupler should be used for coupling."""
        mode = config.get('COUPLING_MODE', 'auto')
        if mode == 'dcoupler':
            return True
        if mode == 'sequential':
            return False
        # Auto-detect
        try:
            from symfluence.core.modeling.coupling import is_dcoupler_available
            return is_dcoupler_available()
        except ImportError:
            return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs,
    ) -> bool:
        """Apply parameters to land surface and MODFLOW files."""
        from .parameter_manager import MODFLOW_DEFAULT_BOUNDS

        config = kwargs.get('config', self.config or {})
        modflow_param_names = set(MODFLOW_DEFAULT_BOUNDS.keys())
        modflow_params_str = config.get('MODFLOW_PARAMS_TO_CALIBRATE', '')
        if modflow_params_str:
            modflow_param_names.update(
                p.strip() for p in str(modflow_params_str).split(',') if p.strip()
            )

        land_params = {k: v for k, v in params.items() if k not in modflow_param_names}
        modflow_params = {k: v for k, v in params.items() if k in modflow_param_names}

        success = True

        # Apply land surface parameters
        if land_params:
            land_settings = self._resolve_land_settings(settings_dir)
            # Resilience: if settings were deleted by a concurrent cleanup,
            # regenerate them from the project-level settings.
            if not self._land_settings_present(land_settings):
                self._regenerate_settings(settings_dir, config)
                land_settings = self._resolve_land_settings(settings_dir)
            try:
                success = self.land_worker.apply_parameters(
                    land_params, land_settings, **kwargs
                )
                if not success:
                    self.logger.error(
                        f"Land surface parameter application returned False. "
                        f"settings_dir={land_settings}, "
                        f"attrs_exists={Path(land_settings / 'attributes.nc').exists()}, "
                        f"trial_exists={Path(land_settings / 'trialParams.nc').exists()}"
                    )
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.error(
                    f"Land surface parameter application raised: {e}",
                    exc_info=True,
                )
                success = False

        # Apply MODFLOW parameters by rewriting text files directly
        # (avoids creating a full CoupledGWParameterManager per iteration)
        if modflow_params and success:
            modflow_settings = self._resolve_modflow_settings(settings_dir)
            try:
                success = self._write_modflow_params(modflow_params, modflow_settings, config)
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.error(f"Failed to apply MODFLOW parameters: {e}", exc_info=True)
                success = False

        return success

    def _land_settings_present(self, land_settings: Path) -> bool:
        if self.land_model_name == 'SUMMA':
            return (land_settings / 'attributes.nc').exists()
        return land_settings.is_dir() and any(land_settings.iterdir())

    def _write_modflow_params(
        self,
        params: Dict[str, float],
        modflow_dir: Path,
        config: Dict[str, Any],
    ) -> bool:
        """Write MODFLOW parameter files directly (NPF, STO, DRN)."""
        from symfluence.models.modflow.basin_model import apply_basin_parameters, params_from_flat
        if apply_basin_parameters(modflow_dir, params, params_from_flat(config)):
            return True

        if 'K' in params:
            k = float(params['K'])
            (modflow_dir / "gwf.npf").write_text(
                "BEGIN OPTIONS\n  SAVE_SPECIFIC_DISCHARGE\nEND OPTIONS\n\n"
                "BEGIN GRIDDATA\n  ICELLTYPE\n    CONSTANT 1\n"
                f"  K\n    CONSTANT {k}\nEND GRIDDATA\n"
            )

        if 'SY' in params:
            sy = float(params['SY'])
            ss = float(config.get('MODFLOW_SS', 1e-5))
            (modflow_dir / "gwf.sto").write_text(
                "BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n"
                "BEGIN GRIDDATA\n  ICONVERT\n    CONSTANT 1\n"
                f"  SS\n    CONSTANT {ss}\n  SY\n    CONSTANT {sy}\n"
                "END GRIDDATA\n\nBEGIN PERIOD 1\n  TRANSIENT\nEND PERIOD 1\n"
            )

        if 'DRAIN_CONDUCTANCE' in params:
            drain_elev = config.get('MODFLOW_DRAIN_ELEVATION')
            if drain_elev is None:
                top = float(config.get('MODFLOW_TOP', 1500.0))
                bot = float(config.get('MODFLOW_BOT', 1400.0))
                drain_elev = (top + bot) / 2.0
            else:
                drain_elev = float(drain_elev)
            cond = float(params['DRAIN_CONDUCTANCE'])
            (modflow_dir / "gwf.drn").write_text(
                "BEGIN OPTIONS\n  PRINT_INPUT\n  PRINT_FLOWS\n  SAVE_FLOWS\n"
                "END OPTIONS\n\nBEGIN DIMENSIONS\n  MAXBOUND 1\nEND DIMENSIONS\n\n"
                f"BEGIN PERIOD 1\n  1 1 1 {drain_elev} {cond}\nEND PERIOD 1\n"
            )

        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Run the coupled land-surface + MODFLOW pipeline.

        Uses dCoupler graph when available, falls back to sequential coupling.
        """
        if self._use_dcoupler(config):
            return self._run_dcoupler(config, settings_dir, output_dir, **kwargs)
        return self._run_sequential(config, settings_dir, output_dir, **kwargs)

    def _run_dcoupler(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Run coupled models via dCoupler CouplingGraph."""
        try:
            from symfluence.core.modeling.coupling import build_coupling_graph
            # Graph builder reads HYDROLOGICAL_MODEL for the land component,
            # so override it with the actual land surface model name.
            # Also remove ROUTING_MODEL if set to 'none' so the graph builder
            # doesn't try to look up 'NONE' as a model.
            graph_config = dict(config)
            graph_config['HYDROLOGICAL_MODEL'] = self.land_model_name
            graph_config['GROUNDWATER_MODEL'] = 'MODFLOW'
            routing = str(graph_config.get('ROUTING_MODEL', '')).upper()
            if routing in ('NONE', 'N/A', ''):
                graph_config.pop('ROUTING_MODEL', None)
            graph = build_coupling_graph(graph_config)

            outputs = graph.forward(
                external_inputs={},
                n_timesteps=kwargs.get('n_timesteps', 1),
                dt=kwargs.get('dt', 86400.0),
            )

            # Save outputs
            import torch
            output_dir.mkdir(parents=True, exist_ok=True)
            for comp_name, comp_outputs in outputs.items():
                comp_dir = output_dir / comp_name.upper()
                comp_dir.mkdir(parents=True, exist_ok=True)
                for flux_name, tensor in comp_outputs.items():
                    torch.save(tensor, comp_dir / f"{flux_name}.pt")

            self.logger.info("Coupled run via dCoupler completed successfully")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"dCoupler execution failed: {e}", exc_info=True)
            self.logger.info("Falling back to sequential coupling")
            return self._run_sequential(config, settings_dir, output_dir, **kwargs)

    def _run_sequential(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Run models sequentially with file-based coupling."""
        from symfluence.models.modflow.runner import MODFLOWRunner

        land_settings = self._resolve_land_settings(settings_dir)
        modflow_settings = self._resolve_modflow_settings(settings_dir)

        land_output = output_dir / self.land_model_name
        modflow_output = output_dir / 'MODFLOW'

        # Fully recreate output directories to prevent stale/corrupted
        # NetCDF files from prior killed iterations causing empty output.
        import shutil
        for d in (land_output, modflow_output):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        # Step 1: Run land surface model (writing into land_output)
        from symfluence.models.modflow.land_adapters import adapter_options, get_land_adapter
        get_land_adapter(self.land_model_name, adapter_options(config)).set_output_dir(land_settings, land_output)
        self.logger.debug(f"Running {self.land_model_name} (land surface)...")
        # Filter kwargs to avoid duplicate keyword arguments when delegating
        land_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ('sim_dir', 'output_dir', 'settings_dir')
        }
        land_success = self.land_worker.run_model(
            config, land_settings, land_output,
            sim_dir=land_output, **land_kwargs,
        )
        if not land_success:
            self.logger.error(
                f"{self.land_model_name} execution failed; aborting coupled run"
            )
            return False

        # Step 2: Run MODFLOW with iteration-specific SUMMA recharge.
        # Pass land_output as coupling_source_dir so the runner reads
        # recharge from THIS iteration's SUMMA output (not the project-
        # level default), writes gwf.rch, and executes mf6.
        self.logger.debug("Running MODFLOW 6 (groundwater)...")
        try:
            runner = MODFLOWRunner(config, self.logger)
            runner.settings_dir = modflow_settings
            result = runner.run_modflow(
                sim_dir=modflow_output,
                coupling_source_dir=land_output,
            )
            if result is None:
                return False
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"MODFLOW execution failed: {e}", exc_info=True)
            return False

        self.logger.debug(
            f"Coupled {self.land_model_name}-MODFLOW run completed successfully"
        )
        return True

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Calculate metrics from combined land-surface + MODFLOW output."""
        try:
            from .targets import CoupledGWStreamflowTarget

            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = config.get('DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"

            target = CoupledGWStreamflowTarget(config, project_dir, self.logger)
            metrics = target.calculate_metrics(output_dir, calibration_only=True)

            if metrics:
                return metrics

            self.logger.warning("Target returned empty metrics")
            return {'kge': self.penalty_score}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating coupled metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score}

    def _resolve_land_settings(self, settings_dir: Path) -> Path:
        """Resolve land surface model settings directory."""
        candidate = settings_dir / self.land_model_name
        return candidate if candidate.exists() else settings_dir

    def _resolve_modflow_settings(self, settings_dir: Path) -> Path:
        """Resolve MODFLOW settings directory."""
        candidate = settings_dir / 'MODFLOW'
        if candidate.exists():
            return candidate
        candidate = settings_dir.parent / 'MODFLOW'
        return candidate if candidate.exists() else settings_dir

    def _regenerate_settings(
        self,
        settings_dir: Path,
        config: Dict[str, Any],
    ) -> None:
        """Regenerate settings from project-level source if deleted by concurrent cleanup."""
        import shutil
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        domain_name = config.get('DOMAIN_NAME')
        project_dir = data_dir / f"domain_{domain_name}"

        # Regenerate land surface settings
        land_dest = settings_dir / self.land_model_name
        land_source = project_dir / 'settings' / self.land_model_name
        if land_source.exists() and not self._land_settings_present(land_dest):
            land_dest.mkdir(parents=True, exist_ok=True)
            for item in land_source.iterdir():
                if item.is_file():
                    shutil.copy2(item, land_dest / item.name)
            self.logger.warning(
                f"Regenerated {self.land_model_name} settings from {land_source}"
            )

        # Regenerate MODFLOW settings
        mf_dest = settings_dir / 'MODFLOW'
        mf_source = project_dir / 'settings' / 'MODFLOW'
        if mf_source.exists() and not mf_dest.exists():
            mf_dest.mkdir(parents=True, exist_ok=True)
            for item in mf_source.iterdir():
                if item.is_file():
                    shutil.copy2(item, mf_dest / item.name)
            self.logger.warning(f"Regenerated MODFLOW settings from {mf_source}")

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for parallel execution."""
        return _evaluate_coupled_gw_worker(task_data)


def _evaluate_coupled_gw_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    from symfluence.core.calibration.workers.base_worker import WorkerTask
    worker = CoupledGWWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
