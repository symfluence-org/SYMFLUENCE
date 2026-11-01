# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Generic coupled-model calibration worker.

Runs an arbitrary chain of coupled standalone models (land + optional snow/groundwater/routing) and
scores one shared objective. Parameters are split + delegated to each model's own manager; the run
executes the dCoupler CouplingGraph when that backend is enabled, otherwise falls back to sequential
file-based delegation in graph order (each downstream model's standalone worker discovers its
upstream model's output). The objective is delegated to the objective model's worker (the routing
model when present, else the land model) -- e.g. multi-gauge streamflow KGE for SUMMA+dRoute.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.core.calibration.workers.base_worker import BaseWorker, run_worker_task
from symfluence.core.exceptions import ConfigurationError
from symfluence.core.registries import R

from .parameter_manager import (
    CoupledModelParameterManager,
    coupled_component_models,
    settings_subdir,
)


@R.workers.add('COUPLED')
class CoupledModelWorker(BaseWorker):
    """Runs the coupled-model chain and scores the objective model's metric."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        cfg = config or {}
        self._models: List[str] = coupled_component_models(cfg, self._cfg_get)
        self.land_model = self._models[0] if self._models else \
            str(cfg.get('HYDROLOGICAL_MODEL', 'SUMMA')).split(',')[0].upper()
        self.objective_model = self._resolve_objective_model(cfg)
        self._workers: Dict[str, BaseWorker] = {}
        self._pm: Optional[CoupledModelParameterManager] = None

    # ---- helpers ----------------------------------------------------------------------------
    def _cfg_get(self, _config, key, default):
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    def _resolve_objective_model(self, cfg: Dict[str, Any]) -> str:
        override = str(self._cfg_get(cfg, 'COUPLED_OBJECTIVE_MODEL', '') or '').upper()
        if override and override in self._models:
            return override
        # Default: the routing model owns the streamflow objective; else the last/only model.
        routing = str(self._cfg_get(cfg, 'ROUTING_MODEL', '') or '').split(',')[0].upper()
        if routing and routing in self._models:
            return routing
        return self._models[-1] if self._models else self.land_model

    def _worker(self, model: str) -> BaseWorker:
        if model not in self._workers:
            cls = R.workers.get(model)
            if cls is None:
                # Workers (unlike parameter managers/optimizers) are not auto-discovered: they
                # register as a side effect of importing the owning package's worker module.
                # Drain the modules packages have declared -- this is the path that works for
                # in-tree AND external plugins, since it asks the registry rather than guessing
                # a filesystem layout.
                R.workers.load_modules()
                cls = R.workers.get(model)
            # There is deliberately no `symfluence.models.<name>` import fallback here. It
            # used to exist, and it was a runtime upward edge from core into the models
            # layer -- invisible to the AST layering guard because the path was built with
            # an f-string, and unable to resolve any external plugin. Every in-tree package
            # now declares its worker module (guarded by
            # tests/unit/core/test_worker_module_declarations.py), so the drain above covers
            # them; an external plugin declares its own the same way.
            if cls is None:
                raise ConfigurationError(
                    f"No worker registered for '{model}'. Available: {sorted(R.workers.keys())}")
            self._workers[model] = cls(config=self.config, logger=self.logger)
        return self._workers[model]

    def _settings_for(self, settings_dir: Path, model: str) -> Path:
        root = Path(settings_dir)
        sub = settings_subdir(model)
        # Sub-model settings live at the project settings root (e.g. settings/SUMMA), as SIBLINGS of
        # the coupled model's own settings dir -- not nested under it. Strip a trailing coupled- or
        # sub-model settings dir so the path resolves to settings/<sub> in per-process run dirs
        # (the COUPLED optimizer passes settings_dir = .../settings/COUPLED).
        if root.name in (model, sub, 'COUPLED', settings_subdir('COUPLED')):
            root = root.parent
        return root / sub

    def _param_manager(self, settings_dir: Path) -> CoupledModelParameterManager:
        if self._pm is None:
            self._pm = CoupledModelParameterManager(self.config or {}, self.logger, Path(settings_dir))
        return self._pm

    # ---- BaseWorker contract ----------------------------------------------------------------
    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        # Split + delegate to each participating model's standalone parameter manager.
        return self._param_manager(settings_dir).update_model_files(params)

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        if self._use_dcoupler(config):
            try:
                return self._run_dcoupler(config, settings_dir, output_dir, **kwargs)
            except (RuntimeError, ValueError, KeyError, TypeError, ImportError, OSError) as e:
                self.logger.error(f"dCoupler coupled run failed ({e}); falling back to sequential",
                                  exc_info=True)
        return self._run_sequential(config, settings_dir, output_dir, **kwargs)

    def _use_dcoupler(self, config: Dict[str, Any]) -> bool:
        backend = str(self._cfg_get(config, 'COUPLING_BACKEND', '') or
                      self._cfg_get(config, 'COUPLING_MODE', '') or '').lower()
        if backend != 'dcoupler':
            return False
        try:
            from symfluence.core.modeling.coupling import is_dcoupler_available
            return bool(is_dcoupler_available())
        except ImportError:
            return False

    def _run_dcoupler(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        import torch

        from symfluence.core.modeling.coupling import build_coupling_graph
        graph = build_coupling_graph(config)
        outputs = graph.forward(external_inputs=kwargs.get('external_inputs', {}),
                                n_timesteps=kwargs.get('n_timesteps', 1),
                                dt=kwargs.get('dt', 86400.0))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for comp_name, comp_outputs in outputs.items():
            cdir = Path(output_dir) / comp_name.upper(); cdir.mkdir(parents=True, exist_ok=True)
            for flux_name, tensor in comp_outputs.items():
                torch.save(tensor, cdir / f"{flux_name}.pt")
        # Graph->metrics handoff: the dCoupler graph returns raw flux tensors, but each model's
        # calculate_metrics reads its own on-disk format. Let the objective model materialize its
        # metric inputs from the graph outputs (optional hook) so calculate_metrics works unchanged.
        obj_worker = self._worker(self.objective_model)
        materialize = getattr(obj_worker, 'materialize_metric_inputs', None)
        if callable(materialize):
            obj_dir = Path(output_dir) / self.objective_model
            obj_dir.mkdir(parents=True, exist_ok=True)
            materialize(outputs, obj_dir, self._settings_for(settings_dir, self.objective_model), config)
        self.logger.info(f"Coupled run via dCoupler completed over {self._models}")
        return True

    def _run_sequential(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Run each model in graph order, handing THIS iteration's upstream output to the next.

        Mirrors the CoupledGW worker: each model runs into ``output_dir/<model>`` and the downstream
        worker receives ``coupling_source_dir`` = the upstream model's fresh output dir, so it routes
        this iteration's runoff (not a stale project-level file). Output dirs are recreated each call
        to avoid stale NetCDF from a prior killed iteration.
        """
        import shutil
        run_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ('sim_dir', 'output_dir', 'settings_dir', 'coupling_source_dir')}
        prev_output: Optional[Path] = None
        for model in self._models:
            mdir = Path(output_dir) / model
            if mdir.exists():
                shutil.rmtree(mdir, ignore_errors=True)
            mdir.mkdir(parents=True, exist_ok=True)
            kw = dict(run_kwargs, sim_dir=mdir)
            if prev_output is not None:
                kw['coupling_source_dir'] = prev_output   # downstream reads THIS iteration's output
            # The objective model owns the final routing/streamflow metric (dRoute when present).
            # Tell every UPSTREAM model not to self-route (e.g. SUMMA must not run its own mizuRoute);
            # the downstream routing model routes this iteration's runoff. Standalone runs never see
            # this flag, so each model's native routing behaviour is preserved outside coupling.
            if model != self.objective_model:
                kw['skip_routing'] = True
            ok = self._worker(model).run_model(config, self._settings_for(settings_dir, model), mdir, **kw)
            if not ok:
                self.logger.error(f"Coupled chain failed at model '{model}'")
                return False
            prev_output = mdir
        return True

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # The objective model (routing if present, else land) owns the shared objective.
        obj_dir = Path(output_dir) / self.objective_model
        if not obj_dir.exists():
            obj_dir = Path(output_dir)
        return self._worker(self.objective_model).calculate_metrics(obj_dir, config, **kwargs)

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static pool entry for parallel (MPI/ProcessPool) coupled evaluation.

        Reconstructs the coupled worker -- which rebuilds the model chain (e.g. SUMMA -> dRoute)
        from ``config`` -- in the worker process and evaluates one joint parameter vector. Without
        this, the parallel ``PopulationEvaluator`` has no worker function to dispatch to, so the
        optimizer runs zero evaluations and produces no results.
        """
        return _evaluate_coupled_parameters_worker(task_data)


def _evaluate_coupled_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level coupled worker function for MPI/ProcessPool execution.

    Resolved by ``PopulationEvaluator`` via the ``_evaluate_<model>_parameters_worker`` naming
    convention (here ``_evaluate_coupled_parameters_worker``); delegates to ``run_worker_task``.
    """
    return run_worker_task(
        CoupledModelWorker,
        task_data,
        start_jitter=None,
        error_prefix='coupled worker exception',
    )


__all__ = ['CoupledModelWorker']
