# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""DCouplerWorker: BaseWorker implementation that delegates to dCoupler CouplingGraph.

Bridges dCoupler's PyTorch-based training loop with SYMFLUENCE's BaseWorker
calibration interface (DDS, Ostrich, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import torch

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.registries import R
from symfluence.coupling.bmi_registry import BMIRegistry
from symfluence.coupling.graph_builder import CouplingGraphBuilder

logger = logging.getLogger(__name__)


@R.workers.add('DCOUPLER')
class DCouplerWorker(BaseWorker):
    """BaseWorker implementation that delegates to a dCoupler CouplingGraph.

    This worker can operate in two modes:

    1. **Standard mode** (process-based models): Applies parameters to files,
       executes external models through the graph, reads outputs.
    2. **Gradient mode** (JAX models): Uses PyTorch autograd through the graph
       for gradient-based optimization.

    The worker constructs the CouplingGraph once on initialization and reuses
    it for all evaluations.
    """

    def __init__(self, config: dict, logger_instance: Optional[logging.Logger] = None):
        super().__init__(config, logger_instance or logger)
        self._registry = BMIRegistry()
        self._builder = CouplingGraphBuilder(registry=self._registry)
        self._graph = None
        self._external_inputs = None
        self._n_timesteps = None
        self._dt = None

    @property
    def graph(self):
        """Lazy-initialize the coupling graph from config."""
        if self._graph is None:
            self._graph = self._builder.build(self.config)
        return self._graph

    def supports_native_gradients(self) -> bool:
        """True if all components in the graph are differentiable."""
        from dcoupler.core.component import GradientMethod
        return all(
            comp.gradient_method != GradientMethod.NONE
            for comp in self.graph.components.values()
        )

    def set_external_inputs(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
    ) -> None:
        """Set the forcing data for graph forward passes.

        Call this before evaluate() when using the graph in calibration loops.

        Args:
            external_inputs: Dict of {component_name: {flux_name: tensor}}
            n_timesteps: Number of timesteps to simulate
            dt: Timestep size in seconds
        """
        self._external_inputs = external_inputs
        self._n_timesteps = n_timesteps
        self._dt = dt

    def apply_parameters(
        self, params: Dict[str, float], settings_dir: Path, **kwargs
    ) -> bool:
        """Apply parameters to graph components.

        For JAX/autograd components, sets the raw nn.Parameters.
        For process components, writes parameter files to settings_dir.
        """
        try:
            for comp_name, comp in self.graph.components.items():
                comp_params = {
                    k: v for k, v in params.items()
                    if any(s.name == k for s in comp.parameters)
                }
                if not comp_params:
                    continue

                # For JAX components: set raw parameters via sigmoid inverse
                from dcoupler.wrappers.jax import JAXComponent
                if isinstance(comp, JAXComponent):
                    for i, spec in enumerate(comp.parameters):
                        if spec.name in comp_params:
                            val = comp_params[spec.name]
                            lo, hi = spec.lower_bound, spec.upper_bound
                            # Inverse sigmoid to get raw value
                            normalized = (val - lo) / (hi - lo)
                            normalized = max(min(normalized, 0.999), 0.001)
                            raw = np.log(normalized / (1 - normalized))
                            comp._raw_params[i].data = torch.tensor(
                                raw, dtype=torch.float32
                            )

                # For process components: delegate to model-specific parameter writing
                from dcoupler.wrappers.process import ProcessComponent
                if isinstance(comp, ProcessComponent):
                    self._apply_process_params(comp, comp_params, settings_dir)

            return True
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Failed to apply parameters: {e}")
            return False

    def _apply_process_params(
        self, comp, params: Dict[str, float], settings_dir: str
    ) -> None:
        """Write parameters to files for process-based components.

        Process models receive their parameters through model-specific
        preprocessors that write config/parameter files before execution.
        This method logs which parameters are being passed through but
        does not duplicate the preprocessor's work.
        """
        if params:
            self.logger.debug(
                f"Process component '{comp.name}' received params "
                f"{list(params.keys())}. Parameter application for process "
                "models is handled by model-specific preprocessors."
            )

    def run_model(
        self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs
    ) -> bool:
        """Execute the coupling graph forward pass."""
        try:
            # For process-only graphs, external_inputs aren't needed
            # (ProcessComponents handle their own I/O from disk)
            external_inputs: Dict[str, Any] = self._external_inputs
            n_timesteps = self._n_timesteps
            dt = self._dt

            if external_inputs is None:
                external_inputs = {}
                n_timesteps = n_timesteps or 1
                dt = dt or 86400.0

            outputs = self.graph.forward(
                external_inputs=external_inputs,
                n_timesteps=n_timesteps,
                dt=dt,
            )

            # Store outputs for metric calculation
            self._last_outputs = outputs

            # Save outputs to disk
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for comp_name, comp_outputs in outputs.items():
                for flux_name, tensor in comp_outputs.items():
                    save_path = output_path / f"{comp_name}_{flux_name}.pt"
                    torch.save(tensor, save_path)

            return True
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Graph forward pass failed: {e}")
            return False

    def calculate_metrics(
        self, output_dir: Path, config: Dict[str, Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate calibration metrics from graph outputs.

        Uses the stored outputs from the last run_model() call.
        Falls back to reading saved tensors from output_dir.
        """
        try:
            outputs = getattr(self, "_last_outputs", None)
            if outputs is None:
                # Try loading from disk
                output_path = Path(output_dir)
                outputs = {}
                for pt_file in output_path.glob("*.pt"):
                    parts = pt_file.stem.split("_", 1)
                    if len(parts) == 2:
                        comp_name, flux_name = parts
                        outputs.setdefault(comp_name, {})[flux_name] = torch.load(
                            pt_file, weights_only=True
                        )

            # Extract simulated discharge
            sim = self._extract_simulation(outputs, config)
            if sim is None:
                return {"KGE": -999.0}

            # Load observations and pair them with the simulation by date
            aligned = self._align_simulation_and_observations(sim, config)
            if aligned is None:
                return {"KGE": -999.0}
            sim, obs = aligned

            # Calculate metrics
            metrics = self._compute_metrics(sim, obs, config)
            return metrics

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Metric calculation failed: {e}")
            return {"KGE": -999.0}

    def _extract_simulation(
        self, outputs: Dict, config: dict
    ) -> Optional[torch.Tensor]:
        """Extract the primary simulation variable from graph outputs."""
        # Look for discharge in routing, then runoff in land
        for comp_name in ("routing", "land"):
            if comp_name in outputs:
                for var in ("discharge", "runoff"):
                    if var in outputs[comp_name]:
                        return outputs[comp_name][var]
        return None

    def _load_observations(self, config: dict) -> Optional[Any]:
        """Load observation data for metric calculation.

        Returns a ``pd.Series`` indexed by date when the file's first column
        parses as dates (value column: the second column, or the first data
        column), otherwise a dateless 1-D tensor of the second column.
        """
        obs_file = config.get("OBSERVATIONS_FILE")
        if obs_file is None:
            return None
        try:
            import pandas as pd
            df = pd.read_csv(obs_file)
            value_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
            values = pd.to_numeric(df[value_col], errors='coerce')
            if df.shape[1] > 1:
                dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                if len(dates) and dates.notna().mean() >= 0.9:
                    series = pd.Series(values.values, index=pd.DatetimeIndex(dates))
                    series = series[series.index.notna()]
                    return series[~series.index.duplicated(keep='first')].sort_index()
            return torch.tensor(values.values, dtype=torch.float32)
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Failed to load observations: {e}")
            return None

    def _simulation_time_index(self, n_steps: int, config: dict):
        """Dates of the simulated steps, from EXPERIMENT_TIME_START and dt.

        Returns None when the simulation start is not configured.
        """
        import pandas as pd
        start = config.get("EXPERIMENT_TIME_START")
        if not start and isinstance(self.config, dict):
            start = self.config.get("EXPERIMENT_TIME_START")
        if not start:
            return None
        try:
            start_ts = pd.Timestamp(start)
        except (TypeError, ValueError):
            return None
        dt = float(self._dt or 86400.0)  # same default as run_model()
        return pd.date_range(start_ts, periods=n_steps, freq=pd.Timedelta(seconds=dt))

    def _align_simulation_and_observations(
        self, sim: torch.Tensor, config: dict
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (sim, obs) tensors paired by date, after warm-up removal.

        With dated observations the simulated steps are stamped from
        EXPERIMENT_TIME_START and the model dt, and only steps whose date
        matches an observation date are kept (sim keeps its autograd graph).
        WARMUP_DAYS is counted in days from the simulation start. A dateless
        observation file (or an unknown simulation start) falls back to the
        legacy positional pairing with a warning.
        """
        import pandas as pd
        obs = self._load_observations(config)
        if obs is None:
            return None
        warmup_days = int(config.get('WARMUP_DAYS', 0) or 0)
        n_sim = int(sim.shape[0])

        sim_index = None
        if isinstance(obs, pd.Series):
            sim_index = self._simulation_time_index(n_sim, config)
            if sim_index is None:
                self.logger.warning(
                    "Observations are dated but EXPERIMENT_TIME_START is not set; "
                    "pairing simulation and observations by position"
                )

        if isinstance(obs, pd.Series) and sim_index is not None:
            obs = obs.dropna()
            if warmup_days > 0:
                cutoff = sim_index[0] + pd.Timedelta(days=warmup_days)
                obs = obs[obs.index >= cutoff]
            pos = sim_index.get_indexer(obs.index)
            keep = pos >= 0
            if not keep.any():
                self.logger.error("No observation dates fall on simulated timesteps")
                return None
            sim_sel = sim[torch.as_tensor(pos[keep], dtype=torch.long)]
            obs_t = torch.tensor(obs.values[keep], dtype=torch.float32)
            return sim_sel, obs_t

        # Legacy positional pairing (dateless observations)
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(
            obs.values, dtype=torch.float32)
        if warmup_days > 0:
            sim = sim[warmup_days:]
            obs_t = obs_t[warmup_days:]
        min_len = min(len(sim), len(obs_t))
        return sim[:min_len], obs_t[:min_len]

    def _compute_metrics(
        self,
        sim: torch.Tensor,
        obs: torch.Tensor,
        config: dict,
    ) -> Dict[str, float]:
        """Compute KGE/NSE with the shared symfluence.core.metrics definitions.

        ``sim`` and ``obs`` must already be paired (see
        _align_simulation_and_observations). Undefined metrics (e.g. constant
        observations) are reported as the -999 failure sentinel.
        """
        from symfluence.core import metrics as core_metrics

        sim_np = sim.detach().cpu().numpy().astype(float).ravel()
        obs_np = obs.detach().cpu().numpy().astype(float).ravel()
        min_len = min(len(sim_np), len(obs_np))
        sim_np, obs_np = sim_np[:min_len], obs_np[:min_len]

        valid = ~(np.isnan(sim_np) | np.isnan(obs_np))
        sim_np, obs_np = sim_np[valid], obs_np[valid]
        if len(sim_np) < 10:
            return {"KGE": -999.0}

        comps = cast("dict[str, float]", core_metrics.kge(obs_np, sim_np, return_components=True))
        kge_value = comps["KGE"]
        if not np.isfinite(kge_value):
            return {"KGE": -999.0}
        return {
            "KGE": float(kge_value),
            "NSE": float(core_metrics.nse(obs_np, sim_np)),
            "r": float(comps["r"]),
            "alpha": float(comps["alpha"]),
            "beta": float(comps["beta"]),
        }

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = "kge",
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Evaluate with gradient computation for differentiable models.

        Only works when all graph components support autograd.

        Returns:
            Tuple of (score, gradients_dict). Gradients is None if unavailable.
        """
        if not self.supports_native_gradients():
            raise RuntimeError(
                "Not all components support native gradients. "
                "Use evaluate() for non-differentiable components."
            )

        # Apply parameters
        self.apply_parameters(params, Path())

        # Forward pass with grad tracking
        for p in self.graph.get_all_parameters():
            p.requires_grad_(True)

        outputs = self.graph.forward(
            external_inputs=self._external_inputs,
            n_timesteps=self._n_timesteps,
            dt=self._dt,
        )

        # Compute differentiable loss
        from dcoupler.losses.hydrological import kge_loss, nse_loss
        sim = self._extract_simulation(outputs, self.config)
        aligned = self._align_simulation_and_observations(sim, self.config)
        if aligned is None:
            raise RuntimeError("No observations aligned with the simulation")
        sim, obs = aligned

        loss_fn = kge_loss if metric == "kge" else nse_loss
        loss = loss_fn(sim, obs)

        # Backward pass
        loss.backward()

        # Collect gradients
        gradients = {}
        for comp_name, comp in self.graph.components.items():
            for spec, param in zip(comp.parameters, comp.get_torch_parameters()):
                if param.grad is not None:
                    gradients[f"{comp_name}.{spec.name}"] = param.grad.detach().numpy()

        return float(loss.item()), gradients if gradients else None
