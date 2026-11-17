# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Process-based adapters wrapping SYMFLUENCE external-executable models.

Each adapter wraps a SYMFLUENCE model's runner as a dCoupler ProcessComponent,
marshalling data between PyTorch tensors and the file formats expected by each
external model.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from dcoupler.core.component import (
    FluxDirection,
    FluxSpec,
)
from dcoupler.wrappers.process import ProcessComponent

from symfluence.core.registries import R

logger = logging.getLogger(__name__)


class SUMMAProcessComponent(ProcessComponent):
    """Wraps SUMMA executable as a dCoupler ProcessComponent with BMI lifecycle.

    Reuses SYMFLUENCE's SUMMARunner for actual execution.
    """

    def __init__(self, name: str = "summa", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("forcing", "mixed", FluxDirection.INPUT, "hru", 3600,
                     ("time", "hru", "var")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("runoff", "kg/m2/s", FluxDirection.OUTPUT, "hru", 3600,
                     ("time", "hru"), conserved_quantity="water_mass"),
            FluxSpec("soil_drainage", "kg/m2/s", FluxDirection.OUTPUT, "hru", 3600,
                     ("time", "hru"), conserved_quantity="water_mass"),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('SUMMA')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("SUMMARunner not registered; execute() will fail")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        # SUMMA reads forcing from NetCDF; inputs are pre-written by the preprocessor
        pass

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("SUMMARunner not initialized. Call bmi_initialize first.")
        try:
            success = self._runner.run_summa()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"SUMMA execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
            output_dir = Path(self._model_config.get('EXPERIMENT_OUTPUT_SUMMA', work_dir))
            output_files = sorted(output_dir.glob("*_output_*.nc"))
            if not output_files:
                raise FileNotFoundError(f"No SUMMA output files in {output_dir}")

            ds = xr.open_mfdataset(output_files, data_vars='minimal', coords='minimal', compat='override')
            runoff = torch.tensor(
                ds["scalarTotalRunoff"].values.astype(np.float32),
                dtype=torch.float32,
            )
            drainage = torch.tensor(
                ds["scalarSoilDrainage"].values.astype(np.float32),
                dtype=torch.float32,
            )
            ds.close()
            return {"runoff": runoff, "soil_drainage": drainage}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read SUMMA outputs: {e}")
            raise RuntimeError("Failed to read SUMMA outputs") from e


class MizuRouteProcessComponent(ProcessComponent):
    """Wraps mizuRoute as ProcessComponent with time-fix logic.

    Reuses SYMFLUENCE's MizuRouteRunner for actual execution.
    """

    def __init__(self, name: str = "mizuroute", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("lateral_inflow", "m3/s", FluxDirection.INPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('MIZUROUTE')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("MizuRouteRunner not registered")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # mizuRoute reads SUMMA output directly

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("MizuRouteRunner not initialized")
        try:
            self._runner.fix_time_precision()
            # No sync_control_file_dimensions() call here: it takes
            # (control_path, netcdf_path), so calling it bare raised TypeError
            # on every invocation — swallowed by the handler below, which then
            # reported "mizuRoute execution failed" and returned 1. This
            # adapter's mizuRoute path therefore never ran at all. The call was
            # redundant as well as wrong: run_mizuroute() performs the same
            # sync itself (models/mizuroute/runner.py), with paths it resolves.
            success = self._runner.run_mizuroute()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"mizuRoute execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
            output_dir = Path(self._model_config.get(
                'EXPERIMENT_OUTPUT_MIZUROUTE', work_dir
            ))
            output_files = sorted(output_dir.glob("*.nc"))
            if not output_files:
                raise FileNotFoundError(f"No mizuRoute output in {output_dir}")

            ds = xr.open_mfdataset(output_files, data_vars='minimal', coords='minimal', compat='override')
            discharge = torch.tensor(
                ds["IRFroutedRunoff"].values.astype(np.float32),
                dtype=torch.float32,
            )
            ds.close()
            return {"discharge": discharge}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read mizuRoute outputs: {e}")
            raise RuntimeError("Failed to read mizuRoute outputs") from e


class ParFlowProcessComponent(ProcessComponent):
    """Wraps ParFlow as ProcessComponent.

    Reuses SUMMAToParFlowCoupler unit conversion logic.
    Unit conversion: kg/m2/s -> m/hr (factor 3.6)
    """

    KG_M2_S_TO_M_HR = 3.6

    def __init__(self, name: str = "parflow", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("recharge", "m/hr", FluxDirection.INPUT, "grid", 3600,
                     ("time", "cell")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("baseflow", "m3/hr", FluxDirection.OUTPUT, "grid", 3600,
                     ("time", "cell")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('PARFLOW')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("ParFlowRunner not registered")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        if "recharge" in inputs:
            recharge = inputs["recharge"].detach().numpy()
            output_path = work_dir / "recharge.csv"
            with open(output_path, "w") as f:
                f.write("sim_hour,recharge_m_hr\n")
                for i, val in enumerate(recharge):
                    f.write(f"{i},{float(val)}\n")

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("ParFlowRunner not initialized")
        try:
            success = self._runner.run_parflow()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"ParFlow execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            extractor_cls = R.result_extractors.get('PARFLOW')
            if extractor_cls is None:
                raise RuntimeError("PARFLOW result extractor not registered")

            output_dir = Path(self._model_config.get('PARFLOW_OUTPUT_DIR', work_dir))
            extractor = extractor_cls()
            kwargs = {
                'start_date': self._model_config.get('SIMULATION_START', '2000-01-01'),
                'timestep_hours': float(self._model_config.get('TIMESTEP_HOURS', 1.0)),
            }

            overland = extractor.extract_variable(
                output_dir, 'overland_flow', **kwargs
            )
            subsurface = extractor.extract_variable(
                output_dir, 'subsurface_drainage', **kwargs
            )
            # Combine into total baseflow (m3/hr for FluxSpec units)
            combined = overland * 3600 + subsurface  # overland m3/s→m3/hr; subsurface already m3/hr
            return {"baseflow": torch.tensor(
                combined.values.astype(np.float32), dtype=torch.float32
            )}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read ParFlow outputs: {e}")
            raise RuntimeError("Failed to read ParFlow outputs") from e


class MODFLOWProcessComponent(ProcessComponent):
    """Wraps MODFLOW 6 (lumped grid) as ProcessComponent.

    The graph's daily recharge (m/d) is written as the model's gwf.rch, one
    block per TDIS stress period (SUMMAToMODFLOWCoupler.write_modflow_recharge_rch),
    and MODFLOW then runs with that file instead of re-deriving recharge from
    the land model's output.
    Unit conversion: kg/m2/s -> m/d (factor 86.4)
    """

    KG_M2_S_TO_M_D = 86.4

    def __init__(self, name: str = "modflow", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None
        self._recharge_written = False

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("recharge", "m/d", FluxDirection.INPUT, "grid", 86400,
                     ("time", "cell")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("drain_discharge", "m3/d", FluxDirection.OUTPUT, "grid", 86400,
                     ("time", "cell")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('MODFLOW')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("MODFLOWRunner not registered")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        self._recharge_written = False
        if "recharge" not in inputs:
            return
        if self._runner is None:
            raise RuntimeError("MODFLOWRunner not initialized. Call bmi_initialize first.")
        if str(self._model_config.get('MODFLOW_GRID_MODE', 'lumped')).lower() == 'basin':
            raise NotImplementedError(
                "Graph-coupled recharge is only supported for the lumped MODFLOW grid; "
                "basin mode takes zone recharge from the land-model adapter")
        recharge = np.asarray(inputs["recharge"].detach().cpu().numpy(), dtype=float)
        if recharge.ndim > 1:
            # Lumped RCH is one CONSTANT per period: average over cells.
            recharge = recharge.reshape(recharge.shape[0], -1).mean(axis=1)
        self._runner.write_lumped_recharge(recharge.ravel())
        self._recharge_written = True

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("MODFLOWRunner not initialized")
        try:
            # Keep the graph's recharge (gwf.rch) rather than rebuilding it from land output.
            success = self._runner.run_modflow(prepare_recharge=not self._recharge_written)
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"MODFLOW execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            extractor_cls = R.result_extractors.get('MODFLOW')
            if extractor_cls is None:
                raise RuntimeError("MODFLOW result extractor not registered")

            output_dir = Path(self._model_config.get('MODFLOW_OUTPUT_DIR', work_dir))
            extractor = extractor_cls()
            kwargs = {
                'start_date': self._model_config.get('SIMULATION_START', '2000-01-01'),
                'stress_period_length': float(
                    self._model_config.get('MODFLOW_STRESS_PERIOD_LENGTH',
                                           self._model_config.get('MODFLOW_STRESS_PERIOD_DAYS', 1.0))
                ),
            }

            drain = extractor.extract_variable(
                output_dir, 'drain_discharge', **kwargs
            )
            # drain is in m3/d (matching FluxSpec units)
            return {"drain_discharge": torch.tensor(
                drain.values.astype(np.float32), dtype=torch.float32
            )}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read MODFLOW outputs: {e}")
            raise RuntimeError("Failed to read MODFLOW outputs") from e


class MESHProcessComponent(ProcessComponent):
    """Wraps MESH executable as a dCoupler ProcessComponent."""

    def __init__(self, name: str = "mesh", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("forcing", "mixed", FluxDirection.INPUT, "subbasin", 3600,
                     ("time", "subbasin", "var")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "subbasin", 86400,
                     ("time",)),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('MESH')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("MESHRunner not registered")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # MESH reads forcing files created by preprocessor

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("MESHRunner not initialized")
        try:
            success = self._runner.run_mesh()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"MESH execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            extractor_cls = R.result_extractors.get('MESH')
            if extractor_cls is None:
                raise RuntimeError("MESH result extractor not registered")

            output_dir = Path(self._model_config.get('EXPERIMENT_OUTPUT_MESH', work_dir))
            extractor = extractor_cls('MESH')
            kwargs = {
                'start_date': self._model_config.get('SIMULATION_START', '2001-01-01'),
                'aggregate': 'daily',
            }

            # Try Basin_average_water_balance.csv first (preferred for lumped)
            basin_wb = output_dir / 'Basin_average_water_balance.csv'
            if basin_wb.exists():
                discharge = extractor.extract_variable(
                    basin_wb, 'streamflow', **kwargs
                )
            else:
                # Fall back to MESH_output_streamflow.csv or GRU_water_balance
                discharge = extractor.extract_variable(
                    output_dir, 'streamflow', **kwargs
                )
            return {"discharge": torch.tensor(
                discharge.values.astype(np.float32), dtype=torch.float32
            )}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read MESH outputs: {e}")
            raise RuntimeError("Failed to read MESH outputs") from e


class TRouteProcessComponent(ProcessComponent):
    """Wraps t-route (NOAA OWP) as ProcessComponent for channel routing.

    Reuses SYMFLUENCE's TRouteRunner for actual execution.
    """

    def __init__(self, name: str = "troute", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("lateral_inflow", "m3/s", FluxDirection.INPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('TROUTE')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("TRouteRunner not registered")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # t-route reads upstream model output directly

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("TRouteRunner not initialized")
        try:
            success = self._runner.run_troute()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"t-route execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
            output_dir = Path(self._model_config.get(
                'EXPERIMENT_OUTPUT_TROUTE', work_dir
            ))
            # Check for both built-in output and nwm_routing output
            output_files = (
                sorted(output_dir.glob("troute_output.nc"))
                + sorted(output_dir.glob("*flowveldepth*.nc"))
                + sorted(output_dir.glob("nex-troute-out.nc"))
            )
            if not output_files:
                raise FileNotFoundError(f"No t-route output in {output_dir}")

            ds = xr.open_dataset(output_files[0])
            for var in ('flow', 'streamflow', 'discharge', 'q_lateral'):
                if var in ds:
                    discharge = torch.tensor(
                        ds[var].values.astype(np.float32),
                        dtype=torch.float32,
                    )
                    ds.close()
                    return {"discharge": discharge}
            ds.close()
            raise KeyError(
                "No supported discharge variable found in t-route output "
                "(expected one of flow, streamflow, discharge, q_lateral)"
            )
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read t-route outputs: {e}")
            raise RuntimeError("Failed to read t-route outputs") from e


class CLMProcessComponent(ProcessComponent):
    """Wraps CLM/CTSM executable as a dCoupler ProcessComponent."""

    def __init__(self, name: str = "clm", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("forcing", "mixed", FluxDirection.INPUT, "grid", 3600,
                     ("time", "grid", "var")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("runoff", "mm/s", FluxDirection.OUTPUT, "grid", 3600,
                     ("time", "grid"), conserved_quantity="water_mass"),
            FluxSpec("evapotranspiration", "mm/s", FluxDirection.OUTPUT, "grid", 3600,
                     ("time", "grid")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        runner_cls = R.runners.get('CLM')
        if runner_cls is not None:
            self._runner = runner_cls(config, logger)
        else:
            logger.warning("CLMRunner not registered; execute() will use raw subprocess")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # CLM reads forcing created by preprocessor

    def execute(self, work_dir: Path) -> int:
        if self._runner is not None:
            try:
                result = self._runner.run()
                return 0 if result else 1
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                logger.error(f"CLM execution failed: {e}")
                return 1

        # Fallback to raw subprocess if CLMRunner unavailable
        cesm_exe = self._model_config.get('CLM_CESM_EXE')
        if cesm_exe is None:
            raise RuntimeError("CLM_CESM_EXE not configured and CLMRunner unavailable")
        try:
            result = subprocess.run(
                [str(cesm_exe)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
            )
            return result.returncode
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"CLM execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            extractor_cls = R.result_extractors.get('CLM')
            if extractor_cls is None:
                raise RuntimeError("CLM result extractor not registered")

            output_dir = Path(self._model_config.get('EXPERIMENT_OUTPUT_CLM', work_dir))
            extractor = extractor_cls()
            kwargs = {
                'catchment_area_km2': self._model_config.get('CATCHMENT_AREA_KM2'),
            }

            runoff = extractor.extract_variable(
                output_dir, 'streamflow', **kwargs
            )
            et = extractor.extract_variable(
                output_dir, 'evapotranspiration', **kwargs
            )
            return {
                "runoff": torch.tensor(
                    runoff.values.astype(np.float32), dtype=torch.float32
                ),
                "evapotranspiration": torch.tensor(
                    et.values.astype(np.float32), dtype=torch.float32
                ),
            }
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read CLM outputs: {e}")
            raise RuntimeError("Failed to read CLM outputs") from e
