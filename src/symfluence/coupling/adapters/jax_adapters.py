# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""JAX-based adapters wrapping SYMFLUENCE's differentiable models.

Each adapter wraps a SYMFLUENCE JAX model's step() function as a dCoupler
JAXComponent, enabling gradient flow through the CouplingGraph.
"""

from __future__ import annotations

import logging
from typing import Optional

from dcoupler.core.component import (
    FluxDirection,
    FluxSpec,
    ParameterSpec,
)
from dcoupler.wrappers.jax import JAXComponent

logger = logging.getLogger(__name__)


def _get_snow17_step():
    """Import Snow-17 step function."""
    from jsnow17.model import snow17_step
    return snow17_step


def _get_snow17_params():
    """Import Snow-17 parameter bounds."""
    from jsnow17.parameters import SNOW17_PARAM_BOUNDS
    return SNOW17_PARAM_BOUNDS


def _get_xaj_step():
    """Import XAJ step function."""
    from jxaj.model import step_jax
    return step_jax


def _get_sacsma_step():
    """Import SAC-SMA step function."""
    from jsacsma.sacsma import sacsma_step
    return sacsma_step


def _get_hbv_step():
    """Import HBV step function."""
    from jhbv.model import step_jax
    return step_jax


def _get_hechms_step():
    """Import HEC-HMS step function."""
    from jhechms.model import step
    return step


def _get_topmodel_step():
    """Import TOPMODEL step function."""
    from jtopmodel.model import step
    return step


class Snow17JAXComponent(JAXComponent):
    """Wraps Snow-17 as JAXComponent with BMI lifecycle.

    Reuses: symfluence.models.snow17.model.snow17_step (the JAX kernel)
    Reuses: symfluence.models.snow17.bmi.Snow17BMI (BMI pattern)

    10 parameters (all linear transform):
        SCF, PXTEMP, MFMAX, MFMIN, NMF, MBASE, TIPM, UADJ, PLWHC, DAYGM
    """

    SNOW17_PARAMS = [
        ("SCF", 0.7, 1.4),
        ("PXTEMP", -2.0, 2.0),
        ("MFMAX", 0.5, 4.0),
        ("MFMIN", 0.05, 2.0),
        ("NMF", 0.001, 0.5),
        ("MBASE", 0.0, 1.0),
        ("TIPM", 0.01, 1.0),
        ("UADJ", 0.01, 0.4),
        ("PLWHC", 0.01, 0.3),
        ("DAYGM", 0.0, 0.3),
    ]

    def __init__(self, name: str = "snow17", config: Optional[dict] = None):
        try:
            jax_step = _get_snow17_step()
        except ImportError:
            raise ImportError("Snow-17 model not available in SYMFLUENCE") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.SNOW17_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("temp", "C", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("doy", "day", FluxDirection.INPUT, "hru", 86400, ("time",),
                     optional=True),
        ]
        output_flux_specs = [
            FluxSpec("rain_plus_melt", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp
            from jsnow17.parameters import (
                DEFAULT_ADC,
                Snow17State,
                params_dict_to_namedtuple,
            )
            snow17_params = params_dict_to_namedtuple(params, use_jax=True)
            adc = jnp.array(DEFAULT_ADC)
            # Unpack flat array → Snow17State namedtuple
            snow_state = Snow17State(
                w_i=state[0], w_q=state[1], w_qx=state[2],
                deficit=state[3], ati=state[4], swe=state[5],
            )
            # Get day-of-year from input (controls seasonal melt factor)
            doy = inputs.get("doy", jnp.float32(1))
            # snow17_step returns (new_state: Snow17State, rain_plus_melt)
            new_snow_state, rain_plus_melt = jax_step(
                precip=inputs["precip"],
                temp=inputs["temp"],
                dt=dt,
                state=snow_state,
                params=snow17_params,
                doy=doy,
                adc=adc,
                xp=jnp,
            )
            # Pack Snow17State back to flat array
            new_state = jnp.stack([
                new_snow_state.w_i, new_snow_state.w_q, new_snow_state.w_qx,
                new_snow_state.deficit, new_snow_state.ati, new_snow_state.swe,
            ])
            return rain_plus_melt, new_state

        # Snow-17 state: (w_i, w_q, w_qx, deficit, ati, swe) = 6 vars
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=6,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class XAJJAXComponent(JAXComponent):
    """Wraps Xinanjiang (XAJ) model as JAXComponent.

    Reuses: symfluence.models.xinanjiang.model.step_jax (per-timestep kernel)

    15 parameters from XAJ parameter registry.
    """

    # Param names match XinanjiangParams namedtuple fields exactly
    XAJ_PARAMS = [
        ("K", 0.0, 1.0),       # PET correction factor
        ("B", 0.1, 0.6),       # Tension water capacity curve exponent
        ("IM", 0.0, 0.1),      # Impervious area fraction
        ("UM", 5.0, 50.0),     # Upper layer tension water capacity (mm)
        ("LM", 10.0, 100.0),   # Lower layer tension water capacity (mm)
        ("DM", 10.0, 100.0),   # Deep layer tension water capacity (mm)
        ("C", 0.05, 0.2),      # Deep layer ET coefficient
        ("SM", 1.0, 100.0),    # Free water capacity (mm)
        ("EX", 0.5, 2.0),      # Free water capacity curve exponent
        ("KI", 0.0, 0.7),      # Interflow outflow coefficient
        ("KG", 0.0, 0.7),      # Groundwater outflow coefficient
        ("CS", 0.0, 1.0),      # Channel recession constant
        ("L", 0.0, 5.0),       # Lag time (timesteps)
        ("CI", 0.0, 1.0),      # Interflow recession constant
        ("CG", 0.0, 1.0),      # Groundwater recession constant
    ]

    def __init__(self, name: str = "xaj", config: Optional[dict] = None):
        try:
            jax_step = _get_xaj_step()
        except ImportError:
            raise ImportError("XAJ model not available in SYMFLUENCE") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.XAJ_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp
            from jxaj.parameters import (
                XinanjiangParams,
                XinanjiangState,
            )
            # Unpack flat array → XinanjiangState namedtuple
            xaj_state = XinanjiangState(
                wu=state[0], wl=state[1], wd=state[2],
                s=state[3], fr=state[4], qi=state[5], qg=state[6],
            )
            # Build XinanjiangParams from dict (names match fields)
            xaj_params = XinanjiangParams(**params)
            # step_jax signature: (precip, pet, state, params) -> (new_state, outflow)
            new_xaj_state, outflow = jax_step(
                precip=inputs["precip"],
                pet=inputs["pet"],
                state=xaj_state,
                params=xaj_params,
            )
            # Pack XinanjiangState back to flat array
            new_state = jnp.stack([
                new_xaj_state.wu, new_xaj_state.wl, new_xaj_state.wd,
                new_xaj_state.s, new_xaj_state.fr,
                new_xaj_state.qi, new_xaj_state.qg,
            ])
            return outflow, new_state

        # XAJ state size: WU, WL, WD, S, FR, QI, QG (7 vars)
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=7,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class SacSmaJAXComponent(JAXComponent):
    """Wraps SAC-SMA as JAXComponent.

    Reuses: symfluence.models.sacsma.sacsma.sacsma_step (per-timestep kernel)

    16 parameters from SAC-SMA parameter registry.
    """

    SACSMA_PARAMS = [
        ("UZTWM", 1.0, 150.0),
        ("UZFWM", 1.0, 150.0),
        ("UZK", 0.1, 0.5),
        ("PCTIM", 0.0, 0.1),
        ("ADIMP", 0.0, 0.4),
        ("RIVA", 0.0, 0.2),
        ("ZPERC", 1.0, 250.0),
        ("REXP", 1.0, 5.0),
        ("LZTWM", 1.0, 500.0),
        ("LZFSM", 1.0, 1000.0),
        ("LZFPM", 1.0, 1000.0),
        ("LZSK", 0.01, 0.25),
        ("LZPK", 0.001, 0.025),
        ("PFREE", 0.0, 0.6),
        ("SIDE", 0.0, 0.5),
        ("RSERV", 0.0, 0.4),
    ]

    def __init__(self, name: str = "sacsma", config: Optional[dict] = None):
        try:
            jax_step = _get_sacsma_step()
        except ImportError:
            raise ImportError("SAC-SMA model not available in SYMFLUENCE") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.SACSMA_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp
            from jsacsma.parameters import SacSmaParameters
            from jsacsma.sacsma import SacSmaState
            # Unpack flat array → SacSmaState namedtuple
            sac_state = SacSmaState(
                uztwc=state[0], uzfwc=state[1], lztwc=state[2],
                lzfpc=state[3], lzfsc=state[4], adimc=state[5],
            )
            sac_params = SacSmaParameters(**params)
            # sacsma_step returns (new_state, surface, interflow, baseflow, et)
            new_sac_state, surface, interflow, baseflow, et = jax_step(
                pxv=inputs["precip"],
                pet=inputs["pet"],
                dt=dt,
                state=sac_state,
                params=sac_params,
                xp=jnp,
            )
            runoff = surface + interflow + baseflow
            # Pack SacSmaState back to flat array
            new_state = jnp.stack([
                new_sac_state.uztwc, new_sac_state.uzfwc, new_sac_state.lztwc,
                new_sac_state.lzfpc, new_sac_state.lzfsc, new_sac_state.adimc,
            ])
            return runoff, new_state

        # SAC-SMA state: UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC (6 vars)
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=6,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class HBVJAXComponent(JAXComponent):
    """Wraps HBV as JAXComponent with BMI lifecycle.

    15 parameters governing snow, soil moisture, and response routines:
        tt, cfmax, sfcf, cfr, cwh, fc, lp, beta, k0, k1, k2, uzl, perc,
        maxbas, smoothing
    """

    HBV_PARAMS = [
        ("tt", -3.0, 3.0),          # Threshold temperature (°C)
        ("cfmax", 1.0, 10.0),       # Degree-day melt factor (mm/°C/dt)
        ("sfcf", 0.5, 1.5),         # Snowfall correction factor
        ("cfr", 0.0, 0.1),          # Refreezing coefficient
        ("cwh", 0.0, 0.2),          # Water holding capacity of snow
        ("fc", 50.0, 700.0),        # Field capacity (mm)
        ("lp", 0.3, 1.0),           # Soil moisture threshold for ET
        ("beta", 1.0, 6.0),         # Shape coefficient for runoff generation
        ("k0", 0.05, 0.5),          # Near-surface flow recession (1/dt)
        ("k1", 0.01, 0.3),          # Upper zone recession (1/dt)
        ("k2", 0.0001, 0.1),        # Lower zone recession (1/dt)
        ("uzl", 0.0, 100.0),        # Upper zone threshold (mm)
        ("perc", 0.0, 20.0),        # Percolation rate (mm/dt)
        ("maxbas", 1.0, 7.0),       # Routing triangle base (dt)
        ("smoothing", 1.0, 50.0),   # Smoothing parameter for thresholds
    ]

    def __init__(self, name: str = "hbv", config: Optional[dict] = None):
        try:
            jax_step = _get_hbv_step()
        except ImportError:
            raise ImportError("HBV model (jhbv) not available") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.HBV_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("temp", "C", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp
            from jhbv.model import HBVState, create_params_from_dict
            hbv_params = create_params_from_dict(params, use_jax=True)
            # Unpack flat array → HBVState namedtuple
            # State: snow, snow_water, sm, suz, slz, routing_buffer
            # Routing buffer length depends on maxbas; use remaining state slots
            n_core = 5  # snow, snow_water, sm, suz, slz
            hbv_state = HBVState(
                snow=state[0],
                snow_water=state[1],
                sm=state[2],
                suz=state[3],
                slz=state[4],
                routing_buffer=state[n_core:],
            )
            timestep_hours = max(1, int(dt / 3600))
            new_hbv_state, runoff = jax_step(
                precip=inputs["precip"],
                temp=inputs["temp"],
                pet=inputs["pet"],
                state=hbv_state,
                params=hbv_params,
                timestep_hours=timestep_hours,
            )
            # Pack HBVState back to flat array
            new_state = jnp.concatenate([
                jnp.stack([
                    new_hbv_state.snow, new_hbv_state.snow_water,
                    new_hbv_state.sm, new_hbv_state.suz, new_hbv_state.slz,
                ]),
                new_hbv_state.routing_buffer,
            ])
            return runoff, new_state

        # HBV state: 5 core + routing buffer (7 slots for maxbas up to 7)
        from jhbv.model import get_routing_buffer_length
        buf_len = get_routing_buffer_length(lag_days=7, timestep_hours=24)
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=5 + buf_len,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class HecHmsJAXComponent(JAXComponent):
    """Wraps HEC-HMS as JAXComponent with BMI lifecycle.

    14 parameters spanning snow accumulation/melt (temperature-index),
    SCS curve-number loss, Clark unit hydrograph transform, and
    linear-reservoir baseflow:
        px_temp, base_temp, ati_meltrate_coeff, meltrate_max, meltrate_min,
        cold_limit, ati_cold_rate_coeff, water_capacity, cn,
        initial_abstraction_ratio, tc, r_coeff, gw_storage_coeff,
        deep_perc_fraction
    """

    HECHMS_PARAMS = [
        ("px_temp", -2.0, 4.0),                # Rain/snow temperature threshold (°C)
        ("base_temp", -3.0, 3.0),              # Base temperature for melt (°C)
        ("ati_meltrate_coeff", 0.5, 1.5),      # ATI melt-rate coefficient
        ("meltrate_max", 2.0, 10.0),           # Maximum melt rate (mm/°C/dt)
        ("meltrate_min", 0.0, 3.0),            # Minimum melt rate (mm/°C/dt)
        ("cold_limit", 0.0, 50.0),             # Cold content limit (mm)
        ("ati_cold_rate_coeff", 0.0, 0.3),     # ATI cold rate coefficient
        ("water_capacity", 0.0, 0.3),          # Liquid water capacity of snowpack
        ("cn", 30.0, 98.0),                    # SCS curve number
        ("initial_abstraction_ratio", 0.05, 0.3),  # Initial abstraction ratio
        ("tc", 0.5, 20.0),                     # Time of concentration (hr)
        ("r_coeff", 0.5, 20.0),                # Clark storage coefficient (hr)
        ("gw_storage_coeff", 1.0, 100.0),      # Groundwater storage coefficient (hr)
        ("deep_perc_fraction", 0.0, 0.5),      # Deep percolation fraction
    ]

    def __init__(self, name: str = "hechms", config: Optional[dict] = None):
        try:
            jax_step = _get_hechms_step()
        except ImportError:
            raise ImportError("HEC-HMS model (jhechms) not available") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.HECHMS_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("temp", "C", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("doy", "day", FluxDirection.INPUT, "hru", 86400, ("time",),
                     optional=True),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp
            from jhechms.model import HecHmsState, create_params_from_dict
            hms_params = create_params_from_dict(params, use_jax=True)
            # Unpack flat array → HecHmsState namedtuple
            hms_state = HecHmsState(
                snow_swe=state[0],
                snow_liquid=state[1],
                snow_ati=state[2],
                snow_cold_content=state[3],
                soil_deficit=state[4],
                clark_storage=state[5],
                gw_storage_1=state[6],
                gw_storage_2=state[7],
            )
            doy = inputs.get("doy", jnp.float32(1))
            new_hms_state, runoff = jax_step(
                precip=inputs["precip"],
                temp=inputs["temp"],
                pet=inputs["pet"],
                state=hms_state,
                params=hms_params,
                day_of_year=doy,
                use_jax=True,
            )
            # Pack HecHmsState back to flat array
            new_state = jnp.stack([
                new_hms_state.snow_swe, new_hms_state.snow_liquid,
                new_hms_state.snow_ati, new_hms_state.snow_cold_content,
                new_hms_state.soil_deficit, new_hms_state.clark_storage,
                new_hms_state.gw_storage_1, new_hms_state.gw_storage_2,
            ])
            return runoff, new_state

        # HecHmsState: 8 variables
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=8,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class TopmodelJAXComponent(JAXComponent):
    """Wraps TOPMODEL as JAXComponent with BMI lifecycle.

    11 parameters governing the topographic-index-based variable
    contributing area formulation with snow and routing:
        m, lnTe, Srmax, Sr0, td, k_route, DDF, T_melt, T_snow, ti_std, S0
    """

    TOPMODEL_PARAMS = [
        ("m", 0.001, 0.3),         # Transmissivity decay parameter (m)
        ("lnTe", -7.0, 10.0),      # Log effective transmissivity (ln m²/hr)
        ("Srmax", 0.005, 0.5),     # Maximum root-zone storage (m)
        ("Sr0", 0.0, 0.1),         # Initial root-zone storage deficit (m)
        ("td", 0.1, 50.0),         # Unsaturated zone time delay (hr/m)
        ("k_route", 1.0, 200.0),   # Channel routing velocity parameter
        ("DDF", 0.5, 10.0),        # Degree-day factor (mm/°C/dt)
        ("T_melt", -2.0, 3.0),     # Snowmelt threshold temperature (°C)
        ("T_snow", -2.0, 3.0),     # Snowfall threshold temperature (°C)
        ("ti_std", 1.0, 10.0),     # Topographic index standard deviation
        ("S0", 0.0, 2.0),          # Initial saturation deficit (m)
    ]

    def __init__(self, name: str = "topmodel", config: Optional[dict] = None):
        try:
            jax_step = _get_topmodel_step()
        except ImportError:
            raise ImportError("TOPMODEL (jtopmodel) not available") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.TOPMODEL_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("temp", "C", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp
            from jtopmodel.model import TopmodelState, create_params_from_dict, generate_ti_distribution
            tm_params = create_params_from_dict(params, use_jax=True)
            # Unpack flat array → TopmodelState namedtuple
            tm_state = TopmodelState(
                s_bar=state[0],
                srz=state[1],
                suz=state[2],
                swe=state[3],
                q_routed=state[4],
            )
            # Generate topographic index distribution from ti_std parameter
            lnaotb, dist_area = generate_ti_distribution(
                ti_mean=jnp.float32(8.0),
                ti_std=params.get("ti_std", jnp.float32(3.0)),
                n_classes=30,
                use_jax=True,
            )
            dt_hours = max(1.0, dt / 3600.0)
            new_tm_state, runoff = jax_step(
                precip=inputs["precip"],
                temp=inputs["temp"],
                pet=inputs["pet"],
                state=tm_state,
                params=tm_params,
                lnaotb=lnaotb,
                dist_area=dist_area,
                dt=dt_hours,
                use_jax=True,
            )
            # Pack TopmodelState back to flat array
            new_state = jnp.stack([
                new_tm_state.s_bar, new_tm_state.srz, new_tm_state.suz,
                new_tm_state.swe, new_tm_state.q_routed,
            ])
            return runoff, new_state

        # TopmodelState: 5 variables (s_bar, srz, suz, swe, q_routed)
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=5,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )
