# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parameter Bounds Registry - the SHARED parameter catalogue.

This module is the single source of truth for parameter bounds that more than
one model resolves. Benefits:
- Eliminates duplication between model-specific parameter managers
- Provides consistent bounds for shared parameters (e.g., soil properties)
- Documents parameter meanings and units
- Keeps cross-model name collisions visible in one file

Architecture Decision:
    This module intentionally contains model-specific functions (get_fuse_bounds,
    get_ngen_bounds, etc.) despite the general pattern of moving model-specific
    code to respective model packages (models/fuse/, models/ngen/, etc.).

    Rationale for centralization:
    - Single source of truth: All parameter bounds in one place for easy comparison
    - Cross-model consistency: Ensures shared parameters use consistent bounds
    - Easier maintenance: Modifying bounds doesn't require editing 11 model packages
    - Better overview: Developers can see all parameter bounds at a glance
    - Scientific documentation: Bounds are documented with units and descriptions

    Alternative considered:
    - Splitting bounds into models/{model}/calibration/parameter_bounds.py
    - Rejected due to increased fragmentation and harder cross-model validation

    Decision affirmed during pre-migration refactoring (January 2026) as part of
    the effort to consolidate model-specific code before the main migration.

    Amended during service-decomposition prep (July 2026): the catalogue stays
    centralized, but external model packages must be able to contribute bounds
    without editing core. ``register_model_bounds()`` is that seam: a model
    package (in-tree or external) registers its parameter definitions and/or
    the catalogue names composing its bound set, and ``get_model_bounds()``
    serves them uniformly.

    REFINED (July 2026, service-decomposition item 2). The centralization
    decision above is kept for what it was actually protecting -- *shared*
    physics -- and dropped for what it was only accumulating: 200 of the 286
    distinct parameter names were resolved by exactly one model, so "one place
    to compare bounds" bought nothing for them while forcing a ``core`` release
    for every model-local bound change. The catalogue is now three tiers:

    * **Tier A -- bound-set composition.** The ``names=[...]`` list (and any
      ``strip_prefix``) that says *which* parameters a model calibrates is
      model identity, not shared physics. Every in-tree model registers its own
      through ``register_model_bounds()`` from its package ``register()``.
    * **Tier B -- solo parameter definitions.** A ``ParameterInfo`` only one
      model ever resolves lives in that model's package and is contributed via
      ``register_model_bounds(params=...)``. It still lands in the same runtime
      catalogue, so ``get_registry().get_bounds(name)`` is unchanged.
    * **Tier C -- shared parameter definitions.** The 86 names resolved by two
      or more models stay HERE, unconditionally. These are shared formulations
      (Snow-17, SAC-SMA, TOPMODEL, CFE/NOAH-OWP) reused by NGEN and by the JAX
      plugin packages, plus every namespaced entry whose *served* name collides
      with another model's. Keeping them together is what caught the
      ``fuse_MBASE`` / Snow-17 ``MBASE`` collision fixed in #368 -- a model
      package must never duplicate one of these locally.

    Consequences worth knowing:
    - ``register_model_bounds()`` keeps the CENTRAL definition when a package
      contributes a name this module already defines, and now records the
      disagreement in :func:`bounds_registration_conflicts` (and logs it), so a
      new #368-style silent override is detectable rather than invisible.
    - The ``get_<model>_bounds()`` helpers for migrated models are retained as
      the stable public API but no longer hold data: they resolve whatever the
      owning package registered. With that package absent they raise ``KeyError``
      instead of serving a stale second copy -- deliberate, since a bound set
      core cannot see is worse than an explicit failure.
    - NGEN, its six ``NGEN_*`` sub-formulations, and the JAX-served SACSMA /
      SNOW17 / TOPMODEL sets have zero solo parameters: they are composed
      entirely from tier C and nothing of theirs moved.
    - HBV, HECHMS, SACSMA, SNOW17, TOPMODEL and XINANJIANG are served by
      EXTERNAL plugin packages (jhbv, jhechms, jsacsma, jsnow17, jtopmodel,
      jxaj) that predate this seam and still import ``get_<model>_bounds()``.
      Their definitions and compositions stay here as compatibility bounds
      until those packages adopt ``register_model_bounds()``.

Usage:
    from symfluence.core.calibration.parameters.parameter_bounds_registry import (
        ParameterBoundsRegistry, get_fuse_bounds, get_ngen_bounds
    )

    # Get all bounds for a model
    fuse_bounds = get_fuse_bounds()

    # Get specific parameter bounds
    registry = ParameterBoundsRegistry()
    mbase_bounds = registry.get_bounds('MBASE')

    # Get bounds for a list of parameters
    bounds = registry.get_bounds_for_params(['MBASE', 'MFMAX', 'maxsmc'])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParameterInfo:
    """Information about a hydrological parameter.

    Attributes:
        min: Minimum bound for the parameter.
        max: Maximum bound for the parameter.
        units: Physical units string (e.g., 'm/day', '°C').
        description: Human-readable description of the parameter.
        category: Parameter category ('snow', 'soil', 'baseflow', etc.).
        transform: Normalization transform type. 'linear' (default) maps
            uniformly between min and max. 'log' maps uniformly in
            log-space, which is appropriate for parameters spanning
            multiple orders of magnitude (e.g., conductivities, loss
            coefficients). Log transform requires min > 0.
    """
    min: float
    max: float
    units: str = ""
    description: str = ""
    category: str = "other"
    transform: str = "linear"


class ParameterBoundsRegistry:
    """
    Central registry for hydrological parameter bounds.

    Organizes parameters by category (snow, soil, baseflow, routing, ET)
    and provides lookups by parameter name or model type.

    The class-level dictionaries below are TIER C only: every entry here is
    resolved by two or more models, or is a namespaced entry whose served name
    collides with another model's (``fuse_MBASE`` vs Snow-17 ``MBASE``). Model
    -local (tier B) definitions live in the owning model package and reach this
    catalogue at runtime through :func:`register_model_bounds`, so
    ``get_bounds()`` still sees every parameter once plugin discovery has run.
    Do not add a single-model parameter here -- put it in the model package.
    """

    # ========================================================================
    # SNOW PARAMETERS
    # ========================================================================
    SNOW_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE snow parameters. MBASE/MFMAX/MFMIN are namespaced: Snow-17 uses
        # the same names with different bounds AND units (mm/°C/6hr vs
        # mm/(°C·day)), and the unnamespaced keys previously collided — the
        # SACSMA definitions silently overrode these in the merged catalogue,
        # so FUSE calibrated against Snow-17 melt bounds. get_fuse_bounds()
        # strips the prefix; bare MBASE/MFMAX/MFMIN retain Snow-17 semantics.
        'fuse_MBASE': ParameterInfo(-5.0, 5.0, '°C', 'Base melt temperature (FUSE)', 'snow'),
        'fuse_MFMAX': ParameterInfo(1.0, 10.0, 'mm/(°C·day)', 'Maximum melt factor (FUSE)', 'snow'),
        'fuse_MFMIN': ParameterInfo(0.5, 5.0, 'mm/(°C·day)', 'Minimum melt factor (FUSE)', 'snow'),
        # Shared: FUSE, SAC-SMA and Snow-17 all calibrate a rain/snow threshold
        # under this name with identical bounds (SACSMA_PARAMS repeats it).
        'PXTEMP': ParameterInfo(-2.0, 2.0, '°C', 'Rain-snow partition temperature', 'snow'),

        # NGEN snow parameters
        'rain_snow_thresh': ParameterInfo(-2.0, 2.0, '°C', 'Rain-snow temperature threshold', 'snow'),
    }

    # ========================================================================
    # SOIL PARAMETERS
    # ========================================================================
    SOIL_PARAMS: Dict[str, ParameterInfo] = {
        # NGEN CFE soil parameters
        # NOTE: Bounds tightened to reduce segfault rate during calibration.
        # Extreme parameter combinations (low satdk + high bb, high routing
        # coefficients) cause numerical instability in CFE's Fortran solver.
        'maxsmc': ParameterInfo(0.3, 0.6, 'fraction', 'Maximum soil moisture content', 'soil'),
        'wltsmc': ParameterInfo(0.02, 0.15, 'fraction', 'Wilting point soil moisture', 'soil'),
        'satdk': ParameterInfo(1e-6, 1e-5, 'm/s', 'Saturated hydraulic conductivity', 'soil', 'log'),
        'satpsi': ParameterInfo(0.05, 0.5, 'm', 'Saturated soil potential', 'soil'),
        'bb': ParameterInfo(3.0, 6.0, '-', 'Pore size distribution index', 'soil'),
        # Note: smcmax defined in NOAH section below with bounds (0.3, 0.6)
        'alpha_fc': ParameterInfo(0.3, 0.8, '-', 'Field capacity coefficient', 'soil'),
        'expon': ParameterInfo(1.0, 6.0, '-', 'Exponent parameter', 'soil'),
        'mult': ParameterInfo(500.0, 2000.0, 'mm', 'Multiplier parameter', 'soil'),
        'slop': ParameterInfo(0.01, 0.3, '-', 'TOPMODEL slope parameter', 'soil'),
        'soil_depth': ParameterInfo(1.0, 5.0, 'm', 'CFE soil depth', 'soil'),

        # NGEN NOAH-OWP soil parameters
        'slope': ParameterInfo(0.1, 1.0, '-', 'NOAH slope parameter', 'soil'),
        'dksat': ParameterInfo(1e-7, 1e-4, 'm/s', 'NOAH saturated conductivity', 'soil', 'log'),
        'psisat': ParameterInfo(0.01, 1.0, 'm', 'NOAH saturated potential', 'soil'),
        'bexp': ParameterInfo(2.0, 14.0, '-', 'NOAH b exponent', 'soil'),
        'smcmax': ParameterInfo(0.3, 0.6, 'm³/m³', 'NOAH maximum soil moisture (should match CFE)', 'soil'),
        'smcwlt': ParameterInfo(0.01, 0.3, 'm³/m³', 'NOAH wilting point', 'soil'),
        'smcref': ParameterInfo(0.1, 0.5, 'm³/m³', 'NOAH reference moisture', 'soil'),
        'noah_refdk': ParameterInfo(1e-7, 1e-3, 'm/s', 'NOAH reference conductivity', 'soil', 'log'),
        'noah_refkdt': ParameterInfo(0.5, 5.0, '-', 'NOAH reference KDT', 'soil'),
        'noah_czil': ParameterInfo(0.02, 0.2, '-', 'NOAH Zilitinkevich coefficient', 'soil'),
        'noah_z0': ParameterInfo(0.001, 1.0, 'm', 'NOAH roughness length', 'soil'),
        'noah_frzk': ParameterInfo(0.0, 10.0, '-', 'NOAH frozen ground parameter', 'soil'),
        'noah_salp': ParameterInfo(-2.0, 2.0, '-', 'NOAH shape parameter', 'soil'),
        'refkdt': ParameterInfo(0.5, 5.0, '-', 'Reference surface runoff parameter (expanded for infiltration control)', 'soil'),

        # ---- Namespaced entries whose SERVED name collides across models ----
        # RHESSys serves 'soil_depth' (prefix stripped) and CFE owns the bare
        # key, deliberately tightened to 1-5 m to avoid solver segfaults; these
        # two once collided and RHESSys' 2-15 m silently won (#368). Both live
        # here so the collision stays visible in one file.
        'rhessys_soil_depth': ParameterInfo(2.0, 15.0, 'm', 'Total soil depth (RHESSys)', 'soil'),
        # 'm' means Ksat lateral decay to RHESSys and transmissivity decay to
        # TOPMODEL (see 'topmodel_m'); different physics, same served name.
        'm': ParameterInfo(0.5, 5.0, '-', 'Lateral decay of Ksat with depth (RHESSys)', 'soil'),
        # GSFLOW serves 'K' (prefix stripped); Xinanjiang serves 'K' from
        # 'xaj_K'. Namespaced on both sides so neither can override the other.
        # Matches what GSFLOW calibration actually searches. This entry read
        # 0.001-100 m/d (log) while GSFLOWParameterManager searched 0.1-5000
        # (linear) from a package-local dict; the catalogue value was inert --
        # nothing in-tree or in any plugin calls get_gsflow_bounds() -- which is
        # precisely how it was free to drift. The in-use range is the documented
        # one (MODFLOW-NWT UPW; Iceland basalt 1e2-1e4 m/d), and the old ceiling
        # of 100 m/d excluded most of it.
        'gsflow_K': ParameterInfo(0.1, 5000.0, 'm/d', 'Hydraulic conductivity (GSFLOW MODFLOW-NWT UPW); Iceland basalt 1e2-1e4 m/d', 'soil'),
    }

    # ========================================================================
    # BASEFLOW / GROUNDWATER PARAMETERS
    # ========================================================================
    BASEFLOW_PARAMS: Dict[str, ParameterInfo] = {
        # NGEN CFE groundwater parameters
        'Cgw': ParameterInfo(1e-5, 0.01, 'm/h', 'Groundwater coefficient', 'baseflow', 'log'),
        'max_gw_storage': ParameterInfo(0.05, 1.0, 'm', 'Maximum groundwater storage', 'baseflow'),

        # ---- Namespaced entries whose SERVED name collides across models ----
        # 'PWR' is MESH's bare baseflow exponent; WATFLOOD serves the same name
        # from 'watflood_PWR' with different bounds. Both stay central.
        'PWR': ParameterInfo(1.0, 5.0, '-', 'Baseflow power exponent (MESH)', 'baseflow'),
        'watflood_PWR': ParameterInfo(0.5, 4.0, '-', 'Power on lower zone function (WATFLOOD)', 'baseflow'),
    }

    # ========================================================================
    # ROUTING PARAMETERS
    # ========================================================================
    ROUTING_PARAMS: Dict[str, ParameterInfo] = {
        # NGEN CFE routing parameters
        'K_lf': ParameterInfo(0.01, 0.5, '1/h', 'Lateral flow coefficient', 'routing'),
        'K_nash': ParameterInfo(0.01, 0.5, '1/h', 'Nash cascade coefficient', 'routing'),
        'Klf': ParameterInfo(0.01, 0.5, '1/h', 'Lateral flow coefficient (alias)', 'routing'),
        'Kn': ParameterInfo(0.01, 0.5, '1/h', 'Nash cascade coefficient (alias)', 'routing'),

        # ---- Namespaced entries whose SERVED name collides across models ----
        # 'R2N' is MESH's bare overland routing roughness; WATFLOOD serves the
        # same name from 'watflood_R2N' with different bounds.
        'R2N': ParameterInfo(0.01, 0.5, '-', 'Overland routing roughness, Manning n (MESH)', 'routing'),
        'watflood_R2N': ParameterInfo(0.01, 0.30, '-', 'Channel Manning roughness multiplier (WATFLOOD)', 'routing'),
    }

    # NOTE: dRoute routing parameter bounds now live in the external ``droute`` package
    # (droute.calibration.bounds) — the model package owns its own bounds (JAX-model pattern).
    # mizuRoute's six routing parameters followed the same pattern in July 2026:
    # they are solo (only MIZUROUTE resolves them) and now live in
    # ``symfluence.models.mizuroute.parameter_bounds``.

    # ========================================================================
    # EVAPOTRANSPIRATION PARAMETERS
    # ========================================================================
    ET_PARAMS: Dict[str, ParameterInfo] = {
        # NGEN PET parameters (BMI config file key names)
        'vegetation_height_m': ParameterInfo(0.1, 30.0, 'm', 'Vegetation height', 'et'),
        'zero_plane_displacement_height_m': ParameterInfo(0.0, 20.0, 'm', 'Zero plane displacement height', 'et'),
        'momentum_transfer_roughness_length': ParameterInfo(0.001, 1.0, 'm', 'Momentum transfer roughness length', 'et'),
        'heat_transfer_roughness_length_m': ParameterInfo(0.0001, 0.1, 'm', 'Heat transfer roughness length', 'et'),
        'surface_shortwave_albedo': ParameterInfo(0.05, 0.5, '-', 'Surface shortwave albedo', 'et'),
        'surface_longwave_emissivity': ParameterInfo(0.9, 1.0, '-', 'Surface longwave emissivity', 'et'),
        'wind_speed_measurement_height_m': ParameterInfo(2.0, 10.0, 'm', 'Wind measurement height', 'et'),
        'humidity_measurement_height_m': ParameterInfo(2.0, 10.0, 'm', 'Humidity measurement height', 'et'),

        # NGEN PET parameters (legacy/alias names)
        'pet_albedo': ParameterInfo(0.05, 0.5, '-', 'PET albedo', 'et'),
        'pet_z0_mom': ParameterInfo(0.001, 1.0, 'm', 'PET momentum roughness', 'et'),
        'pet_z0_heat': ParameterInfo(0.0001, 0.1, 'm', 'PET heat roughness', 'et'),
        'pet_veg_h': ParameterInfo(0.1, 30.0, 'm', 'PET vegetation height', 'et'),
        'pet_d0': ParameterInfo(0.0, 20.0, 'm', 'PET zero plane displacement', 'et'),

        # NGEN NOAH reference height
        'ZREF': ParameterInfo(2.0, 10.0, 'm', 'Reference height for measurements', 'et'),
    }

    # ========================================================================
    # DEPTH PARAMETERS (SUMMA-specific)
    # ========================================================================
    # Kept central by decision, not by omission. 'DEPTH' is not a model: it is
    # SUMMA's soil-depth calibration facet — the only consumers are
    # models/summa/calibration/parameter_manager.py and
    # models/summa/calibration/worker_impl/parameter_application.py, which
    # rescale SUMMA's soil layer thicknesses. The three names are solo, so
    # tier B would move them, but their owner is the SUMMA package (which has
    # no bound set of its own — SUMMA bounds come from localParamInfo.txt), not
    # any of the packages migrated in this pass. Moving them belongs with the
    # SUMMA migration; parking them in another model's package would misattribute
    # ownership. Tracked as a follow-up.
    DEPTH_PARAMS: Dict[str, ParameterInfo] = {
        'total_mult': ParameterInfo(0.1, 5.0, '-', 'Total soil depth multiplier', 'depth'),
        'total_soil_depth_multiplier': ParameterInfo(0.1, 5.0, '-', 'Total soil depth multiplier (alias)', 'depth'),
        'shape_factor': ParameterInfo(0.1, 3.0, '-', 'Soil depth shape factor', 'depth'),
    }

    # ------------------------------------------------------------------
    # MIGRATED (July 2026, tier A+B): the per-model bound sets below moved
    # to the packages that own them. Each package registers its solo
    # ParameterInfo definitions and its bound-set composition from its
    # register() via register_model_bounds(), so the runtime catalogue is
    # unchanged once plugin discovery has run:
    #
    #   HYPE     -> symfluence.models.hype.parameter_bounds      (35 params)
    #   MESH     -> symfluence.models.mesh.parameter_bounds      (27 params)
    #   RHESSYS  -> symfluence.models.rhessys.parameter_bounds   (26 params)
    #   GR       -> symfluence.models.gr.parameter_bounds         (8 params)
    #   VIC      -> symfluence.models.vic.parameter_bounds       (12 params)
    #
    # Their SHARED names stayed behind (tier C): HYPE's "lp" (also HBV's),
    # MESH's "PWR"/"R2N" (also WATFLOOD's), RHESSys' "m" and
    # "rhessys_soil_depth" (also TOPMODEL's / CFE's).
    # ------------------------------------------------------------------

    # ========================================================================
    # SAC-SMA + SNOW-17 PARAMETERS
    # ========================================================================
    SACSMA_PARAMS: Dict[str, ParameterInfo] = {
        # Snow-17 parameters
        'SCF': ParameterInfo(0.7, 1.4, '-', 'Snowfall correction factor', 'snow'),
        'PXTEMP': ParameterInfo(-2.0, 2.0, '°C', 'Rain/snow threshold temperature', 'snow'),
        'MFMAX': ParameterInfo(0.5, 2.0, 'mm/°C/6hr', 'Max melt factor (Jun 21)', 'snow'),
        'MFMIN': ParameterInfo(0.05, 0.6, 'mm/°C/6hr', 'Min melt factor (Dec 21)', 'snow'),
        'NMF': ParameterInfo(0.05, 0.5, 'mm/°C/6hr', 'Negative melt factor', 'snow'),
        'MBASE': ParameterInfo(0.0, 1.0, '°C', 'Base melt temperature', 'snow'),
        'TIPM': ParameterInfo(0.01, 1.0, '-', 'Antecedent temperature index weight', 'snow'),
        'UADJ': ParameterInfo(0.01, 0.2, 'mm/mb/6hr', 'Rain-on-snow wind function', 'snow'),
        'PLWHC': ParameterInfo(0.01, 0.3, '-', 'Liquid water holding capacity', 'snow'),
        'DAYGM': ParameterInfo(0.0, 0.3, 'mm/day', 'Daily ground melt', 'snow'),

        # SAC-SMA upper zone parameters
        'UZTWM': ParameterInfo(10.0, 150.0, 'mm', 'Upper zone tension water max', 'soil'),
        'UZFWM': ParameterInfo(1.0, 150.0, 'mm', 'Upper zone free water max', 'soil'),
        'UZK': ParameterInfo(0.15, 0.75, '1/day', 'Upper zone lateral depletion', 'soil'),

        # SAC-SMA lower zone parameters
        'LZTWM': ParameterInfo(1.0, 500.0, 'mm', 'Lower zone tension water max', 'soil'),
        'LZFPM': ParameterInfo(1.0, 1000.0, 'mm', 'Lower zone primary free water max', 'baseflow', 'log'),
        'LZFSM': ParameterInfo(1.0, 1000.0, 'mm', 'Lower zone supplemental free water max', 'baseflow', 'log'),
        'LZPK': ParameterInfo(0.001, 0.05, '1/day', 'Primary baseflow depletion', 'baseflow', 'log'),
        'LZSK': ParameterInfo(0.01, 0.25, '1/day', 'Supplemental baseflow depletion', 'baseflow', 'log'),

        # SAC-SMA percolation parameters
        'ZPERC': ParameterInfo(1.0, 350.0, '-', 'Maximum percolation rate scaling', 'soil', 'log'),
        'REXP': ParameterInfo(1.0, 5.0, '-', 'Percolation curve exponent', 'soil'),
        'PFREE': ParameterInfo(0.0, 0.8, '-', 'Fraction percolation to free water', 'soil'),

        # SAC-SMA area fractions
        'PCTIM': ParameterInfo(0.0, 0.1, '-', 'Permanent impervious area fraction', 'soil'),
        'ADIMP': ParameterInfo(0.0, 0.4, '-', 'Additional impervious area fraction', 'soil'),
        'RIVA': ParameterInfo(0.0, 0.2, '-', 'Riparian vegetation ET fraction', 'et'),
        'SIDE': ParameterInfo(0.0, 0.5, '-', 'Deep recharge fraction', 'baseflow'),
        'RSERV': ParameterInfo(0.0, 0.4, '-', 'Lower zone free water reserve fraction', 'baseflow'),
    }

    # ========================================================================
    # HBV-96 PARAMETERS
    # ========================================================================
    HBV_PARAMS: Dict[str, ParameterInfo] = {
        # Snow parameters
        'tt': ParameterInfo(-3.0, 3.0, '°C', 'Threshold temperature for snow/rain', 'snow'),
        'cfmax': ParameterInfo(1.0, 10.0, 'mm/°C/day', 'Degree-day factor for snowmelt', 'snow'),
        'sfcf': ParameterInfo(0.5, 1.5, '-', 'Snowfall correction factor', 'snow'),
        'cfr': ParameterInfo(0.0, 0.1, '-', 'Refreezing coefficient', 'snow'),
        'cwh': ParameterInfo(0.0, 0.2, '-', 'Snow water holding capacity', 'snow'),

        # Soil parameters
        'fc': ParameterInfo(50.0, 700.0, 'mm', 'Field capacity / max soil moisture', 'soil'),
        'lp': ParameterInfo(0.3, 1.0, '-', 'ET reduction threshold (fraction of FC)', 'soil'),
        'beta': ParameterInfo(1.0, 6.0, '-', 'Shape coefficient for soil routine', 'soil'),

        # Response/baseflow parameters
        'k0': ParameterInfo(0.05, 0.5, '1/day', 'Fast recession coefficient', 'baseflow'),
        'k1': ParameterInfo(0.01, 0.3, '1/day', 'Slow recession coefficient', 'baseflow'),
        'k2': ParameterInfo(0.0001, 0.1, '1/day', 'Baseflow recession coefficient', 'baseflow'),
        'uzl': ParameterInfo(0.0, 100.0, 'mm', 'Upper zone threshold for fast flow', 'baseflow'),
        'perc': ParameterInfo(0.0, 20.0, 'mm/day', 'Maximum percolation rate', 'baseflow'),

        # Routing parameters
        'maxbas': ParameterInfo(1.0, 7.0, 'days', 'Triangular routing function length', 'routing'),

        # Numerical parameters
        'smoothing': ParameterInfo(1.0, 50.0, '-', 'Smoothing factor for thresholds', 'numerical'),
    }

    # ========================================================================
    # HEC-HMS PARAMETERS
    # ========================================================================
    HECHMS_PARAMS: Dict[str, ParameterInfo] = {
        # Snow (ATI Temperature Index)
        'px_temp': ParameterInfo(-2.0, 4.0, '°C', 'Rain/snow partition temperature', 'snow'),
        'base_temp': ParameterInfo(-3.0, 3.0, '°C', 'Base temperature for snowmelt', 'snow'),
        'ati_meltrate_coeff': ParameterInfo(0.5, 1.5, '-', 'ATI meltrate coefficient', 'snow'),
        'meltrate_max': ParameterInfo(2.0, 10.0, 'mm/°C/day', 'Maximum melt rate', 'snow'),
        'meltrate_min': ParameterInfo(0.0, 3.0, 'mm/°C/day', 'Minimum melt rate', 'snow'),
        'cold_limit': ParameterInfo(0.0, 50.0, 'mm', 'Cold content limit', 'snow'),
        'ati_cold_rate_coeff': ParameterInfo(0.0, 0.3, '-', 'ATI cold rate coefficient', 'snow'),
        'water_capacity': ParameterInfo(0.0, 0.3, '-', 'Snowpack liquid water holding capacity', 'snow'),

        # Loss (SCS Curve Number)
        'cn': ParameterInfo(30.0, 98.0, '-', 'SCS Curve Number', 'soil'),
        'initial_abstraction_ratio': ParameterInfo(0.05, 0.3, '-', 'Initial abstraction ratio Ia/S', 'soil'),

        # Transform (Clark Unit Hydrograph)
        'tc': ParameterInfo(0.5, 20.0, 'days', 'Time of concentration', 'routing'),
        'r_coeff': ParameterInfo(0.5, 20.0, 'days', 'Clark storage coefficient', 'routing'),

        # Baseflow (Linear Reservoir)
        'gw_storage_coeff': ParameterInfo(1.0, 100.0, 'days', 'GW storage coefficient', 'baseflow'),
        'deep_perc_fraction': ParameterInfo(0.0, 0.5, '-', 'Deep percolation fraction', 'baseflow'),
    }

    # ========================================================================
    # TOPMODEL PARAMETERS (Beven & Kirkby 1979)
    # ========================================================================
    TOPMODEL_PARAMS: Dict[str, ParameterInfo] = {
        # Subsurface / transmissivity
        'topmodel_m': ParameterInfo(0.001, 0.3, 'm', 'Transmissivity decay parameter', 'soil'),
        'topmodel_lnTe': ParameterInfo(-7.0, 10.0, 'ln(m²/h)', 'Effective log transmissivity', 'baseflow'),
        'topmodel_Srmax': ParameterInfo(0.005, 0.5, 'm', 'Max root zone storage', 'soil'),
        'topmodel_Sr0': ParameterInfo(0.0, 0.1, 'm', 'Initial root zone deficit', 'soil'),
        'topmodel_td': ParameterInfo(0.1, 50.0, 'h/m', 'Unsaturated zone time delay', 'soil'),

        # Routing
        'topmodel_k_route': ParameterInfo(1.0, 200.0, 'h', 'Routing reservoir coefficient', 'routing'),

        # Snow (degree-day)
        'topmodel_DDF': ParameterInfo(0.5, 10.0, 'mm/°C/day', 'Degree-day melt factor', 'snow'),
        'topmodel_T_melt': ParameterInfo(-2.0, 3.0, '°C', 'Melt threshold temperature', 'snow'),
        'topmodel_T_snow': ParameterInfo(-2.0, 3.0, '°C', 'Snow/rain threshold temperature', 'snow'),

        # TI distribution
        'topmodel_ti_std': ParameterInfo(1.0, 10.0, '-', 'TI distribution spread', 'other'),

        # Initial conditions
        'topmodel_S0': ParameterInfo(0.0, 2.0, 'm', 'Initial mean deficit', 'other'),
    }

    # ========================================================================
    # XINANJIANG (XAJ) PARAMETERS
    # ========================================================================
    XINANJIANG_PARAMS: Dict[str, ParameterInfo] = {
        # Generation parameters
        'xaj_K': ParameterInfo(0.1, 1.5, '-', 'PET correction factor (>1 allows sublimation compensation)', 'et'),
        'xaj_B': ParameterInfo(0.1, 2.0, '-', 'Tension water capacity curve exponent (Zhao 1992)', 'soil'),
        'xaj_IM': ParameterInfo(0.01, 0.1, '-', 'Impervious area fraction', 'soil'),
        'xaj_UM': ParameterInfo(5.0, 50.0, 'mm', 'Upper layer tension water capacity', 'soil'),
        'xaj_LM': ParameterInfo(50.0, 120.0, 'mm', 'Lower layer tension water capacity', 'soil'),
        'xaj_DM': ParameterInfo(50.0, 200.0, 'mm', 'Deep layer tension water capacity', 'soil'),
        'xaj_C': ParameterInfo(0.0, 0.2, '-', 'Deep layer ET coefficient', 'et'),

        # Source separation parameters
        'xaj_SM': ParameterInfo(1.0, 200.0, 'mm', 'Free water capacity', 'soil', 'log'),
        'xaj_EX': ParameterInfo(0.5, 2.0, '-', 'Free water capacity curve exponent', 'soil'),
        'xaj_KI': ParameterInfo(0.0, 0.7, '-', 'Interflow outflow coefficient', 'baseflow'),
        'xaj_KG': ParameterInfo(0.0, 0.7, '-', 'Groundwater outflow coefficient', 'baseflow'),

        # Routing parameters (CS and L excluded — not used in lumped formulation)
        'xaj_CI': ParameterInfo(0.0, 0.9, '-', 'Interflow recession constant', 'routing'),
        'xaj_CG': ParameterInfo(0.98, 0.998, '-', 'Groundwater recession constant', 'routing'),
    }

    # ------------------------------------------------------------------
    # MIGRATED (July 2026, tier A+B):
    #   GSFLOW   -> symfluence.models.gsflow.parameter_bounds     (9 params
    #               composing the bound set, plus 5 catalogue-only entries)
    #   WATFLOOD -> symfluence.models.watflood.parameter_bounds  (14 params)
    #   IGNACIO  -> symfluence.models.ignacio.parameter_bounds    (6 params)
    #   FUSE     -> symfluence.models.fuse.parameter_bounds      (13 params)
    #   MIZUROUTE-> symfluence.models.mizuroute.parameter_bounds  (6 params)
    #   NOAHMP   -> symfluence.models.noahmp.parameter_bounds     (1 param)
    #
    # Shared names stayed above: "gsflow_K" (served "K", also Xinanjiang's),
    # "watflood_PWR"/"watflood_R2N" (served "PWR"/"R2N", also MESH's),
    # FUSE's "fuse_MBASE"/"fuse_MFMAX"/"fuse_MFMIN"/"PXTEMP", and all 11 of
    # NOAH-MP's NOAH-OWP soil/snow names.
    # ------------------------------------------------------------------

    #: Tier-C category dictionaries, in merge order. Package-contributed
    #: (tier-B) definitions are merged FIRST so a central definition always
    #: wins a name clash — the single-source-of-truth rule for shared physics.
    CATEGORY_ATTRS = (
        'SNOW_PARAMS', 'SOIL_PARAMS', 'BASEFLOW_PARAMS', 'ROUTING_PARAMS',
        'ET_PARAMS', 'DEPTH_PARAMS', 'HBV_PARAMS', 'HECHMS_PARAMS',
        'TOPMODEL_PARAMS', 'SACSMA_PARAMS', 'XINANJIANG_PARAMS',
    )

    def __init__(self):
        """Initialize registry with all parameter categories combined."""
        self._all_params: Dict[str, ParameterInfo] = {}
        self._all_params.update(_EXTENSION_PARAMS)
        for attr in self.CATEGORY_ATTRS:
            self._all_params.update(getattr(self, attr))

    def get_bounds(self, param_name: str) -> Optional[Dict]:
        """
        Get bounds for a single parameter.

        Args:
            param_name: Parameter name

        Returns:
            Dictionary with 'min', 'max', and 'transform' keys, or None if not found
        """
        info = self._all_params.get(param_name)
        if info:
            return {'min': info.min, 'max': info.max, 'transform': info.transform}
        return None

    def get_info(self, param_name: str) -> Optional[ParameterInfo]:
        """
        Get full parameter info including description and units.

        Args:
            param_name: Parameter name

        Returns:
            ParameterInfo object or None if not found
        """
        return self._all_params.get(param_name)

    def get_bounds_for_params(self, param_names: List[str]) -> Dict[str, Dict]:
        """
        Get bounds for multiple parameters.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping param_name -> {'min': float, 'max': float, 'transform': str}
        """
        bounds = {}
        for name in param_names:
            b = self.get_bounds(name)
            if b:
                bounds[name] = b
        return bounds

    def get_params_by_category(self, category: str) -> Dict[str, Dict]:
        """
        Get all parameter bounds for a category.

        Args:
            category: One of 'snow', 'soil', 'baseflow', 'routing', 'et', 'depth'

        Returns:
            Dictionary of parameter bounds
        """
        return {
            name: {'min': info.min, 'max': info.max, 'transform': info.transform}
            for name, info in self._all_params.items()
            if info.category == category
        }

    @property
    def all_param_names(self) -> List[str]:
        """Get list of all registered parameter names."""
        return list(self._all_params.keys())


# ============================================================================
# CONVENIENCE FUNCTIONS FOR MODEL-SPECIFIC BOUNDS
# ============================================================================

# Singleton registry instance
_registry: Optional[ParameterBoundsRegistry] = None

# Extension seam (service decomposition): parameter definitions contributed by
# model packages at registration time, and per-model catalogue name lists.
_EXTENSION_PARAMS: Dict[str, ParameterInfo] = {}
_MODEL_PARAM_NAMES: Dict[str, List[str]] = {}
_MODEL_NAME_PREFIXES: Dict[str, str] = {}

# Contributions that disagreed with an already-known definition of the same
# name. The central definition always wins (see register_model_bounds), so a
# disagreement is silent by construction — this list is what makes it visible.
_REGISTRATION_CONFLICTS: List[str] = []

_logger = logging.getLogger(__name__)


def get_registry() -> ParameterBoundsRegistry:
    """Get singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ParameterBoundsRegistry()
    return _registry


def _central_definition(name: str) -> Optional[ParameterInfo]:
    """The definition a new contribution would have to defer to, if any.

    Mirrors the merge order in :meth:`ParameterBoundsRegistry.__init__`: a
    tier-C (central) definition outranks anything a package contributed.
    """
    for attr in ParameterBoundsRegistry.CATEGORY_ATTRS:
        info = getattr(ParameterBoundsRegistry, attr).get(name)
        if info is not None:
            return info
    return _EXTENSION_PARAMS.get(name)


def bounds_registration_conflicts() -> List[str]:
    """Names a model package redefined with bounds that differ from the winner.

    Empty in a healthy install. A non-empty entry means some package's
    ``register_model_bounds(params=...)`` is being silently ignored — the same
    failure mode as the ``fuse_MBASE`` / Snow-17 ``MBASE`` collision (#368),
    but now across package boundaries rather than within this file.
    """
    return list(_REGISTRATION_CONFLICTS)


def register_model_bounds(
    model: str,
    params: Optional[Dict[str, ParameterInfo]] = None,
    names: Optional[List[str]] = None,
    strip_prefix: str = "",
) -> None:
    """Register a model's calibration bounds with the central registry.

    The extension seam for model packages (in-tree or external): call this from
    the package's ``register()`` so the model's bound set is servable through
    :func:`get_model_bounds` without editing core. In-tree models use the same
    path — ``params`` carries their tier-B (solo) definitions and ``names`` /
    ``strip_prefix`` carry their tier-A composition.

    Shared (tier-C) parameters must NOT be passed in ``params``: compose them
    by listing their catalogue names in ``names``. A contribution that disagrees
    with an existing definition is ignored (central wins) and recorded in
    :func:`bounds_registration_conflicts`.

    Args:
        model: Model key, case-insensitive (e.g. ``'FUSE'``, ``'MYMODEL'``).
        params: New :class:`ParameterInfo` definitions to add to the catalogue.
            Names already present in the built-in catalogue keep their central
            definition (single source of truth for shared parameters).
        names: Catalogue parameter names composing this model's bound set.
            Defaults to the keys of ``params`` when omitted.
        strip_prefix: Prefix stripped from returned keys (matches the
            convention of e.g. ``get_gsflow_bounds``/``get_watflood_bounds``,
            whose catalogue entries are namespaced but whose parameter
            managers use unprefixed names).
    """
    key = model.upper()
    if params:
        for name, info in params.items():
            existing = _central_definition(name)
            if existing is not None and (
                (existing.min, existing.max, existing.transform)
                != (info.min, info.max, info.transform)
            ):
                message = (
                    f"{key} redefined shared parameter '{name}' as "
                    f"({info.min}, {info.max}, {info.transform}); the existing "
                    f"definition ({existing.min}, {existing.max}, "
                    f"{existing.transform}) wins. Namespace the entry "
                    f"(e.g. '{key.lower()}_{name}' + strip_prefix) instead."
                )
                _REGISTRATION_CONFLICTS.append(message)
                _logger.warning("parameter bounds conflict: %s", message)
            _EXTENSION_PARAMS.setdefault(name, info)
        if _registry is not None:
            for name, info in params.items():
                _registry._all_params.setdefault(name, info)
    bound_names = list(names) if names is not None else list(params or {})
    if bound_names:
        _MODEL_PARAM_NAMES[key] = bound_names
    if strip_prefix:
        _MODEL_NAME_PREFIXES[key] = strip_prefix


def _registered_bound_set(key: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Resolve a registered composition, or None if the model never registered."""
    names = _MODEL_PARAM_NAMES.get(key)
    if names is None:
        return None
    bounds = get_registry().get_bounds_for_params(names)
    prefix = _MODEL_NAME_PREFIXES.get(key, "")
    if prefix:
        bounds = {k[len(prefix):] if k.startswith(prefix) else k: v
                  for k, v in bounds.items()}
    return bounds


def _owned_by_package(model: str, package: str) -> Dict[str, Dict[str, float]]:
    """Serve a bound set whose definition lives in *package*.

    Used by the retained ``get_<model>_bounds()`` helpers for models migrated
    to tier A/B. Core holds no second copy of these values on purpose, so an
    unregistered model is an explicit failure rather than stale bounds.
    """
    bounds = _registered_bound_set(model)
    if bounds is None:
        raise KeyError(
            f"No parameter bounds registered for model '{model}'. Its bounds "
            f"are owned by the '{package}' package, which contributes them via "
            "register_model_bounds() when plugin discovery runs (on "
            "`import symfluence`). Install/enable the package, or check the "
            "plugin-discovery log for a failed registration."
        )
    return bounds


def get_model_bounds(model: str) -> Dict[str, Dict[str, float]]:
    """Get the bound set for a model, registered or built-in.

    Resolution order: bounds registered via :func:`register_model_bounds`,
    then the built-in ``get_<model>_bounds`` functions.

    Raises:
        KeyError: If the model has neither registered nor built-in bounds.
    """
    key = model.upper()
    bounds = _registered_bound_set(key)
    if bounds is not None:
        return bounds
    getter = _BUILTIN_MODEL_BOUNDS.get(key)
    if getter is not None:
        return getter()
    raise KeyError(
        f"No parameter bounds registered for model '{model}'. Model packages "
        "register bounds via register_model_bounds() at plugin-registration time."
    )


def registered_bound_models() -> List[str]:
    """All model keys servable by :func:`get_model_bounds`."""
    return sorted(set(_MODEL_PARAM_NAMES) | set(_BUILTIN_MODEL_BOUNDS))


def get_fuse_bounds() -> Dict[str, Dict[str, float]]:
    """FUSE bound set (17 parameters).

    13 solo definitions live in ``symfluence.models.fuse.parameter_bounds``;
    the four shared ones (``fuse_MBASE``/``fuse_MFMAX``/``fuse_MFMIN`` and
    ``PXTEMP``) stay in this module because their SERVED names collide with
    Snow-17's — the #368 defect. The ``fuse_`` prefix is stripped on the way out.

    Owned by ``symfluence.models.fuse`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('FUSE', 'symfluence.models.fuse')


def get_ngen_cfe_bounds() -> Dict[str, Dict[str, float]]:
    """CFE module bound set (18 parameters, all tier C).

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN_CFE', 'symfluence.models.ngen')


def get_ngen_noah_bounds() -> Dict[str, Dict[str, float]]:
    """NOAH-OWP module bound set (16 parameters, all tier C).

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN_NOAH', 'symfluence.models.ngen')


def get_ngen_pet_bounds() -> Dict[str, Dict[str, float]]:
    """PET module bound set (13 parameters, all tier C).

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN_PET', 'symfluence.models.ngen')


def get_ngen_topmodel_bounds() -> Dict[str, Dict[str, float]]:
    """TOPMODEL module bound set for NGEN (11 parameters, all tier C).

    Composed from the same ``topmodel_*`` catalogue entries the standalone
    TOPMODEL set uses, with the prefix stripped.

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN_TOPMODEL', 'symfluence.models.ngen')


def get_ngen_sacsma_bounds() -> Dict[str, Dict[str, float]]:
    """SAC-SMA module bound set for NGEN (16 parameters, all tier C).

    Soil-moisture-accounting parameters only — Snow-17 is a separate module.

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN_SACSMA', 'symfluence.models.ngen')


def get_ngen_snow17_bounds() -> Dict[str, Dict[str, float]]:
    """Snow-17 module bound set for NGEN (10 parameters, all tier C).

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN_SNOW17', 'symfluence.models.ngen')


def get_ngen_bounds() -> Dict[str, Dict[str, float]]:
    """Full NGEN bound set: CFE + NOAH + PET + TOPMODEL + SAC-SMA + Snow-17.

    82 parameters, every one of them tier C — NGEN has no solo parameter, so
    nothing of its own moved; only the composition did.

    Owned by ``symfluence.models.ngen`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NGEN', 'symfluence.models.ngen')


def get_mizuroute_bounds() -> Dict[str, Dict[str, float]]:
    """mizuRoute bound set (6 parameters, all solo).

    Owned by ``symfluence.models.mizuroute`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('MIZUROUTE', 'symfluence.models.mizuroute')


def get_depth_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get soil depth calibration parameter bounds.

    Returns:
        Dictionary mapping param_name -> {'min': float, 'max': float}

    NOT a model: this is SUMMA's soil-depth calibration facet, consumed only by
    ``models/summa/calibration``. Its three parameters are solo, but their owner
    is the SUMMA package (which has no bound set of its own — SUMMA reads
    localParamInfo.txt), so they stay here until SUMMA itself migrates.
    """
    depth_params = ['total_mult', 'total_soil_depth_multiplier', 'shape_factor']
    return get_registry().get_bounds_for_params(depth_params)


def get_hype_bounds() -> Dict[str, Dict[str, float]]:
    """HYPE bound set (36 parameters).

    35 solo definitions moved to the package; ``lp`` stays central (HBV
    calibrates the same name).

    Owned by ``symfluence.models.hype`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('HYPE', 'symfluence.models.hype')


def get_mesh_bounds() -> Dict[str, Dict[str, float]]:
    """MESH bound set (29 parameters).

    27 solo definitions moved to the package; ``PWR`` and ``R2N`` stay central
    (WATFLOOD serves the same two names from namespaced entries).

    Owned by ``symfluence.models.mesh`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('MESH', 'symfluence.models.mesh')


def get_gr_bounds() -> Dict[str, Dict[str, float]]:
    """GR4J + CemaNeige bound set (11 parameters incl. forcing corrections and ice DDF, all solo).

    Owned by ``symfluence.models.gr`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('GR', 'symfluence.models.gr')


def get_rhessys_bounds() -> Dict[str, Dict[str, float]]:
    """RHESSys bound set (28 parameters).

    26 solo definitions moved to the package; ``m`` (TOPMODEL serves the same
    name) and ``rhessys_soil_depth`` (CFE owns the bare ``soil_depth``) stay
    central. The ``rhessys_`` prefix is stripped on the way out.

    Owned by ``symfluence.models.rhessys`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('RHESSYS', 'symfluence.models.rhessys')


def get_hbv_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all HBV-96 parameter bounds.

    Returns:
        Dictionary mapping HBV param_name -> {'min': float, 'max': float}

    COMPATIBILITY BOUNDS (tier C by necessity, not by physics). HBV is served
    by the external ``jhbv`` plugin package, which lives in its own repository
    and predates the ``register_model_bounds()`` seam — it still imports this
    helper directly. Its definitions and composition therefore stay in core even
    though 14 of its 15 parameters are solo (only ``lp`` is shared, with HYPE). Upstream issue: have ``jhbv`` register its own bounds, then
    delete this entry.
    """
    hbv_params = [
        # Snow
        'tt', 'cfmax', 'sfcf', 'cfr', 'cwh',
        # Soil
        'fc', 'lp', 'beta',
        # Response/baseflow
        'k0', 'k1', 'k2', 'uzl', 'perc',
        # Routing
        'maxbas',
        # Numerical
        'smoothing',
    ]
    return get_registry().get_bounds_for_params(hbv_params)


def get_hechms_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all HEC-HMS parameter bounds.

    Returns:
        Dictionary mapping HEC-HMS param_name -> {'min': float, 'max': float}

    COMPATIBILITY BOUNDS (tier C by necessity, not by physics). HECHMS is served
    by the external ``jhechms`` plugin package, which lives in its own repository
    and predates the ``register_model_bounds()`` seam — it still imports this
    helper directly. Its definitions and composition therefore stay in core even
    though all 14 of its parameters are solo. Upstream issue: have ``jhechms`` register its own bounds, then
    delete this entry.
    """
    hechms_params = [
        # Snow (ATI)
        'px_temp', 'base_temp', 'ati_meltrate_coeff', 'meltrate_max', 'meltrate_min',
        'cold_limit', 'ati_cold_rate_coeff', 'water_capacity',
        # Loss (SCS-CN)
        'cn', 'initial_abstraction_ratio',
        # Transform (Clark UH)
        'tc', 'r_coeff',
        # Baseflow (Linear Reservoir)
        'gw_storage_coeff', 'deep_perc_fraction',
    ]
    return get_registry().get_bounds_for_params(hechms_params)


def get_topmodel_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all TOPMODEL parameter bounds.

    Returns:
        Dictionary mapping TOPMODEL param_name -> {'min': float, 'max': float}
        Keys use unprefixed names (m, lnTe, ...) matching TOPMODEL parameter conventions.

    COMPATIBILITY BOUNDS (tier C by necessity, not by physics). TOPMODEL is served
    by the external ``jtopmodel`` plugin package, which lives in its own repository
    and predates the ``register_model_bounds()`` seam — it still imports this
    helper directly. Its definitions and composition therefore stay in core even
    though every one of its parameters is shared with NGEN_TOPMODEL, so nothing would move anyway. Upstream issue: have ``jtopmodel`` register its own bounds, then
    delete this entry.
    """
    topmodel_params = [
        'topmodel_m', 'topmodel_lnTe', 'topmodel_Srmax', 'topmodel_Sr0', 'topmodel_td',
        'topmodel_k_route',
        'topmodel_DDF', 'topmodel_T_melt', 'topmodel_T_snow',
        'topmodel_ti_std', 'topmodel_S0',
    ]
    prefixed = get_registry().get_bounds_for_params(topmodel_params)
    # Strip topmodel_ prefix so keys match parameter manager conventions
    return {k.replace('topmodel_', ''): v for k, v in prefixed.items()}


def get_vic_bounds() -> Dict[str, Dict[str, float]]:
    """VIC bound set (12 parameters, all solo).

    Owned by ``symfluence.models.vic`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('VIC', 'symfluence.models.vic')


def get_ignacio_bounds() -> Dict[str, Dict[str, float]]:
    """IGNACIO FBP fire bound set (6 parameters, all solo).

    Owned by ``symfluence.models.ignacio`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('IGNACIO', 'symfluence.models.ignacio')


def get_sacsma_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all SAC-SMA + Snow-17 parameter bounds.

    Returns:
        Dictionary mapping param_name -> {'min': float, 'max': float, 'transform': str}

    COMPATIBILITY BOUNDS (tier C by necessity, not by physics). SACSMA is served
    by the external ``jsacsma`` plugin package, which lives in its own repository
    and predates the ``register_model_bounds()`` seam — it still imports this
    helper directly. Its definitions and composition therefore stay in core even
    though every one of its parameters is shared with NGEN/Snow-17, so nothing would move anyway. Upstream issue: have ``jsacsma`` register its own bounds, then
    delete this entry.
    """
    sacsma_params = [
        # Snow-17
        'SCF', 'PXTEMP', 'MFMAX', 'MFMIN', 'NMF', 'MBASE', 'TIPM', 'UADJ', 'PLWHC', 'DAYGM',
        # SAC-SMA
        'UZTWM', 'UZFWM', 'UZK', 'LZTWM', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK',
        'ZPERC', 'REXP', 'PFREE', 'PCTIM', 'ADIMP', 'RIVA', 'SIDE', 'RSERV',
    ]
    return get_registry().get_bounds_for_params(sacsma_params)


def get_xinanjiang_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all Xinanjiang (XAJ) parameter bounds.

    Returns:
        Dictionary mapping param_name -> {'min': float, 'max': float, 'transform': str}
        Keys use unprefixed names (K, B, SM, ...) matching XAJ parameter conventions.

    COMPATIBILITY BOUNDS (tier C by necessity, not by physics). XINANJIANG is served
    by the external ``jxaj`` plugin package, which lives in its own repository
    and predates the ``register_model_bounds()`` seam — it still imports this
    helper directly. Its definitions and composition therefore stay in core even
    though 12 of its 13 parameters are solo (only ``xaj_K``'s served name ``K`` is shared, with GSFLOW). Upstream issue: have ``jxaj`` register its own bounds, then
    delete this entry.
    """
    xaj_params = [
        'xaj_K', 'xaj_B', 'xaj_IM', 'xaj_UM', 'xaj_LM', 'xaj_DM', 'xaj_C',
        'xaj_SM', 'xaj_EX', 'xaj_KI', 'xaj_KG', 'xaj_CI', 'xaj_CG',
    ]
    prefixed = get_registry().get_bounds_for_params(xaj_params)
    # Strip xaj_ prefix so keys match parameter manager conventions
    return {k.replace('xaj_', ''): v for k, v in prefixed.items()}


def get_snow17_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get Snow-17 parameter bounds (reuses SACSMA_PARAMS entries).

    Returns:
        Dictionary mapping Snow-17 param_name -> {'min': float, 'max': float, 'transform': str}

    COMPATIBILITY BOUNDS (tier C by necessity, not by physics). SNOW17 is served
    by the external ``jsnow17`` plugin package, which lives in its own repository
    and predates the ``register_model_bounds()`` seam — it still imports this
    helper directly. Its definitions and composition therefore stay in core even
    though every one of its parameters is shared with SAC-SMA and NGEN, so nothing would move anyway. Upstream issue: have ``jsnow17`` register its own bounds, then
    delete this entry.
    """
    names = ['SCF', 'PXTEMP', 'MFMAX', 'MFMIN', 'NMF', 'MBASE', 'TIPM', 'UADJ', 'PLWHC', 'DAYGM']
    return get_registry().get_bounds_for_params(names)


def get_gsflow_bounds() -> Dict[str, Dict[str, float]]:
    """GSFLOW (PRMS + MODFLOW-NWT) bound set (10 parameters).

    9 solo definitions moved to the package; ``gsflow_K`` stays central because
    its served name ``K`` collides with Xinanjiang's. The ``gsflow_`` prefix is
    stripped on the way out.

    Owned by ``symfluence.models.gsflow`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('GSFLOW', 'symfluence.models.gsflow')


def get_noahmp_bounds() -> Dict[str, Dict[str, Any]]:
    """NOAH-MP bound set (12 parameters).

    Only ``route_k`` is solo and moved to the package; the other 11 are the
    shared NOAH-OWP soil/snow parameters NGEN also calibrates.

    Owned by ``symfluence.models.noahmp`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('NOAHMP', 'symfluence.models.noahmp')


def get_watflood_bounds() -> Dict[str, Dict[str, float]]:
    """WATFLOOD bound set (16 parameters).

    14 solo definitions moved to the package; ``watflood_PWR``/``watflood_R2N``
    stay central (MESH serves the same two names). The ``watflood_`` prefix is
    stripped on the way out.

    Owned by ``symfluence.models.watflood`` (tier A: composition; tier B: solo
    definitions). Core keeps no second copy — this helper resolves what
    the package registered, and raises ``KeyError`` if it never did.
    """
    return _owned_by_package('WATFLOOD', 'symfluence.models.watflood')


# The completeness ledger for the parity snapshot: every model the framework can
# serve bounds for appears here, so one cannot silently stop being covered
# (tests/unit/core/data/model_bounds_snapshot.json is checked against this set).
#
# It is NOT "what can be served without a registration" -- get_model_bounds()
# consults the registered composition FIRST, so for the migrated in-tree models
# (FUSE, NGEN + NGEN_*, MIZUROUTE, HYPE, MESH, GR, RHESSYS, VIC, IGNACIO,
# GSFLOW, NOAHMP, WATFLOOD) this branch is only reached when the owning package
# is ABSENT, and the delegate then raises a KeyError naming that package. Core
# holds no data for them.
#
# Only DEPTH and the six external-plugin-served sets (HBV, HECHMS, TOPMODEL,
# SACSMA, SNOW17, XINANJIANG) still resolve their values from this module.
_BUILTIN_MODEL_BOUNDS = {
    'FUSE': get_fuse_bounds,
    'NGEN': get_ngen_bounds,
    'NGEN_CFE': get_ngen_cfe_bounds,
    'NGEN_NOAH': get_ngen_noah_bounds,
    'NGEN_PET': get_ngen_pet_bounds,
    'NGEN_TOPMODEL': get_ngen_topmodel_bounds,
    'NGEN_SACSMA': get_ngen_sacsma_bounds,
    'NGEN_SNOW17': get_ngen_snow17_bounds,
    'MIZUROUTE': get_mizuroute_bounds,
    'DEPTH': get_depth_bounds,
    'HYPE': get_hype_bounds,
    'MESH': get_mesh_bounds,
    'GR': get_gr_bounds,
    'RHESSYS': get_rhessys_bounds,
    'HBV': get_hbv_bounds,
    'HECHMS': get_hechms_bounds,
    'TOPMODEL': get_topmodel_bounds,
    'VIC': get_vic_bounds,
    'IGNACIO': get_ignacio_bounds,
    'SACSMA': get_sacsma_bounds,
    'XINANJIANG': get_xinanjiang_bounds,
    'SNOW17': get_snow17_bounds,
    'GSFLOW': get_gsflow_bounds,
    'NOAHMP': get_noahmp_bounds,
    'WATFLOOD': get_watflood_bounds,
}
