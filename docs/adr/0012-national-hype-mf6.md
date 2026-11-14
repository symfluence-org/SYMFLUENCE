# 0012 National HYPE + MODFLOW 6 (Iceland, landslíkan)

**Date:** 2026-11-13 · **Status: IN PROGRESS** · Branch `feature/hype-mf6-national`

## Context

The Tungnaá HYPE + MODFLOW 6 member (`output/tungnaa_modflow/`) is one basin: five HYPE "subbasins" that are
elevation bands, a 3 km MF6 grid of 173 cells, recharge per band, and groundwater discharge added at the single
outlet (`HYPERechargeAdapter._outlet` requires exactly one outlet). The landslíkan runs the same paradigm for all of
Iceland on the landslíkan hydrofabric (v1.0: 10,043 units, 1,433 sea outlets, 1 closed basin, 414 lakes, 19
transfers; repo `landslikan`).

Decisions (user, 2026-11-13):

1. Extend SYMFLUENCE rather than build a separate model package.
2. Two-way exchange of water by two passes: HYPE pass 1 exports deep percolation (AQIN) per subbasin; MF6 runs and
   reports drain discharge per subbasin; HYPE pass 2 takes it as daily point sources into each subbasin's main
   river and routes it through lakes and branches. Exact for the one-way coupling, since point sources do not change
   recharge.
3. Whole country from the start (no pilot subset).

## Design

**Inputs from landslikan (no zonal statistics in SYMFLUENCE).** The fabric is consumed as delivered:
`divides`/`reaches` (topology, length, slope), `attributes` (ice volume, elevation), `hru_classes` (joint area of
land class × soil group × 200 m band per unit, from `landslikan.hru`), `lakes_routing`, `transfers`,
`forcing_carra` (unit → CARRA cell weights) and the daily CARRA land-cell archive (`landslikan.carra daily`).

**HYPE.**
- subid = fabric unit id; maindown = toid (0 = sea or sink), so many outlets are normal.
- SLC = land class × soil group (26 combinations nationally), with `dhslc` (class elevation − unit mean).
  Elevation-band subbasins are a later option; the table already carries bands.
- GlacierData from the fabric's glacier area and ice volume; LakeData for the 414 lakes (outlet lakes at their
  outlet unit, default rating curve until stage–discharge data exist); BranchData from `transfers` (HYPE allows one
  branch per subbasin: transfers sharing a source, e.g. Bjarnalón THJ-9/10, are merged).
- Forcing: Pobs, Tobs, TMIN/TMAXobs, SWobs, RHobs, Uobs, SFobs per unit from the daily cell archive and the fabric
  weights, written directly (no remapping step).
- All geodata built with vectorised pandas (the current per-basin `.loc` loops scale as subbasins × SLCs).

**MODFLOW 6.**
- National grid from the fabric rasters (default 1 km, ~103,000 active cells), one convertible layer first.
- K, SY and thickness by geology zone (NÍ 1:600k groups from the fabric) with calibratable zone multipliers.
- Sea boundary: GHB at stage 0 in coastal cells (submarine discharge leaves the land water balance).
- River and seepage drains are split per (cell, unit) overlap and tagged with the unit id. Discharge per unit is
  computed exactly from the saved heads and the drain table, q = C·max(h − z, 0) (matches the MF6 budget to
  1e-14). Boundname observations were tried first: MF6 re-resolves them every stress period, which cost ~25 s
  per daily period at this size, more than the solve.
- Recharge: the existing volume-conserving zone weights, one TS per subbasin.
- Stress period 1 steady state, then one transient period of daily time steps (recharge varies through the
  time series; package data are read once).

**Transfers over time.** The fabric's transfers carry commissioning years (landslikan `config/network.yaml`:
Búrfell 1969, Köldukvíslarveita 1972, Sigalda 1977, Hrauneyjafoss 1981, Kvíslaveita 1984 and its Þjórsárlón stage
1997, Blanda 1991, Sultartangi 1999, Vatnsfell 2001, Kárahnjúkar 2007, Jökulsárveita/Hraunaveita 2008, Búðarháls
2014, Búrfell II 2018; natural bifurcations always on). HYPE's BranchData is static, so `branch_eras` splits a run
at 1 January of each commissioning year inside it and pass 2 runs era by era, carrying the HYPE state
(`outstatedate` → `state_save<date>.txt` → `instate y`; `glacierini 0` after the first era so the saved glacier
state is not reset). GeoData's routing order is built from every transfer ever operated, so it holds in every era.
Pass 1 runs straight through: branches move river water only and do not change recharge. Lakes exist in every era
(see open questions).

**Coupling driver.** HYPE pass 1 (AQIN) → MF6 → per-subbasin discharge → HYPE pass 2 with
PointSourceData/PSDailySeries (main-river inflow, as the lake-flux coupling already does) → discharge at every
unit. Mass balance check: precipitation − evaporation − storage change = sea outflow + sink + submarine discharge.

## Measured (2026-11-13, v1.0 fabric, 1991)

- HYPE inputs from the fabric: 8 s (10,043 subbasins, 21 SLCs, 16 branches, 412 outlet lakes, 769 glacier
  subbasins). Forcing: 17 s and ~170 MB per year.
- HYPE, one year, default (Tungnaá-calibrated) parameters: ~8 min. Balance 1991: P 2,012 mm, EVAP 115 mm,
  AQIN 561 mm, sea outflow 1,158 mm, snow storage +116 mm.
- MF6 at 1 km: 106,800 active cells, 208,035 unit-cell overlaps, 89,433 stream entries; steady state 31 s,
  converges, discrepancy 3e-7 %; 3 outer iterations per daily step.

Fabric issues found while building (to fix in landslikan): the Bjarnalón lake outlet resolves to a 6.6 km²
unit (Þjórsá bypasses the lake polygon in D8) and Búrfell's tailrace to a 236 km² tributary, so THJ-9/THJ-10
have wrong endpoints.

## Phases

1. landslikan exports: `hru_classes` (done), daily CARRA land-cell archive (running).
2. HYPE national inputs (geodata, classes, glaciers, lakes, branches, forcing) and a default-parameter run.
3. MF6 national grid and per-subbasin drains; steady-state run.
4. Two-pass driver and the national water balance; 1991–2025 default run.
5. Regionalised calibration against LamaH-Ice and IMO gauges (separate ADR).

## Open questions

- MF6 resolution and layering (1 km, one layer first; two layers where Holocene lava overlies older bedrock).
- Glacier beds: recharge under ice and drains beneath glaciers.
- Transfer capacities and reservoir rules (none available yet: defaults are calibration parameters).
- Reservoirs created during the record (Þjórsárlón 1996, Hálslón 2006, Ufsarlón and Kelduárlón 2008,
  Sporðöldulón 2013) are lakes from the start; lakes would need their own commissioning dates.
