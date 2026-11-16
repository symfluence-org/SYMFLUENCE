# 0016 National ParFlow-CLM (Iceland, landslíkan, paradigm 3)

**Date:** 2026-11-16 · **Status: PREPARED** (grid and segment inputs built; first steps run; spin-up and full runs
pending, on the server) · Branch `feature/hype-mf6-national`

## Context

The Tungnaá ParFlow-CLM member (basin mode, `models/clmparflow/basin/`, 72 × 65 cells at 1 km, 1,140 active, 2
ranks, about 2 h per simulated year; hand-tuned KGE 0.87/0.88) has one outlet, no-flow sides and base. Decision
(user, 2026-11-16): the whole country at 1 km, prepared here, run on the server.

## Design (`models/clmparflow/national.py`)

- **Grid:** national MF6's 1 km grid (`modflow.national_model.build_grid`, 371 × 521); active where ≥ 50 % land;
  18 islands under 4 cells dropped; 102,657 active cells.
- **Slopes:** D4 priority flood seeded from every coastal cell, each draining across a sea face, on the block mean
  of `dem_filled.tif` (the fabric's conditioned surface keeps its subglacial divides). Water leaves as overland flow
  at the coast; there is no single outlet.
- **Column:** 15 terrain-following layers, 100 m (basin mode's 14 + 30 m, to hold the MF6 water table); the top 10
  are CLM's soil layers. Aquifer K per cell from the geology groups with the calibrated MF6 values
  (`calibration/gw_v1/best.json`, m/d → m/h); soil K uniform; glacier cells `glacier_bed_k` (Tungnaá 0.001 m/h).
  Coast and base no-flow in this version (MF6 sea outflow was ~20 mm/yr).
- **Initial pressure:** from the calibrated MF6 steady state on the same grid (`settings/PARFLOW/mf6_steady`, 5 s):
  water table at min(h, surface), hydrostatic profile. Median water-table depth 17 m; 20 % deeper than 70 m.
- **Vegetation:** IGBP of each cell's majority unit's dominant land class; snow/ice where ≥ 50 % glacier.
- **Forcing:** each cell takes its unit's CARRA weights (as SUMMA and HYPE), lapsed to the cell elevation; 3-h
  values repeated hourly (`--dt 1`) or used at 3-h steps (`--dt 3`, a third of the steps and files).
  About 330 MB per simulated day at 1 h (PFB doubles over 193k cells × 8 variables), so segments are short and
  their forcing is deleted after the run.
- Basin mode's key database, CLM driver files and PFB writer are reused; `drv_vegm.dat` per rank is a symlink.

## Measured (2026-11-16, Mac)

- Grid 90 s; one-day segment written in 5 s.
- **4 ranks (~48,000 columns each) segfault in `clm_lsm` on the first step**, with or without a 64 MB stack; 16
  ranks (~12,000 each) run. Keep columns per rank ≲ 12,000 (≥ 16 ranks; the server has 24 cores).
- Cold start from a uniform 3 m water table: the first hour needed > 50 Newton iterations (residual 1.8e5 → 6e4).
- Richards-only spin-up stage (`national.py spinup`, 8 ranks, Mac at load 24 from other runs), MF6 initial
  pressure, mean MF6 recharge (639 mm/yr) as a top-layer source:
  - as first written, the first step started at a residual of 6.9e7: 19 % of cells had the MF6 water table at the
    surface, i.e. 5 cm ponded at t = 0, flowing at km/h with the basin Manning's n. The water table is now held
    one top layer (0.1 m) below the surface: residual 2.8e5 (250× lower);
  - hydrostatic suction above deep water tables (to −97 m, kr ≈ 1e-12) is capped at the steady-infiltration
    suction where K(ψ) = recharge (`flux_suction`, about −1 m in the soil);
  - still slow: the 1.2 h first step failed twice and was cut to 0.3 h; the full-Jacobian preconditioner with
    `OverlandKinematic` was no faster on the loaded Mac. Solver tuning moves to the server.

## Open

1. ~~Build on the server~~: done 2026-11-16 (`~/landslikan_models/parflow`, v3.15.0 + 3 patches, hypre 2.31.0).
2. Spin-up: Richards-only with mean recharge (no CLM) from the MF6 state (written; first steps slow, see above:
   try on 24 server ranks with FullJacobian / OverlandKinematic, a smaller initial step, and if needed overland
   flow off for the first years), then a CLM year repeated until storage drift is below tolerance (basin mode's
   `chain.py` logic), then 1991–2025 in segments.
3. Decide 1 h or 3 h steps from a one-month comparison.
4. Sea boundary (Dirichlet at sea level on coastal sides) and lakes (`lake_manning`) are not yet used.
5. Cost: unknown until the spin-up runs; first estimate 15–20 h per simulated year on 24 ranks.
