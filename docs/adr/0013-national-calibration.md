# 0013 National HYPE calibration and glacier ice flow (Iceland, landslíkan)

**Date:** 2026-11-14 · **Status: ACCEPTED** (results 2026-11-16) · Branch `feature/hype-mf6-national` · follows ADR 0012

## Context

The first 1991–2025 national run (ADR 0012) used the Tungnaá-calibrated parameters. Two problems:

1. **Glaciers grew.** National ice volume went from ~3,700 to ~3,810 km³ over 1991–2025, while Icelandic glaciers
   lost −9.6 ± 0.8 Gt/yr over 1994/95–2018/19 (Aðalgeirsdóttir et al. 2020). Two structural causes in HYPE:
   - all precipitation on the glacier part of a class becomes ice, rain included (`glacier_soilmodel.f90`);
   - HYPE keeps one glacier per subbasin with no ice flow between subbasins. On a fabric that splits ice caps
     into many units, accumulation-zone units gain without limit (118 units +421 km³) while ablation-zone units
     melt out (621 units −321 km³, 243 to zero).
2. **No regional calibration.** Parameters came from one glacier-fed highland basin.

## Decisions

**HYPE patch (rain on glaciers runs off).** A patched build `installs/hype_landslikan/bin/hype` (patch
`installs/hype_landslikan/glacier_rain_runoff.patch`): on the glacier part, rain joins the melt in the gross
infiltration and only snowfall accumulates as ice. The stock HYPE build is unchanged and still used by Tungnaá.

**Ice flow by delta-h (`national_glacier.py`).** Once a year (1 October) the run stops, and each ice basin's
volume change over the past year is redistributed over its subbasins with the Huss et al. (2010) delta-h
parameterisation (largest change at the terminus, near zero at the top, basin total conserved; units that melt
out are removed and the rest rescaled). An ice basin is the set of glacier subbasins that drain along `maindown`
to the same last glacier subbasin (769 glacier subbasins, 353 basins, 158 with more than one unit). The volumes
are edited in HYPE's state file (frozen section, after the five snow arrays, GeoData order). `run_eras` splits
eras on the annual date when given an `IceFlux`.

**Final state.** The last era of `run_eras` also saves the state for the day after the run, the starting point
for continuations and the live forecast.

**Calibration (`national_calibration.py`).**
- Gauges: LamaH-Ice river gauges with `area_validation: include`, not an effective catchment, and no transfer
  endpoint or reservoir upstream (74 of 111). Every third by area rank is held out for spatial validation.
- Sub-model: the gauges' upstream areas plus every glacier subbasin, written as a self-contained HYPE settings
  directory (all inputs subset), so each evaluation reads and writes only what it needs.
- Groundwater: MODFLOW discharge from the latest coupled run enters as fixed point sources (pass 2 setup).
  After calibration the two-pass coupling is rerun with the new parameters; one outer iteration is planned.
- Objective: median KGE over the calibration gauges (hydrological years 1995/96–2008/09, two spin-up years)
  minus a penalty on the national glacier surface mass balance: −10.0 ± 1.0 Gt/yr, i.e. the total
  −11.6 ± 0.8 Gt/yr for 1994/95–2009/10 (Aðalgeirsdóttir et al. 2020) less ~1.6 Gt/yr of non-surface balance
  (geothermal, dissipation, calving) that HYPE does not simulate.
- Parameters (18): snow (cmlt, ttmp, cmrad, pcusnow), glacier (glaccmlt, glacttmp, glaccmrad), the within-unit
  temperature lapse rate (tcalt), evaporation (cevp), soil (wcfc, wcep, rrcs1, rrcs2, srrcs, mperc1, mperc2),
  river (rivvel, damp). Class-varying parameters are scaled (or shifted) as a whole so the class pattern is
  kept; regional multipliers are a later step.
- Search: parallel DDS (Tolson & Shoemaker 2007), a batch of candidates per iteration evaluated in parallel.
- Validation: held-out gauges over the same period, all gauges over 2009/10–2020/21 (LamaH ends 2021), and
  the 2010/11–2018/19 glacier rate (about half the 1994/95–2009/10 rate).

**Snow term.** The Tungnaá threshold (ttmp −2.56 °C) melted highland snow through the winter: HYPE built about a
tenth of CARRA's snowpack (unit 5051, Kreppa: ~30 mm peak SWE against 324 mm). The ttmp shift range was widened to
−1..+4 °C and the objective gained −0.25·|log(sim/CARRA)| of mean annual max SWE over 1,853 glacier-free units
(CARRA 1991–2020 normals). Score = median cal KGE − 0.5·|MB − (−10)|/3 − 0.25·snow error.

**Where it ran.** 50 iterations × 8 candidates (401 runs, ~13 h) on a 24-core WSL server, from the self-contained
sub-model (2,492 subbasins, 1993-10..2009-09). Groundwater discharge from the v1 coupled run as fixed point sources.

## Results (2026-11-16)

| | Tungnaá parameters | Calibrated |
|---|---|---|
| Sub-model median KGE, cal / held-out gauges | 0.48 / 0.51 | 0.62 / 0.64 |
| Glacier surface MB 1994/95–2009/10 | −7.7 Gt/yr | −10.0 Gt/yr |
| Peak SWE, sim / CARRA (median) | ~0.1 | 1.06 |

Main changes: ttmp +2.58 °C (melt starts near 0 °C), cmlt ×1.36, wcfc ×1.99, wcep ×0.52, rivvel 4.8 (bound 5.0:
widen next time), glaccmlt 8.66.

Coupled rerun `national_v2_1991_2025` (patched HYPE, calibrated par, annual ice flow in both passes, 10.4 h):
- Balance (mm/yr): recharge 642, groundwater to rivers + seepage 620, to sea 22; sea outflow pass 1 1,036, pass 2 1,655.
- Gauges (median KGE / bias), 1995–2009 and 2009–2021: cal 0.60 (+3 %) and 0.63 (+6 %); held-out 0.61 (+16 %) and
  0.55 (+12 %). The v1 run (Tungnaá parameters, pass 2) had 0.45–0.51.
- Glacier surface MB: −10.0 Gt/yr 1994/95–2009/10, −3.0 Gt/yr 2010/11–2018/19 (observed: about half the earlier
  rate), −8.7 Gt/yr 2019/20–2024/25. ~7 km³ of ice moved down-glacier per year by delta-h.
- The final state (2026-01-01) seeds the live forecast (Water.Landslikan_forecast), which applies the same annual
  ice flow when a cycle crosses 1 October.

## Open

- Per-ice-cap mass balance targets (Aðalgeirsdóttir et al. 2020 supplementary tables S1/S2).
- Regional parameter multipliers (HYPE parameter regions) once the national set is fitted.
- Regulated gauges, once Landsvirkjun capacities and rules are available.
