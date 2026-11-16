# 0014 Groundwater calibration and a spring-fed objective (Iceland, landslíkan)

**Date:** 2026-11-16 · **Status: PROPOSED** (step 1 done; steps 2–3 running) · Branch `feature/hype-mf6-national` ·
follows ADR 0013

## Context

The calibrated coupled run `national_v2_1991_2025` (ADR 0013) scores well on direct-runoff and glacier gauges
(median KGE ~0.65) but poorly on spring-fed ones: gauges with Q10/Q50 > 0.5 have a median daily KGE of 0.21 and
are too flashy (variability ratio ~1.26). The South region is weakest (0.44). Two causes:

- The MODFLOW parameters were never calibrated: hydraulic conductivity, specific yield and the river and seepage
  leakances were literature values per geology group (ADR 0012).
- The HYPE objective (median KGE) is dominated by high flows and by the many direct-runoff gauges, so it gives the
  baseflow of spring-fed rivers almost no weight.

Two gauges are left out of every objective as water-supply areas whose flow is managed: Elliðaár (LamaH 13) and
Hólmsá (LamaH 39). Their simulated flow is 4–7× the observed flow.

## Decisions

Three steps, each one fixing the other's inputs:

1. **Calibrate MODFLOW alone** (`national_gw_calibration.py`), on monthly stress periods, with recharge from the v2
   pass 1 (HYPE AQIN, 1998-10..2010-09) held fixed. Simulated gauge flow = the v2 pass 1 fast flow (HYPE COUT
   without groundwater point sources) + MODFLOW drain discharge from the units upstream. Monthly MF6 matches the
   daily run's national discharge (r 0.998, 1,999 against 2,000 m³/s), and one run takes ~5–15 min instead of hours.
   - Parameters (11): log10 K for six geology groups (Holocene lava, Holocene sediment, upper and lower Pleistocene,
     Tertiary, intrusive), specific yield of lava, sediment and upper Pleistocene, log10 river and seepage leakance.
   - Objective: each gauge scored by the mean of its monthly KGE and log-KGE (evaluated from 2000-10); score =
     0.5 · median over the calibration gauges + 0.5 · median over the spring-fed calibration gauges.
   - Search: asynchronous DDS, 8 workers, a new candidate as each run finishes, journal and resume; runs over
     40 min score as failures (slow runs come from extreme parameters).
2. **Recalibrate HYPE** (`national_calibration.py`, `objective='mixed'`) with the calibrated MODFLOW discharge as
   fixed point sources, the same per-gauge mean(KGE, log-KGE) and the same half weight on spring-fed gauges, plus
   the glacier and snow penalties of ADR 0013. Started from the ADR 0013 parameters; the rivvel bound raised to 10,
   wcfc widened to ×0.5–3.5 and wcep to ×0.25–1.8 (both ended at their bounds; total pore space stays ≤ 0.96).
   - **Per-ice-cap term.** VÍ's glaciological surface balances (glaciers API, CC BY 4.0; annual ba for
     Vatnajökull from 1991/92, Langjökull from 1996/97, Hofsjökull from 1987/88) are compared with each cap's
     simulated balance from the 1 October states. Glacier subbasins are assigned to the 2019 outline (GLIMS id)
     they overlap most (`settings/HYPE/glacier_caps.csv`; model areas 7,785 / 832 / 805 km² against 7,569 / 811 /
     788 km² in the outlines). Penalty 0.25 × mean over caps of |sim − obs| / (3σ), σ = 0.10 (Vatnajökull) and
     0.07 m w.e. (the others), on top of the national −10 Gt/yr term, which still covers Mýrdalsjökull and the
     small glaciers. The weight is moderate because one parameter set cannot fix errors of opposite sign on
     neighbouring caps; those point to the forcing.
   - The v2 run against VÍ, 1995/96–2008/09 (m w.e./yr): Vatnajökull −0.81 vs −0.75, Langjökull −1.47 vs −1.26,
     Hofsjökull −0.92 vs −1.18. Year to year the model follows the measurements closely (r = 0.82, 0.89, 0.93
     over 28–33 years), an independent check of the glacier melt and the ice flow.
   - **Lava soil type (`national_lava.py`).** A diagnosis of the four worst gauges (Svartá, Laxá at Helluvað,
     Hvítá at Kljáfoss, Ytri-Rangá; monthly KGE −0.2 to −4.2 after step 1) found areas within 2 % of LamaH and no
     timing lag, but HYPE fast flow alone varying 2–6× more than the whole observed series (alpha 2.6–6.9). HYPE's
     soil types come from pedology and do not separate young lava (soil 1 is 15 % lava, soil 2 8 %), so lava ran off
     like any other ground. Every ordinary land class of a subbasin is now split by its Holocene-lava fraction into
     the class and a lava twin (soil type 4, classes 22–27; 5,431 km² of the sub-model). The soil rows of par.txt
     get a fourth column copied from soil 1; with neutral lava parameters the split model reproduces the unsplit
     one (one year: median unit identical, 99 % of units within 0.04 % of mean flow). Five lava-only parameters,
     relative to the whole-row values: mperc1, mperc2 ×1–20, rrcs1, rrcs2 ×0.05–1, bfroznsoil ×1–20 (frozen-soil
     infiltration capacity). A strong setting (×10, ×10, ×0.1, ×0.1, ×10) cut the daily CV at Svartá from 0.79 to
     0.50, Laxá 0.74 to 0.45 and Ytri-Rangá 0.60 to 0.29 in a one-year test.
   - **MODFLOW surrogate.** The point sources are fixed during the HYPE calibration, so recharge a candidate adds
     would vanish (Svartá mean 19.9 → 14.2 m³/s in the test). The difference between the candidate's aquifer
     recharge (AQIN) and the reference recharge MODFLOW was run on (v2 pass 1) is summed over each gauge's upstream
     units, passed through a linear reservoir (τ = 180 days) and added to the gauge flow (`gw_return`). It is local
     return only; the coupled rerun replaces it with MODFLOW.
3. **Rerun the coupled model** 1991–2025 with both parameter sets, validate as in ADR 0013, and reseed the live
   forecast.

## Results

### Step 1 — MODFLOW (2026-11-16, local, 240 runs, ~3.5 h)

The run's exclusion list matched the wrong LamaH ids (it left out Hafralónsá, LamaH 28, and Hvítá at Fremstaver,
LamaH 36, instead of Elliðaár, 13, and Hólmsá, 39). The journal keeps every run's per-gauge scores, so all 240 runs
were re-scored with the correct exclusion: score 0.552 (literature values) → 0.624 at best. The top runs lie within
0.003 of each other; e0230 (0.623) was chosen as it is also near the top under the original scoring.

| Median mean(KGE, log-KGE), monthly (gauges) | Literature | Calibrated (e0230) |
|---|---|---|
| Calibration, spring-fed (12) | 0.43 | 0.61 |
| Calibration, direct-runoff (22) | 0.68 | 0.65 |
| Calibration, glacier (12) | 0.68 | 0.58 |
| Held-out, spring-fed (12) | 0.42 | 0.59 |
| Held-out, direct-runoff (5) | 0.63 | 0.61 |
| Held-out, glacier (7) | 0.72 | 0.77 |

The spring-fed gain holds on the held-out gauges, so it is not overfitting. The calibration glacier gauges lose
0.10 while the held-out ones gain 0.05; step 2 refits HYPE against the new groundwater.

| Parameter | Literature | Calibrated |
|---|---|---|
| K Holocene lava (m/d) | 100 | 28 |
| K Holocene sediment (m/d) | 10 | 245 (bound 316) |
| K upper Pleistocene (m/d) | 1 | 11 |
| K lower Pleistocene (m/d) | 0.1 | 0.30 |
| K Tertiary (m/d) | 0.01 | 0.016 |
| K intrusive (m/d) | 0.01 | 0.0002 |
| Sy lava / sediment / upper Pleistocene | 0.10 / 0.15 / 0.05 | 0.15 / 0.08 / 0.19 (bound 0.20) |
| River leakance (m²/d per m) | 10 | 0.73 |
| Seepage leakance (1/d) | 0.1 | 0.0086 |

Lower river leakance and more storage in the upper Pleistocene slow the groundwater response, which is what the
flashy spring-fed gauges needed. Sediment K and upper-Pleistocene Sy near their bounds suggest the groups stand in
for more than their names (e.g. permeable young lava under sediment cover); widen the bounds or split the groups if
a later round needs it.

### Step 2 — HYPE recalibration

*Running.*

### Step 3 — coupled rerun `national_v3_1991_2025`

*Pending.*

## Open

- Per-ice-cap glacier mass balance targets (Aðalgeirsdóttir et al. 2020, S1/S2), and the spring-melt excess at
  Jökulsá á Fjöllum (+21 %, May–June).
- Hydrofabric: Mývatn (38 km²) is split over five units by junction breaks inside the lake, so only 2 km² is an
  outlet lake and the Laxá flow bypasses the lake storage. Stop junction breaks inside lakes ≥ 0.5 km² and merge
  the lake units (fabric v1.2; unit ids change), then give Mývatn its own rating in LakeData.
- Ytri-Rangá: observed runoff 2,318 mm/yr against P − E 1,320 needs ~18 m³/s of groundwater from outside the
  topographic catchment (Hekla, highlands); MODFLOW returns only local recharge. Review lava K anisotropy and the
  Þjórsá/Tungnaá drain conductance; add a gauge note (spring river, compare volumes, not area).
- Hvítá at Kljáfoss: +24 % volume, about its net glacier loss; the per-cap Langjökull term (sim −1.47 vs obs
  −1.26 m w.e./yr) addresses part of it.
