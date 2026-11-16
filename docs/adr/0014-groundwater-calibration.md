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
   the glacier and snow penalties of ADR 0013. Started from the ADR 0013 parameters; the rivvel bound raised to 10.
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
- The remaining poor gauges after step 1: Svartá (3654), Laxá at Helluvað (2677), Hvítá at Kljáfoss (6350),
  Ytri-Rangá (8692); KGE −0.2 to −4 with near-zero bias, i.e. timing/variability, not volume. They need a
  catchment, lake or diversion check (Laxá drains Mývatn) before more parameters.
