# Experiment 12: Parallel Scaling (Section 3.10)

## Scientific Question
How does SYMFLUENCE's computational performance scale with increasing
parallelism across different workflow components?

## Configs
- `configs/config_taudem_scaling.yaml` — TauDEM delineation scaling
- `configs/config_calibration_scaling.yaml` — DDS calibration parallelism
- `configs/config_distributed_scaling.yaml` — SUMMA actors distributed execution

## Three Scaling Experiments

### 12a: TauDEM Domain Delineation
- DEM resolutions: 10m, 30m, 90m
- Core counts: 10 to 100 (North America), 1 to 12 (Iceland)
- Tests: watershed delineation, flow accumulation, stream network extraction

### 12b: Parallel Calibration
- SUMMA calibration at Bow at Banff
- Parallel pool/MPI modes with 1-12 concurrent evaluations
- Tests: DDS with 100 and 20,000 evaluations

### 12c: Distributed Model Actors
- SUMMA actors framework for North America domain
- Core counts: 10, 20, 50, 100, 200, 500, 1000
- Tests: GRU-level parallelism in distributed model execution

## Paper Figures
- Figure 14: Parallel scaling curves (speedup vs. core count)

## Notes
These configs are templates — actual scaling experiments require HPC resources.
The full set of 93 scaling configs is in `../../configs_dir/12_parallel_scaling/`.
