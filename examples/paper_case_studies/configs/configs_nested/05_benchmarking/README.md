# Experiment 5: Benchmarking (Section 3.4)

## Scientific Question
How do calibrated hydrological models compare against simple statistical
benchmarks (persistence, climatology)?

## Configs
Each config calibrates one model on Bow at Banff and evaluates it against
the persistence / climatology / long-term-mean benchmarks. Run them
independently to reproduce one point on Figure 8 each.

- `config_bow_benchmark.yaml` — SUMMA (reference / paper match)
- `config_bow_benchmark_fuse.yaml` — FUSE
- `config_bow_benchmark_hbv.yaml` — HBV
- `config_bow_benchmark_gr4j.yaml` — GR4J (requires R + rpy2 — see top-level README)
- `config_bow_benchmark_hype.yaml` — HYPE

For other models on Figure 8, copy any of the variants above and replace
the `model:` block with the corresponding entry from
`02_model_ensemble/models/config_<model>.yaml`. Keep
`evaluation.analyses: [benchmarking]` so the benchmarks run.

## Key Configuration Choices
- Uses same domain/period as model ensemble (Experiment 2)
- `evaluation.analyses: [benchmarking]` triggers benchmark computation
- Benchmarks: persistence (lag-1), daily/monthly climatology, long-term mean

## Paper Figures
- Figure 8: Model ensemble vs. benchmark performance
