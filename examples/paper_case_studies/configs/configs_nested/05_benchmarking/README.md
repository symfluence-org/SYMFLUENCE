# Experiment 5: Benchmarking (Section 3.4)

## Scientific Question
How do calibrated hydrological models compare against simple statistical
benchmarks (persistence, climatology)?

## Config
- `config_bow_benchmark.yaml` — Benchmark evaluation at Bow at Banff

## Key Configuration Choices
- Uses same domain/period as model ensemble (Experiment 2)
- `ANALYSES: [benchmarking]` triggers benchmark computation
- Benchmarks: persistence (lag-1), daily/monthly climatology, long-term mean

## Paper Figures
- Figure 8: Model ensemble vs. benchmark performance
