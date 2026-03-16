# Experiment 11: Data Processing Pipeline (Section 3.10)

## Scientific Question
How does SYMFLUENCE automate the data acquisition and preprocessing workflow
across different domains and scales?

## Configs
- `configs/config_paradise.yaml` — Point-scale data pipeline
- `configs/config_bow.yaml` — Lumped catchment data pipeline
- `configs/config_iceland.yaml` — Distributed domain data pipeline

## Key Configuration Choices
- `DATA_ACCESS: cloud` enables automated downloads from cloud sources
- Each config demonstrates the full pipeline: DEM download, forcing acquisition,
  domain delineation, observation retrieval
- Minimal model settings (pipeline focus, not calibration)

## What This Demonstrates
1. Automatic DEM download (Copernicus 30m)
2. Watershed delineation (TauDEM)
3. Forcing data acquisition (ERA5, CARRA)
4. Observation data download (SNOTEL, WSC, IMO)
5. Geospatial intersection and remapping
