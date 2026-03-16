# Experiment 9: Large Domain Distributed Modeling (Section 3.8)

## Scientific Question
Can SYMFLUENCE handle regional-scale distributed hydrological modeling with
network routing across Iceland?

## Config
- `config_iceland_distributed.yaml` — Iceland-wide distributed FUSE + mizuRoute

## Key Configuration Choices
- `DOMAIN_DEFINITION_METHOD: subset_geofabric` — uses TDX geofabric
- `FUSE_SPATIAL_MODE: distributed` — spatially distributed FUSE
- `ROUTING_MODEL: mizuRoute` — river network routing
- CARRA forcing at 3-hourly resolution

## Paper Figures
- Figure 12: Distributed model results across Iceland

## Data Requirements
- TDX geofabric for Iceland
- CARRA reanalysis (full Iceland domain)
- IMO streamflow for validation
