# Experiment 8: Large Sample Study (Section 3.7)

## Scientific Question
How does model performance and structural sensitivity vary across diverse
catchments in a large-sample context?

## Configs
- `_template_fuse.yaml` — FUSE template with `DOMAIN_ID` placeholder
- `_template_gr4j.yaml` — GR4J+CemaNeige template
- `generate_configs.py` — Script to generate all 118 catchment configs

## Design
- 59 Icelandic catchments from LamaH-Ice dataset
- FUSE decision analysis: 4 structural combinations per catchment (51 configs)
- GR4J+CemaNeige: baseline comparison model (66 configs)

## Key Configuration Choices
- CARRA reanalysis forcing (3-hourly)
- Calibration: 2001-2007, Evaluation: 2008, Spinup: 2000
- DDS optimization, KGE objective, 1000 iterations
- `SKIP_WARM_START: true` for independent catchment runs

## Usage
```bash
python generate_configs.py --template _template_fuse.yaml --output ./generated/fuse/
python generate_configs.py --template _template_gr4j.yaml --output ./generated/gr4j/

# Run one catchment
symfluence run generated/fuse/config_lamahice_1010_FUSE.yaml
```

## Paper Figures
- Figure 11: Large-sample performance maps and structural sensitivity

## Data Requirements
- CARRA reanalysis for Iceland
- LamaH-Ice catchment shapefiles and attributes
- IMO streamflow observations for 59 gauged catchments
