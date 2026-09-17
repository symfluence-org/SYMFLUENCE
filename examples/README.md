# SYMFLUENCE Examples

This directory provides complete, ready-to-run examples that demonstrate **SYMFLUENCE** across spatial scales — from single-site validation to continental-scale modeling.
Each example includes configuration files, Jupyter notebooks, and optional batch scripts to reproduce the workflows described in the main documentation.

---

## Learning Path

Examples are organized into four progressive stages. Follow them sequentially to build understanding and confidence with SYMFLUENCE.

### 01. Point-Scale Validation

Focuses on single-site process studies:

- **01a** - Snow physics validation using SNOTEL data
- **01b** - Energy balance validation with FLUXNET observations

Covers configuration structure, model setup, and controlled single-point testing.

### 02. Watershed Modeling

Demonstrates transition from lumped to distributed modeling at the basin scale using the Bow River at Banff (~2,600 km²):

- **02a** – Lumped model
- **02b** – Semi-distributed sub-basins (~15 units)
- **02c** – Elevation-based HRU discretization

Highlights trade-offs between spatial complexity and computational cost.

### 03. Large-Domain Applications

Scales up to national and continental domains:

- **03a** – Iceland
- **03b** – North America

Focuses on high-performance execution, large datasets, and scaling efficiency.

### 04. Workshop Notebooks

Hands-on workshop exercises for guided learning:

- **04a** – Logan River watershed
- **04b** – Provo River watershed
- **04c** – Logan River from the command line (shell-only SUMMA workflow, no Python)
- **04d** – [Intelligent Rivers: Himalayan snow](04_workshop_notebooks/04d_intelligent_rivers_snow.ipynb), using the shared [Rohtang config](04_workshop_notebooks/config_rohtang_snow.yaml) for native data acquisition and SUMMA, followed by forcing, snow-storage, MODIS coverage, timing, and native DE calibration exercises with before/after and a second-season holdout comparison and forcing-grid/elevation diagnostics. The notebook runs the full workflow by default; set `RUN_STEPS = False` to inspect prepared outputs during class.

Designed for classroom and workshop settings with step-by-step guidance.

---

## Additional Collections

Beyond the numbered learning path, this directory also contains:

- **`paper_case_studies/`** – Configurations and plotting scripts that reproduce
  the case studies and figures from the SYMFLUENCE papers (see its own README).
- **`iceland_national_model/`** – Ready-to-use configurations for the Iceland
  national domain (SUMMA, SUMMA + glaciers, HYPE, FWI). These complement the
  guided Iceland walkthrough in **03a**.
- **`camels_spat_attributes/`** – Configuration for deriving CAMELS-SPAT
  catchment attributes across North America.
- **`notebook_path_setup_template.ipynb`** – The env-aware path-setup pattern
  used by the example notebooks; copy its first cells when authoring a new
  notebook.

---

## Quick Start

1. **Install dependencies**
   ```bash
   ./scripts/symfluence-bootstrap --install
   ```

2. **Launch an example notebook** directly from the CLI:
   ```bash
   symfluence example launch 1a
   ```
   This command automatically:

   - Locates the corresponding notebook (e.g., `examples/01_point_vertical_flux_estimation/01a_point_scale_snotel.ipynb`)
   - Opens it in Jupyter Lab
   - Initializes it using the **root SYMFLUENCE virtual environment**

   You can substitute any example ID (e.g., `2b`, `3a`) to launch the corresponding workflow.

3. **Run complete workflows** via configuration:
   ```bash
   symfluence workflow run --config config.yaml
   ```

---

## Data Access

Example data for **01a – 02c** are provided as a single ~354 MB bundle for quick testing.

- **Download:** [GitHub Release – Example Data (01a–02c)](https://github.com/symfluence-org/SYMFLUENCE/releases/download/examples-data-v0.5.5/example_data_v0.5.5.zip)

> **Note:** `examples-data-v0.5.5` is the most recent *full* example bundle and
> remains the correct download for these examples. The newer
> `examples-data-v0.6.0`/`v0.7.0` releases are ~18 MB minimal bundles used by CI
> and are not sufficient to run the notebooks.

If you have access to institutional storage (e.g. **FIR** or **UCalgary ARC**),
you may instead point the config paths to your local dataset and set `DATA_ACCESS: 'maf'`.

https://app.globus.org/file-manager?origin_id=1062f558-f976-4d03-b4ef-f6c3b465ed66&origin_path=%2F

> Continental-domain simulations (**03b**) use multi-gigabyte datasets. 1 month of example data is available through globus in this [bundle](https://app.globus.org/file-manager?origin_id=1062f558-f976-4d03-b4ef-f6c3b465ed66&origin_path=%2F).

---

## Example Contents

Each example directory includes:

- A complete and reproducible configuration file
- A Jupyter notebook with contextual explanations
- Notes on expected results and dataset access

---

## Learning Outcomes

By completing these examples, you will learn to:

- Configure and manage SYMFLUENCE workflows
- Apply spatial discretization and calibration strategies
- Execute simulations locally or on HPC systems
- Automate large-sample experiments
- Evaluate results statistically and visually
- Transition confidently from validation studies to large-scale modeling
