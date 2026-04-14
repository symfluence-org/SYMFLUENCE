Examples
========

Overview
--------
This section introduces complete, ready-to-run examples that demonstrate SYMFLUENCE's full range of workflows — from point-scale validation to large-domain modeling.
Each example includes configuration templates, Jupyter notebooks, and batch scripts located in the ``examples/`` directory.

---

Tutorial Structure
------------------
The examples are organized progressively to guide users from simple to advanced workflows:

1. **Point Scale (01)** — Snow and energy balance validation (SNOTEL, FLUXNET)
2. **Basin Scale (02)** — Bow River case studies (lumped, semi-distributed, elevation-based)
3. **Regional and Continental (03)** — Iceland and North America workflows
4. **Workshops (04)** — Guided hands-on exercises (Logan River, Provo River)

Each directory contains a configuration file, notebook, and optional SLURM script.

---

Running the Examples
--------------------
1. Install SYMFLUENCE and activate your environment:
   .. code-block:: bash

      ./scripts/symfluence-bootstrap --install
      source venv/bin/activate

2. Launch an example directly from the CLI:
   .. code-block:: bash

      symfluence example launch 2b

   Or navigate to the example directory manually:
   .. code-block:: bash

      cd examples/02_watershed_modelling/
      jupyter notebook 02b_basin_semidistributed.ipynb

3. Run the notebook or script as described inside.

---

Learning Path
-------------
- **Start simple:** ``01a_point_scale_snotel.ipynb`` — understand configuration and validation
- **Progress spatially:** ``02a–02c`` — from lumped to elevation-band modeling
- **Scale up:** ``03a–03b`` — regional and continental workflows
- **Workshop practice:** ``04a–04b`` — guided hands-on exercises

---

Best Practices
--------------
- Always validate configuration before execution.
- Follow the order: setup → run → evaluate.
- Use logs and plots to verify intermediate outputs.
- Adapt example configurations for your domain and models.

---

References
----------
- Example notebooks: `examples/ <https://github.com/symfluence-org/SYMFLUENCE/tree/main/examples>`_
- Configuration templates: `config_templates/ <https://github.com/symfluence-org/SYMFLUENCE/tree/main/src/symfluence/resources/config_templates>`_
- :doc:`configuration` — Configuration reference
