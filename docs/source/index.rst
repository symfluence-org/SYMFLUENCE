:html_theme.sidebar_secondary.remove:

.. raw:: html

   <div style="text-align: center; margin: 2rem 0 3rem 0;">
     <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 0.25rem; letter-spacing: -0.02em;">SYMFLUENCE</h1>
     <p style="font-size: 1.35rem; color: #5a6f7c; margin-bottom: 1rem; font-weight: 400;">Earth-system modeling. Simplified.</p>
     <p style="font-size: 1rem; color: #6c757d; max-width: 600px; margin: 0 auto;">
       A computational framework for hydrological modeling that integrates data preparation,
       model setup, calibration, and evaluation in unified workflows.
     </p>
   </div>

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Installation, setup, and your first workflow.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      Configure domains, forcing data, and model parameters.

   .. grid-item-card:: Models
      :link: models/index
      :link-type: doc

      Hydrological and routing models supported by SYMFLUENCE.

   .. grid-item-card:: Calibration
      :link: calibration
      :link-type: doc

      Parameter optimization and sensitivity analysis.

   .. grid-item-card:: AI Agent
      :link: agent_guide
      :link-type: doc

      Drive SYMFLUENCE with your installed coding-agent CLI, primed with skills.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Complete Python API documentation.

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   installation
   docker
   getting_started
   examples

.. toctree::
   :maxdepth: 1
   :caption: Configuration
   :hidden:

   configuration
   calibration
   multivariable_evaluation
   multi_model_comparison

.. toctree::
   :maxdepth: 1
   :caption: Models
   :hidden:

   models/index
   routing/index

.. toctree::
   :maxdepth: 1
   :caption: AI Agent
   :hidden:

   agent_guide

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   cli_reference
   api
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   architecture
   developer_guide
   testing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   licensing
