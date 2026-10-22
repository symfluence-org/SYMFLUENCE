Hydrological Models
===================

SYMFLUENCE supports a wide range of hydrological models, from process-based
physical models to data-driven machine learning approaches.

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Process-based Models
      :link: process-based
      :link-type: ref

      Physically-based models with detailed process representation.

      **SUMMA** · **MESH** · **HYPE** · **RHESSys**

   .. grid-item-card:: Conceptual Models
      :link: conceptual
      :link-type: ref

      Bucket and storage-based hydrological models.

      **FUSE** · **GR** · **HBV**

   .. grid-item-card:: Machine Learning
      :link: ml-models
      :link-type: ref

      Data-driven approaches using neural networks.

      **LSTM** · **GNN**

   .. grid-item-card:: Frameworks & Specialized
      :link: frameworks
      :link-type: ref

      Modular frameworks and domain-specific models.

      **NextGen**

----

.. _process-based:

Process-based Models
--------------------

Models with explicit representation of physical processes.

.. toctree::
   :maxdepth: 1

   model_summa
   model_mesh
   model_hype
   model_rhessys
   model_cwatm
   model_noahmp

----

.. _conceptual:

Conceptual Models
-----------------

Simplified representations using storage-based approaches.

.. toctree::
   :maxdepth: 1

   model_fuse
   model_gr
   model_hbv
   model_raven

----

.. _ml-models:

Machine Learning Models
-----------------------

Data-driven approaches for streamflow prediction.

.. toctree::
   :maxdepth: 1

   model_lstm
   model_gnn

----

.. _frameworks:

Frameworks & Specialized
------------------------

Modular frameworks and specialized applications.

.. toctree::
   :maxdepth: 1

   model_ngen
