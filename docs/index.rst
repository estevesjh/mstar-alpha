Mstar: ML Stellar Mass Estimator
=======================================

Mstar is a fast and precise stellar mass estimator for the Dark Energy Survey (DES) galaxies. 
The algorithm consists of a machine learning code based on the Artificial Neural Networks (ANN) architecture. 
The estimator was trained on the DES deep fields matched with the COSMOS dataset.
Our results were cross validated with a local volume sample, the SDSS sample matched with DES.
For more information about the validation process take a look at `Esteves et al. 2023 <https://arxiv.org/>`_.

How to use 
------------

The input information is simply the galaxy redshift colors and the z-band magnitude.

Installation
------------
Mstar-alpha is hosted on PyPI and can be installed using::

   pip install Mstar-alpha

The latest source code is available on `GitHub <https://github.com/estevesjh/Mstar-alpha>`_.


Getting started
---------------
To become familiar with the package, we recommend going through example `Ipython notebooks <https://github.com/smucesh/galpro>`_.
For ease of use, Galpro is built around a single core class :ref:`Model <model>`.


Acknowledgements
----------------
Galpro is built on top of other excellent Python packages such as:

- `scikit-learn <https://scikit-learn.org/stable/>`_: for implementing the random forest algorithm.
- `joblib <https://joblib.readthedocs.io/en/latest/>`_: for saving and loading a trained random forest model.
- `h5py <https://joblib.readthedocs.io/en/latest/>`_: for reading and writing PDFs to disk.

This code was inspired on galpro: `GitHub <https://github.com/smucesh/galpro>`_.

Citation
--------
Don't forget to cite this work If you use Mstar 
(`Esteves et al. 2023 <https://arxiv.org/>`_) in any of your publications.

.. toctree::
   :maxdepth: 2
   :hidden:

   self
   model.rst
