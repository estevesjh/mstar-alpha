Mstar: ML Stellar Mass Estimator
=======================================

Mstar is a fast and precise stellar mass estimator for the Dark Energy Survey (DES) galaxies. 
The algorithm consists of a machine learning code based on the Artificial Neural Networks (ANN) architecture. Checkout our [github repo](https://github.com/estevesjh/mstar-alpha).

The estimator was trained on the DES deep fields matched with the COSMOS dataset.
Our results were cross validated with a local volume sample, the SDSS sample matched with DES.
For more information about the validation process take a look at [Esteves et al. 2023](https://arxiv.org/)

## How to use 

The input information is simply the galaxy redshift colors and the z-band magnitude.


<!-- .. Documentation is

.. Galpro is a novel Python machine learning code based on the Random Forest algorithm for estimating multivariate 
.. posterior probability distributions of galaxy properties (e.g. redshift, stellar mass, star formation rate,
.. metallicity). Documentation for the package is available at `galpro.readthedocs.io <https://galpro.readthedocs.io/>`_.

.. Galpro is hosted on PyPI and can be installed using::

..     pip install galpro


.. .. image:: docs/images/example_plot.png
..    :width: 400

.. Joint redshift - stellar mass posterior PDF
.. (See `Mucesh et al. 2020 <https://arxiv.org/abs/2012.05928>`_). -->