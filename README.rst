CSAM's Readme
=============

.. image:: docs/source/_statc/logo.png
  :width: 400
  :alt: CSAM's logo

.. class:: center

    Constrained Spectral Approximation Method

.. image:: https://img.shields.io/badge/License-AGPL_v3-blue.svg
    :target: https://www.gnu.org/licenses/agpl-3.0
    
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


The Constrained Spectral Approximation Method (CSAM) is a physically sound and robust method for approximating the spectrum of subgrid-scale orography. It operates under the following constraints:

* Utilises a limited number of spectral modes (no more than 100)
* Significantly reduces the complexity of physical terrain by over 500 times
* Maintains the integrity of physical information to a large extent
* Compatible with unstructured geodesic grids
* Inherently scale-aware

This method is primarily used to represent terrain for weather forecasting purposes, but it also shows promise for broader data analysis applications.

----------

Read the documentation here.

----------

Requirements
------------

.. literalinclude:: ../../requirements.txt

.. note::
    The Sphinx dependencies can be found in ``docs/conf.py``.



Usage
-----

Installation
^^^^^^^^^^^^

Make a fork and clone your remote forked repository.

Configuration
^^^^^^^^^^^^^

The user-defined input parameters are in the :mod:`inputs` subpackage. These parameters are imported into the run scripts in :mod:`runs`. 

Usage
^^^^^

A simple setup can be found in :mod:`runs.idealised_isosceles`. To execute this run script:

.. code-block:: console

    python3 ./runs/idealised_isosceles.py

However, the codebase is structured such that the user can easily assemble a run script to define their own experiments. Refer to the documentation for the available APIs.

License
^^^^^^^

GNU GPL v3 (tentative)

Contributions
^^^^^^^^^^^^^

Refer to the open issues that require attention.

Any changes, improvements, or bug fixes can be submitted to upstream via a pull request.

