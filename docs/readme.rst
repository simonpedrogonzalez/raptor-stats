
raptor-stats
============

Raptor (Raster-Vector) Zonal Statistics

This package provides a simple interface to calculate zonal statistics using Raptor Methods with a ``rasterstats``\ -like api.

Installation
------------

You can install the package using pip:

.. code-block:: bash

   pip install raptor-stats

Usage
-----

.. code-block:: python

   from raptorstats import zonal_stats

   # Example usage
   stats = zonal_stats("path/to/raster.tif", "path/to/vector.shp", method="scanline")

Note: see the `API documentation <https://raptor-stats.readthedocs.io/en/latest/>`_ for more details.

Methods
-------


* ``scanline``\ : Uses a scanline algorithm for efficient zonal statistics. Suitable for large datasets in a single pass (large raster, many features).
* ``agqt``\ : Uses the aggregated quadtree method. Suitable for several and repeated queries over a large dataset (large raster, many features).

Credits
-------


* Author: `Simon Pedro Gonzalez <https://simonpedrogonzalez.github.io/>`_
* This package is based on the following `project <https://github.com/yourusername/raptor-stats>`_\ , where you can read more about the zonal stats problem, methods and performance comparison.
* This package API and tests are heavily inspired in the `rasterstats <https://github.com/perrygeo/python-rasterstats>`_ package by `Matthew Perry <https://github.com/perrygeo>`_.
