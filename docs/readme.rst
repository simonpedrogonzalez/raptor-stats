.. role:: raw-html-m2r(raw)
   :format: html


raptor-stats
============


.. image:: https://img.shields.io/badge/Docs-latest-blue
   :target: https://simonpedrogonzalez.github.io/raptor-stats-docs/index.html
   :alt: Docs: latest


.. image:: https://img.shields.io/pypi/v/raptor-stats
   :target: https://pypi.org/project/raptor-stats/
   :alt: PyPI


.. image:: https://img.shields.io/badge/python-3.12.8+-blue.svg
   :target: https://github.com/simonpedrogonzalez/raptor-stats
   :alt: Python  3.12.8+


.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License


Compute zonal statistics using efficient Raptor (Raster+Vector) methods.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
------------

.. code-block:: bash

   pip install raptor-stats

Usage
-----

.. code-block:: python

   from raptorstats import zonal_stats

   stats = zonal_stats("path/to/vector.shp", "path/to/raster.tif", method="scanline")

See the `zonal_stats API docs <https://simonpedrogonzalez.github.io/raptor-stats-docs/raptorstats.api.html#raptorstats.api.zonal_stats>`_ for more details on input types and additional parameters.

Methods
-------


* ``scanline``\ : Scans the raster file once, line by line, and computes all the intersections with the vector layer in a single pass. Suitable for fast one-time run results.
* ``agqt``\ : Builds a QuadTree with precomputed statistics, then combines it with the scanline method to answer queries more efficiently. Suitable for systems that repeatedly query changing vector layers over the same raster.

Performance
-----------

Raptor methods performance advantage increases with the size of the input raster and number of features. For example, with an ~1.9 billion pixel raster and 50 features (US states):


.. image:: assets/total_time_s_states.svg
   :target: assets/total_time_s_states.svg
   :alt: 


For the same raster, on but around 3000 features (US counties):


.. image:: assets/total_time_s_counties.svg
   :target: assets/total_time_s_counties.svg
   :alt: 


NOTES:


* These tests were made on a i7-8750H (2019) 16GB RAM Linux machine.
* The performance of the agqt method depends on the depth of the tree selected and the size of the features.

Credits
-------


* Author: `Simon Pedro Gonzalez <https://simonpedrogonzalez.github.io/>`_.
* This package is based on the following :raw-html-m2r:`<a href="https://simonpedrogonzalez.github.io/raptor-stats-docs/_static/mdml_final_report.pdf" download>project</a>`\ , where you can read more about the zonal stats problem, methods and performance comparison.
* This package API is inspired in the `rasterstats <https://github.com/perrygeo/python-rasterstats>`_ package by `Matthew Perry <https://github.com/perrygeo>`_.
