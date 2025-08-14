# raptor-stats

[Raptor (Raster-Vector) Zonal Statistics](https://simonpedrogonzalez.github.io/raptor-stats-docs/index.html)

This package provides a simple interface to calculate zonal statistics using Raptor Methods with a `rasterstats`-like api.

## Installation

You can install the package using pip:

```bash
pip install raptor-stats
```

## Usage

```python
from raptorstats import zonal_stats

# Example usage
stats = zonal_stats("path/to/vector.shp", "path/to/raster.tif", method="scanline")
```
Note: see the [zonal_stats API docs](https://simonpedrogonzalez.github.io/raptor-stats-docs/raptorstats.api.html#raptorstats.api.zonal_stats) for more details on input types and additional parameters.

## Methods

- `scanline`: Uses a scanline algorithm for efficient zonal statistics. Suitable for large datasets in a single pass (large raster, many features).
- `agqt`: Uses the aggregated quadtree method. Suitable for several and repeated queries over a large dataset (large raster, many features).

## Credits

- Author: [Simon Pedro Gonzalez](https://simonpedrogonzalez.github.io/)
- This package is based on the following <a href="_static/mdml_final_report.pdf" download>project</a>, where you can read more about the zonal stats problem, methods and performance comparison.
- This package API and tests are heavily inspired in the [rasterstats](https://github.com/perrygeo/python-rasterstats) package by [Matthew Perry](https://github.com/perrygeo).