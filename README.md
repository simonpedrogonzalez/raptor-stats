# raptor-stats
Raptor (Raster-Vector) Zonal Statistics

This package provides a simple interface to calculate zonal statistics using Raptor Methods.

## Installation

You can install the package using pip:

```bash
pip install raptor-stats
```

## Usage

```python
from raptor_stats import zonal_stats

# Example usage
stats = zonal_stats("path/to/raster.tif", "path/to/vector.shp", method="scanline")
```

## Methods

- `scanline`: Uses a scanline algorithm for efficient zonal statistics. Suitable for large datasets in a single pass (large raster, many features).
- `aqt`: Uses the aggregated quadtree method for zonal statistics. Suitable for several and repeated queries over a large dataset (large raster, many features).