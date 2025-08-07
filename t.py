
# test zonal stats
import json
import os
import sys

import numpy as np
import pytest
import rasterio
import simplejson
from affine import Affine
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geopandas as gpd

from raptorstats import zonal_stats
from rasterstats.io import read_featurecollection, read_features
from rasterstats.utils import VALID_STATS
import rasterio as rio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/data")
raster = os.path.join(DATA, "slope.tif")



def _assert_dict_eq(a, b):
    """Assert that dicts a and b similar within floating point precision"""
    err = 1e-5
    for k in set(a.keys()).union(set(b.keys())):
        if a[k] == b[k]:
            continue
        try:
            if abs(a[k] - b[k]) > err:
                raise AssertionError(f"{k}: {a[k]} != {b[k]}")
        except TypeError:  # can't take abs, nan
            raise AssertionError(f"{a[k]} != {b[k]}")
        


def test_main():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster)
    for key in ["count", "min", "max", "mean"]:
        assert key in stats[0]
    assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50
    assert round(stats[0]["mean"], 2) == 14.66



test_main()
