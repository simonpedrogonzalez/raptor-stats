
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
        


def test_nan_counts():
    from affine import Affine

    transform = Affine(1, 0, 1, 0, -1, 3)

    data = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [1, 4, 5]])

    # geom extends an additional row to left
    geom = "POLYGON ((1 0, 4 0, 4 3, 1 3, 1 0))"

    # nan stat is requested
    stats = zonal_stats(geom, data, affine=transform, nodata=0.0, stats="*")

    for res in stats:
        assert res["count"] == 3  # 3 pixels of valid data
        assert res["nodata"] == 3  # 3 pixels of nodata
        assert res["nan"] == 3  # 3 pixels of nans

    # nan are ignored if nan stat is not requested
    stats = zonal_stats(geom, data, affine=transform, nodata=0.0, stats="count nodata")

    for res in stats:
        assert res["count"] == 3  # 3 pixels of valid data
        assert res["nodata"] == 3  # 3 pixels of nodata
        assert "nan" not in res


test_nan_counts()
