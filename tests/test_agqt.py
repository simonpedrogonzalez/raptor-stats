# test zonal stats
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import simplejson
from affine import Affine
from shapely.geometry import Polygon, box
from utils import compare_stats_exact_match, compute_file_hash
from raptorstats.agqt import AggQuadTree
from raptorstats.io import open_raster, open_vector
from raptorstats.stats import Stats
import rasterio as rio
    
from raptorstats import zonal_stats
from raptorstats.stats import VALID_STATS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
raster = os.path.join(DATA, "slope.tif")
raster_US = os.path.join(DATA, "US_MSR_resampled_x10.tif")
INDICES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "indices")

def test_builds_index():
    polygons = os.path.join(DATA, "polygons.shp")

    kwargs_new_index = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/new_index.idx"}

    if os.path.exists(f"{INDICES}/new_index.idx"):
        os.remove(f"{INDICES}/new_index.idx")
    if os.path.exists(f"{INDICES}/new_index.dat"):
        os.remove(f"{INDICES}/new_index.dat")

    stats = zonal_stats(polygons, raster, **kwargs_new_index)
    assert os.path.exists(f"{INDICES}/new_index.idx")
    assert os.path.exists(f"{INDICES}/new_index.dat")
    for key in ["count", "min", "max", "mean"]:
        assert key in stats[0]
    assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50
    assert round(stats[0]["mean"], 2) == 14.66
    os.remove(f"{INDICES}/new_index.idx")
    os.remove(f"{INDICES}/new_index.dat")


def test_force_rebuild_index():
    polygons = os.path.join(DATA, "polygons.shp")

    kwargs_new_index = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/new_index.idx", 'build_index': True }

    # garbage index files
    with open(f"{INDICES}/new_index.idx", "w") as f:
        f.write("garbage")
    with open(f"{INDICES}/new_index.dat", "w") as f:
        f.write("garbage")

    hash_before = compute_file_hash(f"{INDICES}/new_index.idx") + compute_file_hash(f"{INDICES}/new_index.dat")

    # force rebuild index
    stats = zonal_stats(polygons, raster, **kwargs_new_index)
    # should be able to calculate correct stats because garbage index files are ignored
    for key in ["count", "min", "max", "mean"]:
        assert key in stats[0]
        assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50
    assert round(stats[0]["mean"], 2) == 14.66

    assert os.path.exists(f"{INDICES}/new_index.idx")
    assert os.path.exists(f"{INDICES}/new_index.dat")

    hash_after = compute_file_hash(f"{INDICES}/new_index.idx") + compute_file_hash(f"{INDICES}/new_index.dat")

    assert hash_before != hash_after, "Index files were not rebuilt as expected."
    
    os.remove(f"{INDICES}/new_index.idx")
    os.remove(f"{INDICES}/new_index.dat")

def test_use_existing_index():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/my_index.idx" }
    hash_before = compute_file_hash(f"{INDICES}/my_index.idx") + compute_file_hash(f"{INDICES}/my_index.dat")
    
    stats = zonal_stats(polygons, raster, build_index=False, **kwargs)
    for key in ["count", "min", "max", "mean"]:
        assert key in stats[0]
        assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50
    assert round(stats[0]["mean"], 2) == 14.66
    # check that the index files were untouched
    hash_after = compute_file_hash(f"{INDICES}/my_index.idx") + compute_file_hash(f"{INDICES}/my_index.dat")
    assert hash_before == hash_after, "Index files were modified unexpectedly."


def test_compute_tree():
    polygons = os.path.join(DATA, "polygons.shp")
    default_file = "index_AggQuadTree_depth_5"
    if os.path.exists(f"{INDICES}/{default_file}.idx"):
        os.remove(f"{INDICES}/{default_file}.idx")
    if os.path.exists(f"{INDICES}/{default_file}.dat"):
        os.remove(f"{INDICES}/{default_file}.dat")

    with open_raster(raster) as raster_ds:
        features = open_vector(polygons)
        agqt = AggQuadTree(max_depth=5, index_path=f"{INDICES}/{default_file}.idx")
        agqt.stats = Stats("*")
        agqt._compute_quad_tree(features, raster_ds)
        idx = agqt.idx
        
        # Correct number of nodes
        n_all_nodes_created = len(list(idx.intersection(idx.bounds, objects=False)))
        win_bounds = raster_ds.window_bounds(rio.windows.Window(0, 0, raster_ds.width, raster_ds.height))
        all_nodes = list(idx.intersection(win_bounds, objects=True)) # within window
        n_nodes = (4**(5 + 1) - 1) // (4 - 1)  # geometric series of 4 divisions
        # All nodes created are inside the window and match the expected count
        assert len(all_nodes) == n_all_nodes_created == n_nodes, \
            "Number of nodes in the index does not match expected count"
        
        # Correct bounds
        idx_bounds = np.array(idx.bounds)
        raster_bounds = np.array(raster_ds.bounds)
        assert np.allclose(idx_bounds, raster_bounds), "Index bounds do not match raster bounds"

        # Correct stats
        all_nodes = [node.object for node in all_nodes]
        all_nodes = sorted(all_nodes, key=lambda x: x.level)[::-1]
        node_stats = [n.stats for n in all_nodes]
        correct_stats = json.load(open(os.path.join(DATA, "test_agqt_tree_results.json")))
        correct, errors = compare_stats_exact_match(correct_stats,node_stats)
        assert correct, f"Node stats do not match truth: {errors}"
        
        # Test each pixel is in one node of each level
        # The most painful way...
        for i in range(raster_ds.width):
            for j in range(raster_ds.height):
                pixel_bounds = raster_ds.xy(j, i)
                nodes = list(idx.intersection(pixel_bounds, objects=True))
                assert len(nodes) == 5+1, f"Pixel at ({i}, {j}) is not in 5 nodes"
                node_levels = [node.object.level for node in nodes]
                assert sorted(node_levels) == list(range(5+1)), f"Pixel at ({i}, {j}) is not in nodes of all levels"
        
        # Test exists files
        assert os.path.exists(f"{INDICES}/{default_file}.idx")
        assert os.path.exists(f"{INDICES}/{default_file}.dat")

        # Remove index files
        os.remove(f"{INDICES}/{default_file}.idx")
        os.remove(f"{INDICES}/{default_file}.dat")


def test_US():
    kwargs_us = { "method": "agqt", "max_depth": 5, "index_path": f"{INDICES}/us_index.idx" }
    states = os.path.join(DATA, "cb_2018_us_state_20m_filtered_filtered.shp")
    correct_result = os.path.join(DATA, "test_US_results.json")
    stats = zonal_stats(states, raster_US, stats="*", categorical=True, **kwargs_us)
    correct_stats = json.load(open(correct_result))
    stats = json.loads(
        json.dumps(stats)
    )  # to ensure histogram keys are strings and such
    correct, errors = compare_stats_exact_match(correct_stats, stats)
    assert correct


def test_zonal_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/my_index.idx" }
    stats = zonal_stats(polygons, raster, nodata=0, **kwargs)
    assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50

# Test multigeoms
def test_multipolygons():
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_multipolygons.idx" }
    multipolygons = os.path.join(DATA, "multipolygons.shp")
    stats = zonal_stats(multipolygons, raster, **kwargs)
    assert len(stats) == 1
    assert stats[0]["count"] == 125
    # cleanup
    os.remove(f"{INDICES}/test_multipolygons.idx")
    os.remove(f"{INDICES}/test_multipolygons.dat")


def test_categorical():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_categorical.idx" }
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    stats = zonal_stats(polygons, categorical_raster, categorical=True, **kwargs)
    assert len(stats) == 2
    assert stats[0]["histogram"][1.0] == 75
    assert 5.0 in stats[1]["histogram"]
    os.remove(f"{INDICES}/test_categorical.idx")
    os.remove(f"{INDICES}/test_categorical.dat")

def test_specify_stats_list():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_specify_stats_list.idx" }
    stats = zonal_stats(polygons, raster, stats=["min", "max"], **kwargs)
    assert sorted(stats[0].keys()) == sorted(["min", "max"])
    assert "count" not in list(stats[0].keys())
    os.remove(f"{INDICES}/test_specify_stats_list.idx")
    os.remove(f"{INDICES}/test_specify_stats_list.dat")


def test_specify_all_stats():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_specify_all_stats.idx" }
    stats = zonal_stats(polygons, raster, stats="ALL", **kwargs)
    # sum_sq and histogram are required for combined stats of std and median
    assert sorted(stats[0].keys()) == sorted(VALID_STATS)
    stats = zonal_stats(polygons, raster, stats="*", **kwargs)
    assert sorted(stats[0].keys()) == sorted(VALID_STATS)
    os.remove(f"{INDICES}/test_specify_all_stats.idx")
    os.remove(f"{INDICES}/test_specify_all_stats.dat")


def test_specify_stats_string():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_specify_stats_string.idx" }
    stats = zonal_stats(polygons, raster, stats="min max", **kwargs)
    assert sorted(stats[0].keys()) == sorted(["min", "max"])
    assert "count" not in list(stats[0].keys())
    os.remove(f"{INDICES}/test_specify_stats_string.idx")
    os.remove(f"{INDICES}/test_specify_stats_string.dat")

def test_optional_stats():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_optional_stats.idx" }
    stats = zonal_stats(polygons, raster, stats="min max sum majority median std", **kwargs)
    assert stats[0]["min"] <= stats[0]["median"] <= stats[0]["max"]
    os.remove(f"{INDICES}/test_optional_stats.idx")
    os.remove(f"{INDICES}/test_optional_stats.dat")


def test_range():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_range.idx" }
    stats = zonal_stats(polygons, raster, stats="range min max", **kwargs)
    for stat in stats:
        assert np.allclose(stat["max"] - stat["min"], stat["range"])
    ranges = [x["range"] for x in stats]
    # without min/max specified
    stats = zonal_stats(polygons, raster, stats="range", **kwargs)
    # assert "min" not in stats[0]
    assert ranges == [x["range"] for x in stats]
    os.remove(f"{INDICES}/test_range.idx")
    os.remove(f"{INDICES}/test_range.dat")


def test_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_nodata.idx" }
    stats = zonal_stats(
        polygons, categorical_raster, stats="*", categorical=True, nodata=1.0, **kwargs
    )
    assert stats[0]["majority"] is np.nan
    assert stats[0]["count"] == 0  # no pixels; they're all null
    assert stats[1]["minority"] == 2.0
    assert stats[1]["count"] == 49  # used to be 50 if we allowed 1.0
    assert "1.0" not in stats[0]["histogram"]
    os.remove(f"{INDICES}/test_nodata.idx")
    os.remove(f"{INDICES}/test_nodata.dat")



def test_dataset_mask():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "dataset_mask.tif")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_dataset_mask.idx" }
    stats = zonal_stats(polygons, raster, stats="*", **kwargs)
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 0
    os.remove(f"{INDICES}/test_dataset_mask.idx")
    os.remove(f"{INDICES}/test_dataset_mask.dat")


# TODO: I knew this would be a problem. Fix it, for scanline too.
# def test_dataset_mask_inverse_polygon_order():
#     polygons = os.path.join(DATA, "polygons.shp")
#     gdf = gpd.read_file(polygons)
#     # reverse the order of the polygons
#     gdf = gdf.iloc[::-1].reset_index(drop=True)
#     kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_dataset_mask_inverse_polygon_order.idx" }
#     raster = os.path.join(DATA, "dataset_mask.tif")
#     stats = zonal_stats(gdf, raster, stats="*", **kwargs)
#     assert stats[0]["count"] == 0
#     assert stats[1]["count"] == 75
#     os.remove(f"{INDICES}/test_dataset_mask_inverse_polygon_order.idx")


def test_partial_overlap():
    polygons = os.path.join(DATA, "polygons_partial_overlap.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_partial_overlap.idx" }  
    stats = zonal_stats(polygons, raster, stats="count", **kwargs)
    for res in stats:
        # each polygon should have at least a few pixels overlap
        assert res["count"] > 0
    os.remove(f"{INDICES}/test_partial_overlap.idx")
    os.remove(f"{INDICES}/test_partial_overlap.dat")


def test_no_overlap():
    polygons = os.path.join(DATA, "polygons_no_overlap.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_no_overlap.idx" }
    stats = zonal_stats(polygons, raster, stats="count", **kwargs)
    for res in stats:
        # no polygon should have any overlap
        assert res["count"] == 0
    os.remove(f"{INDICES}/test_no_overlap.idx")
    os.remove(f"{INDICES}/test_no_overlap.dat")


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


def test_ndarray():
    with rasterio.open(raster) as src:
        arr = src.read(1)
        affine = src.transform

    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_ndarray.idx" , 'build_index': True}
    stats = zonal_stats(polygons, arr, affine=affine, **kwargs)
    stats2 = zonal_stats(polygons, raster, **kwargs)
    for s1, s2 in zip(stats, stats2):
        _assert_dict_eq(s1, s2)
    with pytest.raises(AssertionError):
        _assert_dict_eq(stats[0], stats[1])
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50
    os.remove(f"{INDICES}/test_ndarray.idx")
    os.remove(f"{INDICES}/test_ndarray.dat")

def test_percentile_good():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_percentile_good.idx" }
    stats = zonal_stats(polygons, raster, stats="median percentile_50 percentile_90", **kwargs)
    assert "percentile_50" in stats[0].keys()
    assert "percentile_90" in stats[0].keys()
    assert stats[0]["percentile_50"] == stats[0]["median"]
    assert stats[0]["percentile_50"] <= stats[0]["percentile_90"]
    os.remove(f"{INDICES}/test_percentile_good.idx")
    os.remove(f"{INDICES}/test_percentile_good.dat")


def test_percentile_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_percentile_nodata.idx" }
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    # By setting nodata to 1, one of our polygons is within the raster extent
    # but has an empty masked array
    stats = zonal_stats(polygons, categorical_raster, stats=["percentile_90"], nodata=1, **kwargs)
    assert "percentile_90" in stats[0].keys()
    assert [np.nan, 5.0] == [x["percentile_90"] for x in stats]
    os.remove(f"{INDICES}/test_percentile_nodata.idx")
    os.remove(f"{INDICES}/test_percentile_nodata.dat")


def test_percentile_bad():
    polygons = os.path.join(DATA, "polygons.shp")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_percentile_bad.idx" }
    with pytest.raises(ValueError):
        zonal_stats(polygons, raster, stats="percentile_101", **kwargs)


def test_all_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "all_nodata.tif")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_all_nodata.idx" }
    stats = zonal_stats(polygons, raster, stats=["nodata", "count"], **kwargs)
    assert stats[0]["nodata"] == 75
    assert stats[0]["count"] == 0
    assert stats[1]["nodata"] == 50
    assert stats[1]["count"] == 0
    os.remove(f"{INDICES}/test_all_nodata.idx")
    os.remove(f"{INDICES}/test_all_nodata.dat")


def test_some_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "slope_nodata.tif")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_some_nodata.idx" }
    stats = zonal_stats(polygons, raster, stats=["nodata", "count"], **kwargs)
    assert stats[0]["nodata"] == 36
    assert stats[0]["count"] == 39
    assert stats[1]["nodata"] == 19
    assert stats[1]["count"] == 31
    os.remove(f"{INDICES}/test_some_nodata.idx")
    os.remove(f"{INDICES}/test_some_nodata.dat")


# update this if nan end up being incorporated into nodata
def test_nan_nodata():
    polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
    arr = np.array([[np.nan, 12.25], [-999, 12.75]])
    affine = Affine(1, 0, 0, 0, -1, 2)
    kwargs = { "method": "agqt", "max_depth": 1, "index_path": f"{INDICES}/test_nan_nodata.idx" }
    stats = zonal_stats(
        polygon, arr, affine=affine, nodata=-999, stats="nodata count sum mean min max", **kwargs
    )

    assert stats[0]["nodata"] == 1
    assert stats[0]["count"] == 2
    assert stats[0]["mean"] == 12.5
    assert stats[0]["min"] == 12.25
    assert stats[0]["max"] == 12.75
    os.remove(f"{INDICES}/test_nan_nodata.idx")
    os.remove(f"{INDICES}/test_nan_nodata.dat")


def test_some_nodata_ndarray():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "slope_nodata.tif")
    kwargs = { "method": "agqt", "max_depth": 4, "index_path": f"{INDICES}/test_some_nodata_ndarray.idx", 'build_index': True }
    with rasterio.open(raster) as src:
        arr = src.read(1)
        affine = src.transform

    # without nodata
    stats = zonal_stats(polygons, arr, affine=affine, stats=["nodata", "count", "min"], **kwargs)
    assert stats[0]["min"] == -9999.0
    assert stats[0]["nodata"] == 0
    assert stats[0]["count"] == 75

    # with nodata
    stats = zonal_stats(
        polygons, arr, affine=affine, nodata=-9999.0, stats=["nodata", "count", "min"], **kwargs
    )
    assert stats[0]["min"] >= 0.0
    assert stats[0]["nodata"] == 36
    assert stats[0]["count"] == 39
    os.remove(f"{INDICES}/test_some_nodata_ndarray.idx")
    os.remove(f"{INDICES}/test_some_nodata_ndarray.dat")


# def test_transform():
#     with rasterio.open(raster) as src:
#         arr = src.read(1)
#         affine = src.transform
#     polygons = os.path.join(DATA, "polygons.shp")

#     stats = zonal_stats(polygons, arr, affine=affine, **kwargs)
    # with pytest.deprecated_call():
    #     stats2 = zonal_stats(polygons, arr, transform=affine.to_gdal())
    # assert stats == stats2

def test_nan_counts():
    from affine import Affine
    kwargs = { "method": "agqt", "max_depth": 1, "index_path": f"{INDICES}/test_nan_counts.idx", 'build_index': True }

    transform = Affine(1, 0, 1, 0, -1, 3)

    data = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [1, 4, 5]])

    # geom extends an additional row to left
    geom = "POLYGON ((1 0, 4 0, 4 3, 1 3, 1 0))"

    # nan stat is requested
    stats = zonal_stats(geom, data, affine=transform, nodata=0.0, stats="*", **kwargs)

    for res in stats:
        assert res["count"] == 3  # 3 pixels of valid data
        assert res["nodata"] == 3  # 3 pixels of nodata
        assert res["nan"] == 3  # 3 pixels of nans

    # nan are ignored if nan stat is not requested
    stats = zonal_stats(geom, data, affine=transform, nodata=0.0, stats="count nodata", **kwargs)

    for res in stats:
        assert res["count"] == 3  # 3 pixels of valid data
        assert res["nodata"] == 3  # 3 pixels of nodata
        assert "nan" not in res
    os.remove(f"{INDICES}/test_nan_counts.idx")
    os.remove(f"{INDICES}/test_nan_counts.dat")
