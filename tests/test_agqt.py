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
kwargs = { "method": "agqt", "max_depth": 3, "index_path": f"{INDICES}/my_index.idx" }
kwargs_new_index = { "method": "agqt", "max_depth": 3, "index_path": f"{INDICES}/new_index.idx"}
kwargs_us = { "method": "agqt", "max_depth": 5, "index_path": f"{INDICES}/us_index.idx" }

def test_builds_index():
    polygons = os.path.join(DATA, "polygons.shp")

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
    # garbage index files
    with open(f"{INDICES}/new_index.idx", "w") as f:
        f.write("garbage")
    with open(f"{INDICES}/new_index.dat", "w") as f:
        f.write("garbage")

    hash_before = compute_file_hash(f"{INDICES}/new_index.idx") + compute_file_hash(f"{INDICES}/new_index.dat")

    # force rebuild index
    stats = zonal_stats(polygons, raster, build_index=True, **kwargs_new_index)
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
    hash_before = compute_file_hash(f"{INDICES}/my_index.idx") + compute_file_hash(f"{INDICES}/my_index.dat")
    
    stats = zonal_stats(polygons, raster, **kwargs)
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
    states = os.path.join(DATA, "cb_2018_us_state_20m_filtered_filtered.shp")
    correct_result = os.path.join(DATA, "test_US_results.json")
    stats = zonal_stats(states, raster_US, stats="*", categorical=True, **kwargs_us)
    correct_stats = json.load(open(correct_result))
    stats = json.loads(
        json.dumps(stats)
    )  # to ensure histogram keys are strings and such
    correct, errors = compare_stats_exact_match(correct_stats, stats)
    assert correct


# NOTE: not supporting kwargs that I don't know what they do
# def test_zonal_global_extent():
#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(polygons, raster)
#     global_stats = zonal_stats(polygons, raster, global_src_extent=True)
#     assert stats == global_stats


def test_zonal_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, nodata=0, **kwargs)
    assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50


# NOTE: GEOPANDAS ALREADY RAISES ERROR
# def test_doesnt_exist():
#     nonexistent = os.path.join(DATA, "DOESNOTEXIST.shp")
#     with pytest.raises(ValueError):
#         zonal_stats(nonexistent, raster)

# NOTE: GEOPANDAS ALREADY RAISES ERROR
# def test_nonsense():
#     polygons = os.path.join(DATA, "polygons.shp")
#     with pytest.raises(ValueError):
#         zonal_stats("blaghrlargh", raster)
#     with pytest.raises(OSError):
#         zonal_stats(polygons, "blercherlerch")
#     with pytest.raises(ValueError):
#         zonal_stats(
#             [
#                 "blaghrlargh",
#             ],
#             raster,
#         )


# TODO: Support points and lines maybe with boxify?
# def test_points():
#     points = os.path.join(DATA, "points.shp")
#     stats = zonal_stats(points, raster)
#     # three features
#     assert len(stats) == 3
#     # three pixels
#     assert sum([x["count"] for x in stats]) == 3
#     assert round(stats[0]["mean"], 3) == 11.386
#     assert round(stats[1]["mean"], 3) == 35.547

# def test_points_categorical():
#     points = os.path.join(DATA, "points.shp")
#     categorical_raster = os.path.join(DATA, "slope_classes.tif")
#     stats = zonal_stats(points, categorical_raster, categorical=True)
#     # three features
#     assert len(stats) == 3
#     assert "mean" not in stats[0]
#     assert stats[0][1.0] == 1
#     assert stats[1][2.0] == 1

# def test_lines():
#     lines = os.path.join(DATA, "lines.shp")
#     stats = zonal_stats(lines, raster)
#     assert len(stats) == 2
#     assert stats[0]["count"] == 58
#     assert stats[1]["count"] == 32

# def test_multilines():
#     multilines = os.path.join(DATA, "multilines.shp")
#     stats = zonal_stats(multilines, raster)
#     assert len(stats) == 1
#     # can differ slightly based on platform/gdal version
#     assert stats[0]["count"] in [89, 90]


# def test_multipoints():
#     multipoints = os.path.join(DATA, "multipoints.shp")
#     stats = zonal_stats(multipoints, raster)
#     assert len(stats) == 1
#     assert stats[0]["count"] == 3


# Test multigeoms
def test_multipolygons():
    multipolygons = os.path.join(DATA, "multipolygons.shp")
    stats = zonal_stats(multipolygons, raster, **kwargs)
    assert len(stats) == 1
    assert stats[0]["count"] == 125


def test_categorical():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    stats = zonal_stats(polygons, categorical_raster, categorical=True, **kwargs)
    assert len(stats) == 2
    assert stats[0]["histogram"][1.0] == 75
    assert 5.0 in stats[1]["histogram"]


# TODO: support categorical maps (e.g. mapping value 5.0 to "cat5")
# def test_categorical_map():
#     polygons = os.path.join(DATA, "polygons.shp")
#     categorical_raster = os.path.join(DATA, "slope_classes.tif")
#     catmap = {5.0: "cat5"}
#     stats = zonal_stats(
#         polygons, categorical_raster, categorical=True, category_map=catmap
#     )
#     assert len(stats) == 2
#     assert stats[0][1.0] == 75
#     assert 5.0 not in stats[1]
#     assert "cat5" in stats[1]


def test_specify_stats_list():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats=["min", "max"], **kwargs)
    assert sorted(stats[0].keys()) == sorted(["min", "max"])
    assert "count" not in list(stats[0].keys())


def test_specify_all_stats():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="ALL", **kwargs)
    # sum_sq and histogram are required for combined stats of std and median
    assert sorted(stats[0].keys()) == sorted(VALID_STATS)
    stats = zonal_stats(polygons, raster, stats="*", **kwargs)
    assert sorted(stats[0].keys()) == sorted(VALID_STATS)


def test_specify_stats_string():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="min max")
    assert sorted(stats[0].keys()) == sorted(["min", "max"])
    assert "count" not in list(stats[0].keys())


# def test_specify_stats_invalid():
#     polygons = os.path.join(DATA, "polygons.shp")
#     with pytest.raises(ValueError):
#         zonal_stats(polygons, raster, stats="foo max")


def test_optional_stats():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="min max sum majority median std", **kwargs)
    assert stats[0]["min"] <= stats[0]["median"] <= stats[0]["max"]


def test_range():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="range min max", **kwargs)
    for stat in stats:
        assert np.allclose(stat["max"] - stat["min"], stat["range"])
    ranges = [x["range"] for x in stats]
    # without min/max specified
    stats = zonal_stats(polygons, raster, stats="range", **kwargs)
    # assert "min" not in stats[0]
    assert ranges == [x["range"] for x in stats]


def test_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    stats = zonal_stats(
        polygons, categorical_raster, stats="*", categorical=True, nodata=1.0, **kwargs
    )
    assert stats[0]["majority"] is np.nan
    assert stats[0]["count"] == 0  # no pixels; they're all null
    assert stats[1]["minority"] == 2.0
    assert stats[1]["count"] == 49  # used to be 50 if we allowed 1.0
    assert "1.0" not in stats[0]["histogram"]


def test_dataset_mask():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "dataset_mask.tif")
    stats = zonal_stats(polygons, raster, stats="*", **kwargs)
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 0


def test_partial_overlap():
    polygons = os.path.join(DATA, "polygons_partial_overlap.shp")
    stats = zonal_stats(polygons, raster, stats="count", **kwargs)
    for res in stats:
        # each polygon should have at least a few pixels overlap
        assert res["count"] > 0


def test_no_overlap():
    polygons = os.path.join(DATA, "polygons_no_overlap.shp")
    stats = zonal_stats(polygons, raster, stats="count", **kwargs)
    for res in stats:
        # no polygon should have any overlap
        assert res["count"] == 0


# NOTE: Per definition Scanline is not compatible with all_touched=True
# def test_all_touched():
#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(polygons, raster, all_touched=True)
#     assert stats[0]["count"] == 95  # 75 if ALL_TOUCHED=False
#     assert stats[1]["count"] == 73  # 50 if ALL_TOUCHED=False


# def test_ndarray_without_affine():
#     with rasterio.open(raster) as src:
#         polygons = os.path.join(DATA, "polygons.shp")
#         with pytest.raises(ValueError):
#             zonal_stats(polygons, src.read(1))  # needs affine kwarg


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
    stats = zonal_stats(polygons, arr, affine=affine, **kwargs)
    stats2 = zonal_stats(polygons, raster, **kwargs)
    for s1, s2 in zip(stats, stats2):
        _assert_dict_eq(s1, s2)
    with pytest.raises(AssertionError):
        _assert_dict_eq(stats[0], stats[1])
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50

    # TODO: POINTS NOT SUPPORTED
    # points = os.path.join(DATA, "points.shp")
    # stats = zonal_stats(points, arr, affine=affine)
    # assert stats == zonal_stats(points, raster)
    # assert sum([x["count"] for x in stats]) == 3
    # assert round(stats[0]["mean"], 3) == 11.386
    # assert round(stats[1]["mean"], 3) == 35.547


# TODO: support add_stats? it may require both a from_array and from_partials method
# def test_add_stats():
#     polygons = os.path.join(DATA, "polygons.shp")

#     def mymean(x):
#         return np.ma.mean(x)

#     stats = zonal_stats(polygons, raster, add_stats={"mymean": mymean})
#     for i in range(len(stats)):
#         assert stats[i]["mean"] == stats[i]["mymean"]


# def test_add_stats_prop():
#     polygons = os.path.join(DATA, "polygons.shp")

#     def mymean_prop(x, prop):
#         return np.ma.mean(x) * prop["id"]

#     stats = zonal_stats(polygons, raster, add_stats={"mymean_prop": mymean_prop})
#     for i in range(len(stats)):
#         assert stats[i]["mymean_prop"] == stats[i]["mean"] * (i + 1)

# def test_add_stats_prop_and_array():
#     polygons = os.path.join(DATA, "polygons.shp")

#     def mymean_prop_and_array(x, prop, rv_array):
#         # confirm that the object exists and is accessible.
#         assert rv_array is not None
#         return np.ma.mean(x) * prop["id"]

#     stats = zonal_stats(polygons, raster, add_stats={"mymean_prop_and_array": mymean_prop_and_array})
#     for i in range(len(stats)):
#         assert stats[i]["mymean_prop_and_array"] == stats[i]["mean"] * (i + 1)

# TODO: support raster_out?
# def test_mini_raster():
#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(polygons, raster, raster_out=True)
#     stats2 = zonal_stats(
#         polygons,
#         stats[0]["mini_raster_array"],
#         raster_out=True,
#         affine=stats[0]["mini_raster_affine"],
#     )
#     assert (
#         stats[0]["mini_raster_array"] == stats2[0]["mini_raster_array"]
#     ).sum() == stats[0]["count"]


def test_percentile_good():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="median percentile_50 percentile_90", **kwargs)
    assert "percentile_50" in stats[0].keys()
    assert "percentile_90" in stats[0].keys()
    assert stats[0]["percentile_50"] == stats[0]["median"]
    assert stats[0]["percentile_50"] <= stats[0]["percentile_90"]


# TODO: support zone_func?
# def test_zone_func_has_return():
#     def example_zone_func(zone_arr):
#         return np.ma.masked_array(np.full(zone_arr.shape, 1))

#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(polygons, raster, zone_func=example_zone_func)
#     assert stats[0]["max"] == 1
#     assert stats[0]["min"] == 1
#     assert stats[0]["mean"] == 1


# def test_zone_func_good():
#     def example_zone_func(zone_arr):
#         zone_arr[:] = 0

#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(polygons, raster, zone_func=example_zone_func)
#     assert stats[0]["max"] == 0
#     assert stats[0]["min"] == 0
#     assert stats[0]["mean"] == 0


# def test_zone_func_bad():
#     not_a_func = "jar jar binks"
#     polygons = os.path.join(DATA, "polygons.shp")
#     with pytest.raises(TypeError):
#         zonal_stats(polygons, raster, zone_func=not_a_func)


def test_percentile_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    # By setting nodata to 1, one of our polygons is within the raster extent
    # but has an empty masked array
    stats = zonal_stats(polygons, categorical_raster, stats=["percentile_90"], nodata=1, **kwargs)
    assert "percentile_90" in stats[0].keys()
    assert [np.nan, 5.0] == [x["percentile_90"] for x in stats]


def test_percentile_bad():
    polygons = os.path.join(DATA, "polygons.shp")
    with pytest.raises(ValueError):
        zonal_stats(polygons, raster, stats="percentile_101", **kwargs)


# def test_json_serializable():
#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(
#         polygons, raster, stats=VALID_STATS + ["percentile_90"], categorical=True
#     )
#     try:
#         json.dumps(stats)
#         simplejson.dumps(stats)
#     except TypeError:
#         pytest.fail("zonal_stats returned a list that wasn't JSON-serializable")


# def test_direct_geodataframe():
#     polygons = os.path.join(DATA, "polygons.shp")
#     features = gpd.read_file(polygons)

#     stats_direct = zonal_stats(polygons, raster, **kwargs)
#     stats_features = zonal_stats(features, raster, **kwargs)

#     assert stats_direct == stats_features


def test_all_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "all_nodata.tif")
    stats = zonal_stats(polygons, raster, stats=["nodata", "count"], **kwargs)
    assert stats[0]["nodata"] == 75
    assert stats[0]["count"] == 0
    assert stats[1]["nodata"] == 50
    assert stats[1]["count"] == 0


def test_some_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "slope_nodata.tif")
    stats = zonal_stats(polygons, raster, stats=["nodata", "count"], **kwargs)
    assert stats[0]["nodata"] == 36
    assert stats[0]["count"] == 39
    assert stats[1]["nodata"] == 19
    assert stats[1]["count"] == 31


# update this if nan end up being incorporated into nodata
def test_nan_nodata():
    polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
    arr = np.array([[np.nan, 12.25], [-999, 12.75]])
    affine = Affine(1, 0, 0, 0, -1, 2)

    stats = zonal_stats(
        polygon, arr, affine=affine, nodata=-999, stats="nodata count sum mean min max", **kwargs
    )

    assert stats[0]["nodata"] == 1
    assert stats[0]["count"] == 2
    assert stats[0]["mean"] == 12.5
    assert stats[0]["min"] == 12.25
    assert stats[0]["max"] == 12.75


def test_some_nodata_ndarray():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "slope_nodata.tif")
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


def test_transform():
    with rasterio.open(raster) as src:
        arr = src.read(1)
        affine = src.transform
    polygons = os.path.join(DATA, "polygons.shp")

    stats = zonal_stats(polygons, arr, affine=affine, **kwargs)
    # with pytest.deprecated_call():
    #     stats2 = zonal_stats(polygons, arr, transform=affine.to_gdal())
    # assert stats == stats2


# def test_prefix():
#     polygons = os.path.join(DATA, "polygons.shp")
#     stats = zonal_stats(polygons, raster, prefix="TEST")
#     for key in ["count", "min", "max", "mean"]:
#         assert key not in stats[0]
#     for key in ["TESTcount", "TESTmin", "TESTmax", "TESTmean"]:
#         assert key in stats[0]


# TODO: support geojson_out?
# def test_geojson_out():
#     polygons = os.path.join(DATA, "polygons.shp")
#     features = zonal_stats(polygons, raster, geojson_out=True)
#     for feature in features:
#         assert feature["type"] == "Feature"
#         assert "id" in feature["properties"]  # from orig
#         assert "count" in feature["properties"]  # from zonal stats


# TODO: support geojson_out?
# do not think this is actually testing the line i wanted it to
# since the read_features func for this data type is generating
# the properties field
# def test_geojson_out_with_no_properties():
#     polygon = Polygon([[0, 0], [0, 0.5], [1, 1.5], [1.5, 2], [2, 2], [2, 0]])
#     arr = np.array([[100, 1], [100, 1]])
#     affine = Affine(1, 0, 0, 0, -1, 2)

#     stats = zonal_stats(polygon, arr, affine=affine, geojson_out=True)
#     assert "properties" in stats[0]
#     for key in ["count", "min", "max", "mean"]:
#         assert key in stats[0]["properties"]

#     assert stats[0]["properties"]["mean"] == 34


# TODO: support copy_properties?
# remove when copy_properties alias is removed
# def test_copy_properties_warn():
#     polygons = os.path.join(DATA, "polygons.shp")
#     # run once to trigger any other unrelated deprecation warnings
#     # so the test does not catch them instead
#     stats_a = zonal_stats(polygons, raster)
#     with pytest.deprecated_call():
#         stats_b = zonal_stats(polygons, raster, copy_properties=True)
#     assert stats_a == stats_b


def test_nan_counts():
    from affine import Affine

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


# Optional tests
# def test_geodataframe_zonal():
#     gpd = pytest.importorskip("geopandas")

#     polygons = os.path.join(DATA, "polygons.shp")
#     df = gpd.read_file(polygons)
#     if not hasattr(df, "__geo_interface__"):
#         pytest.skip("This version of geopandas doesn't support df.__geo_interface__")

#     expected = zonal_stats(polygons, raster)
#     assert zonal_stats(df, raster) == expected


# TODO #  gen_zonal_stats(<features_without_props>)
# TODO #  gen_zonal_stats(stats=nodata)
# TODO #  gen_zonal_stats(<raster with non-integer dtype>)
# TODO #  gen_zonal_stats(transform AND affine>)
