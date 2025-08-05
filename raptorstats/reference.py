from rasterstats import zonal_stats
import geopandas as gpd
import rasterio as rio
import numpy as np

def reference_method(raster_file_path, vector_file_path, stats):
    vector = gpd.read_file(vector_file_path)
    zs1 = zonal_stats(vector, raster_file_path, stats=stats, boundless=True)
    zs2 = zonal_stats(vector, raster_file_path, stats=stats, boundless=False, all_touched=True)
    # return one dict for each polygon [ {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'count': 1}, ... ]
    return zs1, zs2

def result_within_tolerance(center_based_result, all_touched_result, result, allow_None=True):
    # per polugon
    for i in range(len(result)):
        # per stat
        if allow_None and (result[i] is None or center_based_result[i] is None or all_touched_result[i] is None):
            continue

        for key in result[i].keys():
            high_bound = all_touched_result[i][key]
            low_bound = center_based_result[i][key]
            value = result[i][key]
            if value < low_bound or value > high_bound:
                return False, {
                    "index": i,
                    "key": key,
                    "value": value,
                    "low_bound": low_bound,
                    "high_bound": high_bound
                }
    return True, {}

def compare_results(result1, result2, allow_None=True, **kwargs):
    # per polugon
    for i in range(len(result1)):
        # per stat
        if allow_None and (result1[i] is None or result2[i] is None):
            continue

        for key in result1[i].keys():
            value1 = result1[i][key]
            value2 = result2[i][key]

            if not np.isclose(value1, value2, **kwargs):
                return False, {
                    "index": i,
                    "key": key,
                    "value1": result1[i][key],
                    "value2": result2[i][key]
                }
    return True, {}