from rasterstats import zonal_stats
import geopandas as gpd
import rasterio as rio
import numpy as np


def reference_method(raster_file_path, vector_file_path, stats):
    vector = gpd.read_file(vector_file_path)
    zs1 = zonal_stats(vector, raster_file_path, stats=stats, boundless=True, categorical=True)
    zs2 = zonal_stats(vector, raster_file_path, stats=stats, boundless=False, all_touched=True, categorical=True)
    # return one dict for each polygon [ {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'count': 1}, ... ]
    return zs1, zs2

def extract_histogram(rasterstat_res):
    # extract all "number" keys from the result
    histograms = []
    omit = []
    for res in rasterstat_res:
        hist = {}
        for key, value in res.items():
            if isinstance(key, (int, float)):
                hist[key] = value
                omit.append(key)
        histograms.append(hist)
    
    new_res = []
    omit = set(omit)
    for i in range(len(rasterstat_res)):
        nr = rasterstat_res[i].copy()
        nr['histogram'] = histograms[i]
        for key in omit:
            if key in nr:
                del nr[key]
        new_res.append(nr)
    return new_res

def compare_histograms(hist1, hist2):
    # compare two histograms
    for key in hist1.keys():
        if key not in hist2:
            return False, {
                "error": "Key not found in second histogram",
                "key": key,
                "value1": hist1[key],
                "value2": None
            }
        if not np.isclose(hist1[key], hist2[key]):
            return False, {
                "error": "Value differs in histograms",
                "key": key,
                "value1": hist1[key],
                "value2": hist2[key]
            }
    return True, {}

def incorrect_val(value, lower_bound, high_bound):
    return not np.allclose(value, high_bound) and \
           not np.allclose(value, lower_bound) and \
           (value < lower_bound or value > high_bound)

def result_within_tolerance(center_based_result, all_touched_result, result, allow_None=True):
    
    all_touched_result = extract_histogram(all_touched_result)
    center_based_result = extract_histogram(center_based_result)

    # per polugon
    for i in range(len(result)):
        # per stat
        if allow_None and (result[i] is None or center_based_result[i] is None or all_touched_result[i] is None):
            continue
        
        h1 = result[i].get("histogram", {})
        h2 = center_based_result[i].get("histogram", {})
        h3 = all_touched_result[i].get("histogram", {})
        correct, errors = compare_histograms(h1, h2)
        if not correct:
            return correct, errors

        for key in result[i].keys():
            if key == "histogram":
                continue
            b1 = all_touched_result[i][key]
            b2 = center_based_result[i][key]
            high_bound = max(b1, b2)
            low_bound = min(b1, b2)
            value = result[i][key]
            
            if incorrect_val(value, low_bound, high_bound):
                if key == "majority" or key == "minority":
                    # ignore tie cases
                    m1 = h1[value]
                    m2 = h2[value]
                    m3 = h3[value]
                    if m1 == m2 or m1 == m3:
                        continue
                    
                return False, {
                    "index": i,
                    "key": key,
                    "value": value,
                    "low_bound": low_bound,
                    "high_bound": high_bound
                }
    return True, {}


def exact_match(center_based_result, result, allow_None=True):
    
    center_based_result = extract_histogram(center_based_result)

    # per polugon
    for i in range(len(result)):
        # per stat
        if allow_None and (result[i] is None or center_based_result[i] is None):
            continue
        
        h1 = result[i].get("histogram", {})
        h2 = center_based_result[i].get("histogram", {})
        correct, errors = compare_histograms(h1, h2)
        if not correct:
            return correct, errors

        for key in result[i].keys():
            if key == "histogram":
                continue
            b2 = center_based_result[i][key]
            value = result[i][key]
            
            if not np.allclose(b2, value):
                if key == "majority" or key == "minority":
                    # ignore tie cases
                    m1 = h1[value]
                    m2 = h2[value]
                    if m1 == m2:
                        continue
                    
                return False, {
                    "index": i,
                    "key": key,
                    "value": value,
                    "truth": b2,
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