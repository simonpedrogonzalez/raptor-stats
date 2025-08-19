import hashlib

import numpy as np
import rasterio as rio


def compute_file_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            h.update(block)
    return h.hexdigest()


def compare_histograms(hist1, hist2):
    # compare two histograms
    if len(hist1) != len(hist2):
        return False, {
            "error": "Histograms have different lengths",
            "length1": len(hist1),
            "length2": len(hist2),
        }

    for key in hist1.keys():
        if key not in hist2:
            return False, {
                "error": "Key not found in second histogram",
                "key": key,
                "value1": hist1[key],
                "value2": None,
            }
        if not np.isclose(hist1[key], hist2[key]):
            return False, {
                "error": "Value differs in histograms",
                "key": key,
                "value1": hist1[key],
                "value2": hist2[key],
            }
    return True, {}


def compare_stats_exact_match(reference, result):

    for i in range(len(result)):
        h1 = result[i].get("histogram", {})
        h2 = reference[i].get("histogram", {})
        h1 = {float(k): v for k, v in h1.items()}
        h2 = {float(k): v for k, v in h2.items()}
        # check histograms
        correct, errors = compare_histograms(h1, h2)
        if not correct:
            errors["index"] = i
            return correct, errors
        for key in result[i].keys():
            if key == "histogram":
                continue
            b2 = reference[i][key]
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


def make_raster(data, transform, crs):
    memfile = rio.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs=crs,
    ) as ds:
        ds.write(data, 1)
    return memfile.open()


def get_reference_mean(geom, raster):
    # Burn polygon into raster and compute masked mean
    mask = rio.features.rasterize(
        [geom], out_shape=raster.shape, transform=raster.transform
    )
    band = raster.read(1, masked=True)
    return band[mask == 1].mean()
