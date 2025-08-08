import math
import os

import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import MultiPolygon, Polygon, box


def validate_raster_vector_compatibility(raster_file_path, vector_file_path):
    with rio.open(raster_file_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        vector = gpd.read_file(vector_file_path)
        vector_crs = vector.crs

        if not str(raster_crs) == str(vector_crs):
            raise ValueError("Raster and vector have different CRS")

        vector_bounds = vector.total_bounds
        intersect = box(*raster_bounds).intersects(box(*vector_bounds))
        if not intersect:
            raise ValueError("Raster and vector do not intersect")


def describe_raster_file(raster_file_path):
    with rio.open(raster_file_path) as src:
        name = os.path.basename(raster_file_path)
        return {
            "name": name,
            "shape": src.shape,
            "size_in_MB": os.path.getsize(raster_file_path) / (1024 * 1024),
            "dtype": src.dtypes[0],
            "bands": src.count,  # this should be always zero but just in case
        }


def describe_vector_file(vector_file_path):
    vector = gpd.read_file(vector_file_path)
    name = os.path.basename(vector_file_path)
    n_features = len(vector)
    geom_size_bytes = vector.geometry.to_wkb().nbytes

    def geom_summary(geom):
        if isinstance(geom, MultiPolygon):
            n_vertices = sum(len(p.exterior.coords) for p in geom.geoms)
            perim = sum(p.length for p in geom.geoms)
            area = sum(p.area for p in geom.geoms)
        elif isinstance(geom, Polygon):
            n_vertices = len(geom.exterior.coords)
            perim = geom.length
            area = geom.area
        else:
            raise ValueError("Invalid geometry type")

        compactness = 4 * math.pi * area / (perim**2) if perim > 0 else 0
        complexity = 1 - compactness

        return {"n_vertices": n_vertices, "complexity": complexity}

    vector_summary = pd.DataFrame(vector.geometry.apply(geom_summary).to_list())
    total_n_vertices = vector_summary["n_vertices"].sum()
    avg_complexity = vector_summary["complexity"].mean()

    return {
        "name": name,
        "n_features": n_features,
        "geom_size_in_B": geom_size_bytes,
        "total_n_vertices": int(total_n_vertices),
        "avg_complexity": float(avg_complexity),
    }
