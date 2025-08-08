
import os
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine
import geopandas as gpd
from shapely.affinity import affine_transform
from constants import US_states, US_MSR_resampled_x10
from rasterio import warp
from rasterio import plot

# -------------------------------------------------------------------------
# CONFIG – tweak to taste
# -------------------------------------------------------------------------
raster_path  = US_MSR_resampled_x10            # e.g. ".../ice_thickness.tif"
vector_path  = US_states                # e.g. ".../us_states.shp"
shear_x_frac = 0.20   # 0.20 ⇒ every row shifts 20 % of a pixel eastward
shear_y_frac = 0.00   # non-zero would tilt columns, too
# -------------------------------------------------------------------------

sheared_raster = f"{raster_path.split('.')[0]}_sheared.tif"
sheared_vect   = f"{vector_path.split('.')[0]}_sheared.shp"

# --- step 1: open source raster ------------------------------------------------
with rio.open(raster_path) as src:
    # (a) build new affine with shear
    a = src.transform.a
    b = src.transform.b
    c = src.transform.c
    d = src.transform.d
    e = src.transform.e
    f = src.transform.f
    
    shear_x = shear_x_frac * a
    shear_y = shear_y_frac * abs(e)     # e is usually negative (north-up)
    dst_transform = Affine(a, shear_x, c,
                           shear_y, e, f)

    # (b) grow the output canvas so nothing falls off the right edge
    #     – conservative option: add |shear_x| * height extra columns
    extra_cols = int(np.ceil(abs(shear_x) * src.height / abs(a)))
    dst_width  = src.width  + extra_cols
    dst_height = src.height

    profile = src.profile.copy()
    profile.update({
        "transform": dst_transform,
        "width":     dst_width,
        "height":    dst_height
    })
    new_profile = profile.copy()

    # (c) write warped raster
    with rio.open(sheared_raster, "w", **profile) as dst:
        warp.reproject(
            source=rio.band(src, 1),
            destination=rio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest
        )

print(f"✅  Sheared raster written → {sheared_raster}")

# --- step 2: shear the vector layer so it still overlays the raster -----------
gdf = gpd.read_file(vector_path).to_crs(new_profile["crs"])

# shapely.affinity.affine_transform expects [a, b, d, e, xoff, yoff]
affine_vec = [1, shear_x / a, shear_y / abs(e), 1, 0, 0]
gdf["geometry"] = gdf["geometry"].apply(
    lambda geom: affine_transform(geom, affine_vec)
)

# gdf.to_file(sheared_vect, driver="GPKG")
print(f"✅  Sheared vector written → {sheared_vect}")

# read sheared files and plot both on the same map
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color="none", edgecolor="red", linewidth=0.5, label="Sheared vector")
with rio.open(sheared_raster) as src:
    rio.plot.show(src, ax=ax, cmap="viridis", label="Sheared raster")
plt.legend()
plt.title("Sheared Raster and Vector Overlay")
plt.tight_layout()
plt.show()
print('done')
