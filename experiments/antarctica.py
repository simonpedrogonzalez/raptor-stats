import rasterio
from constants import RASTER_DATA_PATH, VECTOR_DATA_PATH, glaciers_folder, ice_thickness
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import os
from file_utils import describe_raster_file, describe_vector_file
from rasterstats import zonal_stats
import geopandas as gpd
from shapely import box


def get_resolution_in_meters(tif_path):
    with rasterio.open(tif_path) as src:
        res_x, res_y = src.res
        crs = src.crs

        print(f"CRS: {crs}")
        print(f"Pixel width: {res_x} {crs.linear_units}")
        print(f"Pixel height: {res_y} {crs.linear_units}")

        if crs.is_projected:
            print(f"Approx res: {abs(res_x)} x {abs(res_y)} meters")


def resample_raster(input_path, scale_factors=[2, 4]):
    with rasterio.open(input_path) as src:
        for scale in scale_factors:
            new_res = (src.res[0] * scale, src.res[1] * scale)
            new_width = int(src.width / scale)
            new_height = int(src.height / scale)

            transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )

            output_profile = src.profile.copy()
            output_profile.update({
                'height': new_height,
                'width': new_width,
                'transform': transform
            })

            output_path = os.path.splitext(input_path)[0] + f"_resampled_x{scale}.tif"

            with rasterio.open(output_path, 'w', **output_profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.average
                    )

            print(f"Resampled (x{scale}) saved to: {output_path}")


# raster = ice_thickness
# get_resolution_in_meters(raster)
# print(describe_raster_file(raster))
# resample_raster(raster, scale_factors=[25])

# get_resolution_in_meters(f"{ice_thickness}_resampled_x25.tif")
# print(describe_raster_file(f"{ice_thickness}_resampled_x25.tif"))
# get_resolution_in_meters(f"{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif")
# print(describe_raster_file(f"{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif"))


# with rasterio.open(raster) as src:
#     raster_bounds = src.bounds
#     raster_crs = src.crs
#     raster_geom = box(*raster_bounds)

# print("\n")
# print("--------------------")
# print("VECTOR DATA")

# file1 = f"{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp"
# file2 = f"{VECTOR_DATA_PATH}/cb_2018_us_cd116_500k.shp"
# file3 = f"{VECTOR_DATA_PATH}/cb_2018_us_county_20m.shp"
# file4 = f"{VECTOR_DATA_PATH}/cb_2018_us_county_500k.shp"

# files = [file2, file3, file4]

# import tqdm


# for file in tqdm.tqdm(files):

#     gdf = gpd.read_file(file)
#     # convert to file1 crs
#     gdf.to_crs(gpd.read_file(file1).crs, inplace=True)

#     gdf = gdf[gdf.geometry.within(raster_geom)]

#     raster = f"{RASTER_DATA_PATH}/US_MSR.tif"
#     stats = zonal_stats(gdf, raster, stats="count", geojson_out=True)
#     mask = []
#     for i, s in enumerate(stats):
#         if s["properties"]["count"] > 0:
#             mask.append(1)
#         else:
#             mask.append(0)

#     import numpy as np
#     mask = np.array(mask)
#     filtered_geoms = gdf[mask == 1]
    
#     filtered_gdf = gpd.GeoDataFrame(filtered_geoms, crs=gdf.crs)
#     filtered_gdf.to_file(file.replace(".shp", "_filtered.shp"))


# files1 = f"{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp"
# files2 = f"{VECTOR_DATA_PATH}/cb_2018_us_county_20m_filtered.shp"

# print(describe_vector_file(files1))
# print(describe_vector_file(files2))

# import os
# import sys
# import pandas as pd

# # get all shp files in the glaciers folder
# glaciers_files = [os.path.join(glaciers_folder, f) for f in os.listdir(glaciers_folder) if f.endswith('.shp')]

# # join them in a single geopandas dataframe
# glaciers_gdf = gpd.GeoDataFrame()
# for file in glaciers_files:
#     gdf = gpd.read_file(file)
#     # transform to Polar_Stereographic
#     gdf = gdf.to_crs(epsg=3031)  # Polar Stereographic South
#     glaciers_gdf = pd.concat([glaciers_gdf, gdf])
# # save the combined dataframe to a new file
# output_file = os.path.join(glaciers_folder, "combined_glaciers.shp")
# glaciers_gdf.to_file(output_file)
# print(f"Combined glaciers shapefile saved to: {output_file}")

from constants import antarctica_gjson, antarctica

# make it a polar_stereographic projection
antarctica_gdf = gpd.read_file(antarctica_gjson, driver='GeoJSON')
antarctica_gdf = antarctica_gdf.to_crs(epsg=3031)
antarctica_gdf.to_file(f"{VECTOR_DATA_PATH}/antarctica.shp")


# from constants import glaciers, ice_thickness_resampled_x25

# plot both the glaciers and the ice thickness raster in the same plot
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from constants import glaciers, ice_thickness_resampled_x25
gdf = gpd.read_file(antarctica)
import matplotlib.pyplot as plt
with rasterio.open(ice_thickness_resampled_x25) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, cmap='viridis', title='Ice Thickness')
    gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='black', label='Glaciers')
    plt.legend()
    plt.show()
    print('done')