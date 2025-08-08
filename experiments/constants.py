import os

RESULTS_PATH = "results"
TMP_PATH = "tmp"
DATA_PATH = "data"
RASTER_DATA_PATH = "data/raster"
VECTOR_DATA_PATH = "data/vector"
INDICES_PATH = "data/indices"

US_MSR_resampled_x2 = f"{RASTER_DATA_PATH}/US_MSR_resampled_x2.tif"
US_MSR_resampled_x4 = f"{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif"
US_MSR_resampled_x10 = f"{RASTER_DATA_PATH}/US_MSR_resampled_x10.tif"
US_MSR = f"{RASTER_DATA_PATH}/US_MSR.tif"
US_MSR_upsampled_2 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_2.tif"
US_MSR_upsampled_4 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_4.tif"
US_MSR_upsampled_5 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_5.tif"
US_MSR_upsampled_10 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_10.tif"

US_states = f"{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp"
US_counties = f"{VECTOR_DATA_PATH}/cb_2018_us_county_20m_filtered.shp"

ice_thickness = f"{RASTER_DATA_PATH}/bm3_thickness.tif"
ice_thickness_resampled_x25 = f"{RASTER_DATA_PATH}/bm3_thickness_resampled_x25.tif"
glaciers_folder = f"{VECTOR_DATA_PATH}/mcmlter-gis-watershed_shapefiles-20231020/mcmlter-gis-watershed-shapefiles/MDV_glaciers"
glaciers = f"{glaciers_folder}/combined_glaciers.shp"
antarctica_gjson = f"{VECTOR_DATA_PATH}/geoBoundaries-ATA-ADM0_simplified.geojson"
antarctica = f"{VECTOR_DATA_PATH}/antarctica.shp"

US_states_sheared = f"{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered_sheared.shp"
US_MSR_resampled_x10_sheared = f"{RASTER_DATA_PATH}/US_MSR_resampled_x10_sheared.tif"