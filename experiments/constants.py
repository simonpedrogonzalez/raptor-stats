import os

RESULTS_PATH = "results"
TMP_PATH = "tmp"
DATA_PATH = "data"
RASTER_DATA_PATH = "data/raster"
VECTOR_DATA_PATH = "data/vector"
INDICES_PATH = "data/indices"

US_MSR_resampled_x2 = f"{RASTER_DATA_PATH}/US_MSR_resampled_x2.tif"
US_MSR_resampled_x4 = f"{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif"
US_MSR = f"{RASTER_DATA_PATH}/US_MSR.tif"
US_MSR_upsampled_2 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_2.tif"
US_MSR_upsampled_4 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_4.tif"
US_MSR_upsampled_5 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_5.tif"
US_MSR_upsampled_10 = f"{RASTER_DATA_PATH}/US_MSR_upsampled_10.tif"

US_states = f"{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp"
US_counties = f"{VECTOR_DATA_PATH}/cb_2018_us_county_20m_filtered.shp"