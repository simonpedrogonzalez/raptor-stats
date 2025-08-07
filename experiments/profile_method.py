from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH
from reference import reference_method
from raptorstats import zonal_stats

vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif'

# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)


exp = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=zonal_stats,
    reps=1,
    stats=["count"],
    check_results=True,
)

if __name__ == "__main__":
    exp.run()

