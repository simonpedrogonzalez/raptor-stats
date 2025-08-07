from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH, US_counties, US_MSR_resampled_x4
from reference import reference_method
from raptorstats import zonal_stats


# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)


exp = Experiment(
    raster_path=US_MSR_resampled_x4,
    vector_path=US_counties,
    func=zonal_stats,
    reps=7,
    stats=["count"],
    check_results=True,
)

if __name__ == "__main__":
    exp.run()

