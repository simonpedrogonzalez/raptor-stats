from experiment import Experiment
from constants import antarctica, ice_thickness_resampled_x25
from reference import reference_method
from raptorstats import zonal_stats


# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)


exp = Experiment(
    raster_path=ice_thickness_resampled_x25,
    vector_path=antarctica,
    func=zonal_stats,
    reps=1,
    stats="*",
    check_results=True,
)

if __name__ == "__main__":
    exp.run()

