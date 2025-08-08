from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH, US_counties, US_MSR_upsampled_4, US_states
from raptorstats import zonal_stats
from raptorstats.raster_methods import RasterStatsMasking
from experiment_aggregator import ExperimentAggregator

# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)

methods = [ RasterStatsMasking(), zonal_stats ]

exps = [
    Experiment(
        raster_path=US_MSR_upsampled_4,
        vector_path=US_counties,
        func=method,
        reps=7,
        stats="*",
        check_results=True,
    ) for method in methods
]

if __name__ == "__main__":
    ExperimentAggregator(exps).run()

