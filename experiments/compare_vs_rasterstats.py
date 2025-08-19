from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH, US_counties, US_MSR_upsampled_4, US_states, US_MSR_resampled_x10
from raptorstats import zonal_stats
from raster_methods import RasterStatsMasking
from experiment_aggregator import ExperimentAggregator
from raptorstats.stats import Stats
from raptorstats.scanline import Scanline
from raptorstats.agqt import AggQuadTree
from raptorstats.io import open_raster, open_vector, validate_is_north_up, validate_raster_vector_compatibility

kwargs5 = {
    'max_depth': 5,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_US_counties_depth_5.idx",
    'build_index': False,
}
kwargs7 = {
    'max_depth': 7,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_depth_7.idx",
    'build_index': False,
}
kwargs8 = {
    'max_depth': 8,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_depth_8.idx",
    'build_index': False,
}

kwargs9 = {
    'max_depth': 9,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_depth_9.idx",
    'build_index': False,
}

kwargs10 = {
    'max_depth': 10,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_depth_10.idx",
    'build_index': False,
}

# kwargs5 = {
#     'max_depth': 5,
#     'index_path': f"experiments/indices/US_MSR_resampled_x10_US_states_depth_5.idx",
#     'build_index': False,
# }
# kwargs7 = {
#     'max_depth': 7,
#     'index_path': f"experiments/indices/US_MSR_resampled_x10_US_states_depth_7.idx",
#     'build_index': False,
# }
# kwargs10 = {
#     'max_depth': 10,
#     'index_path': f"experiments/indices/US_MSR_resampled_x10_US_counties_depth_10.idx",
#     'build_index': False,
# }

# Some wrappers because Experiment expects some specific
# objects. I should make a better API later
class Agqt5:

    __name__ = "AggQuadTree Depth=5"
    def __call__(self, vectors, raster, stats):
        stats_conf = Stats(stats)
        func = AggQuadTree(**kwargs5)
        with open_raster(raster) as ds:
            gdf = open_vector(vectors)
            validate_is_north_up(ds.transform)
            validate_raster_vector_compatibility(ds, gdf)
            results = func(ds, gdf, stats=stats_conf)
            results = stats_conf.clean_results(results)
            return results

class Agqt7:

    __name__ = "AggQuadTree Depth=7"
    def __call__(self, vectors, raster, stats):
        stats_conf = Stats(stats)
        func = AggQuadTree(**kwargs7)
        with open_raster(raster) as ds:
            gdf = open_vector(vectors)
            validate_is_north_up(ds.transform)
            validate_raster_vector_compatibility(ds, gdf)
            results = func(ds, gdf, stats=stats_conf)
            results = stats_conf.clean_results(results)
            return results
        
class Agqt10:
    __name__ = "AggQuadTree Depth=10"
    def __call__(self, vectors, raster, stats):
        stats_conf = Stats(stats)
        func = AggQuadTree(**kwargs10)
        with open_raster(raster) as ds:
            gdf = open_vector(vectors)
            validate_is_north_up(ds.transform)
            validate_raster_vector_compatibility(ds, gdf)
            results = func(ds, gdf, stats=stats_conf)
            results = stats_conf.clean_results(results)
            return results
        

class Agqt8:
    __name__ = "AggQuadTree Depth=8"
    def __call__(self, vectors, raster, stats):
        stats_conf = Stats(stats)
        func = AggQuadTree(**kwargs8)
        with open_raster(raster) as ds:
            gdf = open_vector(vectors)
            validate_is_north_up(ds.transform)
            validate_raster_vector_compatibility(ds, gdf)
            results = func(ds, gdf, stats=stats_conf)
            results = stats_conf.clean_results(results)
            return results

class Agqt9:
    __name__ = "AggQuadTree Depth=9"
    def __call__(self, vectors, raster, stats):
        stats_conf = Stats(stats)
        func = AggQuadTree(**kwargs9)
        with open_raster(raster) as ds:
            gdf = open_vector(vectors)
            validate_is_north_up(ds.transform)
            validate_raster_vector_compatibility(ds, gdf)
            results = func(ds, gdf, stats=stats_conf)
            results = stats_conf.clean_results(results)
            return results

class ScanlineMethod:
    __name__ = "Scanline"
    def __call__(self, vectors, raster, stats):
        stats_conf = Stats(stats)
        func = Scanline()
        with open_raster(raster) as ds:
            gdf = open_vector(vectors)
            validate_is_north_up(ds.transform)
            validate_raster_vector_compatibility(ds, gdf)
            results = func(ds, gdf, stats=stats_conf)
            results = stats_conf.clean_results(results)
            return results


methods = [ 
            RasterStatsMasking(),
            ScanlineMethod(),
            Agqt5(),
            Agqt7(),
            # Agqt8(),
            # Agqt9(),
            # Agqt10(),
        ]

exps1 = [
    Experiment(
        raster_path=US_MSR_upsampled_4,
        vector_path=US_states,
        # raster_path=US_MSR_resampled_x10,
        # vector_path=US_states,
        func=method,
        reps=7,
        stats="*",
        check_results=False,
    ) for method in methods
]

methods2 = [
    RasterStatsMasking(),
    ScanlineMethod(),
]

exps2 = [
    Experiment(
        raster_path=US_MSR_upsampled_4,
        vector_path=US_counties,
        func=method,
        reps=7,
        stats="*",
        check_results=False,
    ) for method in methods2
]

if __name__ == "__main__":
    ExperimentAggregator(exps1).run()
    ExperimentAggregator(exps2).run()
