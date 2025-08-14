from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH, US_counties, US_MSR_upsampled_4, US_states
from raptorstats import zonal_stats
from raptorstats.raster_methods import RasterStatsMasking
from experiment_aggregator import ExperimentAggregator
from raptorstats.stats import Stats
from raptorstats.scanline import Scanline
from raptorstats.agqt import AggQuadTree

kwargs5 = {
    'method': "agqt",
    'max_depth': 5,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_US_counties_depth_5.idx",
    'build_index': True,
}

kwargs7 = {
    'method': "agqt",
    'max_depth': 7,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_US_counties_depth_7.idx",
    'build_index': True,
}

kwargs10 = {
    'method': "agqt",
    'max_depth': 10,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_US_counties_depth_10.idx",
    'build_index': True,
}

zonal_stats(US_counties, US_MSR_upsampled_4, stats="*", **kwargs5)
zonal_stats(US_counties, US_MSR_upsampled_4, stats="*", **kwargs7)
zonal_stats(US_counties, US_MSR_upsampled_4, stats="*", **kwargs10)

