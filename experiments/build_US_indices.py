from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH, US_states, US_MSR_upsampled_4, US_states, US_MSR_resampled_x10, US_counties
from raptorstats import zonal_stats, build_agqt_index
from experiment_aggregator import ExperimentAggregator
from raptorstats.stats import Stats
from raptorstats.scanline import Scanline
from raptorstats.agqt import AggQuadTree

# kwargs5 = {
#     'method': "agqt",
#     'max_depth': 5,
#     'index_path': f"experiments/indices/US_MSR_upsampled_4_US_states_depth_5.idx",
#     'build_index': True,
# }

# kwargs7 = {
#     'method': "agqt",
#     'max_depth': 7,
#     'index_path': f"experiments/indices/US_MSR_upsampled_4_US_states_depth_7.idx",
#     'build_index': True,
# }

kwargs8 = {
    'method': "agqt",
    'max_depth': 8,
    'index_path': f"experiments/indices/US_MSR_upsampled_4_US_states_depth_8.idx",
    'build_index': True,
}

# kwargs5 = {
#     'method': "agqt",
#     'max_depth': 5,
#     'index_path': f"experiments/indices/US_MSR_resampled_x10_US_states_depth_5.idx",
#     'build_index': True,
# }
# kwargs7 = {
#     'method': "agqt",
#     'max_depth': 7,
#     'index_path': f"experiments/indices/US_MSR_resampled_x10_US_states_depth_7.idx",
#     'build_index': True,
# }
# kwargs10 = {
#     'method': "agqt",
#     'max_depth': 10,
#     'index_path': f"experiments/indices/US_MSR_resampled_x10_US_states_depth_10.idx",
#     'build_index': True,
# }


# zonal_stats(US_states, US_MSR_resampled_x10, stats="*", **kwargs5)
# zonal_stats(US_states, US_MSR_resampled_x10, stats="*", **kwargs7)
build_agqt_index(US_MSR_upsampled_4, stats="*", max_depth=10, index_path=f"experiments/indices/US_MSR_upsampled_4_depth_10.idx")

