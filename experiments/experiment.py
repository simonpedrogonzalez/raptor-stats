from constants import RESULTS_PATH
import pandas as pd
import gc
import tqdm
from profiling_tools import profile
from reference import reference_method, result_within_tolerance
import datetime
from file_utils import describe_raster_file, describe_vector_file, validate_raster_vector_compatibility
import json
import seaborn as sns
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self,
    raster_path, vector_path, func, reps=1, stats=['count'],
    check_results=False, plot_results=False,
    csv_output=True, json_output=False,
    ):
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.func = func
        self.reps = reps
        self.stats = stats
        self.check_results = check_results
        self.plot_results = plot_results
        self.result_files = None
        self.csv_output = csv_output
        self.json_output = json_output


    def _reset(self):
        gc.collect()
        # I deactivated cache clearing until we are ready to run experiments
        # I should create a flag to enable/disable this
        # Clears both page cache and inodes/dentries cache assuming linux os
        # os.system("sync; echo 3 > /proc/sys/vm/drop_caches")
        # I dont think random seeds needs to be setted since all algos are deterministic

    def _write_results(self, metric_list, raw_data):

        print("Writing results...")

        now_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        now_string = f"{self.func.__name__}_{now_string}"

        exp_name = f"exp_{now_string}.json"
        metrics_df = pd.DataFrame(metric_list)
        metrics_summary = metrics_df.describe()

        self.result_files = {}

        if self.plot_results:
            # boxplot per metric in the same figure
            n_metrics = len(metrics_df.columns)
            fig, axs = plt.subplots(1, n_metrics, figsize=(15, 5))
            for i, col in enumerate(metrics_df.columns):
                sns.boxplot(data=metrics_df[col], ax=axs[i])
                axs[i].set_title(col)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_PATH}/boxplot_{now_string}.png")
            plt.clf()
            self.result_files["boxplot"] = f"{RESULTS_PATH}/boxplot_{now_string}.png"
            # lineplot per metric
            fig, axs = plt.subplots(n_metrics, 1, figsize=(5, 10))
            for i, col in enumerate(metrics_df.columns):
                sns.lineplot(data=metrics_df[col], ax=axs[i])
                # axs[i].set_title(col)
                # remove x label from all but the last plot
                if i < n_metrics - 1:
                    axs[i].set_xlabel("")
            plt.tight_layout()
            # remove vertical space between plots
            plt.subplots_adjust(hspace=0)
            plt.savefig(f"{RESULTS_PATH}/lineplot_{now_string}.png")
            plt.clf()
            plt.close()
            self.result_files["lineplot"] = f"{RESULTS_PATH}/lineplot_{now_string}.png"

        vector_summary = describe_vector_file(self.vector_path)
        raster_summary = describe_raster_file(self.raster_path)

        res = {
            "func": self.func.__name__,
            # "func_params": self.func._get_params(),
            "reps": self.reps,
            "stats": self.stats,
            "raster": raster_summary,
            "vector": vector_summary,
            "metrics": metrics_summary.to_dict()
        }

        if self.json_output:
            # write a json file with the results
            res_path = f"{RESULTS_PATH}/{exp_name}"
            with open(res_path, "w") as f:
                json.dump(res, f)
            self.result_files["json"] = res_path
        if self.csv_output:
            # write csv with raw data adding function name, raster and vector name and metrics
            raw_data["func"] = self.func.__name__
            raw_data["raster"] = raster_summary["name"]
            raw_data["vector"] = vector_summary["name"]
            for k,v in raster_summary.items():
                if k != "name":
                    if k == "shape":
                        v = str(v)
                    raw_data["raster_" + k] = v
            for k,v in vector_summary.items():
                if k != "name":
                    raw_data["vector_" + k] = v
            
            # for k,v in self.func._get_params().items():
            #     raw_data["func_param_" + k] = v

            raw_data.to_csv(f"{RESULTS_PATH}/exp_{now_string}.csv", index=False)
            self.result_files["csv"] = f"{RESULTS_PATH}/exp_{now_string}.csv"

        print(f"Results written to {self.result_files}")

    def run(self):

        # if 'count' not in self.stats:
        #     self.stats.append("count")

        print(f"Starting experiment with {self.reps} repetitions")
        print(f"Function: {self.func.__name__}")
        print(f"Raster: {self.raster_path}")
        print(f"Vector: {self.vector_path}")
        print(f"Stats: {self.stats}")

        validate_raster_vector_compatibility(self.raster_path, self.vector_path)
        truth = reference_method(self.raster_path, self.vector_path, self.stats)

        metric_list = []
        
        for i in tqdm.tqdm(range(self.reps)):
            self._reset()
            
            # I'm wrapping the func call with the profiler
            result, metrics = profile(self.func)(self.vector_path, self.raster_path, self.stats, categorical=True)
            
            if self.check_results:
                correct, errors = result_within_tolerance(truth[0], truth[1], result)
                if not correct:
                    try:
                        e_ind = errors["index"]
                        import geopandas as gpd
                        gdf = gpd.read_file(self.vector_path)
                        name = gdf.iloc[e_ind]["NAME"]
                        # show the name of the polygon
                        print(f"Error in polygon {e_ind} ({name}): {errors}")
                    except Exception as e:
                        pass
                    raise ValueError(f"Error in result: {errors}")
            
            metric_list.append(metrics)
        
        print("Experiment finished")
        print("Results:")
        raw_data = pd.DataFrame(metric_list)
        desc = raw_data.describe()
        print(desc.loc[["mean", "std"], :])

        self._write_results(metric_list, raw_data)
        return
