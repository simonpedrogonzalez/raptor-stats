from constants import RESULTS_PATH
import pandas as pd
import gc
import tqdm
from reference import reference_method, result_within_tolerance
import datetime
from file_utils import describe_raster_file, describe_vector_file, validate_raster_vector_compatibility
import json
import seaborn as sns
import matplotlib.pyplot as plt

class ExperimentAggregator:

    def __init__(self, exp_list):
        self.exp_list = exp_list
        self.results_files = []
    
    def run(self):
        for exp in self.exp_list:
            exp.run()
            self.results_files.append(exp.result_files)
        self._write_results()

    def _write_results(self):

        df = pd.DataFrame()
        for rf in self.results_files:
            res = pd.read_csv(rf['csv'])
            df = pd.concat([df, res], ignore_index=True)
        
        # Get comparison groups
        # That is, am I comparing different functions, different datasets, etc
        possible_comparison_cols = [
            'raster',
            'vector',
            'func'
        ]
        actual_comparison_cols = []
        for col in possible_comparison_cols:
            if df[col].nunique() > 1:
                actual_comparison_cols.append(col)
        # Group by function and check if param columns, starting with func_param_
        # are different
        param_cols = [col for col in df.columns if col.startswith('func_param_')]
        for col in param_cols:
            if df[col].nunique() > 1:
                actual_comparison_cols.append(col)
        
        # Create a new column with the comparison groups
        df['comparison_group'] = df[actual_comparison_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        # remove any _nan from the comparison group
        df['comparison_group'] = df['comparison_group'].str.replace('_nan', '')

        metrics = [
            "total_time_s",
            "cpu_time_s",
            "memory_peak_GB",
            "io_read_MB",
            "io_read_time_s",
        ]

        n_groups = df['comparison_group'].nunique()
        width = 7
        height = 5
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # create a directory for the results
        dir_name = f"{RESULTS_PATH}/{run_name}"
        # create the directory if it doesn't exist
        import os
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        metric_dict = {
            "total_time_s": "Total Time (s)",
            "cpu_time_s": "CPU Time (s)",
            "memory_peak_GB": "Memory Peak (GB)",
            "io_read_MB": "IO Read (MB)",
            "io_read_time_s": "IO Read Time (s)",
        }

        all_groups = df['comparison_group'].unique()
        palette_colors = sns.color_palette("Set1", n_colors=len(all_groups))
        group_color_mapping = dict(zip(all_groups, palette_colors))

        sns.set(style="whitegrid")

        for metric in metrics:
            # create a boxplot for this metric comparing
            # the different comparison groups

            means_PerGroup = df.groupby('comparison_group')[metric].mean()
            # get a list of the ordered groups by desc mean
            ordered_groups = means_PerGroup.sort_values(ascending=False).index.tolist()


            fig, ax = plt.subplots(figsize=(width, height))
            
            sns.boxplot(y='comparison_group', x=metric, data=df, ax=ax, order=ordered_groups, palette=group_color_mapping, width=0.5)
            # ax.set_title(metric)
            ax.set_ylabel("")
            ax.set_xlabel(metric_dict[metric])
            plt.tight_layout()
            plt.savefig(f"{dir_name}/{metric}.svg")
            plt.clf()
            plt.close()


def test():
    expagg = ExperimentAggregator([])
    import os
    for file in os.listdir(RESULTS_PATH):
        if file.endswith(".csv"):
            expagg.results_files.append(os.path.join(RESULTS_PATH, file))
    expagg.results_files = [{
        "csv": file,
    } for file in expagg.results_files]
    expagg._write_results()

# test()
    
