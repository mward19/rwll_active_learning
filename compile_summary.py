#compile_summary.py
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import os
import numpy as np
from glob import glob
from functools import reduce
import yaml
from utils_representations import get_representation_config, get_representation_tag

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Summary Stats of Active Learning Tests")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    
    #-----
    # New representation tagging
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    rep_cfg = get_representation_config(config)
    rep_tag = get_representation_tag(rep_cfg)
    print("Using representation:", rep_tag)
    #-----
    rate = config.get("initial_labels_per_class", 1)

    # Get average and std curves over all tests
    overall_results_dir = os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_overall_{args.iters}") # Changed
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)

    results_models_directories = glob(os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_results_*_{args.iters}", "*/")) # Changed
    # acc_model_names_list = np.unique([fpath.split("/")[-2] for fpath in results_models_directories]) # This was designed for mac
    acc_model_names_list = np.unique([os.path.basename(os.path.dirname(fpath)) for fpath in results_models_directories]) # This works on windows (and should generally i think)
    for acc_model_name in tqdm(acc_model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(acc_model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats.csv")
        acc_files = glob(os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_results_*_{args.iters}", acc_model_name, "accs.csv")) # Changed
        dfs = []
        err_string = ""
        for f in sorted(acc_files):
            try:
                dfs.append(pd.read_csv(f))
            except:
                run_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
                err_string += run_name.split("_results_")[-1].split("_")[0] + ", "
        
        if len(dfs) == 0:
            continue
        possible_columns = reduce(np.union1d, [df.columns for df in dfs])
        all_columns = {}
        for col in possible_columns:
            vals = np.array([df[col].values for df in dfs if col in df.columns])
            all_columns[col + " : avg"] = np.average(vals, axis=0)
            all_columns[col + " : std"] = np.std(vals, axis=0)

        all_df = pd.DataFrame(all_columns)
        all_df.to_csv(overall_results_file, index=None)
        
        if len(err_string) > 0:
            print(f"Error with {acc_model_name} and seeds {err_string}")
        
        
        
        # Do metrics summary
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats_metrics.csv")
        metric_files = glob(os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_results_*_{args.iters}", acc_model_name, "metrics.csv"))
        dfs = []
        err_string = ""
        for f in sorted(metric_files):
            try:
                dfs.append(pd.read_csv(f))
            except:
                err_string += f.split("/")[1].split("_")[-2] + ", "
                
        if len(dfs) == 0:
            continue
        possible_columns = reduce(np.union1d, [df.columns for df in dfs])
        all_columns = {}
        for col in possible_columns:
            vals = np.array([df[col].values for df in dfs if col in df.columns])
            all_columns[col + " : avg"] = np.average(vals, axis=0)
            all_columns[col + " : std"] = np.std(vals, axis=0)

        all_df = pd.DataFrame(all_columns)
        all_df.to_csv(overall_results_file, index=None)
    print("-"*40)
    
    
    

    # Get average and std curves over all tests
    overall_results_dir = os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_overall_{args.iters}") # Changed
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)

    results_models_directories = glob(os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_results_*_{args.iters}", "*/")) # Changed
    # acc_model_names_list = np.unique([fpath.split("/")[-2] for fpath in results_models_directories]) # This was designed for mac
    acc_model_names_list = np.unique([os.path.basename(os.path.dirname(fpath)) for fpath in results_models_directories]) # This works on windows (and should generally i think)

    for acc_model_name in tqdm(acc_model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(acc_model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats_metrics.csv")
        metric_files = glob(os.path.join(args.resultsdir, f"{args.dataset}_{rep_tag}_r{rate}_results_*_{args.iters}", acc_model_name, "metrics.csv"))
        dfs = [pd.read_csv(f) for f in sorted(metric_files)]
        if len(dfs) == 0:
            continue
        possible_columns = reduce(np.union1d, [df.columns for df in dfs])
        all_columns = {}
        for col in possible_columns:
            vals = np.array([df[col].values for df in dfs if col in df.columns])
            all_columns[col + " : avg"] = np.average(vals, axis=0)
            all_columns[col + " : std"] = np.std(vals, axis=0)

        all_df = pd.DataFrame(all_columns)
        all_df.to_csv(overall_results_file, index=None)
    print("-"*40)