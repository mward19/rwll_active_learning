import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import subprocess
import re

from utils_representations import get_representation_config, get_representation_tag


def _get_rep_info(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    rep_cfg = get_representation_config(config)
    rep_tag = get_representation_tag(rep_cfg)
    rate = config.get("initial_labels_per_class", 1)
    return rep_cfg, rep_tag, rate


def _summary_path(
    resultsdir: str,
    dataset: str,
    iters: int,
    modelname: str,
    config_path: str = "./config.yaml",
) -> Path:
    _, rep_tag, rate = _get_rep_info(config_path)
    return Path(resultsdir) / f"{dataset}_{rep_tag}_r{rate}_overall_{iters}" / f"{modelname}_stats.csv"


def _load_summary(
    resultsdir: str,
    dataset: str,
    iters: int,
    modelname: str,
    config_path: str = "./config.yaml",
) -> pd.DataFrame:
    path = _summary_path(resultsdir, dataset, iters, modelname, config_path=config_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find summary file:\n{path}")
    return pd.read_csv(path)


def _get_method_names(df: pd.DataFrame) -> list[str]:
    methods = []
    for col in df.columns:
        if col.endswith(" : avg"):
            methods.append(col[:-6])
    return methods


def get_file_path(
        resultsdir: str = "results",
        config_type: str="nn", 
        nn_layer: int=2, 
        nn_update_interval: int=0
    ):
    cmd = f"grep -ril 'type: regular' ./{resultsdir}"
    if config_type == "nn":
        cmd = f"grep -ril 'type: {config_type}' ./{resultsdir} | xargs grep -ril 'nn_layer: {nn_layer}' | xargs grep -ril 'nn_update_interval: {nn_update_interval}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    paths = result.stdout.strip().split('\n')

    return paths[0]


def get_data_from_path(path: str, decay: bool=False):
    with open(path,'r') as f:
        data = f.read()

    yaml_text = re.search(r"acqs_models[\s\S]*",data).group(0)

    with open('temp.yaml','w') as f:
        f.write(yaml_text) 

    base_path = path.split('/')[:-1]
    df = pd.read_csv('/'.join(base_path) + '/rwll_stats.csv')
    all_methods = _get_method_names(df)

    if decay and 'uncnormdecaytau : rwll0010' not in all_methods:
        df2 = pd.read_csv('/'.join(base_path) + '/rwll0010_stats.csv')
        df = pd.concat([df, df2], axis=1)
        all_methods = _get_method_names(df)

    _, rep_tag, rate = _get_rep_info("temp.yaml")
    
    return df, rep_tag, rate, all_methods


def plot_summary(
    dataset: str,
    iters: int,
    modelname: str,
    resultsdir: str = "results",
    config_path: str = "./config.yaml",
    methods: Iterable[str] | None = None,
    show_std: bool = True,
    title: str | None = None,
    config_type: str="nn",
    nn_layer: int=2,
    nn_update_interval: int=0,
    decay: bool=True
) -> None:
    # getting the correct file path
    path = get_file_path(resultsdir, config_type, nn_layer, nn_update_interval)
    df, rep_tag, rate, all_methods = get_data_from_path(path, decay)


    if methods is None:
        methods = all_methods
    else:
        methods = [m for m in methods if m in all_methods]

    if not methods:
        raise ValueError("No matching methods found in the summary CSV.")

    x = list(range(len(df)))

    plt.figure(figsize=(9, 5))

    for method in methods:
        avg_col = f"{method} : avg"
        std_col = f"{method} : std"

        y = df[avg_col]
        plt.plot(x, y, marker="o", label=method)

        if show_std and std_col in df.columns:
            s = df[std_col]
            plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Active Learning Step")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x[::5])
    plt.title(title or f"{dataset} — {modelname} — {rep_tag} — r={rate}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_methods_across_update_intervals(
    dataset: str,
    iters: int,
    modelname: str,
    resultsdir: str = "results",
    config_path: str = "./config.yaml",
    method: Iterable[str] | None = None,
    show_std: bool = True,
    title: str | None = None,
    config_type: str="nn",
    nn_layer: int=2,
    nn_update_intervals: Iterable[int]=[0,5,50],
    decay: bool=True
):
    plt.figure(figsize=(9, 5))

    path = get_file_path(resultsdir, config_type="regular")
    df, rep_tag, rate, _ = get_data_from_path(path, decay)
    x = list(range(len(df)))
    avg_col = f"{method} : avg"
    std_col = f"{method} : std"

    y = df[avg_col]
    plt.plot(x, y, marker="o", label=f"{method} : regular")

    if show_std and std_col in df.columns:
        s = df[std_col]
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    for nn_update_interval in nn_update_intervals:
        path = get_file_path(resultsdir, config_type, nn_layer, nn_update_interval)
        df, rep_tag, rate, _ = get_data_from_path(path, decay)

        x = list(range(len(df)))

        avg_col = f"{method} : avg"
        std_col = f"{method} : std"

        y = df[avg_col]
        plt.plot(x, y, marker="o", label=f"{method} : {nn_update_interval}")

        if show_std and std_col in df.columns:
            s = df[std_col]
            plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Active Learning Step")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x[::5])
    plt.title(title or f"{dataset} — {modelname} — {rep_tag} — r={rate}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_methods_across_nn_layers(
    dataset: str,
    iters: int,
    modelname: str,
    resultsdir: str = "results",
    config_path: str = "./config.yaml",
    method: Iterable[str] | None = None,
    show_std: bool = True,
    title: str | None = None,
    config_type: str="nn",
    nn_layers: Iterable[int]=[2,4,6,8,10,12],
    nn_update_interval: Iterable[int]=0,
    decay: bool=True
):
    plt.figure(figsize=(9, 5))

    path = get_file_path(resultsdir, config_type="regular")
    df, rep_tag, rate, _ = get_data_from_path(path, decay)
    x = list(range(len(df)))
    avg_col = f"{method} : avg"
    std_col = f"{method} : std"

    y = df[avg_col]
    plt.plot(x, y, marker="o", label=f"{method} : regular")

    if show_std and std_col in df.columns:
        s = df[std_col]
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    for nn_layer in nn_layers:
        path = get_file_path(resultsdir, config_type, nn_layer, nn_update_interval)
        df, rep_tag, rate, _ = get_data_from_path(path, decay)

        x = list(range(len(df)))

        avg_col = f"{method} : avg"
        std_col = f"{method} : std"

        y = df[avg_col]
        plt.plot(x, y, marker="o", label=f"{method} : {nn_layer}")

        if show_std and std_col in df.columns:
            s = df[std_col]
            plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Active Learning Step")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x[::5])
    plt.title(title or f"{dataset} — {modelname} — {rep_tag} — r={rate}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def list_methods(
    dataset: str,
    iters: int,
    modelname: str,
    resultsdir: str = "results",
    config_path: str = "./config.yaml",
    slurm_id: str | None = None
) -> list[str]:
    path = subprocess.run(['grep','-ril',f'{slurm_id}','./results'],capture_output=True, text=True).stdout.strip()
    df = pd.read_csv(path)

    # df = _load_summary(resultsdir, dataset, iters, modelname, config_path=config_path)
    methods = _get_method_names(df)
    print("\n".join(methods))
    return methods