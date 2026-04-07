import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _summary_path(resultsdir: str, dataset: str, iters: int, modelname: str) -> Path:
    """Return the expected summary CSV path for a dataset/model."""
    return Path(resultsdir) / f"{dataset}_overall_{iters}" / f"{modelname}_stats.csv"


def _load_summary(resultsdir: str, dataset: str, iters: int, modelname: str) -> pd.DataFrame:
    """Load the summary CSV for a dataset/model."""
    path = _summary_path(resultsdir, dataset, iters, modelname)
    if not path.exists():
        raise FileNotFoundError(f"Could not find summary file:\n{path}")
    return pd.read_csv(path)


def _get_method_names(df: pd.DataFrame) -> list[str]:
    """Extract method names from columns ending in ' : avg'."""
    methods = []
    for col in df.columns:
        if col.endswith(" : avg"):
            methods.append(col[:-6])  # strip " : avg"
    return methods


def plot_summary(
    dataset: str,
    iters: int,
    modelname: str,
    resultsdir: str = "results",
    methods: Iterable[str] | None = None,
    show_std: bool = True,
    title: str | None = None,
) -> None:
    """
    Plot average accuracy curves from a summary CSV.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. 'mnistimb-mod3'
    iters : int
        Number of AL iterations used in the run, e.g. 3
    modelname : str
        Accuracy model summary to plot, e.g. 'rwll' or 'rwll0010'
    resultsdir : str
        Root results directory
    methods : iterable of str or None
        Exact method names to include, e.g. ['unc : rwll', 'random : rwll0010'].
        If None, plot all methods found.
    show_std : bool
        Whether to shade ±1 std when available.
    title : str or None
        Custom title. If None, generate one.
    """
    df = _load_summary(resultsdir, dataset, iters, modelname)
    all_methods = _get_method_names(df)

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
    plt.xticks(x)
    plt.title(title or f"{dataset} — {modelname}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def list_methods(dataset: str, iters: int, modelname: str, resultsdir: str = "results") -> list[str]:
    """Print and return the method names available in a summary CSV."""
    df = _load_summary(resultsdir, dataset, iters, modelname)
    methods = _get_method_names(df)
    print("\n".join(methods))
    return methods
