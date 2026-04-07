import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA




def get_representation_config(config: dict) -> dict:
    """
    Return a normalized representation config.
    """
    rep = config.get("representation", {})

    rep_type = rep.get("type", "regular")
    if rep_type not in {"regular", "noise", "pca", "nn"}:
        raise ValueError(f"Unknown representation type: {rep_type}")

    return {
        "type": rep_type,
        "noise_std": float(rep.get("noise_std", 0.05)),
        "pca_components": rep.get("pca_components", 20),
        "nn_name": rep.get("nn_name", None),
        "nn_layer": rep.get("nn_layer", None),
        "nn_update_interval": rep.get("nn_update_interval", None),
        "random_seed": int(rep.get("random_seed", 0)),
    }

def get_representation_tag(rep_cfg: dict) -> str:
    """
    Return a short tag for filenames/results/caches.
    """
    rep_type = rep_cfg["type"]

    if rep_type == "regular":
        return "regular"

    if rep_type == "noise":
        noise_std = rep_cfg["noise_std"]
        return f"noise{str(noise_std).replace('.', 'p')}"

    if rep_type == "pca":
        ncomp = rep_cfg["pca_components"]
        return f"pca{ncomp}"

    if rep_type == "nn":
        nn_name = rep_cfg["nn_name"] or "stub"
        nn_layer = rep_cfg["nn_layer"]
        layer_tag = "none" if nn_layer is None else str(nn_layer)
        return f"nn_{nn_name}_layer{layer_tag}"

    raise ValueError(f"Unknown representation type: {rep_type}")


def get_base_features(dataset, metric):
    X, clusters = gl.datasets.load(dataset.split("-")[0], metric=metric)
    return X, clusters

def get_regular_features(X: np.ndarray, rep_cfg: dict | None = None) -> np.ndarray:
    """
    Baseline representation: unchanged features.
    """
    return np.asarray(X, dtype=float)


def get_noisy_features(X: np.ndarray, rep_cfg: dict) -> np.ndarray:
    """
    Add Gaussian noise to features.
    """
    X = np.asarray(X, dtype=float)
    noise_std = rep_cfg["noise_std"]
    seed = rep_cfg.get("random_seed", 0)

    rng = np.random.default_rng(seed)
    noise = noise_std * rng.standard_normal(X.shape)
    return X + noise


def get_pca_features(X: np.ndarray, rep_cfg: dict) -> np.ndarray:
    """
    PCA representation.
    """
    X = np.asarray(X, dtype=float)
    n_components = rep_cfg["pca_components"]
    seed = rep_cfg.get("random_seed", 0)

    if n_components is None:
        return X

    n_components = min(int(n_components), X.shape[0], X.shape[1])
    if n_components <= 0:
        raise ValueError(f"Invalid PCA n_components: {n_components}")

    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(X)


def get_nn_features(X: np.ndarray, rep_cfg: dict, labels=None, labeled_ind=None) -> np.ndarray:
    """
    Placeholder for future NN embeddings.

    For now, just return X unchanged so the plumbing works.
    Later this can:
      1. train/fine-tune a network on labeled data
      2. extract embeddings from a chosen layer
      3. return those embeddings for all points
    """
    return np.asarray(X, dtype=float)


def apply_representation(
    X: np.ndarray,
    rep_cfg: dict,
    *,
    labels=None,
    labeled_ind=None,
) -> np.ndarray:
    """
    Transform base features according to the representation config.
    """
    rep_type = rep_cfg["type"]

    if rep_type == "regular":
        X_rep = get_regular_features(X, rep_cfg)
    elif rep_type == "noise":
        X_rep = get_noisy_features(X, rep_cfg)
    elif rep_type == "pca":
        X_rep = get_pca_features(X, rep_cfg)
    elif rep_type == "nn":
        X_rep = get_nn_features(X, rep_cfg, labels=labels, labeled_ind=labeled_ind)
    else:
        raise ValueError(f"Unknown representation type: {rep_type}")

    print(
        f"Representation '{get_representation_tag(rep_cfg)}': "
        f"shape {X.shape} -> {X_rep.shape}"
    )
    return X_rep


def get_features(
    dataset: str,
    metric: str,
    rep_cfg: dict,
    *,
    labels=None,
    labeled_ind=None,
):
    """
    Load base features and apply the selected representation.
    Returns:
        X_rep, clusters
    """
    X, clusters = get_base_features(dataset, metric)
    X_rep = apply_representation(
        X,
        rep_cfg,
        labels=labels,
        labeled_ind=labeled_ind,
    )
    return X_rep, clusters