import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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

def representation_depends_on_seed(rep_cfg: dict) -> bool:
    return rep_cfg["type"] == "nn"

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
        nn_name = rep_cfg["nn_name"] or "mlp"
        nn_layer = rep_cfg["nn_layer"]
        return f"nn_{nn_name}_h{nn_layer}"

    raise ValueError(f"Unknown representation type: {rep_type}")


# NN model helpers
class MLPEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_embedding(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        if layer == 1:
            return x

        x = self.relu(self.fc2(x))
        if layer == 2:
            return x

        x = self.relu(self.fc3(x))
        if layer == 3:
            return x

        x = self.relu(self.fc4(x))
        if layer == 4:
            return x

        raise ValueError(f"nn_layer must be 1, 2, 3, or 4. Got {layer}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.classifier(x)

def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlp_embedding_model(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    rep_cfg: dict,
) -> MLPEmbeddingNet:
    if rep_cfg["nn_name"] != "mlp":
        raise NotImplementedError(f"NN type '{rep_cfg['nn_name']}' is not implemented yet.")

    device = get_torch_device()
    seed = rep_cfg.get("random_seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_labeled = np.asarray(X_labeled, dtype=np.float32)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)

    input_dim = X_labeled.shape[1]
    num_classes = len(np.unique(y_labeled))
    hidden_dim = 128
    epochs = 20
    batch_size = 64
    lr = 1e-3

    model = MLPEmbeddingNet(input_dim, hidden_dim, num_classes).to(device)

    dataset = TensorDataset(
        torch.from_numpy(X_labeled),
        torch.from_numpy(y_labeled),
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"[NN] Training {rep_cfg['nn_name']} on {len(X_labeled)} labeled points")
    print(f"[NN] Input dim = {X_labeled.shape[1]}, output classes = {len(np.unique(y_labeled))}")
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model

def extract_mlp_embeddings(
    model: MLPEmbeddingNet,
    X_all: np.ndarray,
    layer: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    X_all = np.asarray(X_all, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_all).to(device)
        emb = model.get_embedding(x, layer)
        emb = emb.cpu().numpy()

    return emb

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
    Train a fixed NN once on the initial labeled set and use the chosen hidden
    layer embedding for all points.
    """
    if labels is None or labeled_ind is None:
        raise ValueError("NN representation requires labels and labeled_ind.")

    if rep_cfg["nn_name"] == "cnn":
        raise NotImplementedError("CNN is not implemented yet.")

    layer = int(rep_cfg["nn_layer"])
    if layer not in {1, 2, 3, 4}:
        raise ValueError(f"nn_layer must be 1, 2, 3, or 4. Got {layer}.")

    X = np.asarray(X, dtype=np.float32)
    labeled_ind = np.asarray(labeled_ind, dtype=int)

    model = train_mlp_embedding_model(X[labeled_ind], labels[labeled_ind], rep_cfg)
    print(f"[NN] Finished training. Extracting hidden layer {rep_cfg['nn_layer']} embeddings.")
    X_emb = extract_mlp_embeddings(model, X, layer)

    return X_emb


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
