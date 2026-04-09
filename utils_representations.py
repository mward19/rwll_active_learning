#utils_representations.py
import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


def get_representation_config(config: dict) -> dict:
    """
    Return a normalized representation config.
    """
    rep = config.get("representation", {})

    rep_type = rep.get("type", "regular")
    if rep_type not in {"regular", "noise", "pca", "nn"}:
        raise ValueError(f"Unknown representation type: {rep_type}")

    nn_name = rep.get("nn_name", "mlp")
    if nn_name not in {"mlp", "cnn"}:
        raise ValueError(f"Unknown nn_name: {nn_name}")

    return {
        "type": rep_type,
        "noise_std": float(rep.get("noise_std", 0.05)),
        "pca_components": rep.get("pca_components", 20),

        "nn_name": nn_name,
        "nn_layer": int(rep.get("nn_layer", 4)),
        "nn_hidden_dim": int(rep.get("nn_hidden_dim", 128)),
        "nn_num_hidden_layers": int(rep.get("nn_num_hidden_layers", 4)),
        "nn_epochs": int(rep.get("nn_epochs", 20)),
        "nn_batch_size": int(rep.get("nn_batch_size", 64)),
        "nn_lr": float(rep.get("nn_lr", 1e-3)),

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
        seed = rep_cfg.get("random_seed", 0)
        return f"noise{str(noise_std).replace('.', 'p')}_s{seed}"

    if rep_type == "pca":
        ncomp = rep_cfg["pca_components"]
        seed = rep_cfg.get("random_seed", 0)
        return f"pca{ncomp}_s{seed}"

    if rep_type == "nn":
        nn_name = rep_cfg["nn_name"]
        nn_layer = rep_cfg["nn_layer"]
        hidden_dim = rep_cfg["nn_hidden_dim"]
        num_hidden_layers = rep_cfg["nn_num_hidden_layers"]
        epochs = rep_cfg["nn_epochs"]
        lr = str(rep_cfg["nn_lr"]).replace(".", "p")
        seed = rep_cfg.get("random_seed", 0)
        return (
            f"nn_{nn_name}"
            f"_L{num_hidden_layers}"
            f"_H{hidden_dim}"
            f"_h{nn_layer}"
            f"_e{epochs}"
            f"_lr{lr}"
            f"_s{seed}"
        )

    raise ValueError(f"Unknown representation type: {rep_type}")


# NN model helpers
class MLPEmbeddingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_hidden_layers: int = 4,
    ):
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

        self.num_hidden_layers = num_hidden_layers

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_embedding(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        if layer < 1 or layer > self.num_hidden_layers:
            raise ValueError(
                f"nn_layer must be between 1 and {self.num_hidden_layers}. Got {layer}."
            )

        for i, linear in enumerate(self.hidden_layers, start=1):
            x = linear(x)
            x = self.relu(x)
            if i == layer:
                return x

        raise RuntimeError("Failed to extract embedding layer.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.hidden_layers:
            x = linear(x)
            x = self.relu(x)
        return self.classifier(x)

def get_torch_device() -> torch.device:
    device = None
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

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
    hidden_dim = rep_cfg["nn_hidden_dim"]
    num_hidden_layers = rep_cfg["nn_num_hidden_layers"]
    epochs = rep_cfg["nn_epochs"]
    batch_size = rep_cfg["nn_batch_size"]
    lr = rep_cfg["nn_lr"]


    model = MLPEmbeddingNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_hidden_layers,
    ).to(device)

    dataset = TensorDataset(
        torch.from_numpy(X_labeled),
        torch.from_numpy(y_labeled),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    print(f"[NN] Training {rep_cfg['nn_name']} on {len(X_labeled)} labeled points")
    print(
        f"[NN] device={device}, input_dim={input_dim}, hidden_dim={hidden_dim}, "
        f"num_hidden_layers={num_hidden_layers}, num_classes={num_classes}"
    )
    print(
        f"[NN] epochs={epochs}, batch_size={batch_size}, lr={lr}"
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size_actual = xb.shape[0]
            epoch_loss += loss.item() * batch_size_actual
            total += batch_size_actual

        if epoch % 5 == 0:
            avg_loss = epoch_loss / max(total, 1)
            print(f"[NN] epoch {epoch + 1:02d}/{epochs} loss={avg_loss:.4f}")

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

    if rep_cfg["nn_name"] != "mlp":
        raise ValueError(f"Unknown nn_name: {rep_cfg['nn_name']}")

    layer = int(rep_cfg["nn_layer"])
    num_hidden_layers = int(rep_cfg["nn_num_hidden_layers"])

    if layer < 1 or layer > num_hidden_layers:
        raise ValueError(
            f"nn_layer must be between 1 and nn_num_hidden_layers={num_hidden_layers}. Got {layer}."
        )

    X = np.asarray(X, dtype=np.float32)
    labeled_ind = np.asarray(labeled_ind, dtype=int)

    model = train_mlp_embedding_model(X[labeled_ind], labels[labeled_ind], rep_cfg)
    print(f"[NN] Finished training. Extracting hidden layer {layer} embeddings.")
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
