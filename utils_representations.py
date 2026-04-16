import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

NN_TRAIN_CALL_COUNT = 0

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

    nn_num_hidden_layers = int(rep.get("nn_num_hidden_layers", 4))
    nn_hidden_dim_raw = rep.get("nn_hidden_dim", 128)

    if isinstance(nn_hidden_dim_raw, list):
        nn_hidden_dims = [int(x) for x in nn_hidden_dim_raw]
        if len(nn_hidden_dims) != nn_num_hidden_layers:
            raise ValueError(
                f"If nn_hidden_dim is a list, its length must equal "
                f"nn_num_hidden_layers={nn_num_hidden_layers}. "
                f"Got {len(nn_hidden_dims)} entries."
            )
    else:
        nn_hidden_dims = [int(nn_hidden_dim_raw)] * nn_num_hidden_layers

    return {
        "type": rep_type,
        "noise_std": float(rep.get("noise_std", 0.05)),
        "pca_components": rep.get("pca_components", 20),

        "nn_name": nn_name,
        "nn_layer": int(rep.get("nn_layer", 4)),
        "nn_hidden_dim": nn_hidden_dims,
        "nn_num_hidden_layers": nn_num_hidden_layers,
        "nn_epochs": int(rep.get("nn_epochs", 20)),
        "nn_batch_size": int(rep.get("nn_batch_size", 64)),
        "nn_lr": float(rep.get("nn_lr", 1e-3)),
        "nn_update_interval": int(rep.get("nn_update_interval", 0)),

        "random_seed": int(rep.get("random_seed", 0)),
    }


def get_nn_update_interval(rep_cfg: dict) -> int:
    return int(rep_cfg.get("nn_update_interval", 0))


def representation_depends_on_seed(rep_cfg: dict) -> bool:
    return rep_cfg["type"] == "nn"


def representation_is_dynamic(rep_cfg: dict) -> bool:
    return rep_cfg["type"] == "nn" and get_nn_update_interval(rep_cfg) > 0


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
        hidden_dims = rep_cfg["nn_hidden_dim"]
        num_hidden_layers = rep_cfg["nn_num_hidden_layers"]
        epochs = rep_cfg["nn_epochs"]
        lr = str(rep_cfg["nn_lr"]).replace(".", "p")
        seed = rep_cfg.get("random_seed", 0)

        if len(set(hidden_dims)) == 1:
            hidden_part = f"H{hidden_dims[0]}"
        else:
            hidden_part = "Hvar"

        tag = (
            f"nn_{nn_name}"
            f"_L{num_hidden_layers}"
            f"_{hidden_part}"
            f"_h{nn_layer}"
            f"_e{epochs}"
            f"_lr{lr}"
            f"_s{seed}"
        )
        update_interval = get_nn_update_interval(rep_cfg)
        if update_interval > 0:
            tag += f"_upd{update_interval}"
        return tag

    raise ValueError(f"Unknown representation type: {rep_type}")


# NN model helpers
class MLPEmbeddingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
    ):
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer width.")

        self.num_hidden_layers = len(hidden_dims)

        self.hidden_layers = nn.ModuleList()

        prev_dim = input_dim
        for hdim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim

        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mlp_embedding_model(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    rep_cfg: dict,
) -> MLPEmbeddingNet:
    global NN_TRAIN_CALL_COUNT
    NN_TRAIN_CALL_COUNT += 1
    print(f"[NN] TRAIN CALL #{NN_TRAIN_CALL_COUNT}")
    
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
    hidden_dims = rep_cfg["nn_hidden_dim"]
    num_hidden_layers = rep_cfg["nn_num_hidden_layers"]
    epochs = rep_cfg["nn_epochs"]
    batch_size = rep_cfg["nn_batch_size"]
    lr = rep_cfg["nn_lr"]

    model = MLPEmbeddingNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
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
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"[NN] Training {rep_cfg['nn_name']} on {len(X_labeled)} labeled points")
    print(
        f"[NN] device={device}, input_dim={input_dim}, hidden_dims={hidden_dims}, "
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


def train_embedding_model(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    rep_cfg: dict,
):
    if rep_cfg["nn_name"] == "mlp":
        return train_mlp_embedding_model(X_labeled, y_labeled, rep_cfg)
    if rep_cfg["nn_name"] == "cnn":
        raise NotImplementedError("CNN is not implemented yet.")
    raise ValueError(f"Unknown nn_name: {rep_cfg['nn_name']}")


def extract_embeddings(
    model,
    X_all: np.ndarray,
    rep_cfg: dict,
) -> np.ndarray:
    if rep_cfg["nn_name"] == "mlp":
        return extract_mlp_embeddings(model, X_all, int(rep_cfg["nn_layer"]))
    if rep_cfg["nn_name"] == "cnn":
        raise NotImplementedError("CNN is not implemented yet.")
    raise ValueError(f"Unknown nn_name: {rep_cfg['nn_name']}")


def get_base_features(dataset, metric):
    X, clusters = gl.datasets.load(dataset.split("-")[0], metric=metric)
    return X, clusters


def get_regular_features(X: np.ndarray, rep_cfg: dict | None = None) -> np.ndarray:
    return np.asarray(X, dtype=float)


def get_noisy_features(X: np.ndarray, rep_cfg: dict) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    noise_std = rep_cfg["noise_std"]
    seed = rep_cfg.get("random_seed", 0)

    rng = np.random.default_rng(seed)
    noise = noise_std * rng.standard_normal(X.shape)
    return X + noise


def get_pca_features(X: np.ndarray, rep_cfg: dict) -> np.ndarray:
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
    if labels is None or labeled_ind is None:
        raise ValueError("NN representation requires labels and labeled_ind.")

    layer = int(rep_cfg["nn_layer"])
    num_hidden_layers = int(rep_cfg["nn_num_hidden_layers"])

    if layer < 1 or layer > num_hidden_layers:
        raise ValueError(
            f"nn_layer must be between 1 and nn_num_hidden_layers={num_hidden_layers}. Got {layer}."
        )

    X = np.asarray(X, dtype=np.float32)
    labeled_ind = np.asarray(labeled_ind, dtype=int)

    model = train_embedding_model(X[labeled_ind], labels[labeled_ind], rep_cfg)
    print(f"[NN] Finished training. Extracting hidden layer {layer} embeddings.")
    X_emb = extract_embeddings(model, X, rep_cfg)

    return X_emb


def apply_representation(
    X: np.ndarray,
    rep_cfg: dict,
    *,
    labels=None,
    labeled_ind=None,
) -> np.ndarray:
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
    X, clusters = get_base_features(dataset, metric)
    X_rep = apply_representation(
        X,
        rep_cfg,
        labels=labels,
        labeled_ind=labeled_ind,
    )
    return X_rep, clusters