from copy import deepcopy
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.methods.knn_distinguisher import (
    compute_knn_candidate_accuracies,
    rank_key_candidates,
    split_nonprofiling_attack_data,
)
from src.models.model_zoo import build_backbone
from src.utils.key_rank import metadata_true_key
from src.utils.trace_transforms import (
    ensure_trace_matrix,
    make_trace_view,
    parse_augmentation_family,
    prepare_model_input,
)
from src.utils.experiment_logger import append_experiment_result
from src.utils.get_device import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class BYOL1D(nn.Module):
    def __init__(
        self,
        backbone_name: str = "shared_cnn_v1",
        pool_mode: str = "mean_max",
        proj_dim: int = 128,
        hidden_dim: int = 512,
        ema_decay: float = 0.996,
        input_length: int = 700,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay
        self.input_length = input_length

        self.online_encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
        )

        self.pooled_repr_dim = (
            self.online_encoder.get_output_dim(
                pool=pool_mode,
            )
        )

        self.online_projector = MLPHead(
            input_dim=self.pooled_repr_dim,
            hidden_dim=hidden_dim,
            output_dim=proj_dim,
        )

        self.online_predictor = MLPHead(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            output_dim=proj_dim,
        )

        self.target_encoder = deepcopy(
            self.online_encoder
        )

        self.target_projector = deepcopy(
            self.online_projector
        )

        self._set_target_requires_grad(False)

    def _set_target_requires_grad(
        self,
        requires_grad: bool,
    ) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = requires_grad

        for parameter in self.target_projector.parameters():
            parameter.requires_grad = requires_grad

    @torch.no_grad()
    def update_target_network(self) -> None:
        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_parameter.data.mul_(
                self.ema_decay
            ).add_(
                online_parameter.data,
                alpha=1.0 - self.ema_decay,
            )

        for online_parameter, target_parameter in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            target_parameter.data.mul_(
                self.ema_decay
            ).add_(
                online_parameter.data,
                alpha=1.0 - self.ema_decay,
            )

    @staticmethod
    def byol_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = F.normalize(
            prediction,
            dim=1,
        )

        target = F.normalize(
            target.detach(),
            dim=1,
        )

        return (
            2.0
            - 2.0
            * (
                prediction
                * target
            ).sum(dim=1).mean()
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        online_h1 = self.online_encoder.encode(
            x1,
            pool=self.pool_mode,
        )

        online_h2 = self.online_encoder.encode(
            x2,
            pool=self.pool_mode,
        )

        online_z1 = self.online_projector(
            online_h1
        )

        online_z2 = self.online_projector(
            online_h2
        )

        prediction1 = self.online_predictor(
            online_z1
        )

        prediction2 = self.online_predictor(
            online_z2
        )

        with torch.no_grad():
            target_h1 = self.target_encoder.encode(
                x1,
                pool=self.pool_mode,
            )

            target_h2 = self.target_encoder.encode(
                x2,
                pool=self.pool_mode,
            )

            target_z1 = self.target_projector(
                target_h1
            )

            target_z2 = self.target_projector(
                target_h2
            )

        loss = 0.5 * (
            self.byol_loss(
                prediction1,
                target_z2,
            )
            + self.byol_loss(
                prediction2,
                target_z1,
            )
        )

        return loss

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.online_encoder.encode(
            x,
            pool=self.pool_mode,
        )


def train_byol(
    X_train,
    device,
    trace_window,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    proj_dim: int = 128,
    hidden_dim: int = 512,
    ema_decay: float = 0.996,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    max_shift: int = 5,
    noise_std: float = 0.05,
    denoise_kernel_size: int = 5,
    view_augmentation: str = "random",
    augmentation_family=("random_shift", "denoise", "gaussian_noise"),
    augmentation_probability: float = 0.5,
    input_length: int = 700,
):
    X_train = ensure_trace_matrix(X_train)

    model = BYOL1D(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
        input_length=input_length,
    ).to(device)

    dataset = TensorDataset(
        torch.from_numpy(X_train).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=lr,
    )

    backbone_params = sum(
        parameter.numel()
        for parameter
        in model.online_encoder.parameters()
        if parameter.requires_grad
    )

    full_model_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Shared backbone trainable parameters:",
        backbone_params,
    )

    print(
        "Full BYOL trainable parameters:",
        full_model_params,
    )

    model.eval()

    with torch.no_grad():
        sample_x = torch.from_numpy(
            X_train[:8]
        ).float().to(device)

        sample_x = prepare_model_input(
            sample_x,
            trace_window=trace_window,
        )

        temporal_features = (
            model.online_encoder.forward_features(
                sample_x
            )
        )

        pooled_features = model.encode(
            sample_x
        )

        projected_features = (
            model.online_projector(
                pooled_features
            )
        )

        predicted_features = (
            model.online_predictor(
                projected_features
            )
        )

    print(
        "Sample input shape:",
        sample_x.shape,
    )

    print(
        "Temporal feature shape:",
        temporal_features.shape,
    )

    print(
        "Pooled representation shape:",
        pooled_features.shape,
    )

    print(
        "Projection shape:",
        projected_features.shape,
    )

    print(
        "Prediction shape:",
        predicted_features.shape,
    )

    model.train()

    loss_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            x1 = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation=view_augmentation,
                augmentation_family=augmentation_family,
                augmentation_probability=augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )

            x2 = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation=view_augmentation,
                augmentation_family=augmentation_family,
                augmentation_probability=augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )

            loss = model(
                x1,
                x2,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=1.0,
            )

            optimizer.step()

            model.update_target_network()

            total_loss += loss.item()
            num_batches += 1

        average_loss = (
            total_loss
            / max(num_batches, 1)
        )

        loss_log.append(
            average_loss
        )

        print(
            f"Epoch #{epoch}: "
            f"byol_loss={average_loss:.6f}"
        )

    return model, loss_log


def encode_representations(
    model,
    X,
    device,
    trace_window,
    batch_size: int = 256,
):
    X = ensure_trace_matrix(X)

    dataset = TensorDataset(
        torch.from_numpy(X).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    representations = []

    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_x = prepare_model_input(
                batch_x,
                trace_window=trace_window,
            )

            representation = model.encode(
                batch_x
            )

            representations.append(
                representation.cpu().numpy()
            )

    return np.concatenate(
        representations,
        axis=0,
    )


def main():
    ascad_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "ascad"
        / "ASCAD.h5"
    )

    seed = 42

    n_train = 500
    n_finetune = 500
    n_attack = 25

    n_epochs = 100
    batch_size = 128
    lr = 3e-4

    backbone_name = "shared_cnn_v1"
    pool_mode = "mean_max"

    proj_dim = 128
    hidden_dim = 512
    ema_decay = 0.996

    max_shift = 5
    noise_std = 0.05
    denoise_kernel_size = 5
    view_augmentation = "random"
    augmentation_family = parse_augmentation_family(
        "random_shift,denoise,gaussian_noise"
    )
    augmentation_probability = 0.5
    knn_neighbors = 3
    knn_weights = "distance"
    leakage_model = "HW"

    target_byte = 2
    normalize_mode = None

    trace_window = (0, 700)

    window_start, window_end = (
        trace_window
    )

    window_size = (
        window_end
        - window_start
    )

    set_seed(seed)

    run_name = (
        f"byol_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_{pool_mode}"
        f"_weakaug"
        f"_shift{max_shift}"
        f"_noise{str(noise_std).replace('.', 'p')}"
        f"_ema{str(ema_decay).replace('.', 'p')}"
        f"_proj{proj_dim}"
        f"_ep{n_epochs}"
        f"_seed{seed}"
    )

    figure_dir = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / run_name
    )

    repr_dir = (
        PROJECT_ROOT
        / "outputs"
        / "representations"
        / run_name
    )

    checkpoint_dir = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / run_name
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repr_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading ASCAD attack traces "
        "with metadata..."
    )

    (
        X_attack,
        _,
        metadata_attack,
    ) = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=False,
        normalize=normalize_mode,
        load_metadata=True,
        trace_window=None,
    )

    (
        X_ssl_train,
        X_knn_train,
        metadata_knn_train,
        X_knn_eval,
        metadata_knn_eval,
    ) = split_nonprofiling_attack_data(
        X_attack=X_attack,
        metadata_attack=metadata_attack,
        n_ssl_train=n_train,
        n_knn_train=n_finetune,
        n_knn_eval=n_attack,
        n_neighbors=knn_neighbors,
    )

    print(
        "Trace window:",
        trace_window,
    )

    print(
        "Window size:",
        window_size,
    )

    print(
        "Full trace length:",
        X_attack.shape[1],
    )

    print(
        "X_ssl_train shape:",
        X_ssl_train.shape,
    )

    print(
        "X_knn_train shape:",
        X_knn_train.shape,
    )

    print(
        "metadata_knn_train shape:",
        metadata_knn_train.shape,
    )

    print(
        "X_knn_eval shape:",
        X_knn_eval.shape,
    )

    print(
        "metadata_knn_eval shape:",
        metadata_knn_eval.shape,
    )

    device = get_device(
        prefer_mps=False,
    )

    print(
        "Using device:",
        device,
    )

    print(
        "Training BYOL-1D..."
    )

    train_start_time = time.time()

    model, loss_log = train_byol(
        X_train=X_ssl_train,
        device=device,
        trace_window=trace_window,
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        max_shift=max_shift,
        noise_std=noise_std,
        denoise_kernel_size=denoise_kernel_size,
        view_augmentation=view_augmentation,
        augmentation_family=augmentation_family,
        augmentation_probability=augmentation_probability,
        input_length=window_size,
    )

    train_end_time = time.time()

    train_time_sec = (
        train_end_time
        - train_start_time
    )

    train_time_ms = (
        train_time_sec
        * 1000
    )

    print(
        "BYOL loss log:",
        loss_log,
    )

    print(
        f"Training start time: "
        f"{train_start_time}"
    )

    print(
        f"Training end time: "
        f"{train_end_time}"
    )

    print(
        f"Training time: "
        f"{train_time_sec:.2f} sec"
    )

    print(
        f"Training time: "
        f"{train_time_ms:.2f} ms"
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{run_name}_encoder.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(
        "Saved checkpoint to:",
        checkpoint_path,
    )

    print(
        "Encoding KNN-train representations..."
    )

    repr_knn_train = encode_representations(
        model=model,
        X=X_knn_train,
        device=device,
        trace_window=trace_window,
        batch_size=256,
    )

    print(
        "Encoding KNN-eval representations..."
    )

    repr_knn_eval = encode_representations(
        model=model,
        X=X_knn_eval,
        device=device,
        trace_window=trace_window,
        batch_size=256,
    )

    print(
        "repr_knn_train shape:",
        repr_knn_train.shape,
    )

    print(
        "repr_knn_eval shape:",
        repr_knn_eval.shape,
    )

    expected_repr_dim = (
        model.pooled_repr_dim
    )

    if repr_knn_train.shape[1] != expected_repr_dim:
        raise ValueError(
            "Unexpected representation dimension: "
            f"expected {expected_repr_dim}, "
            f"received {repr_knn_train.shape[1]}"
        )

    np.save(
        repr_dir / "repr_knn_train.npy",
        repr_knn_train,
    )

    np.save(
        repr_dir / "repr_knn_eval.npy",
        repr_knn_eval,
    )

    print(
        "Training 256 candidate KNNs "
        "and computing candidate accuracies..."
    )

    candidate_accuracies = compute_knn_candidate_accuracies(
        repr_train=repr_knn_train,
        metadata_train=metadata_knn_train,
        repr_eval=repr_knn_eval,
        metadata_eval=metadata_knn_eval,
        target_byte=target_byte,
        n_neighbors=knn_neighbors,
        leakage_model=leakage_model,
        weights=knn_weights,
    )

    true_key = metadata_true_key(
        metadata_knn_eval,
        target_byte=target_byte,
    )

    ranked_keys, true_key_rank = rank_key_candidates(
        candidate_scores=candidate_accuracies,
        true_key=true_key,
    )

    best_key = int(ranked_keys[0])
    best_accuracy = float(candidate_accuracies[best_key])
    true_key_accuracy = float(candidate_accuracies[true_key])

    print("True key:", true_key)
    print("Best key:", best_key)
    print("Best candidate accuracy:", best_accuracy)
    print("True-key candidate accuracy:", true_key_accuracy)
    print("True-key rank among candidate KNNs:", true_key_rank)

    candidate_scores_path = (
        repr_dir
        / f"{run_name}_candidate_accuracies.npy"
    )

    ranked_keys_path = (
        repr_dir
        / f"{run_name}_ranked_keys.npy"
    )

    np.save(
        candidate_scores_path,
        candidate_accuracies,
    )

    np.save(
        ranked_keys_path,
        ranked_keys,
    )

    print(
        "Saved candidate accuracies to:",
        candidate_scores_path,
    )

    print(
        "Saved ranked keys to:",
        ranked_keys_path,
    )

    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / "experiment_summary.csv"
    )

    backbone_params = sum(
        parameter.numel()
        for parameter
        in model.online_encoder.parameters()
        if parameter.requires_grad
    )

    append_experiment_result(
        summary_path,
        {
            "method": "BYOL-nonprofiling-shared-backbone",
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "seed": seed,
            "n_train": n_train,
            "n_finetune": n_finetune,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "backbone_name": backbone_name,
            "backbone_params": backbone_params,
            "encoder_output_channels": (
                model.online_encoder.output_channels
            ),
            "pool_mode": pool_mode,
            "pooled_repr_dim": (
                model.pooled_repr_dim
            ),
            "proj_dim": proj_dim,
            "hidden_dim": hidden_dim,
            "ema_decay": ema_decay,
            "max_shift": max_shift,
            "noise_std": noise_std,
            "denoise_kernel_size": denoise_kernel_size,
            "view_augmentation": view_augmentation,
            "augmentation_family": ",".join(augmentation_family),
            "augmentation_probability": augmentation_probability,
            "normalize": normalize_mode,
            "window_start": window_start,
            "window_end": window_end,
            "window_size": window_size,
            "classifier": "candidate-key KNeighborsClassifier",
            "knn_neighbors": knn_neighbors,
            "knn_weights": knn_weights,
            "leakage_model": leakage_model,
            "target_byte": target_byte,
            "device": str(device),
            "train_start_time": (
                train_start_time
            ),
            "train_end_time": (
                train_end_time
            ),
            "train_time_sec": round(
                train_time_sec,
                2,
            ),
            "train_time_ms": round(
                train_time_ms,
                2,
            ),
            "true_key": true_key,
            "best_key": best_key,
            "best_accuracy": round(best_accuracy, 6),
            "true_key_accuracy": round(true_key_accuracy, 6),
            "true_key_candidate_rank": true_key_rank,
            "checkpoint_path": str(
                checkpoint_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "candidate_scores_path": str(
                candidate_scores_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "ranked_keys_path": str(
                ranked_keys_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    )

    print(
        "Saved experiment summary to:",
        summary_path,
    )


if __name__ == "__main__":
    main()
