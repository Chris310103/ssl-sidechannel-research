from pathlib import Path
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
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


class FCMAESharedCNN1D(nn.Module):
    def __init__(
        self,
        backbone_name: str = "shared_cnn_v1",
        pool_mode: str = "mean_max",
        patch_size: int = 5,
        mask_ratio: float = 0.30,
        input_length: int = 700,
    ):
        super().__init__()

        if patch_size <= 0:
            raise ValueError(
                "patch_size must be greater than 0"
            )

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(
                "mask_ratio must be between 0 and 1"
            )

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.input_length = input_length

        if backbone_name != "shared_cnn_v1":
            raise ValueError(
                "1D-FCMAE currently supports backbone_name='shared_cnn_v1'. "
                f"Received {backbone_name}."
            )

        self.encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
        )

        self.encoder_output_channels = (
            self.encoder.get_temporal_output_dim()
        )

        self.pooled_repr_dim = (
            self.encoder.get_output_dim(
                pool=pool_mode,
            )
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                self.encoder_output_channels,
                256,
                kernel_size=11,
                stride=2,
                padding=5,
                output_padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(
                256,
                128,
                kernel_size=11,
                stride=2,
                padding=5,
                output_padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(
                128,
                64,
                kernel_size=11,
                stride=2,
                padding=5,
                output_padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(
                64,
                1,
                kernel_size=11,
                stride=2,
                padding=5,
                output_padding=1,
            ),
        )

    def make_patch_mask(
        self,
        batch_size: int,
        trace_length: int,
        device,
    ) -> torch.Tensor:
        number_of_patches = math.ceil(
            trace_length / self.patch_size
        )

        number_of_masked_patches = max(
            1,
            int(
                round(
                    number_of_patches
                    * self.mask_ratio
                )
            ),
        )

        random_values = torch.rand(
            batch_size,
            number_of_patches,
            device=device,
        )

        shuffled_patch_indices = torch.argsort(
            random_values,
            dim=1,
        )

        patch_mask = torch.zeros(
            batch_size,
            number_of_patches,
            dtype=torch.bool,
            device=device,
        )

        patch_mask.scatter_(
            1,
            shuffled_patch_indices[
                :,
                :number_of_masked_patches,
            ],
            True,
        )

        sample_mask = patch_mask.repeat_interleave(
            self.patch_size,
            dim=1,
        )

        sample_mask = sample_mask[
            :,
            :trace_length,
        ]

        return sample_mask.unsqueeze(-1)

    def reconstruct(
        self,
        x: torch.Tensor,
        sample_mask: torch.Tensor,
    ):
        visible_mask = (
            ~sample_mask
        ).to(
            dtype=x.dtype,
        )

        masked_trace = (
            x
            * visible_mask
        )

        if not hasattr(
            self.encoder,
            "forward_masked_features",
        ):
            raise ValueError(
                "The selected encoder does not support mask-aware FCMAE "
                "feature extraction."
            )

        temporal_features = (
            self.encoder.forward_masked_features(
                masked_trace,
                visible_mask,
            )
        )
        if isinstance(
            temporal_features,
            tuple,
        ):
            temporal_features = temporal_features[0]

        decoder_input = temporal_features.transpose(
            1,
            2,
        )

        reconstruction = self.decoder(
            decoder_input
        )

        reconstruction = reconstruction.transpose(
            1,
            2,
        )

        trace_length = masked_trace.size(1)

        reconstruction = reconstruction[
            :,
            :trace_length,
            :,
        ]

        return temporal_features, reconstruction

    def forward(
        self,
        x: torch.Tensor,
    ):
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [B, L, 1], "
                f"received {tuple(x.shape)}"
            )

        batch_size = x.size(0)
        trace_length = x.size(1)

        sample_mask = self.make_patch_mask(
            batch_size=batch_size,
            trace_length=trace_length,
            device=x.device,
        )

        (
            temporal_features,
            reconstruction,
        ) = self.reconstruct(
            x,
            sample_mask,
        )

        squared_error = (
            reconstruction - x
        ) ** 2

        mask_float = sample_mask.float()

        reconstruction_loss = (
            squared_error
            * mask_float
        ).sum() / mask_float.sum().clamp_min(1.0)

        return (
            reconstruction_loss,
            reconstruction,
            sample_mask,
            temporal_features,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder.encode(
            x,
            pool=self.pool_mode,
        )


MAESharedCNN1D = FCMAESharedCNN1D


def train_mae(
    X_train,
    device,
    trace_window,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    patch_size: int = 5,
    mask_ratio: float = 0.30,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    input_length: int = 700,
):
    X_train = ensure_trace_matrix(X_train)

    model = FCMAESharedCNN1D(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    backbone_params = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
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
        "Full 1D-FCMAE trainable parameters:",
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

        (
            sample_loss,
            sample_reconstruction,
            sample_mask,
            sample_temporal_features,
        ) = model(sample_x)

        sample_representation = model.encode(
            sample_x
        )

    print(
        "Sample input shape:",
        sample_x.shape,
    )

    print(
        "Temporal feature shape:",
        sample_temporal_features.shape,
    )

    print(
        "Reconstruction shape:",
        sample_reconstruction.shape,
    )

    print(
        "Mask shape:",
        sample_mask.shape,
    )

    print(
        "Downstream representation shape:",
        sample_representation.shape,
    )

    print(
        "Sample masked reconstruction loss:",
        float(sample_loss.item()),
    )

    if sample_temporal_features.shape[-1] != model.encoder_output_channels:
        raise ValueError(
            "Unexpected MAE temporal channel dimension: "
            f"expected {model.encoder_output_channels}, "
            f"received {sample_temporal_features.shape[-1]}"
        )

    if sample_reconstruction.shape != sample_x.shape:
        raise ValueError(
            "Unexpected MAE reconstruction shape: "
            f"expected {tuple(sample_x.shape)}, "
            f"received {tuple(sample_reconstruction.shape)}"
        )

    if sample_representation.shape[-1] != model.pooled_repr_dim:
        raise ValueError(
            "Unexpected MAE downstream representation dimension: "
            f"expected {model.pooled_repr_dim}, "
            f"received {sample_representation.shape[-1]}"
        )

    model.train()

    loss_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        number_of_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_x = prepare_model_input(
                batch_x,
                trace_window=trace_window,
            )

            (
                loss,
                _,
                _,
                _,
            ) = model(batch_x)

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            number_of_batches += 1

        average_loss = (
            total_loss
            / max(number_of_batches, 1)
        )

        loss_log.append(
            average_loss
        )

        print(
            f"Epoch #{epoch}: "
            f"fcmae_loss={average_loss:.6f}"
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
    lr = 1e-4
    weight_decay = 1e-4

    backbone_name = "shared_cnn_v1"
    pool_mode = "mean_max"

    patch_size = 5
    mask_ratio = 0.30

    target_byte = 2
    normalize_mode = None
    knn_neighbors = 3
    knn_weights = "distance"
    leakage_model = "HW"

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
        f"fcmae_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_{pool_mode}"
        f"_patch{patch_size}"
        f"_mask{int(mask_ratio * 100)}"
        f"_ep{n_epochs}"
        f"_seed{seed}"
    )

    figure_dir = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / run_name
    )

    representation_dir = (
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

    representation_dir.mkdir(
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
        "Training 1D-FCMAE Shared CNN..."
    )

    train_start_time = time.time()

    model, loss_log = train_mae(
        X_train=X_ssl_train,
        device=device,
        trace_window=trace_window,
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
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
        "1D-FCMAE loss log:",
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
        representation_dir / "repr_knn_train.npy",
        repr_knn_train,
    )

    np.save(
        representation_dir / "repr_knn_eval.npy",
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
        representation_dir
        / f"{run_name}_candidate_accuracies.npy"
    )

    ranked_keys_path = (
        representation_dir
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
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )

    decoder_params = sum(
        parameter.numel()
        for parameter in model.decoder.parameters()
        if parameter.requires_grad
    )

    full_model_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    append_experiment_result(
        summary_path,
        {
            "method": "1D-FCMAE-nonprofiling-shared-backbone",
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "seed": seed,
            "n_train": n_train,
            "n_finetune": n_finetune,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "backbone_name": backbone_name,
            "backbone_params": backbone_params,
            "decoder_params": decoder_params,
            "full_model_params": full_model_params,
            "encoder_output_channels": (
                model.encoder_output_channels
            ),
            "pool_mode": pool_mode,
            "pooled_repr_dim": (
                model.pooled_repr_dim
            ),
            "patch_size": patch_size,
            "mask_ratio": mask_ratio,
            "decoder_type": (
                "ConvTranspose1d"
            ),
            "reconstruction_loss": (
                "masked_mse"
            ),
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
            "final_mae_loss": round(
                loss_log[-1],
                6,
            ),
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
