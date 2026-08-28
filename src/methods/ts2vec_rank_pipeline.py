from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
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


def instance_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, _ = z1.shape

    if batch_size == 1:
        return z1.new_tensor(0.0)

    representations = torch.cat(
        [z1, z2],
        dim=0,
    ).transpose(0, 1)

    similarity = torch.matmul(
        representations,
        representations.transpose(1, 2),
    )

    logits = torch.tril(
        similarity,
        diagonal=-1,
    )[:, :, :-1]

    logits = logits + torch.triu(
        similarity,
        diagonal=1,
    )[:, :, 1:]

    logits = -F.log_softmax(
        logits,
        dim=-1,
    )

    indices = torch.arange(
        batch_size,
        device=z1.device,
    )

    loss_1 = logits[
        :,
        indices,
        batch_size + indices - 1,
    ].mean()

    loss_2 = logits[
        :,
        batch_size + indices,
        indices,
    ].mean()

    return 0.5 * (
        loss_1 + loss_2
    )


def temporal_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    _, sequence_length, _ = z1.shape

    if sequence_length == 1:
        return z1.new_tensor(0.0)

    representations = torch.cat(
        [z1, z2],
        dim=1,
    )

    similarity = torch.matmul(
        representations,
        representations.transpose(1, 2),
    )

    logits = torch.tril(
        similarity,
        diagonal=-1,
    )[:, :, :-1]

    logits = logits + torch.triu(
        similarity,
        diagonal=1,
    )[:, :, 1:]

    logits = -F.log_softmax(
        logits,
        dim=-1,
    )

    time_indices = torch.arange(
        sequence_length,
        device=z1.device,
    )

    loss_1 = logits[
        :,
        time_indices,
        sequence_length + time_indices - 1,
    ].mean()

    loss_2 = logits[
        :,
        sequence_length + time_indices,
        time_indices,
    ].mean()

    return 0.5 * (
        loss_1 + loss_2
    )


def hierarchical_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float = 0.5,
    temporal_unit: int = 0,
) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError(
            "TS2Vec views must have matching shapes, "
            f"received {tuple(z1.shape)} and {tuple(z2.shape)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1"
        )

    loss = z1.new_tensor(0.0)
    hierarchy_depth = 0

    while z1.size(1) > 1:
        if alpha > 0:
            loss = loss + (
                alpha
                * instance_contrastive_loss(
                    z1,
                    z2,
                )
            )

        if (
            hierarchy_depth >= temporal_unit
            and alpha < 1.0
        ):
            loss = loss + (
                (1.0 - alpha)
                * temporal_contrastive_loss(
                    z1,
                    z2,
                )
            )

        hierarchy_depth += 1

        z1 = F.max_pool1d(
            z1.transpose(1, 2),
            kernel_size=2,
        ).transpose(1, 2)

        z2 = F.max_pool1d(
            z2.transpose(1, 2),
            kernel_size=2,
        ).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha > 0:
            loss = loss + (
                alpha
                * instance_contrastive_loss(
                    z1,
                    z2,
                )
            )

        hierarchy_depth += 1

    return loss / max(
        hierarchy_depth,
        1,
    )


class TS2VecSharedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "shared_cnn_v1",
        pool_mode: str = "mean_max",
        input_length: int = 700,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.input_length = input_length

        self.encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
        )

        self.temporal_repr_dim = (
            self.encoder.get_temporal_output_dim()
        )

        self.pooled_repr_dim = (
            self.encoder.get_output_dim(
                pool=pool_mode,
            )
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1)
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder.forward_features(x)

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder.encode(
            x,
            pool=self.pool_mode,
        )

    def apply_timestamp_mask(
        self,
        x: torch.Tensor,
        keep_probability: float = 0.5,
    ) -> torch.Tensor:
        if not 0.0 < keep_probability <= 1.0:
            raise ValueError(
                "keep_probability must be in (0, 1]"
            )

        batch_size, sequence_length, _ = x.shape

        keep_mask = (
            torch.rand(
                batch_size,
                sequence_length,
                device=x.device,
            )
            < keep_probability
        )

        fully_masked = ~keep_mask.any(
            dim=1
        )

        if fully_masked.any():
            rows = torch.nonzero(
                fully_masked,
                as_tuple=False,
            ).squeeze(1)

            columns = torch.randint(
                low=0,
                high=sequence_length,
                size=(rows.numel(),),
                device=x.device,
            )

            keep_mask[
                rows,
                columns,
            ] = True

        mask_values = self.mask_token.expand_as(x)

        return torch.where(
            keep_mask.unsqueeze(-1),
            x,
            mask_values,
        )


def random_shared_crop(
    x: torch.Tensor,
    minimum_crop_ratio: float = 0.5,
) -> torch.Tensor:
    if not 0.0 < minimum_crop_ratio <= 1.0:
        raise ValueError(
            "minimum_crop_ratio must be in (0, 1]"
        )

    batch_size, sequence_length, _ = x.shape

    minimum_crop_length = max(
        32,
        int(
            round(
                sequence_length
                * minimum_crop_ratio
            )
        ),
    )

    minimum_crop_length = min(
        minimum_crop_length,
        sequence_length,
    )

    crop_length = int(
        torch.randint(
            low=minimum_crop_length,
            high=sequence_length + 1,
            size=(1,),
            device=x.device,
        ).item()
    )

    maximum_start = (
        sequence_length
        - crop_length
    )

    starts = torch.randint(
        low=0,
        high=maximum_start + 1,
        size=(batch_size,),
        device=x.device,
    )

    offsets = torch.arange(
        crop_length,
        device=x.device,
    ).unsqueeze(0)

    time_indices = (
        starts.unsqueeze(1)
        + offsets
    )

    batch_indices = torch.arange(
        batch_size,
        device=x.device,
    ).unsqueeze(1)

    return x[
        batch_indices,
        time_indices,
        :,
    ]


def train_ts2vec(
    X_train,
    device,
    trace_window,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    alpha: float = 0.5,
    temporal_unit: int = 0,
    minimum_crop_ratio: float = 0.5,
    timestamp_keep_probability: float = 0.5,
    input_length: int = 700,
):
    X_train = ensure_trace_matrix(X_train)

    model = TS2VecSharedModel(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        input_length=input_length,
    ).to(device)

    averaged_model = AveragedModel(
        model
    )

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
        for parameter
        in model.encoder.parameters()
        if parameter.requires_grad
    )

    full_model_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Shared backbone trainable parameters:",
        backbone_params,
    )

    print(
        "Full TS2Vec trainable parameters:",
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

        sample_temporal = (
            model.forward_features(
                sample_x
            )
        )

        sample_pooled = model.encode(
            sample_x
        )

    print(
        "Sample input shape:",
        sample_x.shape,
    )

    print(
        "Temporal feature shape:",
        sample_temporal.shape,
    )

    print(
        "Downstream representation shape:",
        sample_pooled.shape,
    )

    if sample_temporal.shape[-1] != model.temporal_repr_dim:
        raise ValueError(
            "Unexpected TS2Vec temporal representation dimension: "
            f"expected {model.temporal_repr_dim}, "
            f"received {sample_temporal.shape[-1]}"
        )

    if sample_pooled.shape[-1] != model.pooled_repr_dim:
        raise ValueError(
            "Unexpected TS2Vec downstream representation dimension: "
            f"expected {model.pooled_repr_dim}, "
            f"received {sample_pooled.shape[-1]}"
        )

    model.train()

    loss_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        number_of_batches = 0

        for batch_index, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_x = prepare_model_input(
                batch_x,
                trace_window=trace_window,
            )

            cropped_x = random_shared_crop(
                batch_x,
                minimum_crop_ratio=minimum_crop_ratio,
            )

            view_1 = model.apply_timestamp_mask(
                cropped_x,
                keep_probability=(
                    timestamp_keep_probability
                ),
            )

            view_2 = model.apply_timestamp_mask(
                cropped_x,
                keep_probability=(
                    timestamp_keep_probability
                ),
            )

            representation_1 = (
                model.forward_features(
                    view_1
                )
            )

            representation_2 = (
                model.forward_features(
                    view_2
                )
            )

            if (
                epoch == 0
                and batch_index == 0
            ):
                print(
                    "Training crop shape:",
                    cropped_x.shape,
                )

                print(
                    "View 1 shape:",
                    view_1.shape,
                )

                print(
                    "View 2 shape:",
                    view_2.shape,
                )

                print(
                    "TS2Vec temporal output shape:",
                    representation_1.shape,
                )

            loss = hierarchical_contrastive_loss(
                z1=representation_1,
                z2=representation_2,
                alpha=alpha,
                temporal_unit=temporal_unit,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            averaged_model.update_parameters(
                model
            )

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
            f"ts2vec_loss={average_loss:.6f}"
        )

    trained_model = averaged_model.module

    trained_model.eval()

    return trained_model, loss_log


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
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-2

    backbone_name = "shared_cnn_v1"
    pool_mode = "mean_max"

    alpha = 0.5
    temporal_unit = 0

    minimum_crop_ratio = 0.5
    timestamp_keep_probability = 0.5

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
        f"ts2vec_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_{pool_mode}"
        f"_crop{int(minimum_crop_ratio * 100)}"
        f"_keep{int(timestamp_keep_probability * 100)}"
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
        "Training TS2Vec shared backbone..."
    )

    train_start_time = time.time()

    model, loss_log = train_ts2vec(
        X_train=X_ssl_train,
        device=device,
        trace_window=trace_window,
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha,
        temporal_unit=temporal_unit,
        minimum_crop_ratio=(
            minimum_crop_ratio
        ),
        timestamp_keep_probability=(
            timestamp_keep_probability
        ),
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
        "TS2Vec loss log:",
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
        for parameter
        in model.encoder.parameters()
        if parameter.requires_grad
    )

    full_model_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    append_experiment_result(
        summary_path,
        {
            "method": (
                "TS2Vec-nonprofiling-shared-backbone"
            ),
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
            "full_model_params": (
                full_model_params
            ),
            "temporal_repr_dim": (
                model.temporal_repr_dim
            ),
            "pool_mode": pool_mode,
            "pooled_repr_dim": (
                model.pooled_repr_dim
            ),
            "alpha": alpha,
            "temporal_unit": temporal_unit,
            "minimum_crop_ratio": (
                minimum_crop_ratio
            ),
            "timestamp_keep_probability": (
                timestamp_keep_probability
            ),
            "context_view": (
                "shared_crop_independent_timestamp_mask"
            ),
            "parameter_averaging": (
                "AveragedModel"
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
            "final_ts2vec_loss": round(
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
