from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.evaluation.rank_eval import (
    compute_rank_curve,
    expand_proba_to_256,
    plot_rank_curve,
)
from src.models.cnn_zoo import build_cnn_backbone
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
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode

        self.encoder = build_cnn_backbone(
            name=backbone_name,
            input_channels=1,
        )

        self.temporal_repr_dim = (
            self.encoder.output_channels
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
):
    model = TS2VecSharedModel(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
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

    model.train()

    loss_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        number_of_batches = 0

        for batch_index, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)

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
    batch_size: int = 256,
):
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

    n_train = 50000
    n_attack = 10000

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
        "Loading ASCAD profiling traces..."
    )

    X_profiling, y_profiling = load_ascad_split(
        h5_path=ascad_path,
        split="profiling",
        add_channel=True,
        normalize=normalize_mode,
        load_metadata=False,
        trace_window=trace_window,
    )

    print(
        "Loading ASCAD attack traces "
        "with metadata..."
    )

    (
        X_attack,
        y_attack,
        metadata_attack,
    ) = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=True,
        normalize=normalize_mode,
        load_metadata=True,
        trace_window=trace_window,
    )

    X_train = X_profiling[:n_train]
    y_train = y_profiling[:n_train]

    X_attack_small = (
        X_attack[:n_attack]
    )

    metadata_attack_small = (
        metadata_attack[:n_attack]
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
        "X_train shape:",
        X_train.shape,
    )

    print(
        "y_train shape:",
        y_train.shape,
    )

    print(
        "X_attack shape:",
        X_attack_small.shape,
    )

    print(
        "metadata_attack shape:",
        metadata_attack_small.shape,
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
        X_train=X_train,
        device=device,
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
        "Encoding profiling representations..."
    )

    repr_train = encode_representations(
        model=model,
        X=X_train,
        device=device,
        batch_size=256,
    )

    print(
        "Encoding attack representations..."
    )

    repr_attack = encode_representations(
        model=model,
        X=X_attack_small,
        device=device,
        batch_size=256,
    )

    print(
        "repr_train shape:",
        repr_train.shape,
    )

    print(
        "repr_attack shape:",
        repr_attack.shape,
    )

    expected_repr_dim = (
        model.pooled_repr_dim
    )

    if repr_train.shape[1] != expected_repr_dim:
        raise ValueError(
            "Unexpected representation dimension: "
            f"expected {expected_repr_dim}, "
            f"received {repr_train.shape[1]}"
        )

    np.save(
        representation_dir / "repr_train.npy",
        repr_train,
    )

    np.save(
        representation_dir / "repr_attack.npy",
        repr_attack,
    )

    np.save(
        representation_dir / "y_train.npy",
        y_train,
    )

    print(
        "Training linear classifier "
        "on TS2Vec shared-backbone representations..."
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    classifier.fit(
        repr_train,
        y_train,
    )

    train_accuracy = float(
        classifier.score(
            repr_train,
            y_train,
        )
    )

    print(
        "Linear probe train accuracy:",
        train_accuracy,
    )

    print(
        "Predicting attack probabilities..."
    )

    attack_probabilities_seen = (
        classifier.predict_proba(
            repr_attack
        )
    )

    attack_probabilities = expand_proba_to_256(
        attack_probabilities_seen,
        classes=classifier.classes_,
    )

    print(
        "attack_probas shape:",
        attack_probabilities.shape,
    )

    print(
        "Computing key rank curve..."
    )

    ranks = compute_rank_curve(
        probas=attack_probabilities,
        metadata=metadata_attack_small,
        target_byte=target_byte,
        max_traces=n_attack,
        use_log=True,
    )

    final_rank = int(
        ranks[-1]
    )

    minimum_rank = int(
        ranks.min()
    )

    rank_zero_indices = np.where(
        ranks == 0
    )[0]

    rank_zero_trace = (
        int(rank_zero_indices[0] + 1)
        if len(rank_zero_indices) > 0
        else -1
    )

    print(
        "Final rank:",
        final_rank,
    )

    print(
        "Minimum rank:",
        minimum_rank,
    )

    print(
        "Rank-0 trace:",
        rank_zero_trace,
    )

    rank_path = (
        figure_dir
        / f"{run_name}_linear_probe_rank.png"
    )

    ranks_path = (
        representation_dir
        / f"{run_name}_linear_probe_ranks.npy"
    )

    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title=(
            "TS2Vec Shared CNN "
            "+ Linear Probe Key Rank"
        ),
    )

    np.save(
        ranks_path,
        ranks,
    )

    print(
        "Saved rank curve to:",
        rank_path,
    )

    print(
        "Saved ranks to:",
        ranks_path,
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
                "TS2Vec-shared-backbone"
            ),
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "seed": seed,
            "n_train": n_train,
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
            "classifier": (
                "LogisticRegression"
            ),
            "linear_probe_train_acc": round(
                train_accuracy,
                6,
            ),
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
            "final_rank": final_rank,
            "min_rank": minimum_rank,
            "rank0_trace": rank_zero_trace,
            "figure_path": str(
                rank_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "checkpoint_path": str(
                checkpoint_path.relative_to(
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