from copy import deepcopy
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
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


def random_shift_1d(
    x: torch.Tensor,
    max_shift: int = 3,
) -> torch.Tensor:
    if max_shift <= 0:
        return x

    batch_size = x.size(0)

    shifts = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(batch_size,),
        device=x.device,
    )

    output = torch.empty_like(x)

    for index, shift in enumerate(shifts):
        output[index] = torch.roll(
            x[index],
            shifts=int(shift.item()),
            dims=0,
        )

    return output


def random_time_mask_1d(
    x: torch.Tensor,
    mask_ratio: float = 0.0,
) -> torch.Tensor:
    if mask_ratio <= 0:
        return x

    output = x.clone()

    batch_size, trace_length, _ = output.shape
    mask_length = max(
        1,
        int(round(trace_length * mask_ratio)),
    )

    if mask_length >= trace_length:
        output.zero_()
        return output

    starts = torch.randint(
        low=0,
        high=trace_length - mask_length + 1,
        size=(batch_size,),
        device=output.device,
    )

    for index, start in enumerate(starts):
        start = int(start.item())

        output[
            index,
            start : start + mask_length,
            :,
        ] = 0.0

    return output


def augment_trace(
    x: torch.Tensor,
    max_shift: int = 3,
    noise_std: float = 0.01,
    scale_std: float = 0.0,
    mask_ratio: float = 0.0,
) -> torch.Tensor:
    augmented = random_shift_1d(
        x,
        max_shift=max_shift,
    )

    if scale_std > 0:
        scale = (
            1.0
            + scale_std
            * torch.randn(
                augmented.size(0),
                1,
                1,
                device=augmented.device,
            )
        )

        augmented = augmented * scale

    if noise_std > 0:
        trace_std = augmented.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        noise = (
            torch.randn_like(augmented)
            * trace_std
            * noise_std
        )

        augmented = augmented + noise

    augmented = random_time_mask_1d(
        augmented,
        mask_ratio=mask_ratio,
    )

    return augmented


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
        embedding_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 512,
        ema_decay: float = 0.996,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.repr_dim = embedding_dim
        self.pooled_repr_dim = embedding_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        self.online_encoder = build_cnn_backbone(
            input_channels=1,
            input_length=700,
        )

        self.temporal_channels = (
            self.online_encoder.get_output_channels()
        )
        self.temporal_length = (
            self.online_encoder.get_temporal_length()
        )
        self.flatten_dim = (
            self.temporal_channels
            * self.temporal_length
        )

        self.online_embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.flatten_dim,
                4096,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                4096,
                4096,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                4096,
                embedding_dim,
            ),
            nn.ReLU(inplace=True),
        )

        self.online_projector = MLPHead(
            input_dim=embedding_dim,
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

        self.target_embedding_head = deepcopy(
            self.online_embedding_head
        )

        self.target_projector = deepcopy(
            self.online_projector
        )

        self._set_target_requires_grad(False)

    def _embed_with(
        self,
        encoder,
        embedding_head,
        x: torch.Tensor,
    ) -> torch.Tensor:
        temporal = encoder.forward_features(x)
        return embedding_head(temporal)

    def _set_target_requires_grad(
        self,
        requires_grad: bool,
    ) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = requires_grad

        for parameter in self.target_embedding_head.parameters():
            parameter.requires_grad = requires_grad

        for parameter in self.target_projector.parameters():
            parameter.requires_grad = requires_grad

    @staticmethod
    @torch.no_grad()
    def _copy_buffers(
        online_module: nn.Module,
        target_module: nn.Module,
    ) -> None:
        for online_buffer, target_buffer in zip(
            online_module.buffers(),
            target_module.buffers(),
        ):
            target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def update_target_network(
        self,
        ema_decay: float | None = None,
    ) -> None:
        decay = (
            self.ema_decay
            if ema_decay is None
            else ema_decay
        )

        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_parameter.data.mul_(
                decay
            ).add_(
                online_parameter.data,
                alpha=1.0 - decay,
            )

        for online_parameter, target_parameter in zip(
            self.online_embedding_head.parameters(),
            self.target_embedding_head.parameters(),
        ):
            target_parameter.data.mul_(
                decay
            ).add_(
                online_parameter.data,
                alpha=1.0 - decay,
            )

        for online_parameter, target_parameter in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            target_parameter.data.mul_(
                decay
            ).add_(
                online_parameter.data,
                alpha=1.0 - decay,
            )

        self._copy_buffers(
            self.online_encoder,
            self.target_encoder,
        )

        self._copy_buffers(
            self.online_embedding_head,
            self.target_embedding_head,
        )

        self._copy_buffers(
            self.online_projector,
            self.target_projector,
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
        online_h1 = self._embed_with(
            self.online_encoder,
            self.online_embedding_head,
            x1,
        )

        online_h2 = self._embed_with(
            self.online_encoder,
            self.online_embedding_head,
            x2,
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
            target_h1 = self._embed_with(
                self.target_encoder,
                self.target_embedding_head,
                x1,
            )

            target_h2 = self._embed_with(
                self.target_encoder,
                self.target_embedding_head,
                x2,
            )

            target_z1 = self.target_projector(
                target_h1
            )

            target_z2 = self.target_projector(
                target_h2
            )

        return 0.5 * (
            self.byol_loss(
                prediction1,
                target_z2,
            )
            + self.byol_loss(
                prediction2,
                target_z1,
            )
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._embed_with(
            self.online_encoder,
            self.online_embedding_head,
            x,
        )


def cosine_ema_decay(
    step: int,
    total_steps: int,
    base_decay: float,
) -> float:
    if total_steps <= 1:
        return 1.0

    progress = min(
        max(step / (total_steps - 1), 0.0),
        1.0,
    )

    return 1.0 - (
        1.0 - base_decay
    ) * (
        math.cos(math.pi * progress) + 1.0
    ) * 0.5


def build_weight_decay_groups(
    model: nn.Module,
    weight_decay: float,
):
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim == 1
            or name.endswith(".bias")
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
):
    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        remaining_steps = max(
            1,
            total_steps - warmup_steps,
        )

        progress = (
            step - warmup_steps
        ) / remaining_steps

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def train_byol(
    X_train,
    device,
    embedding_dim: int = 256,
    proj_dim: int = 128,
    hidden_dim: int = 512,
    ema_decay: float = 0.996,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    warmup_epochs: int = 5,
    max_shift: int = 3,
    noise_std: float = 0.01,
    scale_std: float = 0.0,
    mask_ratio: float = 0.0,
):
    model = BYOL1D(
        embedding_dim=embedding_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
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

    parameter_groups = build_weight_decay_groups(
        model=model,
        weight_decay=weight_decay,
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=lr,
    )

    total_steps = (
        len(loader) * n_epochs
    )

    warmup_steps = (
        len(loader) * warmup_epochs
    )

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    global_step = 0

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
        "Backbone trainable parameters:",
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

        temporal_features = (
            model.online_encoder.forward_features(
                sample_x
            )
        )

        embedding_features = model.encode(
            sample_x
        )

        projected_features = (
            model.online_projector(
                embedding_features
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
        "BYOL embedding shape:",
        embedding_features.shape,
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

            x1 = augment_trace(
                batch_x,
                max_shift=max_shift,
                noise_std=noise_std,
                scale_std=scale_std,
                mask_ratio=mask_ratio,
            )

            x2 = augment_trace(
                batch_x,
                max_shift=max_shift,
                noise_std=noise_std,
                scale_std=scale_std,
                mask_ratio=mask_ratio,
            )

            loss = model(
                x1,
                x2,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()

            optimizer.step()

            ema_decay_now = cosine_ema_decay(
                step=global_step,
                total_steps=total_steps,
                base_decay=ema_decay,
            )

            model.update_target_network(
                ema_decay=ema_decay_now,
            )

            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            num_batches += 1

        average_loss = (
            total_loss
            / max(num_batches, 1)
        )

        loss_log.append(
            average_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch #{epoch}: "
            f"byol_loss={average_loss:.6f}, "
            f"lr={current_lr:.8f}, "
            f"ema={ema_decay_now:.6f}"
        )

    return model, loss_log


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

    n_epochs = 30
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-6
    warmup_epochs = 5

    backbone_name = "triplet_network_cnn"
    embedding_dim = 256
    proj_dim = 128
    hidden_dim = 512
    ema_decay = 0.996

    max_shift = 10
    noise_std = 0.05
    scale_std = 0.0
    mask_ratio = 0.0

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
        f"_emb{embedding_dim}"
        f"_scheduled"
        f"_simclr_aug"
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
        "Training BYOL-1D..."
    )

    train_start_time = time.time()

    model, loss_log = train_byol(
        X_train=X_train,
        device=device,
        embedding_dim=embedding_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        max_shift=max_shift,
        noise_std=noise_std,
        scale_std=scale_std,
        mask_ratio=mask_ratio,
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
        repr_dir / "repr_train.npy",
        repr_train,
    )

    np.save(
        repr_dir / "repr_attack.npy",
        repr_attack,
    )

    np.save(
        repr_dir / "y_train.npy",
        y_train,
    )

    print(
        "Training linear classifier "
        "on BYOL representations..."
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    classifier.fit(
        repr_train,
        y_train,
    )

    train_acc = float(
        classifier.score(
            repr_train,
            y_train,
        )
    )

    print(
        "Linear probe train accuracy:",
        train_acc,
    )

    print(
        "Predicting attack probabilities..."
    )

    attack_probas_seen = (
        classifier.predict_proba(
            repr_attack
        )
    )

    attack_probas = expand_proba_to_256(
        attack_probas_seen,
        classes=classifier.classes_,
    )

    print(
        "attack_probas shape:",
        attack_probas.shape,
    )

    print(
        "Computing key rank curve..."
    )

    ranks = compute_rank_curve(
        probas=attack_probas,
        metadata=metadata_attack_small,
        target_byte=target_byte,
        max_traces=n_attack,
        use_log=True,
    )

    final_rank = int(
        ranks[-1]
    )

    min_rank = int(
        ranks.min()
    )

    rank0_indices = np.where(
        ranks == 0
    )[0]

    rank0_trace = (
        int(rank0_indices[0] + 1)
        if len(rank0_indices) > 0
        else -1
    )

    print(
        "Final rank:",
        final_rank,
    )

    print(
        "Minimum rank:",
        min_rank,
    )

    print(
        "Rank-0 trace:",
        rank0_trace,
    )

    rank_path = (
        figure_dir
        / f"{run_name}_linear_probe_rank.png"
    )

    ranks_path = (
        repr_dir
        / f"{run_name}_linear_probe_ranks.npy"
    )

    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title=(
            "BYOL Shared Triplet CNN + Triplet-Style 256-D Embedding "
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
        in model.online_encoder.parameters()
        if parameter.requires_grad
    )

    append_experiment_result(
        summary_path,
        {
            "method": "BYOL-shared-triplet-cnn",
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "seed": seed,
            "n_train": n_train,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": "AdamW",
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "lr_schedule": "warmup_cosine",
            "ema_schedule": "cosine_to_1",
            "backbone_name": backbone_name,
            "backbone_params": backbone_params,
            "encoder_output_channels": (
                model.online_encoder.get_temporal_output_dim()
            ),
            "embedding_head": "flatten_4096_4096_256",
            "flatten_dim": model.flatten_dim,
            "embedding_dim": model.embedding_dim,
            "pooled_repr_dim": (
                model.pooled_repr_dim
            ),
            "backbone_temporal_length": (
                model.online_encoder.get_temporal_length()
            ),
            "proj_dim": proj_dim,
            "hidden_dim": hidden_dim,
            "ema_decay": ema_decay,
            "max_shift": max_shift,
            "noise_std": noise_std,
            "scale_std": scale_std,
            "mask_ratio": mask_ratio,
            "normalize": normalize_mode,
            "window_start": window_start,
            "window_end": window_end,
            "window_size": window_size,
            "classifier": (
                "LogisticRegression"
            ),
            "linear_probe_train_acc": round(
                train_acc,
                6,
            ),
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
            "final_rank": final_rank,
            "min_rank": min_rank,
            "rank0_trace": rank0_trace,
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