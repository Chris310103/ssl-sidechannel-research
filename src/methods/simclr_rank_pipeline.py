from pathlib import Path
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
from src.models.cnn_zoo import build_cnn_backbone, pool_temporal
from src.utils.experiment_logger import append_experiment_result
from src.utils.get_device import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimCLRModel(nn.Module):
    def __init__(
        self,
        projector_hidden_dim,
        proj_dim,
    ):
        super().__init__()

        self.encoder = build_cnn_backbone(
            input_channels=1,
            input_length=700,
        )

        self.readout_mode = "mean_max"

        self.repr_dim = self.encoder.get_readout_dim(
            mode=self.readout_mode,
        )

        self.projector = nn.Sequential(
            nn.Linear(
                self.repr_dim,
                projector_hidden_dim,
            ),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                projector_hidden_dim,
                proj_dim,
            ),
        )

    def forward(self, x):
        temporal = self.encoder.forward_temporal(x)

        h = pool_temporal(
            temporal,
            mode=self.readout_mode,
        )

        z = self.projector(h)
        z = F.normalize(z, dim=1)

        return h, z

    def encode(self, x):
        temporal = self.encoder.forward_temporal(x)

        return pool_temporal(
            temporal,
            mode=self.readout_mode,
        )


def random_shift(
    x: torch.Tensor,
    max_shift: int = 10,
) -> torch.Tensor:
    if max_shift <= 0:
        return x

    shifted = torch.empty_like(x)

    shifts = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(x.shape[0],),
        device=x.device,
    )

    for i, shift in enumerate(shifts):
        shifted[i] = torch.roll(
            x[i],
            shifts=int(shift.item()),
            dims=0,
        )

    return shifted


def add_gaussian_noise(
    x: torch.Tensor,
    noise_std: float = 0.05,
) -> torch.Tensor:
    if noise_std <= 0:
        return x

    trace_std = x.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    noise = (
        torch.randn_like(x)
        * trace_std
        * noise_std
    )

    return x + noise


def augment_traces(
    x: torch.Tensor,
    max_shift: int = 10,
    noise_std: float = 0.05,
) -> torch.Tensor:
    x = random_shift(
        x,
        max_shift=max_shift,
    )

    x = add_gaussian_noise(
        x,
        noise_std=noise_std,
    )

    return x


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape, "
            f"received {tuple(z1.shape)} and {tuple(z2.shape)}"
        )

    batch_size = z1.shape[0]

    if batch_size < 2:
        raise ValueError(
            "NT-Xent loss requires batch_size >= 2"
        )

    z = torch.cat(
        [z1, z2],
        dim=0,
    )

    similarity = torch.matmul(
        z,
        z.T,
    ) / temperature

    self_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z.device,
    )

    similarity = similarity.masked_fill(
        self_mask,
        -1e9,
    )

    labels = torch.arange(
        2 * batch_size,
        device=z.device,
    )

    labels = (
        labels + batch_size
    ) % (2 * batch_size)

    return F.cross_entropy(
        similarity,
        labels,
    )


def train_simclr(
    X_train,
    device,
    projector_hidden_dim: int = 320,
    proj_dim: int = 128,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.2,
    max_shift: int = 10,
    noise_std: float = 0.05,
):
    model = SimCLRModel(
        projector_hidden_dim=projector_hidden_dim,
        proj_dim=proj_dim,
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
        model.parameters(),
        lr=lr,
    )

    loss_log = []

    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    backbone_params = sum(
        param.numel()
        for param in model.encoder.parameters()
        if param.requires_grad
    )

    print(
        "Shared backbone trainable parameters:",
        backbone_params,
    )

    print(
        "Full SimCLR trainable parameters:",
        trainable_params,
    )

    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_index, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)

            x1 = augment_traces(
                batch_x,
                max_shift=max_shift,
                noise_std=noise_std,
            )

            x2 = augment_traces(
                batch_x,
                max_shift=max_shift,
                noise_std=noise_std,
            )

            h1, z1 = model(x1)
            _, z2 = model(x2)

            if epoch == 0 and batch_index == 0:
                temporal_features = (
                    model.encoder.forward_features(batch_x)
                )

                print(
                    "Batch input shape:",
                    batch_x.shape,
                )

                print(
                    "Temporal feature shape:",
                    temporal_features.shape,
                )

                print(
                    "Common readout representation shape:",
                    h1.shape,
                )
                print(
                    "Projection shape:",
                    z1.shape,
                )

            loss = nt_xent_loss(
                z1=z1,
                z2=z2,
                temperature=temperature,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = (
            total_loss
            / max(num_batches, 1)
        )

        loss_log.append(avg_loss)

        print(
            f"Epoch #{epoch}: "
            f"simclr_loss={avg_loss:.6f}"
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

            h = model.encode(batch_x)

            representations.append(
                h.cpu().numpy()
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

    backbone_name = "triplet_network_cnn"
    projector_hidden_dim = 320
    proj_dim = 128

    temperature = 0.2
    max_shift = 10
    noise_std = 0.05

    target_byte = 2
    normalize_mode = None

    trace_window = (0, 700)
    window_start, window_end = trace_window
    window_size = window_end - window_start

    set_seed(seed)

    run_name = (
        f"simclr_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_readout_meanmax"
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
        "Loading ASCAD attack traces with metadata..."
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

    X_attack_small = X_attack[:n_attack]

    metadata_attack_small = (
        metadata_attack[:n_attack]
    )

    print("Trace window:", trace_window)
    print("Window size:", window_size)

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

    print("Using device:", device)
    print("Training SimCLR...")

    train_start_time = time.time()

    model, loss_log = train_simclr(
        X_train=X_train,
        device=device,
        projector_hidden_dim=projector_hidden_dim,
        proj_dim=proj_dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        max_shift=max_shift,
        noise_std=noise_std,
    )

    train_end_time = time.time()

    train_time_sec = (
        train_end_time
        - train_start_time
    )

    train_time_ms = (
        train_time_sec * 1000
    )

    print(
        "SimCLR loss log:",
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

    expected_repr_dim = model.repr_dim

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
        "on SimCLR representations..."
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

    final_rank = int(ranks[-1])
    min_rank = int(ranks.min())

    rank0_indices = np.where(
        ranks == 0
    )[0]

    rank0_trace = (
        int(rank0_indices[0] + 1)
        if len(rank0_indices) > 0
        else -1
    )

    print("Final rank:", final_rank)
    print("Minimum rank:", min_rank)
    print("Rank-0 trace:", rank0_trace)

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
            "SimCLR Triplet CNN Backbone"
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
        param.numel()
        for param in model.encoder.parameters()
        if param.requires_grad
    )

    append_experiment_result(
        summary_path,
        {
            "method": "SimCLR-shared-triplet-cnn",
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "seed": seed,
            "n_train": n_train,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "backbone_name": backbone_name,
            "backbone_params": backbone_params,
            "encoder_output_channels": (
                model.encoder.get_temporal_output_dim()
            ),
            "pool_mode": "mean_max",
            "pooled_repr_dim": model.repr_dim,
            "backbone_temporal_length": (
                model.encoder.get_temporal_length()
            ),
            "projector_hidden_dim": (
                projector_hidden_dim
            ),
            "proj_dim": proj_dim,
            "temperature": temperature,
            "max_shift": max_shift,
            "noise_std": noise_std,
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