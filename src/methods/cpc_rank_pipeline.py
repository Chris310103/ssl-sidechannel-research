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


class CPCSharedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "shared_cnn_v1",
        pool_mode: str = "mean_max",
        context_dim: int = 320,
        prediction_steps: int = 6,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.context_dim = context_dim
        self.prediction_steps = prediction_steps

        self.encoder = build_cnn_backbone(
            name=backbone_name,
            input_channels=1,
        )

        self.latent_dim = self.encoder.output_channels

        self.pooled_repr_dim = self.encoder.get_output_dim(
            pool=pool_mode,
        )

        self.gru = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
        )

        self.predictor = nn.Linear(
            context_dim,
            self.latent_dim * prediction_steps,
            bias=False,
        )

    def forward(self, x):
        z = self.encoder.forward_features(x)

        c, _ = self.gru(z)

        prediction = self.predictor(c)

        return z, c, prediction

    def encode(self, x):
        return self.encoder.encode(
            x,
            pool=self.pool_mode,
        )


def cpc_reference_loss(
    z: torch.Tensor,
    prediction: torch.Tensor,
    prediction_steps: int = 6,
    negative_samples: int = 10,
):
    batch_size, sequence_length, latent_dim = z.shape
    device = z.device

    if sequence_length <= prediction_steps:
        raise ValueError(
            f"Sequence length {sequence_length} is too short "
            f"for prediction_steps={prediction_steps}"
        )

    if negative_samples < 1:
        raise ValueError(
            "negative_samples must be at least 1"
        )

    latent_pool = z.reshape(
        batch_size * sequence_length,
        latent_dim,
    )

    pool_size = latent_pool.size(0)

    total_loss = z.new_tensor(0.0)
    total_accuracy = 0.0
    loss_count = 0

    for step in range(1, prediction_steps + 1):
        prediction_step = prediction[
            :,
            :-step,
            (step - 1) * latent_dim : step * latent_dim,
        ]

        target_step = z[
            :,
            step:,
            :,
        ]

        prediction_flat = prediction_step.reshape(
            -1,
            latent_dim,
        )

        target_flat = target_step.reshape(
            -1,
            latent_dim,
        )

        number_of_predictions = prediction_flat.size(0)

        positive_scores = torch.sum(
            prediction_flat * target_flat,
            dim=1,
            keepdim=True,
        )

        target_times = torch.arange(
            step,
            sequence_length,
            device=device,
        )

        batch_offsets = (
            torch.arange(
                batch_size,
                device=device,
            ).unsqueeze(1)
            * sequence_length
        )

        positive_pool_indices = (
            batch_offsets
            + target_times.unsqueeze(0)
        ).reshape(-1)

        negative_offsets = torch.randint(
            low=1,
            high=pool_size,
            size=(
                number_of_predictions,
                negative_samples,
            ),
            device=device,
        )

        negative_indices = (
            positive_pool_indices.unsqueeze(1)
            + negative_offsets
        ) % pool_size

        negative_latents = latent_pool[
            negative_indices
        ]

        negative_scores = torch.bmm(
            negative_latents,
            prediction_flat.unsqueeze(2),
        ).squeeze(2)

        logits = torch.cat(
            [
                positive_scores,
                negative_scores,
            ],
            dim=1,
        )

        labels = torch.zeros(
            number_of_predictions,
            dtype=torch.long,
            device=device,
        )

        loss = F.cross_entropy(
            logits,
            labels,
        )

        with torch.no_grad():
            accuracy = (
                logits.argmax(dim=1)
                == labels
            ).float().mean()

        total_loss = total_loss + loss
        total_accuracy += accuracy.item()
        loss_count += 1

    average_loss = total_loss / loss_count
    average_accuracy = total_accuracy / loss_count

    return average_loss, average_accuracy


def train_cpc(
    X_train,
    device,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    context_dim: int = 320,
    prediction_steps: int = 6,
    negative_samples: int = 10,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 2e-4,
):
    model = CPCSharedModel(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        context_dim=context_dim,
        prediction_steps=prediction_steps,
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
        "Full CPC trainable parameters:",
        full_model_params,
    )

    model.eval()

    with torch.no_grad():
        sample_x = torch.from_numpy(
            X_train[:8]
        ).float().to(device)

        sample_z, sample_c, sample_prediction = model(
            sample_x
        )

        sample_representation = model.encode(
            sample_x
        )

    print(
        "Sample input shape:",
        sample_x.shape,
    )

    print(
        "Latent sequence shape:",
        sample_z.shape,
    )

    print(
        "Context sequence shape:",
        sample_c.shape,
    )

    print(
        "Prediction shape:",
        sample_prediction.shape,
    )

    print(
        "Downstream representation shape:",
        sample_representation.shape,
    )

    model.train()

    loss_log = []
    accuracy_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        number_of_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            z, _, prediction = model(
                batch_x
            )

            loss, cpc_accuracy = cpc_reference_loss(
                z=z,
                prediction=prediction,
                prediction_steps=prediction_steps,
                negative_samples=negative_samples,
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

            total_loss += loss.item()
            total_accuracy += cpc_accuracy
            number_of_batches += 1

        average_loss = (
            total_loss
            / max(number_of_batches, 1)
        )

        average_accuracy = (
            total_accuracy
            / max(number_of_batches, 1)
        )

        loss_log.append(
            average_loss
        )

        accuracy_log.append(
            average_accuracy
        )

        print(
            f"Epoch #{epoch}: "
            f"cpc_loss={average_loss:.6f}, "
            f"cpc_acc={average_accuracy:.4f}"
        )

    return model, loss_log, accuracy_log


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
    lr = 2e-4

    backbone_name = "shared_cnn_v1"
    pool_mode = "mean_max"

    context_dim = 320
    prediction_steps = 6
    negative_samples = 10

    target_byte = 2
    normalize_mode = None

    trace_window = (0, 700)

    window_start, window_end = trace_window
    window_size = window_end - window_start

    set_seed(seed)

    run_name = (
        f"cpc_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_{pool_mode}"
        f"_context{context_dim}"
        f"_pred{prediction_steps}"
        f"_neg{negative_samples}"
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

    X_attack_small = X_attack[:n_attack]

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
        "Training CPC..."
    )

    train_start_time = time.time()

    model, loss_log, accuracy_log = train_cpc(
        X_train=X_train,
        device=device,
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        context_dim=context_dim,
        prediction_steps=prediction_steps,
        negative_samples=negative_samples,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
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
        "CPC loss log:",
        loss_log,
    )

    print(
        "CPC accuracy log:",
        accuracy_log,
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
        "on CPC shared-backbone representations..."
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
            "CPC Shared CNN "
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
        for parameter in model.encoder.parameters()
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
            "method": "CPC-shared-backbone",
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
            "full_model_params": full_model_params,
            "encoder_output_channels": (
                model.latent_dim
            ),
            "pool_mode": pool_mode,
            "pooled_repr_dim": (
                model.pooled_repr_dim
            ),
            "context_dim": context_dim,
            "prediction_steps": (
                prediction_steps
            ),
            "negative_samples": (
                negative_samples
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
            "final_cpc_loss": round(
                loss_log[-1],
                6,
            ),
            "final_cpc_acc": round(
                accuracy_log[-1],
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