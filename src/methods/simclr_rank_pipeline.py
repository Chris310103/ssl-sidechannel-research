import sys
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
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
    VALID_AUGMENTATIONS,
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


class SimCLRModel(nn.Module):
    def __init__(self, backbone_name, pool_mode, input_length, projector_hidden_dim, proj_dim):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode

        self.encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
        )

        self.repr_dim = self.encoder.get_output_dim(pool=pool_mode)

        self.projector = nn.Sequential(
            nn.Linear(self.repr_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder.encode(x, pool=self.pool_mode)
        z = self.projector(h)
        z = F.normalize(z, dim=1)

        return h, z

    def encode(self, x):
        return self.encoder.encode(x, pool=self.pool_mode)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape, "
            f"received {tuple(z1.shape)} and {tuple(z2.shape)}"
        )

    batch_size = z1.shape[0]

    if batch_size < 2:
        raise ValueError("NT-Xent loss requires batch_size >= 2")

    z = torch.cat([z1, z2], dim=0)

    similarity = torch.matmul(z, z.T) / temperature

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

    similarity = similarity.masked_fill(self_mask, -1e9)

    labels = torch.arange(2 * batch_size, device=z.device)

    labels = (labels + batch_size) % (2 * batch_size)

    return F.cross_entropy(similarity, labels)


def train_simclr(
    X_train,
    device,
    trace_window,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    projector_hidden_dim: int = 320,
    proj_dim: int = 128,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.2,
    max_shift: int = 5,
    noise_std: float = 0.05,
    denoise_kernel_size: int = 5,
    view_augmentation: str = "random",
    augmentation_family=("random_shift", "denoise", "gaussian_noise"),
    augmentation_probability: float = 0.5,
    input_length: int = 700
):
    X_train = ensure_trace_matrix(X_train)

    model = SimCLRModel(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        projector_hidden_dim=projector_hidden_dim,
        proj_dim=proj_dim,
        input_length=input_length
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train).float())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_log = []

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    backbone_params = sum(param.numel() for param in model.encoder.parameters() if param.requires_grad)

    print("Shared backbone trainable parameters:", backbone_params)

    print("Full SimCLR trainable parameters:", trainable_params)

    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_index, (batch_x,) in enumerate(loader):
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

            h1, z1 = model(x1)
            _, z2 = model(x2)

            if epoch == 0 and batch_index == 0:
                model_input = prepare_model_input(
                    batch_x,
                    trace_window=trace_window,
                )

                temporal_features = (model.encoder.forward_features(model_input))

                print("Batch input shape:", batch_x.shape)
                print("Model input shape:", model_input.shape)

                print("Temporal feature shape:", temporal_features.shape)

                print("Backbone representation shape:", h1.shape)
                print("Projection shape:", z1.shape)

            loss = nt_xent_loss(z1=z1, z2=z2, temperature=temperature)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = (total_loss / max(num_batches, 1))

        loss_log.append(avg_loss)

        print(f"Epoch = #{epoch}, simclr_loss = {avg_loss:.6f}")

    return model, loss_log


def encode_representations(
    model,
    X,
    device,
    trace_window,
    batch_size: int = 256,
    ):
    X = ensure_trace_matrix(X)
    
    dataset = TensorDataset(torch.from_numpy(X).float())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    representations = []

    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_x = prepare_model_input(
                batch_x,
                trace_window=trace_window,
            )

            h = model.encode(batch_x)

            representations.append(h.cpu().numpy())

    return np.concatenate(representations, axis=0)


def main(opts):
    ascad_path = Path(opts.input) if opts.input else PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"

    seed = opts.seed

    n_train = opts.n_train
    n_finetune = opts.n_finetune
    n_attack = opts.n_attack

    n_epochs = opts.epochs
    batch_size = opts.batch_size
    lr = opts.lr

    backbone_name = opts.backbone_name
    pool_mode = opts.pool_mode

    projector_hidden_dim = opts.projector_hidden_dim
    proj_dim = opts.proj_dim

    temperature = opts.temperature
    max_shift = opts.max_shift
    noise_std = opts.noise_std
    denoise_kernel_size = opts.denoise_kernel_size
    view_augmentation = opts.view_augmentation
    augmentation_family = parse_augmentation_family(opts.augmentations)
    augmentation_probability = opts.augmentation_probability

    target_byte = opts.target_byte
    normalize_mode = opts.normalize
    leakage_model = opts.leakage_model
    knn_neighbors = opts.knn_neighbors
    knn_weights = opts.knn_weights

    trace_window = (opts.window_start, opts.window_end)
    window_start, window_end = trace_window
    window_size = window_end - window_start

    set_seed(seed)

    run_name = (
        f"simclr_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_{pool_mode}"
        f"_proj{proj_dim}"
        f"_ep{n_epochs}"
        f"_knn{knn_neighbors}"
        f"_seed{seed}"
    )

    figure_dir = PROJECT_ROOT / "outputs" / "figures" / run_name
    repr_dir = PROJECT_ROOT / "outputs" / "representations" / run_name
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / run_name

    figure_dir.mkdir(parents=True, exist_ok=True)
    repr_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading full ASCAD attack traces with metadata...")

    X_attack, _, metadata_attack = load_ascad_split(
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

    print("Trace window:", trace_window)
    print("Window size:", window_size)
    print("Full trace length:", X_attack.shape[1])
    print("X_ssl_train shape:", X_ssl_train.shape)
    print("X_knn_train shape:", X_knn_train.shape)
    print("metadata_knn_train shape:", metadata_knn_train.shape)
    print("X_knn_eval shape:", X_knn_eval.shape)
    print("metadata_knn_eval shape:", metadata_knn_eval.shape)

    device = get_device(prefer_mps=False)

    print("Using device:", device)
    print("Training SimCLR...")

    train_start_time = time.time()

    model, loss_log = train_simclr(
        X_train=X_ssl_train,
        device=device,
        trace_window=trace_window,
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        projector_hidden_dim=projector_hidden_dim,
        proj_dim=proj_dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        max_shift=max_shift,
        noise_std=noise_std,
        denoise_kernel_size=denoise_kernel_size,
        view_augmentation=view_augmentation,
        augmentation_family=augmentation_family,
        augmentation_probability=augmentation_probability,
        input_length=window_size,
    )

    train_end_time = time.time()

    train_time_sec = train_end_time - train_start_time

    train_time_ms = train_time_sec * 1000

    print("SimCLR loss log:", loss_log)

    print(f"Training start time: {train_start_time}")

    print(f"Training end time: {train_end_time}")

    print(f"Training time: {train_time_sec:.2f} sec")

    print(f"Training time: {train_time_ms:.2f} ms")

    checkpoint_path = checkpoint_dir / f"{run_name}_encoder.pt"

    torch.save(model.state_dict(), checkpoint_path)

    print("Saved checkpoint to:", checkpoint_path)

    print("Encoding KNN-train representations...")

    repr_knn_train = encode_representations(
        model=model,
        X=X_knn_train,
        device=device,
        trace_window=trace_window,
        batch_size=opts.encode_batch_size,
    )

    print("Encoding KNN-eval representations...")

    repr_knn_eval = encode_representations(
        model=model,
        X=X_knn_eval,
        device=device,
        trace_window=trace_window,
        batch_size=opts.encode_batch_size,
    )

    print("repr_knn_train shape:", repr_knn_train.shape)

    print("repr_knn_eval shape:", repr_knn_eval.shape)

    expected_repr_dim = model.repr_dim

    if repr_knn_train.shape[1] != expected_repr_dim:
        raise ValueError(
            f"Unexpected representation dimension: expected {expected_repr_dim}, "
            f"received {repr_knn_train.shape[1]}"
        )

    np.save(repr_dir / "repr_knn_train.npy", repr_knn_train)
    np.save(repr_dir / "repr_knn_eval.npy", repr_knn_eval)

    print("Training 256 candidate KNNs and computing candidate accuracies...")

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

    candidate_scores_path = repr_dir / f"{run_name}_candidate_accuracies.npy"
    ranked_keys_path = repr_dir / f"{run_name}_ranked_keys.npy"

    np.save(candidate_scores_path, candidate_accuracies)
    np.save(ranked_keys_path, ranked_keys)

    print("Saved candidate accuracies to:", candidate_scores_path)
    print("Saved ranked keys to:", ranked_keys_path)

    summary_path = PROJECT_ROOT / "outputs" / "logs" / "experiment_summary.csv"

    backbone_params = sum(param.numel() for param in model.encoder.parameters() if param.requires_grad)

    append_experiment_result(
        summary_path,
        {
            "method": "SimCLR-nonprofiling-shared-backbone",
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
            "encoder_output_channels": (model.encoder.get_temporal_output_dim()),
            "pool_mode": pool_mode,
            "pooled_repr_dim": model.repr_dim,
            "projector_hidden_dim": (projector_hidden_dim),
            "proj_dim": proj_dim,
            "temperature": temperature,
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
            "train_start_time": (train_start_time),
            "train_end_time": (train_end_time),
            "train_time_sec": round(train_time_sec, 2),
            "train_time_ms": round(train_time_ms, 2),
            "true_key": true_key,
            "best_key": best_key,
            "best_accuracy": round(best_accuracy, 6),
            "true_key_accuracy": round(true_key_accuracy, 6),
            "true_key_candidate_rank": true_key_rank,
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "candidate_scores_path": str(candidate_scores_path.relative_to(PROJECT_ROOT)),
            "ranked_keys_path": str(ranked_keys_path.relative_to(PROJECT_ROOT)),
        },
    )

    print("Saved experiment summary to:", summary_path)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None, help="Path to ASCAD.h5")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-finetune", type=int, default=500)
    parser.add_argument("--n-attack", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-name", default="shared_cnn_v1")
    parser.add_argument("--pool-mode", default="mean_max")
    parser.add_argument("--projector-hidden-dim", type=int, default=320)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-shift", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--denoise-kernel-size", type=int, default=5)
    parser.add_argument(
        "--view-augmentation",
        choices=("random",) + VALID_AUGMENTATIONS,
        default="random",
        help="Transformation policy sampled independently for both SimCLR views.",
    )
    parser.add_argument("--augmentations", default="random_shift,denoise,gaussian_noise")
    parser.add_argument("--augmentation-probability", type=float, default=0.5)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--window-end", type=int, default=700)
    parser.add_argument("--target-byte", type=int, default=2)
    parser.add_argument("--normalize", choices=("divide128", "zscore"), default=None)
    parser.add_argument("--leakage-model", choices=("ID", "HW"), default="HW")
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--knn-weights", choices=("uniform", "distance"), default="distance")
    
    opts = parser.parse_args(argv[1:])
    return opts 


if __name__ == "__main__":
    opts = parse_args(sys.argv)
    main(opts)
