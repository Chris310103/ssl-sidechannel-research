import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.methods.byol_rank_pipeline import BYOL1D
from src.methods.cpc_rank_pipeline import CPCSharedModel
from src.methods.mae_rank_pipeline import FCMAESharedCNN1D
from src.methods.simclr_rank_pipeline import SimCLRModel
from src.methods.ts2vec_rank_pipeline import TS2VecSharedModel
from src.utils.get_device import get_device
from src.utils.key_rank import (
    compute_rank_curve,
    leakage_labels,
    metadata_plaintext,
    metadata_true_key,
    plot_rank_curve,
)
from src.utils.trace_transforms import (
    ensure_trace_matrix,
    prepare_model_input,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_trace_window(value: str):
    try:
        window_start, window_end = value.split("_")
        return int(window_start), int(window_end)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "trace window must use the form START_END, for example 200_900"
        ) from error


def build_ssl_model(opts, input_length: int):
    if opts.method == "simclr":
        return SimCLRModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
            projector_hidden_dim=opts.projector_hidden_dim,
            proj_dim=opts.proj_dim,
        )

    if opts.method == "byol":
        return BYOL1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            proj_dim=opts.proj_dim,
            hidden_dim=opts.hidden_dim,
            ema_decay=opts.ema_decay,
            input_length=input_length,
        )

    if opts.method == "cpc":
        return CPCSharedModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            context_dim=opts.context_dim,
            prediction_steps=opts.prediction_steps,
            input_length=input_length,
        )

    if opts.method == "mae":
        return FCMAESharedCNN1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            patch_size=opts.patch_size,
            mask_ratio=opts.mask_ratio,
            input_length=input_length,
        )

    if opts.method == "ts2vec":
        return TS2VecSharedModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
        )

    raise ValueError(f"Unsupported method: {opts.method}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


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

            representation = model.encode(batch_x)
            representations.append(
                representation.cpu().numpy()
            )

    return np.concatenate(representations, axis=0)


def get_profile_labels(metadata, target_byte: int, leakage_model: str):
    plaintext = metadata_plaintext(
        metadata,
        target_byte=target_byte,
    )
    true_key = metadata_true_key(
        metadata,
        target_byte=target_byte,
    )

    return leakage_labels(
        plaintext,
        key_guess=true_key,
        leakage_model=leakage_model,
    )


def expand_classifier_proba(
    probas,
    classes,
    n_classes: int,
    eps: float = 1e-40,
):
    full_probas = np.full(
        (probas.shape[0], n_classes),
        eps,
        dtype=np.float64,
    )
    full_probas[:, np.asarray(classes, dtype=np.int64)] = probas

    row_sum = full_probas.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)

    return full_probas / row_sum


def main(opts):
    set_seed(opts.seed)

    trace_window = parse_trace_window(opts.trace_window)
    window_start, window_end = trace_window
    window_size = window_end - window_start

    ascad_path = (
        Path(opts.input)
        if opts.input
        else PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"
    )

    output_dir = (
        Path(opts.output_dir)
        if opts.output_dir
        else PROJECT_ROOT / "outputs" / "profiling_attacks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ASCAD profiling traces with metadata...")
    X_profiling, _, metadata_profiling = load_ascad_split(
        h5_path=ascad_path,
        split="profiling",
        add_channel=False,
        normalize=opts.normalize,
        load_metadata=True,
        trace_window=None,
    )

    print("Loading ASCAD attack traces with metadata...")
    X_attack, _, metadata_attack = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=False,
        normalize=opts.normalize,
        load_metadata=True,
        trace_window=None,
    )

    X_profile_small = X_profiling[: opts.n_profile]
    metadata_profile_small = metadata_profiling[: opts.n_profile]
    X_attack_small = X_attack[: opts.n_attack]
    metadata_attack_small = metadata_attack[: opts.n_attack]

    print("Method:", opts.method)
    print("Trace window:", trace_window)
    print("Window size:", window_size)
    print("X_profile shape:", X_profile_small.shape)
    print("X_attack shape:", X_attack_small.shape)

    device = get_device(prefer_mps=False)
    print("Using device:", device)

    model = build_ssl_model(
        opts,
        input_length=window_size,
    ).to(device)
    model = load_checkpoint(
        model,
        checkpoint_path=opts.checkpoint,
        device=device,
    )

    encode_start_time = time.time()

    print("Encoding profiling representations...")
    repr_profile = encode_representations(
        model=model,
        X=X_profile_small,
        device=device,
        trace_window=trace_window,
        batch_size=opts.encode_batch_size,
    )

    print("Encoding attack representations...")
    repr_attack = encode_representations(
        model=model,
        X=X_attack_small,
        device=device,
        trace_window=trace_window,
        batch_size=opts.encode_batch_size,
    )

    encode_time_sec = time.time() - encode_start_time

    y_profile = get_profile_labels(
        metadata_profile_small,
        target_byte=opts.target_byte,
        leakage_model=opts.leakage_model,
    )

    print("repr_profile shape:", repr_profile.shape)
    print("repr_attack shape:", repr_attack.shape)
    print("Training profiling classifier...")

    classifier = LogisticRegression(
        max_iter=opts.max_iter,
        solver="lbfgs",
    )
    classifier.fit(
        repr_profile,
        y_profile,
    )

    train_accuracy = float(
        classifier.score(
            repr_profile,
            y_profile,
        )
    )

    print("Profiling classifier train accuracy:", train_accuracy)
    print("Predicting attack probabilities...")

    attack_probas_seen = classifier.predict_proba(repr_attack)
    n_classes = 256 if opts.leakage_model == "ID" else 9
    attack_probas = expand_classifier_proba(
        attack_probas_seen,
        classes=classifier.classes_,
        n_classes=n_classes,
    )

    print("Computing profiling key-rank curve...")
    ranks = compute_rank_curve(
        probas=attack_probas,
        metadata=metadata_attack_small,
        target_byte=opts.target_byte,
        max_traces=opts.n_attack,
        leakage_model=opts.leakage_model,
    )

    final_rank = int(ranks[-1])
    min_rank = int(ranks.min())
    rank0_indices = np.where(ranks == 0)[0]
    rank0_trace = (
        int(rank0_indices[0] + 1)
        if len(rank0_indices) > 0
        else -1
    )

    print("Final rank:", final_rank)
    print("Minimum rank:", min_rank)
    print("Rank-0 trace:", rank0_trace)

    run_name = (
        f"{opts.method}_{opts.backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_{opts.pool_mode}"
        f"_{opts.leakage_model}"
        f"_profile{opts.n_profile}"
        f"_attack{opts.n_attack}"
        f"_seed{opts.seed}"
    )
    rank_path = output_dir / f"{run_name}_rank.png"
    ranks_path = output_dir / f"{run_name}_ranks.npy"
    probas_path = output_dir / f"{run_name}_attack_probas.npy"

    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title=f"{opts.method.upper()} Profiling Attack Key Rank",
    )

    np.save(ranks_path, ranks)
    np.save(probas_path, attack_probas)

    print("Saved rank curve to:", rank_path)
    print("Saved ranks to:", ranks_path)
    print("Saved attack probabilities to:", probas_path)
    print(f"Encoding time: {encode_time_sec:.2f} sec")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("simclr", "byol", "cpc", "mae", "ts2vec"),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trace-window", default="0_700")
    parser.add_argument("--n-profile", type=int, default=50000)
    parser.add_argument("--n-attack", type=int, default=10000)
    parser.add_argument("--target-byte", type=int, default=2)
    parser.add_argument("--leakage-model", choices=("ID", "HW"), default="ID")
    parser.add_argument("--normalize", choices=("divide128", "zscore"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=2000)

    parser.add_argument("--backbone-name", default="shared_cnn_v1")
    parser.add_argument("--pool-mode", default="mean_max")

    parser.add_argument("--projector-hidden-dim", type=int, default=320)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--context-dim", type=int, default=320)
    parser.add_argument("--prediction-steps", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--mask-ratio", type=float, default=0.30)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
