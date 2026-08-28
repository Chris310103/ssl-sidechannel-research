import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.methods.knn_distinguisher import (
    compute_knn_candidate_accuracies,
    rank_key_candidates,
    split_nonprofiling_attack_data,
)
from src.methods.ssl_factory import SSL_METHODS, build_ssl_model
from src.utils.cli_parsers import parse_count_pair, parse_range
from src.utils.get_device import get_device
from src.utils.key_rank import metadata_true_key
from src.utils.trace_transforms import ensure_trace_matrix, prepare_model_input


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ASCAD_PATH = PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one SSL encoder with 256 non-profiling KNN distinguishers."
    )

    parser.add_argument("--method", choices=SSL_METHODS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=str(DEFAULT_ASCAD_PATH))
    parser.add_argument("--trace-window", type=parse_range, default=(0, 700))
    parser.add_argument(
        "--nv",
        type=parse_count_pair,
        default=(500, 25),
        help="KNN train/eval trace counts as TRAIN_EVAL, for example 500_25.",
    )
    parser.add_argument("--normalize", choices=("divide128", "zscore"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-byte", type=int, default=2)
    parser.add_argument("--leakage-model", choices=("ID", "HW"), default="HW")
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--knn-weights", choices=("uniform", "distance"), default="distance")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "nonprofiling_attacks"),
    )

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

    return parser


def load_checkpoint(model, checkpoint_path: str, device: str) -> dict:
    payload = torch.load(checkpoint_path, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload
    else:
        state_dict = payload
        metadata = {}

    model.load_state_dict(state_dict)
    return metadata


def encode_representations(
    model,
    X,
    device: str,
    trace_window,
    batch_size: int,
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
            representations.append(model.encode(batch_x).cpu().numpy())

    return np.concatenate(representations, axis=0)


def make_run_name(opts, true_key: int, true_key_rank: int) -> str:
    window_start, window_end = opts.trace_window
    n_knn_train, n_knn_eval = opts.nv
    return (
        f"{opts.method}_knn_w{window_start}-{window_end}"
        f"_nv{n_knn_train}-{n_knn_eval}_byte{opts.target_byte}"
        f"_key{true_key}_rank{true_key_rank}"
    )


def save_attack_artifacts(
    opts,
    checkpoint_metadata,
    candidate_scores,
    ranked_keys,
    true_key: int,
    true_key_rank: int,
) -> None:
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = make_run_name(
        opts=opts,
        true_key=true_key,
        true_key_rank=true_key_rank,
    )

    scores_path = output_dir / f"{run_name}_candidate_accuracies.npy"
    ranked_keys_path = output_dir / f"{run_name}_ranked_keys.npy"
    summary_path = output_dir / f"{run_name}_summary.json"

    np.save(scores_path, candidate_scores)
    np.save(ranked_keys_path, ranked_keys)

    summary = {
        "method": opts.method,
        "checkpoint": opts.checkpoint,
        "checkpoint_metadata": {
            key: value
            for key, value in checkpoint_metadata.items()
            if key != "model_state_dict"
        },
        "trace_window": tuple(opts.trace_window),
        "nv": tuple(opts.nv),
        "target_byte": opts.target_byte,
        "leakage_model": opts.leakage_model,
        "knn_neighbors": opts.knn_neighbors,
        "knn_weights": opts.knn_weights,
        "true_key": true_key,
        "best_key": int(ranked_keys[0]),
        "best_accuracy": float(candidate_scores[ranked_keys[0]]),
        "true_key_accuracy": float(candidate_scores[true_key]),
        "true_key_candidate_rank": true_key_rank,
        "candidate_scores_path": str(scores_path),
        "ranked_keys_path": str(ranked_keys_path),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"Saved summary: {summary_path}")
    print(f"Best key: {summary['best_key']} accuracy={summary['best_accuracy']:.6f}")
    print(
        f"True key: {true_key} rank={true_key_rank} "
        f"accuracy={summary['true_key_accuracy']:.6f}"
    )


def main(opts) -> None:
    set_seed(opts.seed)

    window_start, window_end = opts.trace_window
    input_length = window_end - window_start
    n_knn_train, n_knn_eval = opts.nv

    X_attack, _, metadata_attack = load_ascad_split(
        h5_path=opts.input,
        split="attack",
        add_channel=False,
        normalize=opts.normalize,
        load_metadata=True,
        trace_window=None,
    )

    _, X_knn_train, metadata_knn_train, X_knn_eval, metadata_knn_eval = (
        split_nonprofiling_attack_data(
            X_attack=X_attack,
            metadata_attack=metadata_attack,
            n_ssl_train=n_knn_train,
            n_knn_train=n_knn_train,
            n_knn_eval=n_knn_eval,
            n_neighbors=opts.knn_neighbors,
        )
    )

    device = get_device(prefer_mps=False)
    model = build_ssl_model(opts, input_length=input_length).to(device)
    checkpoint_metadata = load_checkpoint(
        model=model,
        checkpoint_path=opts.checkpoint,
        device=device,
    )

    repr_train = encode_representations(
        model=model,
        X=X_knn_train,
        device=device,
        trace_window=opts.trace_window,
        batch_size=opts.encode_batch_size,
    )
    repr_eval = encode_representations(
        model=model,
        X=X_knn_eval,
        device=device,
        trace_window=opts.trace_window,
        batch_size=opts.encode_batch_size,
    )

    candidate_scores = compute_knn_candidate_accuracies(
        repr_train=repr_train,
        metadata_train=metadata_knn_train,
        repr_eval=repr_eval,
        metadata_eval=metadata_knn_eval,
        target_byte=opts.target_byte,
        n_neighbors=opts.knn_neighbors,
        leakage_model=opts.leakage_model,
        weights=opts.knn_weights,
    )

    true_key = metadata_true_key(
        metadata_knn_eval,
        target_byte=opts.target_byte,
    )
    ranked_keys, true_key_rank = rank_key_candidates(
        candidate_scores=candidate_scores,
        true_key=true_key,
    )

    save_attack_artifacts(
        opts=opts,
        checkpoint_metadata=checkpoint_metadata,
        candidate_scores=candidate_scores,
        ranked_keys=ranked_keys,
        true_key=true_key,
        true_key_rank=true_key_rank,
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
