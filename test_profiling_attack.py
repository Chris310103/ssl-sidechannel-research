import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.utils.cli_parsers import parse_range
from src.utils.experiment_config import (
    DEFAULT_METHOD_OPTIONS,
    SSL_METHODS,
    build_method_options,
    get_method_metadata,
)
from src.utils.get_device import get_device
from src.utils.key_rank import (
    compute_rank_curve,
    leakage_labels,
    metadata_plaintext,
    metadata_true_key,
    plot_rank_curve,
)
from src.utils.trace_transforms import ensure_trace_matrix, prepare_model_input


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ASCAD_PATH = PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_payload(checkpoint_path: str, device: str):
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload
    return payload, {}


def encode_representations(model, X, device, trace_window, batch_size: int):
    X = ensure_trace_matrix(X)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    representations = []
    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = prepare_model_input(
                batch_x.to(device),
                trace_window=trace_window,
            )
            representations.append(model.encode(batch_x).cpu().numpy())

    return np.concatenate(representations, axis=0)


def expand_classifier_proba(probas, classes, n_classes: int, eps: float = 1e-40):
    full_probas = np.full(
        (probas.shape[0], n_classes),
        eps,
        dtype=np.float64,
    )
    full_probas[:, np.asarray(classes, dtype=np.int64)] = probas
    return full_probas / full_probas.sum(axis=1, keepdims=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an SSL encoder in a profiling side-channel attack."
    )
    parser.add_argument("--method", choices=SSL_METHODS, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=str(DEFAULT_ASCAD_PATH))
    parser.add_argument("--trace-window", type=parse_range, default=None)
    parser.add_argument("--n-profile", type=int, default=50000)
    parser.add_argument("--n-attack", type=int, default=10000)
    parser.add_argument("--target-byte", type=int, default=2)
    parser.add_argument("--leakage-model", choices=("ID", "HW"), default="ID")
    parser.add_argument("--normalize", choices=("divide128", "zscore"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "profiling_attacks"),
    )
    return parser


def main(opts) -> None:
    from sklearn.linear_model import LogisticRegression
    from src.methods.ssl_factory import build_ssl_model

    set_seed(opts.seed)
    device = get_device(prefer_mps=False)
    state_dict, checkpoint_metadata = load_checkpoint_payload(
        opts.checkpoint,
        device=device,
    )
    method = opts.method or checkpoint_metadata.get("method")
    if method is None:
        raise ValueError(
            "--method is required when the checkpoint has no method metadata"
        )

    trace_window = opts.trace_window or tuple(
        checkpoint_metadata.get("trace_window", (0, 700))
    )
    input_length = trace_window[1] - trace_window[0]
    model_options = build_method_options(
        method=method,
        overrides=checkpoint_metadata.get("args", {}),
    )
    model = build_ssl_model(model_options, input_length=input_length).to(device)
    model.load_state_dict(state_dict)

    X_profile, _, metadata_profile = load_ascad_split(
        h5_path=opts.input,
        split="profiling",
        add_channel=False,
        normalize=opts.normalize,
        load_metadata=True,
        trace_window=None,
    )
    X_attack, _, metadata_attack = load_ascad_split(
        h5_path=opts.input,
        split="attack",
        add_channel=False,
        normalize=opts.normalize,
        load_metadata=True,
        trace_window=None,
    )
    X_profile = X_profile[: opts.n_profile]
    metadata_profile = metadata_profile[: opts.n_profile]
    X_attack = X_attack[: opts.n_attack]
    metadata_attack = metadata_attack[: opts.n_attack]

    encode_start = time.time()
    repr_profile = encode_representations(
        model,
        X_profile,
        device,
        trace_window,
        DEFAULT_METHOD_OPTIONS["encode_batch_size"],
    )
    repr_attack = encode_representations(
        model,
        X_attack,
        device,
        trace_window,
        DEFAULT_METHOD_OPTIONS["encode_batch_size"],
    )
    encode_time_sec = time.time() - encode_start

    true_key = metadata_true_key(metadata_profile, target_byte=opts.target_byte)
    plaintext_profile = metadata_plaintext(
        metadata_profile,
        target_byte=opts.target_byte,
    )
    profile_labels = leakage_labels(
        plaintext_profile,
        key_guess=true_key,
        leakage_model=opts.leakage_model,
    )
    classifier = LogisticRegression(max_iter=opts.max_iter, solver="lbfgs")
    classifier.fit(repr_profile, profile_labels)

    observed_probas = classifier.predict_proba(repr_attack)
    number_of_classes = 256 if opts.leakage_model == "ID" else 9
    attack_probas = expand_classifier_proba(
        observed_probas,
        classifier.classes_,
        n_classes=number_of_classes,
    )
    ranks = compute_rank_curve(
        probas=attack_probas,
        metadata=metadata_attack,
        target_byte=opts.target_byte,
        max_traces=opts.n_attack,
        leakage_model=opts.leakage_model,
    )

    rank_zero = np.where(ranks == 0)[0]
    summary = {
        "method": method,
        "method_metadata": get_method_metadata(method),
        "checkpoint": opts.checkpoint,
        "trace_window": tuple(trace_window),
        "n_profile": opts.n_profile,
        "n_attack": opts.n_attack,
        "target_byte": opts.target_byte,
        "leakage_model": opts.leakage_model,
        "true_key": true_key,
        "final_rank": int(ranks[-1]),
        "minimum_rank": int(ranks.min()),
        "rank_zero_trace": int(rank_zero[0] + 1) if rank_zero.size else -1,
        "profiling_train_accuracy": float(
            classifier.score(repr_profile, profile_labels)
        ),
        "encoding_time_sec": encode_time_sec,
    }

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{method}_profile{opts.n_profile}_attack{opts.n_attack}"
        f"_byte{opts.target_byte}_seed{opts.seed}"
    )
    np.save(output_dir / f"{run_name}_ranks.npy", ranks)
    np.save(output_dir / f"{run_name}_attack_probas.npy", attack_probas)
    plot_rank_curve(
        ranks,
        save_path=output_dir / f"{run_name}_rank.png",
        title=f"{get_method_metadata(method)['display_name']} Profiling Key Rank",
    )
    with (output_dir / f"{run_name}_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(build_parser().parse_args())
