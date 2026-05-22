from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.datasets.ascad_loader import load_ascad_split
from src.evaluation.rank_eval import (
    expand_proba_to_256,
    compute_rank_curve,
    plot_rank_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TS2VEC_ROOT = PROJECT_ROOT / "external" / "ts2vec"

if str(TS2VEC_ROOT) not in sys.path:
    sys.path.insert(0, str(TS2VEC_ROOT))

from ts2vec import TS2Vec


def get_device(prefer_mps: bool = False) -> str:
    """
    Select PyTorch device.

    Priority:
        CUDA -> MPS if enabled -> CPU
    """
    if torch.cuda.is_available():
        return "cuda"

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main():
    ascad_path = PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"

    output_dir = PROJECT_ROOT / "outputs" / "ts2vec_rank"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start small first. Later we can increase these.
    n_train = 50000
    n_attack = 10000
    n_epochs = 10

    print("Loading ASCAD profiling traces...")
    X_profiling, y_profiling = load_ascad_split(
        h5_path=ascad_path,
        split="profiling",
        add_channel=True,
        normalize=None,
        load_metadata=False,
    )

    print("Loading ASCAD attack traces with metadata...")
    X_attack, y_attack, metadata_attack = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=True,
        normalize=None,
        load_metadata=True,
    )

    X_train = X_profiling[:n_train]
    y_train = y_profiling[:n_train]

    X_attack_small = X_attack[:n_attack]
    metadata_attack_small = metadata_attack[:n_attack]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_attack shape:", X_attack_small.shape)
    print("metadata_attack shape:", metadata_attack_small.shape)

    device = get_device(prefer_mps=False)
    print("Using device:", device)

    model = TS2Vec(
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=6,
        device=device,
        lr=0.001,
        batch_size=64,
    )

    print("Training TS2Vec...")
    loss_log = model.fit(
        X_train,
        n_epochs=n_epochs,
        verbose=True,
    )
    print("TS2Vec loss log:", loss_log)

    print("Encoding profiling representations...")
    repr_train = model.encode(
        X_train,
        encoding_window="full_series",
    )

    print("Encoding attack representations...")
    repr_attack = model.encode(
        X_attack_small,
        encoding_window="full_series",
    )

    print("repr_train shape:", repr_train.shape)
    print("repr_attack shape:", repr_attack.shape)

    np.save(output_dir / "repr_train.npy", repr_train)
    np.save(output_dir / "repr_attack.npy", repr_attack)
    np.save(output_dir / "y_train.npy", y_train)

    print("Training linear classifier on TS2Vec representations...")
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(repr_train, y_train)

    print("Predicting attack probabilities...")
    attack_probas_seen = clf.predict_proba(repr_attack)
    attack_probas = expand_proba_to_256(
        attack_probas_seen,
        classes=clf.classes_,
    )

    print("attack_probas shape:", attack_probas.shape)

    print("Computing key rank curve...")
    ranks = compute_rank_curve(
        probas=attack_probas,
        metadata=metadata_attack_small,
        target_byte=2,
        max_traces=n_attack,
        use_log=True,
    )

    print("Final rank:", ranks[-1])
    print("Minimum rank:", ranks.min())

    rank_path = output_dir / "ts2vec_linear_probe_rank.png"
    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title="TS2Vec + Linear Probe Key Rank",
    )

    np.save(output_dir / "ts2vec_linear_probe_ranks.npy", ranks)

    print("Saved rank curve to:", rank_path)
    print("Saved ranks to:", output_dir / "ts2vec_linear_probe_ranks.npy")


if __name__ == "__main__":
    main()