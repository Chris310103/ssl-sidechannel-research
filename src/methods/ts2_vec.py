from pathlib import Path
import sys
import numpy as np
import torch
from src.utils.get_device import get_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TS2VEC_ROOT = PROJECT_ROOT / "external" / "ts2vec"

if str(TS2VEC_ROOT) not in sys.path:
    sys.path.insert(0, str(TS2VEC_ROOT))

from ts2vec import TS2Vec  

from src.datasets.ascad_loader import load_ascad_for_ts2vec

def main():
    ascad_path = PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"

    print("Loading ASCAD...")
    X, y = load_ascad_for_ts2vec(
        h5_path=ascad_path,
        split="profiling",
        normalize=None,
    )

    print("Full X shape:", X.shape)
    print("Full y shape:", y.shape)

    n_train = 2048
    X_small = X[:n_train]
    y_small = y[:n_train]

    print("Small X shape:", X_small.shape)
    print("Small y shape:", y_small.shape)

    device = get_device(prefer_mps=False)
    print("Using device:", device)

    model = TS2Vec(
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=6,
        device=device,
        lr=0.001,
        batch_size=16,
    )

    print("Training TS2Vec on small ASCAD subset...")
    loss_log = model.fit(
        X_small,
        n_epochs=1,
        verbose=True,
    )

    print("Loss log:", loss_log)

    print("Encoding traces...")
    representations = model.encode(
        X_small,
        encoding_window="full_series",
    )

    print("Representations shape:", representations.shape)
    print("Representations dtype:", representations.dtype)

    output_dir = PROJECT_ROOT / "outputs" / "ts2vec"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "ascad_ts2vec_repr_2048.npy", representations)
    np.save(output_dir / "ascad_ts2vec_labels_2048.npy", y_small)

    print("Saved representations to:", output_dir / "ascad_ts2vec_repr_2048.npy")
    print("Saved labels to:", output_dir / "ascad_ts2vec_labels_2048.npy")


if __name__ == "__main__":
    main()