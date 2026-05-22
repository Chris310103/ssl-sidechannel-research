
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict, Any

import h5py
import numpy as np


SplitName = Literal["profiling", "attack"]


def _resolve_group_name(split: SplitName) -> str:
    
    if split == "profiling":
        return "Profiling_traces"
    if split == "attack":
        return "Attack_traces"
    raise ValueError(f"Unsupported split: {split}. Use 'profiling' or 'attack'.")


def _apply_normalization(
    X: np.ndarray,
    normalize: Optional[Literal["divide128", "zscore"]] = None,
) -> np.ndarray:

    X = X.astype(np.float32)

    if normalize is None:
        return X

    if normalize == "divide128":
        return X / 128.0

    if normalize == "zscore":
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (X - mean) / std

    raise ValueError(f"Unsupported normalization: {normalize}")


def load_ascad_split(
    h5_path: Union[str, Path],
    split: SplitName = "profiling",
    add_channel: bool = True,
    normalize: Optional[Literal["divide128", "zscore"]] = None,
    load_metadata: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"ASCAD file not found: {h5_path}")

    group_name = _resolve_group_name(split)

    with h5py.File(h5_path, "r") as f:
        if group_name not in f:
            raise KeyError(f"Group '{group_name}' not found in {h5_path}")

        group = f[group_name]

        X = np.array(group["traces"], dtype=np.float32)
        y = np.array(group["labels"])

        X = _apply_normalization(X, normalize=normalize)

        if add_channel:
            X = X[..., None]  # (N, T) -> (N, T, 1)

        if load_metadata:
            metadata = np.array(group["metadata"])
            return X, y, metadata

    return X, y


def load_ascad_for_ts2vec(
    h5_path: Union[str, Path],
    split: SplitName = "profiling",
    normalize: Optional[Literal["divide128", "zscore"]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    X, y = load_ascad_split(
        h5_path=h5_path,
        split=split,
        add_channel=True,
        normalize=normalize,
        load_metadata=False,
    )
    return X, y


def load_ascad_all(
    h5_path: Union[str, Path],
    add_channel: bool = True,
    normalize: Optional[Literal["divide128", "zscore"]] = None,
    load_metadata: bool = False,
) -> Dict[str, Any]:
    
    if load_metadata:
        X_profiling, y_profiling, metadata_profiling = load_ascad_split(
            h5_path=h5_path,
            split="profiling",
            add_channel=add_channel,
            normalize=normalize,
            load_metadata=True,
        )

        X_attack, y_attack, metadata_attack = load_ascad_split(
            h5_path=h5_path,
            split="attack",
            add_channel=add_channel,
            normalize=normalize,
            load_metadata=True,
        )

        return {
            "X_profiling": X_profiling,
            "y_profiling": y_profiling,
            "metadata_profiling": metadata_profiling,
            "X_attack": X_attack,
            "y_attack": y_attack,
            "metadata_attack": metadata_attack,
        }

    X_profiling, y_profiling = load_ascad_split(
        h5_path=h5_path,
        split="profiling",
        add_channel=add_channel,
        normalize=normalize,
        load_metadata=False,
    )

    X_attack, y_attack = load_ascad_split(
        h5_path=h5_path,
        split="attack",
        add_channel=add_channel,
        normalize=normalize,
        load_metadata=False,
    )

    return {
        "X_profiling": X_profiling,
        "y_profiling": y_profiling,
        "X_attack": X_attack,
        "y_attack": y_attack,
    }


if __name__ == "__main__":
    ascad_path = "data/raw/ascad/ASCAD.h5"

    X, y = load_ascad_for_ts2vec(ascad_path, split="profiling")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)
    print("First labels:", y[:10])