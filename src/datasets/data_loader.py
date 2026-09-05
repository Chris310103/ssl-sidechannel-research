from typing import Literal, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict, Any

import h5py
import numpy as np

TraceWindow = Optional[Tuple[int, int]]

def _apply_trace_window(
    X: np.ndarray,
    trace_window: TraceWindow = None,
) -> np.ndarray:
    if trace_window is None:
        return X

    if len(trace_window) != 2:
        raise ValueError(
            "trace_window must be a tuple: (window_start, window_end)"
        )

    window_start, window_end = trace_window
    trace_length = X.shape[1]

    if window_start < 0:
        raise ValueError("window_start must be non-negative")

    if window_end <= window_start:
        raise ValueError("window_end must be greater than window_start")

    if window_end > trace_length:
        raise ValueError(
            f"window_end={window_end} exceeds trace length={trace_length}"
        )

    return X[:, window_start:window_end]

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
    trace_window: TraceWindow = None,
):
    
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

        X = _apply_trace_window(
            X,
            trace_window=trace_window,
        )

        X = _apply_normalization(X, normalize=normalize)

        if add_channel:
            X = X[..., None]  # (N, T) -> (N, T, 1)

        if load_metadata:
            metadata = np.array(group["metadata"])
            return X, y, metadata

    return X, y


def load_from_hdf5(h5_path: Union[str, Path]) -> Dict[str, Any]:
    h5_path = Path(h5_path)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except Exception:
        raise ValueError("Error: can't open HDF5 file {} for reading (it might be malformed) ...".format(ascad_database_file))

    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])

    # using numpy to save the data in .npz format
    profiling_plaintext = in_file['Profiling_traces/metadata']
    attack_plaintext = in_file['Attack_traces/metadata']

    data_dict = {
        "X_profiling": X_profiling,
        "Y_profiling": Y_profiling,
        "X_attack": X_attack,
        "Y_attack": Y_attack,
        "profiling_plaintext": profiling_plaintext,
        "attack_plaintext": attack_plaintext,
    }
    return data_dict


def load_from_npz(npz_path: Union[str, Path]) -> Dict[str, Any]:
    npz_path = Path(npz_path)

    data_dict = np.load(npz_path, allow_pickle=True)
    return data_dict


def load_dataset(inp_path: Union[str, Path], data_source: str = "hdf5", db_name: str = "ascad") -> Dict[str, Any]:
    """
        So, this function is loading the dataset based on the data source and database name.
        It currently supports loading the ASCAD dataset or DF dataset from an HDF5 file, or npz file.
        It returns a dictionary containing the profiling and attack traces, labels, and optionally metadata.
    """
    # first, we check if the input path exists
    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"Input path not found: {inp_path}")

    # second, we load the dataset based on the data source
    if data_source == "hdf5":
        data_dict = load_from_hdf5(inp_path, db_name)
    elif data_source == "npz":
        data_dict = load_from_npz(inp_path, db_name)
    else:
        raise ValueError(f"Unsupported data source: {data_source}. Use 'hdf5' or 'npz'.")

    # third, return the data dictionary
    return data_dict


if __name__ == "__main__":
    ascad_path = "data/raw/ascad/ASCAD.h5"

    X, y = load_ascad_split(
            h5_path=ascad_path,
            split="profiling",
            add_channel=True,
            normalize=None,
            load_metadata=False,
            trace_window=None,
        )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)
    print("First labels:", y[:10])