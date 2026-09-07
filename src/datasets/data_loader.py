import os
import sys
import argparse
import h5py
import numpy as np

from typing import Literal, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict, Any


def get_trace_window(trace_window_str):
    tmp = trace_window_str.split("_")
    start = int(tmp[0])
    end = int(tmp[1])
    return (start, end)


def load_from_hdf5(h5_path):
    h5_path = Path(h5_path)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(h5_path, "r")
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


def load_from_npz(npz_path):
    npz_path = Path(npz_path)

    data_dict = np.load(npz_path, allow_pickle=True)
    return data_dict


def load_dataset(data_path, trace_window = None):
    """ function to load the dataset from the given path. It supports loading from HDF5 or NPZ files. """
    if data_path.endswith(".h5"):
        data_dict = load_from_hdf5(data_path)
        X_data = data_dict["X_profiling"]
        y_data = data_dict["Y_profiling"]
        plaintext = data_dict["profiling_plaintext"]

        # save the attack/testing data
        save_path = Path(data_path).with_suffix(".npz")
        np.savez_compressed(
            save_path,
            X_test=data_dict["X_attack"],
            y_test=data_dict["Y_attack"],
            plaintext=data_dict["attack_plaintext"],
        )
    elif data_path.endswith(".npz")
        data_dict = load_from_npz(data_path)
        X_data = data_dict["X_train"]
        y_data = data_dict["y_train"]
        plaintext = data_dict["plaintext"]
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Use .h5 or .npz.")

    if trace_window is not None:
        start, end = trace_window
        X_data = X_data[:, start:end]
        y_data = y_data[:, start:end]

    return X_data, y_data, plaintext


def parse_opts(argv):
    parser = argparse.ArgumentParser(description="Load dataset from HDF5 or NPZ file.")
    parser.add_argument("-i", "input_path", type=str, help="Path to the dataset file (.h5 or .npz).")
    parser.add_argument("--trace_window", type=int, nargs=2, default=None,
                        help="Optional trace window as two integers: start end.")
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    # Example usage of the load_dataset function and for testing the loading of the dataset
    opts = parse_opts(sys.argv[1:])
    data_path = opts.input_path
    trace_window_str = opts.trace_window
    trace_window = get_trace_window(trace_window_str)

    X_data, y_data, plaintext = load_dataset(data_path, trace_window)

    print("X shape:", X_data.shape)
    print("y shape:", y_data.shape)
    print("Plaintext shape:", plaintext.shape)
    print("Plaintext:", plaintext[:10])  # Print first 10 plaintext values for verification
