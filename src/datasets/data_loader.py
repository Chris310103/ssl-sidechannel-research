import os
import sys
import argparse
import h5py
import numpy as np

from typing import Literal, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict, Any

from utils.transform_ascad_data import extract_plaintext_from_metadata


def get_trace_window(trace_window_str):
    tmp = trace_window_str.split("_")
    start = int(tmp[0])
    end = int(tmp[1])
    return (start, end)


def load_from_hdf5(h5_path, which_one):
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
    profiling_plaintext, profiling_key = extract_plaintext_from_metadata(in_file['Profiling_traces/metadata'])
    attack_plaintext, attack_key = extract_plaintext_from_metadata(in_file['Attack_traces/metadata'])

    if "train" == which_one:
        train_data_dict = {
            "X_train": X_profiling,
            "y_train": Y_profiling,
            "plaintext": profiling_plaintext,
            "key": profiling_key,
        }
        return train_data_dict
    elif "test" == which_one:
        test_data_dict = {
            "X_test": X_attack,
            "y_test": Y_attack,
            "plaintext": attack_plaintext,
            "key": attack_key
        }
        return test_data_dict
    else:
        raise ValueError(f"Unsupported value for 'which_one': {which_one}. Use 'train' or 'test'.") 


def load_from_npz(data_path):
    data_name = os.path.basename(data_path).split(".")[0]

    if "train" in data_name:
        train_data_dict = np.load(data_path, allow_pickle=True)
        return train_data_dict
    elif "test" in data_name:
        test_data_dict = np.load(data_path, allow_pickle=True)
        return test_data_dict
    else:
        raise ValueError(f"Unsupported NPZ file name: {data_name}. It should contain 'train' or 'test' in the filename.")
    


def load_dataset(data_path, which_one="train"):
    """ function to load the dataset from the given path. It supports loading from HDF5 or NPZ files. """
    if data_path.endswith(".h5"):
        data_dict = load_from_hdf5(data_path, which_one=which_one)
    elif data_path.endswith(".npz"):
        data_dict = load_from_npz(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Use .h5 or .npz.")

    return data_dict


def dissemble_data_dict(data_dict, tracewindow, which_one="train"):
    """ function to dissemble the data dictionary into X, y, and plaintext. """
    if "train" == which_one:
        X = data_dict.get("X_train")
        y = data_dict.get("y_train")
    elif "test" == which_one:
        X = data_dict.get("X_test")
        y = data_dict.get("y_test")
    else:
        raise ValueError(f"Unsupported value for 'which_one': {which_one}. Use 'train' or 'test'.")

    plaintext = data_dict.get("plaintext")
    key = data_dict.get("key")

    if tracewindow is not None:
        start, end = tracewindow
        X = X[:, start:end]

    return X, y, plaintext, key


def parse_opts(argv):
    parser = argparse.ArgumentParser(description="Load dataset from HDF5 or NPZ file.")
    parser.add_argument("-i", "--input_path", type=str, help="Path to the dataset file (.h5 or .npz).")
    parser.add_argument("-t", "--trace_window", type=str, help="Optional trace window as two integers: start end.")
    parser.add_argument("-w", "--which_one", type=str, choices=["train", "test"], default="train", help="Specify whether to load 'train' or 'test' data.")
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    # Example usage of the load_dataset function and for testing the loading of the dataset
    opts = parse_opts(sys.argv[1:])
    data_path = opts.input_path
    trace_window_str = opts.trace_window
    trace_window = get_trace_window(trace_window_str)

    data_dict = load_dataset(data_path, which_one=opts.which_one)

    X_data, y_data, plaintext, key = dissemble_data_dict(data_dict, trace_window, which_one=opts.which_one)

    print("X shape:", X_data.shape)
    print("y shape:", y_data.shape)
    print("Plaintext shape:", plaintext.shape)
    print("Plaintext:", plaintext[:10])  # Print first 10 plaintext values for verification
    print("Key shape:", key.shape)
