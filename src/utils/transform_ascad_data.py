import os
import sys
import argparse

import numpy as np
import h5py
import pdb


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise ValueError("Error: provided file path '%s' does not exist!" % file_path)
    return


def load_and_save_ascad(ascad_database_file, output_dir):
    check_file_exists(ascad_database_file)

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
    profile_metadata = in_file['Profiling_traces/metadata']
    attack_metadata = in_file['Attack_traces/metadata']
    pdb.set_trace()
    # save the training data to a .npz file
    output_train_file = os.path.join(output_dir, os.path.splitext(os.path.basename(ascad_database_file))[0] + "_train.npz")
    np.savez_compressed(
        output_train_file,
        X_train=X_profiling,
        y_train=Y_profiling,
        plaintext=profile_metadata
    )

    print("Successfully converted ASCAD database to .npz format with plaintext and saved training data to '%s'." % output_train_file)

    # save the testing data to a .npz file
    output_test_file = os.path.join(output_dir, os.path.splitext(os.path.basename(ascad_database_file))[0] + "_test.npz")
    np.savez_compressed(
        output_test_file,
        X_test=X_attack,
        y_test=Y_attack,
        plaintext=attack_metadata,
    )

    print("Successfully converted ASCAD database to .npz format with plaintext and saved testing data to '%s'." % output_test_file)
    

def main(opts):
    input_file = opts.input
    output_dir = opts.output
    os.makedirs(output_dir, exist_ok=True)

    print("Loading metadata from the ASCAD database...")
    print("Loading ASCAD database from '%s' and saving to '%s'..." % (input_file, output_dir))
    load_and_save_ascad(input_file, output_dir)


def parse_opts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    opts = parse_opts(sys.argv[1:])
    main(opts)
