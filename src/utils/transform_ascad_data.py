import os
import sys
import argparse

import numpy as np
import h5py


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise ValueError("Error: provided file path '%s' does not exist!" % file_path)
    return


def load_and_save_ascad(ascad_database_file, output_file):
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
    np.savez_compressed(
        output_file,
        X_profiling=X_profiling,
        Y_profiling=Y_profiling,
        X_attack=X_attack,
        Y_attack=Y_attack,
        profiling_metadata=profile_metadata,
        attack_metadata=attack_metadata,
    )

    print("Successfully converted ASCAD database to .npz format with metadata and saved to '%s'." % output_file)


def main(opts):
    input_file = opts.input
    output_dir = opts.output
    os.makedirs(output_dir, exist_ok=True)

    input_name = os.path.basename(input_file)
    input_name_no_ext = os.path.splitext(input_name)[0]
    output_file = os.path.join(output_dir, input_name_no_ext + ".npz")

    print("Loading metadata from the ASCAD database...")
    print("Loading ASCAD database from '%s' and saving to '%s'..." % (input_file, output_file))
    load_and_save_ascad(input_file, output_file)


def parse_opts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    opts = parse_opts(sys.argv[1:])
    main(opts)
