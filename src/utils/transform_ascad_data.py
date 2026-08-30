import os
import sys
import argparse

import numpy as np
import h5py


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_ascad(ascad_database_file, output_file, load_metadata=False):
    check_file_exists(ascad_database_file)

    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except Exception:
        raise ValueError(
            "Error: can't open HDF5 file {} for reading (it might be malformed) ...".format(
                ascad_database_file
            )
        )

    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])

    # using numpy to save the data in .npz format
    if not load_metadata:
        np.savez_compressed(
            output_file,
            X_profiling=X_profiling,
            Y_profiling=Y_profiling,
            X_attack=X_attack,
            Y_attack=Y_attack,
        )

        print("Successfully converted ASCAD database to .npz format and saved to '%s'." % output_file)
    else:
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

    if opts.load_metadata:
        print("Loading metadata from the ASCAD database...")
        output_file = os.path.join(output_dir, "ascad_data_with_metadata.npz")
    else:
        output_file = os.path.join(output_dir, "ascad_data.npz")

    load_ascad(input_file, output_file, load_metadata=opts.load_metadata)


def parse_opts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('--load_metadata', action='store_true', help='Load metadata from the ASCAD database')
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    opts = parse_opts(sys.argv[1:])
    main(opts)
