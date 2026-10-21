"""Dataset loading helpers for ASCAD HDF5 and NPZ splits."""
import sys
import pdb
import argparse
import numpy as np
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.transform_ascad_data import extract_plaintext_from_metadata
import src.utils.supporting_functions as supporting_func


def get_trace_window(trace_window_str: str):
    """Parse START_END and return both the slice and its model input length."""
    try:
        start_text, end_text = trace_window_str.split("_", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except (AttributeError, ValueError) as error:
        raise ValueError("trace_window must use START_END, for example 0_700") from error
    if start < 0 or end <= start:
        raise ValueError("trace_window must have 0 <= start < end")
    return (start, end), end - start


def apply_n_limit(X, y, n_train):
    """Apply the n_train limit to the training traces."""
    data_dict = defaultdict(list)
    n_train = int(n_train)  # Ensure n_train is an integer

    # using a dict to group traces by label, then limit to n_train per label
    for i in range(len(X)):
        tmp_label = y[i]
        if len(data_dict[tmp_label]) < n_train:
            data_dict[tmp_label].append(X[i])

    # convert the grouped traces back to a single array
    X_limited = []
    y_limited = []
    for label, traces in data_dict.items():
        X_limited.extend(traces)
        y_limited.extend([label] * len(traces))

    X_limited, y_limited = np.array(X_limited), np.array(y_limited)

    return X_limited, y_limited


sbox = [
    # 0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,  # 0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,  # 1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,  # 2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,  # 3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,  # 4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,  # 5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,  # 6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,  # 7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,  # 8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,  # 9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,  # a
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,  # b
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,  # c
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,  # d
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,  # e
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16  # f
]


def calc_hamming_weight(n):
    return bin(n).count("1")


def get_HW():
    HW = []
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW.append(hw_val)
    return HW


def aes_internal(inp_data_byte, key_byte):
    inp_data_byte = int(inp_data_byte)
    return sbox[inp_data_byte ^ key_byte]


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def get_labels(plain_text, key, target_byte, leakage_model):
    if 'HW' == leakage_model:
        HW = get_HW()

    key_byte = key[target_byte]

    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = aes_internal(text_i[target_byte], key_byte)
        if 'HW' == leakage_model:
            label = HW[label]
        elif 'MSB' == leakage_model:
            tmp = (label & 0x80)
            if 0 == tmp:
                label = tmp
            elif 128 == tmp:
                label = 1
            else:
                raise ValueError()
        else:
            # using standard model
            pass
        labels.append(label)

    if 'HW' == leakage_model:
        try:
            assert(set(labels) == set(list(range(9))))
        except Exception:
            print('[LOG] -- HW leakage model, but not all class have data: ', set(labels))
    elif 'MSB' == leakage_model:
        try:
            assert(set(labels) == set([0,1]))
        except Exception:
            print('[LOG] -- MSB leakage model, but not all class have data: ', set(labels))
    else:
        try:
            assert(set(labels) == set(range(256)))
        except Exception:
            print('[LOG] -- standard leakage model, but not all class have data: ', set(labels))
    labels = np.array(labels)
    return labels


def dissemble_data_dict(data_dict):
    """Extract a raw trace matrix and its optional metadata from a split."""
    if "X_train" in data_dict.keys():
        trace_key, label_key = "X_train", "y_train"
    else:
        trace_key, label_key = "X_test", "y_test"

    traces, labels, textin, key = data_dict.get(trace_key), data_dict.get(label_key), data_dict.get("plaintext"), data_dict.get("key")

    return traces, labels, textin, key


def load_dataset_base(data_path):
    """Load one ASCAD split from an HDF5 or NPZ file."""
    supporting_func.check_file_exists(data_path)
    print('[LOG] -- load the training data from path {} now...'.format(data_path))
    data_path = str(data_path)
    assert data_path.endswith(".npz")
    
    # Load a split by its stored keys, not by a fragile filename convention.
    archive = np.load(data_path, allow_pickle=True)
    datadict = {key: archive[key] for key in archive.files}
    return datadict


def random_shift_the_data(trace_mat, textin_mat, attack_window, random_shift, trace_num=0):
    """random shift the data, one of the mitigation of sca attack."""    
    if trace_num:
        trace_mat = trace_mat[:trace_num, :]
        textin_mat = textin_mat[:trace_num, :]

    start_idx, end_idx = attack_window[0], attack_window[1]
    if random_shift >= 0:
        print(f'[LOG] -- data will be shifted in range: [{0}, {random_shift}]')
        shifted_traces, new_textins = [], []
        for i in range(textin_mat.shape[0]):
            textin_i = textin_mat[i, :]

            random_int = random.randint(0, random_shift)
            trace_i = trace_mat[i, start_idx+random_int:end_idx+random_int]

            shifted_traces.append(trace_i)
            new_textins.append(textin_i)

        trace_mat, textin_mat = np.array(shifted_traces), np.array(new_textins)

    else:
        print(f'[LOG] -- data is not shifted, original data is used!')
        trace_mat = trace_mat[:, start_idx:end_idx]

    return trace_mat, textin_mat


def train_val_split(data_dict, split_ratio, seed=47):
    """return (train_dict, val_dict), shuffle before splitting."""
    trace_key, label_key, textin_key = "X_train", "y_train", "plaintext"
    X, y, textin = data_dict.get(trace_key), data_dict.get(label_key), data_dict.get(textin_key)
    n_samples = X.shape[0]
    n_val = int(n_samples * split_ratio)

    idx = np.random.default_rng(seed).permutation(n_samples)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def take(ii):
        out = dict(data_dict)
        out[trace_key] = X[ii]
        out[label_key] = y[ii]
        out[textin_key] = textin[ii]
        return out

    return take(train_idx), take(val_idx)


class PowertraceDataset(Dataset):
    """Power trace dataset."""

    def __init__(self, data_dict, target_byte, attack_window, leakage_model, random_shift=0, trace_num=0):
        self.target_byte = target_byte
        self.random_shift = random_shift
        self.leakage_model = leakage_model
        self.trace_num = trace_num
        self.HW = get_HW()

        tmp = attack_window.split('_')
        self.attack_window = [int(tmp[0]), int(tmp[1])]
        print('[LOG] -- using self-defined attack window, attack window is:', self.attack_window)

        traces, ori_labels, textin, key = dissemble_data_dict(data_dict)
        
        self.key = key
        self.key_byte = key[target_byte]
        self.which_one = "train" if "X_train" in data_dict.keys() else "test"

        # random shift and apply the attack window and limit the trace num to the data
        trace_mat, self.textin_mat = random_shift_the_data(traces, textin, self.attack_window, random_shift, trace_num)

        # Precompute labels once (vectorized if aes_internal supports it)
        intermediate = get_labels(self.textin_mat, self.key, target_byte, leakage_model)
        self.labels = np.asarray(intermediate, dtype=np.int64)
        self.class_num = len(list(set(intermediate)))
        self.data_num = trace_mat.shape[0]

        # (N, 1, L) — compute inp_shape from the *final* length
        self.trace_mat = trace_mat[:, np.newaxis, :].astype(np.float32)
        self.inp_shape = self.trace_mat.shape[1:]   # (1, L)

    def __len__(self):
        return self.trace_mat.shape[0]

    def __getitem__(self, idx):
        trace = torch.from_numpy(self.trace_mat[idx])
        label = int(self.labels[idx])
        textin = torch.from_numpy(self.textin_mat[idx])
        return trace, label, textin

    def get_inp_shape(self):
        return self.inp_shape

    def get_data_key(self):
        return self.key

    def get_attack_window(self):
        return self.attack_window

    def get_class_num(self):
        return self.class_num

    def get_data_num(self):
        return self.data_num

    def get_which_one(self):
        return self.which_one


def build_loader(data_dict, target_byte, attack_window, leakage_model, random_shift=0, trace_num=0):
    """function to build data loader from the data dict"""
    batch_size = 100

    # data_dict, target_byte, attack_window, leakage_model, random_shift=0, trace_num=0
    dataset = PowertraceDataset(data_dict, target_byte, attack_window, leakage_model, random_shift, trace_num)
    key = dataset.get_data_key()
    class_num = dataset.get_class_num()
    if_train = dataset.get_which_one()

    # # only drop the partial batch while training
    if 'train' == if_train:
        print("creating data loader for training...")
        shuffle = True
        drop_last = True
        num_workers = 3
        pin_memory = True
    else:
        print("creating data loader for testing...")
        shuffle = False   
        drop_last = False         
        num_workers = 5  
        pin_memory = True

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    inp_shape = dataset.get_inp_shape()
    data_num = dataset.get_data_num()

    out_params = {"inp_shape":inp_shape, "key":key, "data_num":data_num, "class_num":class_num}
    return data_loader, out_params


def parse_opts(argv):
    parser = argparse.ArgumentParser(description="Load one ASCAD dataset split.")
    parser.add_argument("-i", "--input_path", required=True, help="")
    parser.add_argument("-t", "--trace_window", default="0_700", help="")
    parser.add_argument("-m", "--method", default="norm", choices=("norm", "scaling"), help="")
    parser.add_argument("-tn", "--trace_num", type=int, default=0, help="")
    parser.add_argument('-lm', '--leakage_model', default="HW", help="")
    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    opts = parse_opts(sys.argv[1:])
    attack_window = opts.trace_window
    trace_num = opts.trace_num
    random_shift = 0
    target_byte = 2
    leakage_model = "ID"
    data_dict = load_dataset_base(opts.input_path)
    data_loader, out_params = build_loader(data_dict, target_byte, attack_window, leakage_model, random_shift, trace_num)
    device = supporting_func.get_device()

    print("\n##############################################################################\n")
    print("data input shape: ", out_params['inp_shape'])
    print("data number: ", out_params['data_num'])
    print("Total class numbers:", out_params['class_num'])
    key = out_params['key']
    print("Key shape:", None if key is None else key.shape)

    print("\n##############################################################################\n")
    for x_data, y_target, textin in data_loader:        
        print("X shape:", x_data.shape)
        print("y shape:", None if y_target is None else y_target.shape)
        print("Plaintext shape:", None if textin is None else textin.shape)

        n_train = 10  # Example limit for training traces
        x_limited, y_limited = apply_n_limit(x_data, y_target, n_train)
        print("Limited X shape:", x_limited.shape)
        print("Limited y shape:", y_limited.shape)                                         
        break                                              
