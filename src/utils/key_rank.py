#!/usr/bin/python3
import os
import sys
import pdb
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.utils.supporting_functions as supporting_funcs


Sbox = [
    # 0    1    2     3     4     5     6     7     8     9     a     b     c     d     e     f
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
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16   # f
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
    return Sbox[inp_data_byte ^ key_byte]


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def get_labels(plain_text, key_byte, target_byte, leakage_model):
    if 'HW' == leakage_model:
        HW = get_HW()

    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = aes_internal(text_i[target_byte], key_byte)
        if 'ID' == leakage_model:
            pass 
        elif 'HW' == leakage_model:
            label = HW[label]
        elif 'MSB' == leakage_model:
            tmp = (label & 0x80)
            if 0 == tmp:
                label = tmp
            elif 128 == tmp:
                label = 1
            else:
                raise ValueError()
        labels.append(label)

    if 'HW' == leakage_model:
        try:
            assert(set(labels) == set(list(range(9))))
        except Exception:
            print('[LOG] -- not all class have data: ', set(labels))
    elif 'MSB' == leakage_model:
        try:
            assert(set(labels) == set([0,1]))
        except Exception:
            print('[LOG] -- not all class have data: ', set(labels))
    else:  # this is the route for ID leakage model
        try:
            assert(set(labels) == set(range(256)))
        except Exception:
            print('[LOG] -- not all class have data: ', set(labels))
    labels = np.array(labels)
    return labels


def shuffle_and_pick(mat1, mat2, choice_num):
    perm = np.random.permutation(mat1.shape[0])
    new_mat1 = mat1[perm]
    new_mat2 = mat2[perm]

    new_mat1 = new_mat1[:choice_num]
    new_mat2 = new_mat2[:choice_num]
    return new_mat1, new_mat2


class KeyRanking():
    def __init__(self, leakage_model, min_trace_idx, max_trace_idx, target_byte, rank_step, output_dir):
        self.HW = get_HW()
        self.hw_mapping = create_hw_label_mapping()
        
        self.leakage_model = leakage_model
        self.min_trace_idx = min_trace_idx
        self.max_trace_idx = max_trace_idx
        self.target_byte = target_byte
        self.rank_step = rank_step

        self.fig_save_name = os.path.join(output_dir, "ranking_curve.jpg")
        tmp = os.path.basename(output_dir).split("_")
        self.dataset_name = 'ASCAD'
        self.model_name = 'cnn'

    def rank(self, predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    # Compute the rank
        if len(last_key_bytes_proba) == 0:
            # If this is the first rank we compute, initialize all the estimates to zero
            key_bytes_proba = np.zeros(256)
        else:
            # This is not the first rank we compute: we optimize things by using the
            # previous computations to save time!
            key_bytes_proba = last_key_bytes_proba

        for p in range(0, max_trace_idx-min_trace_idx):
            # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
            plaintext = plaintext_list[p][target_byte]
            for i in range(0, 256):
                # Our candidate key byte probability is the sum of the predictions logs
                # AES_Sbox[plaintext ^ i]
                if 'HW' == self.leakage_model:
                    tmp_label = self.HW[aes_internal(plaintext, i)]
                    tmp_hw_list = self.hw_mapping[tmp_label]
                elif 'ID' == self.leakage_model:
                    tmp_label = aes_internal(plaintext, i)
                else:
                    raise ValueError("only support the HW and ID leakage model for now.")

                proba = predictions[p][tmp_label]
                if proba != 0:
                    if 'ID' == self.leakage_model:
                        key_bytes_proba[i] += np.log(proba)
                    elif 'HW' == self.leakage_model:
                        proba_log_share = np.log(proba) / len(tmp_hw_list)
                        for elem in tmp_hw_list:
                            key_bytes_proba[elem] += proba_log_share
                    else:
                        raise ValueError("only support the HW and ID leakage model for now.")
                else:
                    # We do not want an -inf here, put a very small epsilon
                    # that corresponds to a power of our min non zero proba
                    min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                    if len(min_proba_predictions) == 0:
                        raise ValueError("Error: got a prediction with only zeroes ... this should not happen!")
                    
                    min_proba = min(min_proba_predictions)
                    if 'ID' == self.leakage_model:
                        key_bytes_proba[i] += np.log(min_proba**2)
                    elif 'HW' == self.leakage_model:
                        min_proba_log_share = np.log(min_proba**2) / len(tmp_hw_list)
                        for elem in tmp_hw_list:
                            key_bytes_proba[elem] += min_proba_log_share
                    else:
                        raise ValueError("only support the HW and ID leakage model for now.")

        # Now we find where our real key candidate lies in the estimation.
        # We do this by sorting our estimates and find the rank in the sorted array.
        sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
        real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
        return real_key_rank, key_bytes_proba

    def full_ranks(self, proba_mat, key, plaintext_attack):
        # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
        real_key = key[self.target_byte]
        # Predict our probabilities                                      
        predictions = proba_mat[self.min_trace_idx:self.max_trace_idx, :]
        # Check for overflow
        if self.max_trace_idx > proba_mat.shape[0]:
            raise ValueError(f"Error: asked trace index {self.max_trace_idx} overflows the total traces number {proba_mat.shape[0]}.")

        index = np.arange(self.min_trace_idx+self.rank_step, self.max_trace_idx, self.rank_step)
        f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
        key_bytes_proba = []

        for t, i in zip(index, range(0, len(index))):
            real_key_rank, key_bytes_proba = self.rank(predictions[t-self.rank_step:t], plaintext_attack[t-self.rank_step:t], real_key, t-self.rank_step, t, key_bytes_proba, self.target_byte)
            f_ranks[i] = [t - self.min_trace_idx, real_key_rank]
        return f_ranks

    def ranking_curve(self, preds_proba, textin_mat, key):
        """function to compute the rank and plot the curve"""
        # run rank key curve function
        assert(preds_proba.shape[0] == textin_mat.shape[0])
        y_list = []
        for i in tqdm(range(5)):
            proba_mat_x, textin_mat_x = shuffle_and_pick(preds_proba, textin_mat, choice_num=self.max_trace_idx)
            f_ranks = self.full_ranks(proba_mat_x, key, textin_mat_x)

            # We plot the results  f_ranks[i] = [t, real_key_rank]
            x = [f_ranks[i][0] for i in range(0, f_ranks.shape[0])]
            y_tmp = [f_ranks[i][1] for i in range(0, f_ranks.shape[0])]
            y_list.append(y_tmp)

        y_arr = np.array(y_list)
        y = np.mean(y_arr, axis=0)

        supporting_funcs.plot_key_rank_curve(x, y, self.dataset_name, self.model_name, self.fig_save_name)
        print('[LOG] -- figure save to file: {}'.format(self.fig_save_name))
        return x, y

    def save_rank_data(self, x, y):
        # save the raw data
        rank_dir = os.path.dirname(self.fig_save_name)
        self.raw_rank_data = os.path.join(rank_dir, 'rank_byte_{}.npz'.format(self.target_byte))
        np.savez(self.raw_rank_data, x=x, y=y)
        print('[LOG] -- raw data save to path: {}'.format(self.raw_rank_data))


def compute_key_ranks(preds_proba, plaintext, key, target_byte, leakage_model, output_dir):
    """Function to compute the key guess rank, plot the figure and save the raw data."""
    # now key_rank ranking curve is a class
    min_trace_idx = 0
    max_trace_idx = min(10000, len(preds_proba))
    rank_step = 10

    rank_curve_obj = KeyRanking(leakage_model, min_trace_idx, max_trace_idx, target_byte, rank_step, output_dir)
    x, y = rank_curve_obj.ranking_curve(preds_proba, plaintext, key)
    rank_curve_obj.save_rank_data(x, y)

    print('[LOG] -- all done for model testing!') # type: ignore
