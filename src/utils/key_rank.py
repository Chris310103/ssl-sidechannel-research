import os
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


Sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71,
        240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216,
        49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160,
        82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208,
        239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
        182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96,
        129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211,
        172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
        120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97,
        53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140,
        161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]

AES_SBOX = np.asarray(Sbox, dtype=np.uint8)


HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

HW_BYTE = np.asarray(HW_byte, dtype=np.uint8)


def calc_hamming_weight(n):
    return bin(n).count("1")


def get_HW():
    HW = []
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW.append(hw_val)
    return HW


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def _metadata_field(metadata, field_name):
    if metadata is None:
        raise ValueError("metadata is required for key-rank calculation")

    if getattr(metadata, "dtype", None) is not None and metadata.dtype.names:
        if field_name not in metadata.dtype.names:
            raise ValueError(f"metadata does not contain field: {field_name}")
        return metadata[field_name]

    if isinstance(metadata, dict):
        if field_name not in metadata:
            raise ValueError(f"metadata does not contain field: {field_name}")
        return metadata[field_name]

    raise ValueError(
        "metadata must be a structured numpy array or a dict with plaintext/key"
    )


def metadata_plaintext(metadata, target_byte):
    plaintext = np.asarray(_metadata_field(metadata, "plaintext"))
    return plaintext[:, target_byte].astype(np.uint8)


def metadata_true_key(metadata, target_byte):
    key = np.asarray(_metadata_field(metadata, "key"))
    return int(key[0, target_byte])


def leakage_labels(plaintext_byte, key_guess, leakage_model="ID"):
    plaintext_byte = np.asarray(plaintext_byte, dtype=np.uint8)
    sbox_output = AES_SBOX[np.bitwise_xor(plaintext_byte, np.uint8(key_guess))]

    if leakage_model == "ID":
        return sbox_output.astype(np.int64)

    if leakage_model == "HW":
        return HW_BYTE[sbox_output].astype(np.int64)

    raise ValueError(f"Unsupported leakage_model: {leakage_model}")


def key_rank_from_log_scores(log_scores, true_key, max_traces=None):
    log_scores = np.asarray(log_scores, dtype=np.float64)

    if log_scores.ndim != 2 or log_scores.shape[1] != 256:
        raise ValueError(f"log_scores must have shape (N, 256), got {log_scores.shape}")

    num_traces = log_scores.shape[0]
    if max_traces is not None:
        num_traces = min(num_traces, max_traces)

    cumulative_scores = np.zeros(256, dtype=np.float64)
    ranks = np.zeros(num_traces, dtype=np.int64)

    for trace_index in tqdm(range(num_traces), desc="Computing key rank"):
        cumulative_scores += log_scores[trace_index]
        ranks[trace_index] = int(np.sum(cumulative_scores > cumulative_scores[true_key]))

    return ranks


def compute_rank_curve(probas, metadata, target_byte=2, max_traces=None, leakage_model="ID", eps=1e-40):
    probas = np.asarray(probas, dtype=np.float64)

    if probas.ndim != 2:
        raise ValueError(f"probas must be a matrix, got {probas.shape}")

    expected_classes = 256 if leakage_model == "ID" else 9
    if probas.shape[1] != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} probability columns for {leakage_model}, "
            f"got {probas.shape[1]}"
        )

    num_traces = probas.shape[0]
    if max_traces is not None:
        num_traces = min(num_traces, max_traces)

    if isinstance(metadata, dict):
        metadata_subset = {
            key: np.asarray(value)[:num_traces]
            for key, value in metadata.items()
        }
    else:
        metadata_subset = metadata[:num_traces]

    plaintext = metadata_plaintext(metadata_subset, target_byte)
    true_key = metadata_true_key(metadata, target_byte)
    key_candidates = np.arange(256, dtype=np.uint8)
    log_scores = np.zeros((num_traces, 256), dtype=np.float64)
    hw_mapping = create_hw_label_mapping()

    for trace_index in range(num_traces):
        candidate_labels = leakage_labels(
            plaintext[trace_index],
            key_candidates,
            leakage_model=leakage_model,
        )
        key_probas = probas[trace_index, candidate_labels]

        if leakage_model == "HW":
            key_probas = np.asarray(
                [
                    key_proba / len(hw_mapping[int(label)])
                    for key_proba, label in zip(key_probas, candidate_labels)
                ],
                dtype=np.float64,
            )

        log_scores[trace_index] = np.log(key_probas + eps)

    return key_rank_from_log_scores(
        log_scores=log_scores,
        true_key=true_key,
        max_traces=num_traces,
    )


def plot_rank_curve(ranks, save_path, title="Key Rank Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.plot(np.arange(1, len(ranks) + 1), ranks)
    plt.xlabel("Number of attack traces")
    plt.ylabel("Rank")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def ranking_curve(preds, key, plaintext, target_byte, rank_root, leakage_model='HW', trace_num_max=500):
    """
    - preds : the probability for each class (n*256 for a byte, n*9 for Hamming weight)
    - real_key : the key of the target device
    - device_id : id of the target device
    - model_flag : a string for naming GE result
    """
    hw_mapping = create_hw_label_mapping()

    # GE/SR is averaged over 100 attacks
    num_averaged = 100
    # max trace num for attack
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    real_key = key[target_byte]
    plaintext = plaintext[:, target_byte]

    # attack multiples times for average
    for time in tqdm(range(num_averaged)):
        # select the attack traces randomly
        random_index = list(range(plaintext.shape[0]))

        #         ## customized by HL
        #         print(f"random_index shape {len(random_index)}, max value {max(random_index)}, min value {min(random_index)}")

        random.shuffle(random_index)
        random_index = random_index[0:trace_num_max]

        #         ## customized by HL
        #         print(f"random_index shape after slicing {len(random_index)}, max value {max(random_index)}, min value {min(random_index)}")

        # initialize score matrix
        score_mat = np.zeros((trace_num_max, 256))
        for key_guess in range(0, 256):
            for i in range(0, trace_num_max):
                initialState = int(plaintext[random_index[i]]) ^ key_guess
                sout = Sbox[initialState]
                if leakage_model == 'ID':
                    label = sout
                elif leakage_model == 'HW':
                    label = HW_byte[sout]
                try:
                    prob_value = preds[random_index[i], label]
                    if leakage_model == 'HW':
                        prob_value = prob_value / len(hw_mapping[label])
                    score_mat[i, key_guess] = prob_value
                except Exception as e:
                    print(e)
                    raise
        score_mat = np.log(score_mat + 1e-40)

        #         ## customized by HL
        #         print(f"score_mat {score_mat}")

        for i in range(0, trace_num_max):
            log_likelihood = np.sum(score_mat[0:i+1, :], axis=0)
            ranked = np.argsort(log_likelihood)[::-1]
            guessing_entropy[time, i] = list(ranked).index(real_key)
            if list(ranked).index(real_key) == 0:
                success_flag[time, i] = 1

    guessing_entropy = np.mean(guessing_entropy, axis=0)

    # define the saving path
    os.makedirs(rank_root, exist_ok=True)

    # only plot guess entry
    plt.figure(figsize=(8, 6))
    plt.plot(guessing_entropy[0:trace_num_max], color='red')
    plt.title('Leakage model: {}, target byte: {}'.format(leakage_model, target_byte))
    plt.xlabel('Number of trace')
    plt.ylabel('Key Rank')
    fig_save_path = os.path.join(rank_root, 'ranking_curve.png')
    plt.savefig(fig_save_path)
    plt.show()
    plt.close()
    print('[LOG] -- ranking curve save to path: ', fig_save_path)

    # saving the ranking raw data
    raw_save_path = os.path.join(rank_root, 'ranking_raw_data.npz')
    x = list(range(len(guessing_entropy)))
    np.savez(raw_save_path, x=x, y=guessing_entropy)
    print('[LOG] -- ranking raw data save to path: ', raw_save_path)


def get_the_labels(textins, key, target_byte):
    return leakage_labels(
        np.asarray(textins)[:, target_byte],
        int(key[target_byte]),
        leakage_model="ID",
    )
