import os
import sys
import argparse
import pdb
import torch
import random
import time
import json

import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt


def get_attack_window(attack_window_str):
    tmp = attack_window_str.split("_")
    start, end = int(tmp[0]), int(tmp[1])
    attack_window = (start, end)
    input_length = end - start
    return attack_window, input_length


def get_device(prefer_mps: bool = False):
    if torch.cuda.is_available():
        return "cuda"

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def set_seed(seed=47):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_run_time(func):
    """ time decrator """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        time_elapes = end_time - start_time

        m, s = divmod(time_elapes, 60)
        h, m = divmod(m, 60)
        formatted_time = "{:d}:{:02d}:{:02d}".format(int(h), int(m), int(s))
 
        print('[LOG] -- function {} run time is: {}'.format(func.__name__, formatted_time))   # type: ignore
        return result
    return wrapper


def plot_distinguisher_figure(x, y, real_key_index, real_key_value, title, fig_save_path):
    """Plot the distinguisher figure with the given data and save it to a file.
    The y axis is the accuracy, and x axis is the number of traces used for testing.
    The hope is that the accuracy of the correct key guess will be higher than the 
    other key guesses, and the accuracy of the correct key guess will increase as the
    number of traces increases.
    example input:
        x = [25, 50, 100, 200, 400, 800]
        y = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
             [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
             [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    """
    plt.title(title)
    plt.xlabel("No. of traces used for testing")
    plt.ylabel("Accuracy")
    plt.xticks([25, 50, 100, 200, 400, 800], fontsize=12)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12)
    n_lines = len(y)
    count = 0
    for i in range(n_lines):
        if i == real_key_index:
            plt.plot(x, y[i], label=f"Correct key {real_key_value}", alpha=0.5, linewidth=2.5, color='red', marker='o')
        else:
            if 0 == count:
                plt.plot(x, y[i], label=f"Incorrect keys", color='gray', alpha=0.5)
                count += 1
            else:
                plt.plot(x, y[i], color='gray', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()
    plt.close()


def plot_key_rank_curve(x, y, dataset_name, model_name, save_file):
    """
    We plot the results based on the ranks of the key guesses.
        The x axis is the number of traces used for testing
        The y axis is the rank of the correct key guess. 
    The hope is that the rank of the correct key guess 
    will decrease as the number of traces increases.
    """

    plt.title(f'Performance of {model_name} against {dataset_name}')
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    if (save_file != ""):
        plt.savefig(save_file)
        print("Saved figure to {}".format(save_file))
    else:
        plt.show(block=False)
    plt.close()


def plot_loss_curve(history, save_dir):
    """save the loss and acc figure."""
    train_loss, train_acc, val_loss, val_acc = history['loss'], history['accuracy'], history['val_loss'], history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # --- left y-axis: Loss ---
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_loss, color="tab:red", linestyle="-", marker="o", label="Train Loss")
    ax1.plot(epochs, val_loss, color="tab:red", linestyle="--", marker="*", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True)

    # --- right y-axis: Accuracy ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, train_acc, color="tab:blue", linestyle="-", marker="+", label="Train Accuracy")
    ax2.plot(epochs, val_acc, color="tab:blue", linestyle="--", marker="^", label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # --- combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training and Val Loss and Acc")
    fig.tight_layout()
    save_path = os.path.join(save_dir, "train_val_loss_and_acc_curve.png")
    fig.savefig(save_path, dpi=150)
    print(f"[LOG] -- loss and acc curve are save to path: {save_path}")
    plt.show()
    plt.close()


def load_or_save_ml_model(model_path, mode, model=None):
    if 'load' == mode:
        model = load(model_path)
        print("[LOG] -- model load from path: {}".format(model_path))
        return model
    elif 'save' == mode:
        dump(model, model_path)
        print("[LOG] -- model saved to path: {}".format(model_path))
    else:
        raise ValueError(" The mode has either to be load or save! ")


def load_json_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


if __name__ == "__main__":
    # Assume these come from your training loop
    train_losses = [0.9, 0.7, 0.5, 0.35, 0.28, 0.22, 0.18, 0.15, 0.13, 0.11]
    val_losses   = [0.95, 0.75, 0.55, 0.42, 0.36, 0.32, 0.30, 0.29, 0.29, 0.30]
    epochs = range(1, len(train_losses) + 1)
    plot_key_rank_curve()
