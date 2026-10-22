import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.utils.data_loader import get_trace_window, load_dataset
from src.methods.ssl_factory import get_ssl_method_preset, train_ssl_model
from src.utils.experiment_config import SSL_METHODS, get_method_metadata
from src.utils.supporting_functions import get_device, set_seed, plot_loss_figure

PROJECT_ROOT = Path(__file__).resolve().parent


def make_run_name(trace_window, method, train_num, epochs):
    (window_start, window_end), _ = get_trace_window(trace_window)
    return f"{method}_w{window_start}_{window_end}_n{train_num}_ep{epochs}"


def save_training_artifacts(opts, run_name, model):
    output_dir = opts.output
    os.mkdir(output_dir, exist_ok=True)
    trace_window, input_length = get_trace_window(opts.trace_window)

    checkpoint_path = os.path.join(output_dir, f"{run_name}.pt")
    checkpoint_payload = {
        "method": opts.method,
        "method_metadata": get_method_metadata(opts.method),
        "ssl_method_preset": get_ssl_method_preset(opts.method),
        "model_state_dict": model.state_dict(),
        "input_length": input_length,
        "trace_window": opts.trace_window,
        "args": vars(opts)
    }

    torch.save(checkpoint_payload, checkpoint_path)

    # plot the loss figure for check
    loss_fig_path = os.path.join(output_dir)
    plot_loss_figure(loss_log, loss_fig_path)

    return str(checkpoint_path)


def main(opts):
    """main entry point for training a single SSL baseline on a given dataset."""
    set_seed()
    trace_window, input_length = get_trace_window(opts.trace_window)

    data_path = opts.input
    output_dir = opts.output
    ssl_method = opts.method
    train_num = opts.n_train

    # Keep raw traces here.  The SSL trainers apply trace_window exactly once
    # while creating model input and augmented views.
    X_train, y_train, plaintext, key = load_dataset(data_path, which_one="train", attack_window=trace_window, method="norm", trace_num=train_num)

    print(f"Training SSL model {opts.method} on {len(X_train)} traces with trace window {trace_window}...")
    ssl_model, loss_log, run_name = train_ssl_model(method=ssl_method, X_train=X_train, device=get_device(), trace_window=trace_window)

    print("Training complete. Saving artifacts...")
    checkpoint_path = save_training_artifacts(opts=opts, model=ssl_model, run_name=run_name)

    print(f"Saved model checkpoint in path: {checkpoint_path}")


def parser_opts(argv):
    parser = argparse.ArgumentParser(description="Train one fixed SSL baseline for side-channel traces.")
    parser.add_argument("-i", "--input", help="Path to the input dataset (npz file).")
    parser.add_argument("-o", "--output", default=os.path.join(PROJECT_ROOT, "outputs", "checkpoints"))
    parser.add_argument("-m", "--method", choices=SSL_METHODS, required=True)
    parser.add_argument("-tw", "--trace_window", type=str, default="0_700", help="")
    parser.add_argument("-nt", "--n_train", type=int, default=0, help="Number of unlabeled profiling traces to use.")

    opts = parser.parse_args(argv)
    return opts


if __name__ == "__main__":
    opts = parser_opts(sys.argv[1:])
    main(opts)
