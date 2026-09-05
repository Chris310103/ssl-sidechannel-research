import os
import sys
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.datasets.data_loader import load_dataset
from src.utils.cli_parsers import parse_range
from src.utils.experiment_config import (
    SSL_METHODS,
    build_method_options,
    get_method_metadata,
)
from src.utils.get_device import get_device
from src.utils.trace_transforms import parse_augmentation_family
from src.methods.ssl_factory import train_ssl_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ASCAD_PATH = os.path.join(PROJECT_ROOT, "data", "npz_data", "ASCAD", "ATM_AES_v1_fixed_key", "ASCAD.npz")


def set_seed():
    seed = 47
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_trace_window(opts):
    tmp = opts.trace_window.split("_")
    window_start, window_end = int(tmp[0]), int(tmp[1])
    return window_start, window_end


def make_run_name(opts, input_length: int) -> str:
    window_start, window_end = get_trace_window(opts)
    return f"{opts.method}_w{window_start}-{window_end}_n{opts.n_train}_ep{opts.epochs}"


def save_training_artifacts(opts, model, train_result, input_length: int) -> Path:
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = make_run_name(opts, input_length=input_length)
    checkpoint_path = os.path.join(output_dir, f"{run_name}.pt")

    torch.save(
        {
            "method": opts.method,
            "method_metadata": get_method_metadata(opts.method),
            "model_state_dict": model.state_dict(),
            "input_length": input_length,
            "trace_window": opts.trace_window,
            "args": vars(opts),
        },
        checkpoint_path,
    )

    if len(train_result) > 1 and train_result[1] is not None:
        train_res_save_path = os.path.join(output_dir, f"{run_name}_loss.npy")
        np.save(train_res_save_path, np.asarray(train_result[1]))

    json_save_path = os.path.join(output_dir, f"{run_name}_config.json")
    with (json_save_path).open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(opts), "method_metadata": get_method_metadata(opts.method)}, handle, indent=2, default=str)

    return checkpoint_path


def main(opts):
    set_seed()

    window_start, window_end = get_trace_window(opts)
    input_length = window_end - window_start

    X_train, y_train, plaintext = load_dataset(opts.input)

    augmentation_family = parse_augmentation_family(opts.augmentations)
    device = get_device()

    train_result = train_ssl_model(
        opts=opts,
        X_train=X_train,
        y_train=y_train,
        device=device,
        trace_window=opts.trace_window,
        input_length=input_length,
        augmentation_family=augmentation_family,
    )

    model = train_result[0]
    checkpoint_path = save_training_artifacts(
        opts=opts,
        model=model,
        train_result=train_result,
        input_length=input_length,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


def parser_opts(argv):
    parser = argparse.ArgumentParser(description="Train one SSL encoder for side-channel traces.")

    parser.add_argument("-m", "--method", choices=SSL_METHODS, required=True)
    parser.add_argument("-i", "--input", default=str(DEFAULT_ASCAD_PATH))
    parser.add_argument("-o", "--output_dir", default=os.path.join(PROJECT_ROOT, "outputs", "checkpoints"))
    parser.add_argument("-tw", "--trace_window", type=str, default="0_700")
    parser.add_argument("-nt", "--n_train", type=int, default=500)
    parser.add_argument("-e", "--epochs", type=int, default=None)

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parser_opts(sys.argv[1:])
    main(opts)
