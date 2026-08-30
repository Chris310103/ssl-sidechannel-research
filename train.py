import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.datasets.ascad_loader import load_ascad_split
from src.utils.cli_parsers import parse_range
from src.utils.experiment_config import (
    SSL_METHODS,
    build_method_options,
    get_method_metadata,
)
from src.utils.get_device import get_device
from src.utils.trace_transforms import parse_augmentation_family


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ASCAD_PATH = PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one SSL encoder for side-channel traces."
    )

    parser.add_argument("--method", choices=SSL_METHODS, required=True)
    parser.add_argument("--input", default=str(DEFAULT_ASCAD_PATH))
    parser.add_argument("--split", choices=("attack", "profiling"), default="attack")
    parser.add_argument("--trace-window", type=parse_range, default=(0, 700))
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--normalize", choices=("divide128", "zscore"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "checkpoints"),
    )

    return parser


def make_run_name(opts, input_length: int) -> str:
    window_start, window_end = opts.trace_window
    return (
        f"{opts.method}_{opts.backbone_name}_{opts.pool_mode}"
        f"_w{window_start}-{window_end}_l{input_length}"
        f"_n{opts.n_train}_ep{opts.epochs}_seed{opts.seed}"
    )


def save_training_artifacts(opts, model, train_result, input_length: int) -> Path:
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = make_run_name(opts, input_length=input_length)
    checkpoint_path = output_dir / f"{run_name}.pt"

    torch.save(
        {
            "method": opts.method,
            "method_metadata": get_method_metadata(opts.method),
            "model_state_dict": model.state_dict(),
            "input_length": input_length,
            "trace_window": tuple(opts.trace_window),
            "args": vars(opts),
        },
        checkpoint_path,
    )

    if len(train_result) > 1 and train_result[1] is not None:
        np.save(output_dir / f"{run_name}_loss.npy", np.asarray(train_result[1]))

    with (output_dir / f"{run_name}_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(opts),
                "method_metadata": get_method_metadata(opts.method),
            },
            handle,
            indent=2,
            default=str,
        )

    return checkpoint_path


def main(opts) -> None:
    from src.methods.ssl_factory import train_ssl_model

    set_seed(opts.seed)

    window_start, window_end = opts.trace_window
    input_length = window_end - window_start

    X_train, _ = load_ascad_split(
        h5_path=opts.input,
        split=opts.split,
        add_channel=False,
        normalize=opts.normalize,
        load_metadata=False,
        trace_window=None,
    )

    X_train = X_train[: opts.n_train]
    augmentation_family = parse_augmentation_family(opts.augmentations)
    device = get_device(prefer_mps=False)

    train_result = train_ssl_model(
        opts=opts,
        X_train=X_train,
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


if __name__ == "__main__":
    cli_opts = build_parser().parse_args()
    main(
        build_method_options(
            method=cli_opts.method,
            overrides=vars(cli_opts),
        )
    )
