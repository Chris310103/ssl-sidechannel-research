import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import get_trace_window, load_dataset
from src.utils.experiment_config import DEFAULT_METHOD_OPTIONS, SSL_METHODS, get_method_metadata
from src.utils.supporting_functions import get_device
from src.utils.trace_transforms import ensure_trace_matrix, prepare_model_input
from src.utils.supporting_functions import set_seed, load_ssl_model, load_params

from src.methods.knn_distinguisher import compute_knn_candidate_accuracies


PROJECT_ROOT = Path(__file__).resolve().parent
PARAM_config_path = os.path.join(PROJECT_ROOT, 'src', 'config', 'config.json')


def get_x_feat(model, X, device, batch_size):
    X = ensure_trace_matrix(X)
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    representations = []
    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_x = prepare_model_input(batch_x, trace_window=trace_window)
            representations.append(model.encode(batch_x).cpu().numpy())

    X_feat = np.concatenate(representations, axis=0)
    return X_feat


def make_run_name(opts) -> str:
    """ generate the run name for the testing """
    (window_start, window_end), trace_length = get_trace_window(opts.trace_window) 
    n_knn_train, n_knn_eval = opts.nv
    return (f"{opts.method}_knn_w{window_start}-{window_end}_nv{n_knn_train}-{n_knn_eval}_byte{opts.target_byte}")


def save_attack_artifacts(opts, checkpoint_metadata, candidate_scores, ranked_keys, true_key, true_key_rank):
    """ function to save testing results """
    run_name = make_run_name(opts=opts, true_key=true_key, true_key_rank=true_key_rank)
    output_path = os.path.join(opts.output_dir, run_name)
    os.mkdir(output_path, exist_ok=True)

    scores_path = f"candidate_accuracies.npy"
    ranked_keys_path = f"ranked_keys.npy"
    summary_path = f"summary.json"

    np.save(scores_path, candidate_scores)
    np.save(ranked_keys_path, ranked_keys)

    summary = {
        "method": opts.method,
        "method_metadata": get_method_metadata(opts.method),
        "checkpoint": opts.checkpoint,
        "checkpoint_metadata": {key: value for key, value in checkpoint_metadata.items() if key != "model_state_dict"},
        "trace_window": tuple(opts.trace_window),
        "nv": tuple(opts.nv),
        "target_byte": opts.target_byte,
        "leakage_model": opts.leakage_model,
        "knn_neighbors": opts.knn_neighbors,
        "knn_weights": opts.knn_weights,
        "true_key": true_key,
        "best_key": int(ranked_keys[0]),
        "best_accuracy": float(candidate_scores[ranked_keys[0]]),
        "true_key_accuracy": float(candidate_scores[true_key]),
        "true_key_candidate_rank": true_key_rank,
        "candidate_scores_path": str(scores_path),
        "ranked_keys_path": str(ranked_keys_path),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"Saved summary: {summary_path}")         #type: ignore
    print(f"Best key: {summary['best_key']} accuracy={summary['best_accuracy']:.6f}")
    print(f"True key: {true_key} rank={true_key_rank} " f"accuracy={summary['true_key_accuracy']:.6f}")




def main(opts) -> None:
    """ entry point for testing only purpose """
    params = load_params(PARAM_config_path)
    input_path = opts.input
    trace_window, input_length = get_trace_window(opts.trace_window)
    model_name = opts.method

    # so the data in the test dataset will be all used for testing, data in training dataset 
    # will be used for KNN training as well
    X_test, y_test, plaintext, key = load_dataset(input_path, trace_window, which_one="test")

    device = get_device()
    batch_size = params['batch_size']
    feat_model = load_ssl_model(opts.model_path, model_name)
    x_data_feat, plaintext, key = get_x_feat(feat_model, X_test, device, batch_size)

    compute_knn_candidate_accuracies(opts, x_data_feat, plaintext, key)

    print("[LOG] -- all test done!")


def build_parser(argv):
    parser = argparse.ArgumentParser(description="Evaluate one SSL encoder with 256 non-profiling KNN distinguishers.")

    parser.add_argument("-i", "--input", help="")
    parser.add_argument("-o", "--output_dir", help="")
    parser.add_argument("-m", "--method", choices=SSL_METHODS, help="")
    parser.add_argument("-c", "--checkpoint", required=True, help="path to the place that model is saved")
    parser.add_argument("-tw", "--trace_window", type=str, help="")
    parser.add_argument("-tb", "--target_byte", type=int, default=3)
    parser.add_argument("-l", "--leakage_model", choices=("ID", "HW"), default="HW")

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = build_parser(sys.argv)
    main(opts)
