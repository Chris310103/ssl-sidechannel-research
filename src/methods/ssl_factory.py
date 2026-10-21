"""
Fixed baseline definitions and dispatch for the supported SSL methods.

The project compares published SSL methods.  Their architecture and
method-specific algorithm settings are baselines, not CLI options.  Keeping
both build and training dispatch tied to this table prevents checkpoint-shape
drift.
"""
import os
import inspect
from copy import deepcopy
import torch
import pdb

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.utils.supporting_functions import load_json_file
from src.utils.data_loader import get_trace_window
from src.utils.trace_transforms import ensure_trace_matrix, prepare_model_input
from src.methods.byol import BYOL1D, train_byol
from src.methods.lhf_bootstrap import LHFBootstrap1D, train_lhf_bootstrap
from src.methods.mae import MAETransformer1D, train_mae
from src.methods.simclr import SimCLRModel, train_simclr
from src.methods.time_mae import TimeMAE1D, train_time_mae
from src.methods.ts2vec import TS2VecSharedModel, train_ts2vec
from src.methods.ts_tcc import TSTCC1D, train_ts_tcc

DEFAULT_COMMON_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ssl_shared_params.json")

SSL_MODEL_CLASSES = {
    "simclr": SimCLRModel,
    "ts2vec": TS2VecSharedModel,
    "ts_tcc": TSTCC1D,
    "byol": BYOL1D,
    "lhf_bootstrap": LHFBootstrap1D,
    "mae": MAETransformer1D,
    "time_mae": TimeMAE1D,
}

# Fixed ASCAD 1-D baselines.  Change this table deliberately when adopting a
# new literature baseline; do not expose its entries through argparse options.
SSL_METHOD_PRESETS = {
    "simclr": {
        "backbone_name": "best_cnn_norm",
        "proj_dim": 128,
        "projector_hidden_dim": 320,
        "view_augmentation": "random",
        "augmentation_family": ("random_shift", "denoise", "gaussian_noise"),
        "augmentation_probability": 0.5,
    },
    "ts2vec": {
        "backbone_name": "shared_cnn_v1",
        "alpha": 0.5,
        "temporal_unit": 0,
        "minimum_crop_ratio": 0.0,
        "timestamp_keep_probability": 0.5,
    },
    "ts_tcc": {
        "backbone_name": "shared_cnn_v1",
        "context_dim": 256,
        "prediction_steps": 6,
        "context_depth": 2,
        "context_heads": 4,
        "temporal_temperature": 0.2,
        "contextual_temperature": 0.2,
        "contextual_weight": 0.7,
        "augmentation_family": ("random_shift", "denoise", "gaussian_noise"),
        "weak_augmentation_probability": 0.3,
        "strong_augmentation_probability": 0.8,
    },
    "byol": {
        "backbone_name": "shared_cnn_v1", 
        "proj_dim": 128,
        "hidden_dim": 512,
        "ema_decay": 0.996,
        "view_augmentation": "random",
        "augmentation_family": ("random_shift", "gaussian_noise"),
        "augmentation_probability": 0.5,
    },
    "lhf_bootstrap": {
        "backbone_name": "shared_cnn_v1",
        "proj_dim": 128,
        "hidden_dim": 512,
        "tcn_hidden_dim": 160,
        "ema_decay": 0.996,
        "view_augmentation": "random",
        "augmentation_family": ("random_shift", "denoise", "gaussian_noise"),
        "augmentation_probability": 0.5,
    },
    "mae": {
        "backbone_name": "trace_transformer_v1",
        "patch_size": 5,
        "mask_ratio": 0.30,
        "transformer_dim": 320,
        "transformer_depth": 4,
        "transformer_heads": 8,
        "transformer_mlp_ratio": 4.0,
        "transformer_dropout": 0.1,
        "decoder_dim": 128,
        "decoder_depth": 2,
        "decoder_heads": 4,
    },
    "time_mae": {
        "backbone_name": "trace_transformer_v1",
        "patch_size": 10,
        "mask_ratio": 0.60,
        "transformer_dim": 192,
        "transformer_depth": 4,
        "transformer_heads": 6,
        "transformer_mlp_ratio": 4.0,
        "transformer_dropout": 0.1,
        "masked_encoder_depth": 2,
        "target_ema_decay": 0.996,
    },
}


def get_ssl_method_preset(method: str) -> dict:
    """Return a copy so callers cannot mutate a fixed research baseline."""
    try:
        return deepcopy(SSL_METHOD_PRESETS[method])
    except KeyError as error:
        raise ValueError(f"Unsupported SSL method: {method}") from error


def build_ssl_model(method, input_length, params=None):
    """Build the fixed baseline model used for both training and reloads.

    params defaults to the method preset merged with the shared JSON, the same
    merge train_ssl_model uses. Only keys the model constructor accepts are passed.
    """
    print("[LOG] -- start to build model: {}".format(method))
    if method not in SSL_MODEL_CLASSES:
        raise ValueError(f"Unsupported SSL method: {method}")

    if params is None:
        params = {**get_ssl_method_preset(method), **load_json_file(DEFAULT_COMMON_CONFIG_PATH)}

    model_class = SSL_MODEL_CLASSES[method]
    accepted = inspect.signature(model_class.__init__).parameters
    model_options = {key: value for key, value in params.items() if key in accepted and key != "input_length"}
    return model_class(input_length=input_length, **model_options)


def train_ssl_model(common_config_path, method, X_train, device, trace_window):
    """Train one fixed SSL baseline over an unwindowed trace matrix."""
    # load common params
    preset_training = load_json_file(common_config_path)
    preset_model = get_ssl_method_preset(method)
    params = {**preset_model, **preset_training}

    window_start, window_end = trace_window
    input_length = window_end - window_start
    if window_start < 0 or input_length <= 0:
        raise ValueError("trace_window must have 0 <= start < end")

    if method == "simclr":
        # train_simclr(params, X_train, device, trace_window, backbone_name="shared_cnn_v1"):
        return train_simclr(params, X_train, device, trace_window)
    elif method == "ts2vec":
        return train_ts2vec(params, X_train, device, trace_window)
    elif method == "ts_tcc":
        return train_ts_tcc(params, X_train, device, trace_window)
    elif method == "byol":
        return train_byol(params, X_train, device, trace_window)
    elif method == "lhf_bootstrap":
        return train_lhf_bootstrap(params, X_train, device, trace_window)
    elif method == "mae":
        return train_mae(params, X_train, device, trace_window)
    elif method == "time_mae":
        return train_time_mae(params, X_train, device, trace_window)
    else:
        raise ValueError(f"Unsupported SSL method: {method}")


def load_ssl_encoder(checkpoint_path, device=None):
    """Rebuild a trained encoder from a checkpoint written by train.py."""
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"]

    method = payload["method"]
    input_length = payload["input_length"]
    trace_window = payload["trace_window"]      # "0_700"

    model = build_ssl_model(method, input_length, params=payload.get("params"))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for p in model.parameters():        # freeze the params in the model
        p.requires_grad_(False)

    return model, trace_window, payload


def extract_features(model_or_path, X, device=None, batch_size=256):
    """Load a trained encoder from a checkpoint and encode X into feature vectors."""
    if isinstance(model_or_path, str):
        model, trace_window_str, _ = load_ssl_encoder(model_or_path, device)
    else:
        model = model_or_path
    trace_window, _ = get_trace_window(trace_window_str)
    loader = DataLoader(TensorDataset(torch.from_numpy(ensure_trace_matrix(X)).float()),
                        batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = prepare_model_input(batch_x.to(device), trace_window=trace_window)
            feature_batch = model.encode(batch_x).cpu().numpy()
            features.append(feature_batch)
    features = np.concatenate(features, axis=0)
    return features
