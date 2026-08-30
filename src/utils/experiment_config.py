from types import SimpleNamespace


METHOD_REGISTRY = {
    "simclr": {
        "display_name": "SimCLR",
        "family": "contrastive",
        "role": "generic",
        "backbone": "shared_cnn_v1",
    },
    "ts2vec": {
        "display_name": "TS2Vec",
        "family": "contrastive",
        "role": "time_series_specialized",
        "backbone": "shared_cnn_v1",
    },
    "ts_tcc": {
        "display_name": "TS-TCC",
        "family": "contrastive",
        "role": "time_series_specialized",
        "backbone": "shared_cnn_v1",
    },
    "byol": {
        "display_name": "BYOL",
        "family": "bootstrap",
        "role": "generic",
        "backbone": "shared_cnn_v1",
    },
    "lhf_bootstrap": {
        "display_name": "L/H-Frequency Bootstrapping",
        "family": "bootstrap",
        "role": "time_series_specialized",
        "backbone": "shared_cnn_v1",
    },
    "mae": {
        "display_name": "Transformer MAE",
        "family": "masked_modeling",
        "role": "generic",
        "backbone": "trace_transformer_v1",
    },
    "time_mae": {
        "display_name": "TimeMAE-MRR",
        "family": "masked_modeling",
        "role": "time_series_specialized",
        "backbone": "trace_transformer_v1",
    },
}


SSL_METHODS = tuple(METHOD_REGISTRY)


DEFAULT_METHOD_OPTIONS = {
    "backbone_name": "shared_cnn_v1",
    "pool_mode": "mean_max",
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "augmentations": "random_shift,denoise,gaussian_noise",
    "view_augmentation": "random",
    "augmentation_probability": 0.5,
    "weak_augmentation_probability": 0.3,
    "strong_augmentation_probability": 0.8,
    "max_shift": 5,
    "noise_std": 0.05,
    "denoise_kernel_size": 5,
    "projector_hidden_dim": 320,
    "proj_dim": 128,
    "temperature": 0.2,
    "hidden_dim": 512,
    "tcn_hidden_dim": 160,
    "ema_decay": 0.996,
    "context_dim": 256,
    "prediction_steps": 6,
    "context_depth": 2,
    "context_heads": 4,
    "temporal_temperature": 0.2,
    "contextual_temperature": 0.2,
    "contextual_weight": 0.7,
    "patch_size": 10,
    "mask_ratio": 0.75,
    "transformer_dim": 192,
    "transformer_depth": 4,
    "transformer_heads": 6,
    "transformer_mlp_ratio": 4.0,
    "transformer_dropout": 0.1,
    "decoder_dim": 128,
    "decoder_depth": 2,
    "decoder_heads": 4,
    "masked_encoder_depth": 2,
    "target_ema_decay": 0.996,
    "alpha": 0.5,
    "temporal_unit": 0,
    "minimum_crop_ratio": 0.5,
    "timestamp_keep_probability": 0.5,
    "knn_neighbors": 3,
    "knn_weights": "distance",
    "encode_batch_size": 256,
}


METHOD_OVERRIDES = {
    "ts2vec": {
        "weight_decay": 1e-2,
    },
    "ts_tcc": {
        "lr": 3e-4,
    },
    "byol": {
        "lr": 3e-4,
        "batch_size": 128,
    },
    "lhf_bootstrap": {
        "lr": 3e-4,
        "batch_size": 128,
    },
    "mae": {
        "backbone_name": "trace_transformer_v1",
        "pool_mode": "mean",
        "lr": 1e-4,
        "batch_size": 128,
        "mask_ratio": 0.75,
    },
    "time_mae": {
        "backbone_name": "trace_transformer_v1",
        "pool_mode": "mean",
        "lr": 1e-4,
        "batch_size": 128,
        "mask_ratio": 0.60,
    },
}


def get_method_metadata(method: str) -> dict:
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown SSL method: {method}")
    return dict(METHOD_REGISTRY[method])


def build_method_options(method: str, overrides=None):
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown SSL method: {method}")

    values = dict(DEFAULT_METHOD_OPTIONS)
    values.update(METHOD_OVERRIDES.get(method, {}))

    if overrides:
        values.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )

    values["method"] = method
    return SimpleNamespace(**values)
