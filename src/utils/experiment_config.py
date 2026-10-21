"""Method metadata and non-model defaults used by evaluation utilities."""

METHOD_REGISTRY = {
    "simclr": {"display_name": "SimCLR", "family": "contrastive", "role": "generic", "backbone": "shared_cnn_v1"},
    "ts2vec": {"display_name": "TS2Vec", "family": "contrastive", "role": "time_series_specialized", "backbone": "shared_cnn_v1"},
    "ts_tcc": {"display_name": "TS-TCC", "family": "contrastive", "role": "time_series_specialized", "backbone": "shared_cnn_v1"},
    "byol": {"display_name": "BYOL", "family": "bootstrap", "role": "generic", "backbone": "shared_cnn_v1"},
    "lhf_bootstrap": {
        "display_name": "L/H-Frequency Bootstrapping",
        "family": "bootstrap",
        "role": "time_series_specialized",
        "backbone": "shared_cnn_v1",
    },
    "mae": {"display_name": "Transformer MAE", "family": "masked_modeling", "role": "generic", "backbone": "trace_transformer_v1"},
    "time_mae": {"display_name": "TimeMAE-MRR", "family": "masked_modeling", "role": "time_series_specialized", "backbone": "trace_transformer_v1"},
}


SSL_METHODS = tuple(METHOD_REGISTRY)


# These concern evaluation only.  SSL architecture and training baselines live
# in src.methods.ssl_factory.SSL_METHOD_PRESETS.
DEFAULT_METHOD_OPTIONS = {"knn_neighbors": 3, "knn_weights": "distance", "encode_batch_size": 256}


def get_method_metadata(method: str) -> dict:
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown SSL method: {method}")
    return dict(METHOD_REGISTRY[method])
