from src.methods.byol_rank_pipeline import BYOL1D, train_byol
from src.methods.lhf_bootstrap_rank_pipeline import (
    LHFBootstrap1D,
    train_lhf_bootstrap,
)
from src.methods.mae_rank_pipeline import MAETransformer1D, train_mae
from src.methods.simclr_rank_pipeline import SimCLRModel, train_simclr
from src.methods.time_mae_rank_pipeline import TimeMAE1D, train_time_mae
from src.methods.ts2vec_rank_pipeline import TS2VecSharedModel, train_ts2vec
from src.methods.ts_tcc_rank_pipeline import TSTCC1D, train_ts_tcc


def _transformer_options(opts):
    return {
        "transformer_dim": opts.transformer_dim,
        "transformer_depth": opts.transformer_depth,
        "transformer_heads": opts.transformer_heads,
        "transformer_mlp_ratio": opts.transformer_mlp_ratio,
        "transformer_dropout": opts.transformer_dropout,
    }


def build_ssl_model(opts, input_length: int):
    if opts.method == "simclr":
        return SimCLRModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
            projector_hidden_dim=opts.projector_hidden_dim,
            proj_dim=opts.proj_dim,
        )

    if opts.method == "ts2vec":
        return TS2VecSharedModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
        )

    if opts.method == "ts_tcc":
        return TSTCC1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
            context_dim=opts.context_dim,
            prediction_steps=opts.prediction_steps,
            context_depth=opts.context_depth,
            context_heads=opts.context_heads,
            temporal_temperature=opts.temporal_temperature,
            contextual_temperature=opts.contextual_temperature,
            contextual_weight=opts.contextual_weight,
        )

    if opts.method == "byol":
        return BYOL1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            proj_dim=opts.proj_dim,
            hidden_dim=opts.hidden_dim,
            ema_decay=opts.ema_decay,
            input_length=input_length,
        )

    if opts.method == "lhf_bootstrap":
        return LHFBootstrap1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            proj_dim=opts.proj_dim,
            hidden_dim=opts.hidden_dim,
            tcn_hidden_dim=opts.tcn_hidden_dim,
            ema_decay=opts.ema_decay,
            input_length=input_length,
        )

    if opts.method == "mae":
        return MAETransformer1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            patch_size=opts.patch_size,
            mask_ratio=opts.mask_ratio,
            input_length=input_length,
            decoder_dim=opts.decoder_dim,
            decoder_depth=opts.decoder_depth,
            decoder_heads=opts.decoder_heads,
            **_transformer_options(opts),
        )

    if opts.method == "time_mae":
        return TimeMAE1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            patch_size=opts.patch_size,
            mask_ratio=opts.mask_ratio,
            input_length=input_length,
            masked_encoder_depth=opts.masked_encoder_depth,
            target_ema_decay=opts.target_ema_decay,
            **_transformer_options(opts),
        )

    raise ValueError(f"Unsupported SSL method: {opts.method}")


def train_ssl_model(
    opts,
    X_train,
    device,
    trace_window,
    input_length: int,
    augmentation_family,
):
    common_augmentation_options = {
        "max_shift": opts.max_shift,
        "noise_std": opts.noise_std,
        "denoise_kernel_size": opts.denoise_kernel_size,
        "augmentation_family": augmentation_family,
    }

    if opts.method == "simclr":
        return train_simclr(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            projector_hidden_dim=opts.projector_hidden_dim,
            proj_dim=opts.proj_dim,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            temperature=opts.temperature,
            view_augmentation=opts.view_augmentation,
            augmentation_probability=opts.augmentation_probability,
            input_length=input_length,
            **common_augmentation_options,
        )

    if opts.method == "ts2vec":
        return train_ts2vec(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            alpha=opts.alpha,
            temporal_unit=opts.temporal_unit,
            minimum_crop_ratio=opts.minimum_crop_ratio,
            timestamp_keep_probability=opts.timestamp_keep_probability,
            input_length=input_length,
        )

    if opts.method == "ts_tcc":
        return train_ts_tcc(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            weak_augmentation_probability=opts.weak_augmentation_probability,
            strong_augmentation_probability=opts.strong_augmentation_probability,
            context_dim=opts.context_dim,
            prediction_steps=opts.prediction_steps,
            context_depth=opts.context_depth,
            context_heads=opts.context_heads,
            temporal_temperature=opts.temporal_temperature,
            contextual_temperature=opts.contextual_temperature,
            contextual_weight=opts.contextual_weight,
            input_length=input_length,
            **common_augmentation_options,
        )

    if opts.method == "byol":
        return train_byol(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            proj_dim=opts.proj_dim,
            hidden_dim=opts.hidden_dim,
            ema_decay=opts.ema_decay,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            view_augmentation=opts.view_augmentation,
            augmentation_probability=opts.augmentation_probability,
            input_length=input_length,
            **common_augmentation_options,
        )

    if opts.method == "lhf_bootstrap":
        return train_lhf_bootstrap(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            proj_dim=opts.proj_dim,
            hidden_dim=opts.hidden_dim,
            tcn_hidden_dim=opts.tcn_hidden_dim,
            ema_decay=opts.ema_decay,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            view_augmentation=opts.view_augmentation,
            augmentation_probability=opts.augmentation_probability,
            input_length=input_length,
            **common_augmentation_options,
        )

    if opts.method == "mae":
        return train_mae(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            patch_size=opts.patch_size,
            mask_ratio=opts.mask_ratio,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            input_length=input_length,
            decoder_dim=opts.decoder_dim,
            decoder_depth=opts.decoder_depth,
            decoder_heads=opts.decoder_heads,
            **_transformer_options(opts),
        )

    if opts.method == "time_mae":
        return train_time_mae(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            patch_size=opts.patch_size,
            mask_ratio=opts.mask_ratio,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            input_length=input_length,
            masked_encoder_depth=opts.masked_encoder_depth,
            target_ema_decay=opts.target_ema_decay,
            **_transformer_options(opts),
        )

    raise ValueError(f"Unsupported SSL method: {opts.method}")
