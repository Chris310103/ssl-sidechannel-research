from src.methods.byol_rank_pipeline import BYOL1D, train_byol
from src.methods.cpc_rank_pipeline import CPCSharedModel, train_cpc
from src.methods.mae_rank_pipeline import FCMAESharedCNN1D, train_mae
from src.methods.simclr_rank_pipeline import SimCLRModel, train_simclr
from src.methods.ts2vec_rank_pipeline import TS2VecSharedModel, train_ts2vec


SSL_METHODS = ("simclr", "byol", "cpc", "mae", "ts2vec")


def build_ssl_model(opts, input_length: int):
    if opts.method == "simclr":
        return SimCLRModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
            projector_hidden_dim=opts.projector_hidden_dim,
            proj_dim=opts.proj_dim,
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

    if opts.method == "cpc":
        return CPCSharedModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            context_dim=opts.context_dim,
            prediction_steps=opts.prediction_steps,
            input_length=input_length,
        )

    if opts.method == "mae":
        return FCMAESharedCNN1D(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            patch_size=opts.patch_size,
            mask_ratio=opts.mask_ratio,
            input_length=input_length,
        )

    if opts.method == "ts2vec":
        return TS2VecSharedModel(
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            input_length=input_length,
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
            max_shift=opts.max_shift,
            noise_std=opts.noise_std,
            denoise_kernel_size=opts.denoise_kernel_size,
            view_augmentation=opts.view_augmentation,
            augmentation_family=augmentation_family,
            augmentation_probability=opts.augmentation_probability,
            input_length=input_length,
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
            max_shift=opts.max_shift,
            noise_std=opts.noise_std,
            denoise_kernel_size=opts.denoise_kernel_size,
            view_augmentation=opts.view_augmentation,
            augmentation_family=augmentation_family,
            augmentation_probability=opts.augmentation_probability,
            input_length=input_length,
        )

    if opts.method == "cpc":
        return train_cpc(
            X_train=X_train,
            device=device,
            trace_window=trace_window,
            backbone_name=opts.backbone_name,
            pool_mode=opts.pool_mode,
            context_dim=opts.context_dim,
            prediction_steps=opts.prediction_steps,
            negative_samples=opts.negative_samples,
            n_epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            input_length=input_length,
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

    raise ValueError(f"Unsupported SSL method: {opts.method}")
