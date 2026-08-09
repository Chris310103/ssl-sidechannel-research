from copy import deepcopy
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.evaluation.rank_eval import (
    compute_rank_curve,
    expand_proba_to_256,
    plot_rank_curve,
)
from src.models.cnn_zoo import build_cnn_backbone
from src.utils.experiment_logger import append_experiment_result
from src.utils.get_device import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_shift_1d(
    x: torch.Tensor,
    max_shift: int = 10,
) -> torch.Tensor:
    if max_shift <= 0:
        return x

    batch_size = x.size(0)

    shifts = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(batch_size,),
        device=x.device,
    )

    output = torch.empty_like(x)

    for index, shift in enumerate(shifts):
        output[index] = torch.roll(
            x[index],
            shifts=int(shift.item()),
            dims=0,
        )

    return output


def add_gaussian_noise(
    x: torch.Tensor,
    noise_std: float = 0.05,
) -> torch.Tensor:
    if noise_std <= 0:
        return x

    trace_std = x.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    noise = (
        torch.randn_like(x)
        * trace_std
        * noise_std
    )

    return x + noise


def augment_trace(
    x: torch.Tensor,
    max_shift: int = 10,
    noise_std: float = 0.05,
) -> torch.Tensor:
    x = random_shift_1d(
        x,
        max_shift=max_shift,
    )

    x = add_gaussian_noise(
        x,
        noise_std=noise_std,
    )

    return x


class ReferenceMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(
                hidden_dim,
                eps=1e-5,
                momentum=0.1,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                output_dim,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class BYOLReference1D(nn.Module):
    def __init__(
        self,
        projector_hidden_dim: int = 4096,
        projection_dim: int = 256,
        predictor_hidden_dim: int = 4096,
    ):
        super().__init__()

        self.online_encoder = build_cnn_backbone(
            input_channels=1,
            input_length=700,
        )

        self.repr_dim = (
            self.online_encoder.get_output_channels()
        )

        self.projector_hidden_dim = (
            projector_hidden_dim
        )
        self.projection_dim = projection_dim
        self.predictor_hidden_dim = (
            predictor_hidden_dim
        )

        self.online_projector = ReferenceMLP(
            input_dim=self.repr_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projection_dim,
        )

        self.online_predictor = ReferenceMLP(
            input_dim=projection_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projection_dim,
        )

        self.target_encoder = deepcopy(
            self.online_encoder
        )

        self.target_projector = deepcopy(
            self.online_projector
        )

        for parameter in (
            self.target_encoder.parameters()
        ):
            parameter.requires_grad = False

        for parameter in (
            self.target_projector.parameters()
        ):
            parameter.requires_grad = False

    @staticmethod
    def _encoder_representation(
        encoder: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        temporal = encoder.forward_features(x)

        return temporal.mean(dim=-1)

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._encoder_representation(
            self.online_encoder,
            x,
        )

    def online_forward(
        self,
        x: torch.Tensor,
    ):
        representation = self.encode(x)

        projection = self.online_projector(
            representation
        )

        prediction = self.online_predictor(
            projection
        )

        return (
            representation,
            projection,
            prediction,
        )

    @torch.no_grad()
    def target_forward(
        self,
        x: torch.Tensor,
    ):
        representation = (
            self._encoder_representation(
                self.target_encoder,
                x,
            )
        )

        projection = self.target_projector(
            representation
        )

        return representation, projection

    @staticmethod
    def regression_loss(
        prediction: torch.Tensor,
        target_projection: torch.Tensor,
    ) -> torch.Tensor:
        prediction = F.normalize(
            prediction,
            dim=1,
        )

        target_projection = F.normalize(
            target_projection.detach(),
            dim=1,
        )

        return (
            2.0
            - 2.0
            * (
                prediction
                * target_projection
            ).sum(dim=1)
        ).mean()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        _, _, prediction1 = (
            self.online_forward(x1)
        )

        _, _, prediction2 = (
            self.online_forward(x2)
        )

        with torch.no_grad():
            _, target_projection1 = (
                self.target_forward(x1)
            )

            _, target_projection2 = (
                self.target_forward(x2)
            )

        loss1 = self.regression_loss(
            prediction1,
            target_projection2,
        )

        loss2 = self.regression_loss(
            prediction2,
            target_projection1,
        )

        return loss1 + loss2

    @torch.no_grad()
    def update_target_network(
        self,
        tau: float,
    ) -> None:
        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_parameter.data.add_(
                online_parameter.data
                - target_parameter.data,
                alpha=1.0 - tau,
            )

        for online_parameter, target_parameter in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            target_parameter.data.add_(
                online_parameter.data
                - target_parameter.data,
                alpha=1.0 - tau,
            )


class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 1e-3,
        eps: float = 1e-9,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
            lars_adaptation=True,
        )

        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eta = group["eta"]
            eps = group["eps"]
            use_lars = group.get(
                "lars_adaptation",
                True,
            )

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                update = parameter.grad

                if weight_decay != 0:
                    update = (
                        update
                        + weight_decay * parameter
                    )

                if use_lars:
                    parameter_norm = torch.norm(
                        parameter
                    )
                    update_norm = torch.norm(
                        update
                    )

                    if (
                        parameter_norm > 0
                        and update_norm > 0
                    ):
                        trust_ratio = (
                            eta
                            * parameter_norm
                            / (
                                update_norm
                                + eps
                            )
                        )

                        update = (
                            update
                            * trust_ratio
                        )

                state = self.state[parameter]

                if "momentum_buffer" not in state:
                    state[
                        "momentum_buffer"
                    ] = torch.zeros_like(
                        parameter
                    )

                buffer = state[
                    "momentum_buffer"
                ]

                buffer.mul_(
                    momentum
                ).add_(
                    update
                )

                parameter.add_(
                    buffer,
                    alpha=-lr,
                )

        return loss


def build_lars_parameter_groups(
    model: nn.Module,
    weight_decay: float,
):
    regular = []
    excluded = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        normalized_name = name.lower()

        exclude = (
            parameter.ndim == 1
            or normalized_name.endswith(
                ".bias"
            )
            or "bn" in normalized_name
            or "norm" in normalized_name
        )

        if exclude:
            excluded.append(parameter)
        else:
            regular.append(parameter)

    return [
        {
            "params": regular,
            "weight_decay": weight_decay,
            "lars_adaptation": True,
        },
        {
            "params": excluded,
            "weight_decay": 0.0,
            "lars_adaptation": False,
        },
    ]


def reference_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
    batch_size: int,
) -> float:
    scaled_lr = (
        base_learning_rate
        * batch_size
        / 256.0
    )

    if (
        warmup_steps > 0
        and step < warmup_steps
    ):
        return (
            step
            / warmup_steps
            * scaled_lr
        )

    decay_steps = max(
        1,
        total_steps - warmup_steps,
    )

    progress = (
        step - warmup_steps
    ) / decay_steps

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        0.5
        * scaled_lr
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


def reference_target_ema(
    step: int,
    total_steps: int,
    base_ema: float,
) -> float:
    progress = min(
        max(
            step / max(
                total_steps,
                1,
            ),
            0.0,
        ),
        1.0,
    )

    cosine_decay = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        1.0
        - (
            1.0 - base_ema
        )
        * cosine_decay
    )


def reference_preset(
    n_epochs: int,
):
    presets = {
        40: {
            "base_learning_rate": 0.45,
            "weight_decay": 1e-6,
            "base_target_ema": 0.97,
        },
        100: {
            "base_learning_rate": 0.45,
            "weight_decay": 1e-6,
            "base_target_ema": 0.99,
        },
        300: {
            "base_learning_rate": 0.30,
            "weight_decay": 1e-6,
            "base_target_ema": 0.99,
        },
        1000: {
            "base_learning_rate": 0.20,
            "weight_decay": 1.5e-6,
            "base_target_ema": 0.996,
        },
    }

    if n_epochs not in presets:
        raise ValueError(
            "Reference BYOL presets support "
            "40, 100, 300, or 1000 epochs."
        )

    return presets[n_epochs]


def train_byol(
    X_train,
    device,
    n_epochs: int = 100,
    batch_size: int = 128,
    projector_hidden_dim: int = 4096,
    projection_dim: int = 256,
    predictor_hidden_dim: int = 4096,
    warmup_epochs: int = 10,
    lars_momentum: float = 0.9,
    lars_eta: float = 1e-3,
    max_shift: int = 10,
    noise_std: float = 0.05,
):
    preset = reference_preset(
        n_epochs
    )

    base_learning_rate = preset[
        "base_learning_rate"
    ]

    weight_decay = preset[
        "weight_decay"
    ]

    base_target_ema = preset[
        "base_target_ema"
    ]

    model = BYOLReference1D(
        projector_hidden_dim=(
            projector_hidden_dim
        ),
        projection_dim=projection_dim,
        predictor_hidden_dim=(
            predictor_hidden_dim
        ),
    ).to(device)

    dataset = TensorDataset(
        torch.from_numpy(
            X_train
        ).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    parameter_groups = (
        build_lars_parameter_groups(
            model,
            weight_decay=weight_decay,
        )
    )

    optimizer = LARS(
        parameter_groups,
        lr=0.0,
        momentum=lars_momentum,
        eta=lars_eta,
    )

    total_steps = (
        len(loader) * n_epochs
    )

    warmup_steps = (
        len(loader)
        * warmup_epochs
    )

    trainable_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    backbone_params = sum(
        parameter.numel()
        for parameter
        in model.online_encoder.parameters()
        if parameter.requires_grad
    )

    print(
        "Backbone trainable parameters:",
        backbone_params,
    )

    print(
        "Full online trainable parameters:",
        trainable_params,
    )

    model.eval()

    with torch.no_grad():
        sample = torch.from_numpy(
            X_train[:8]
        ).float().to(device)

        temporal = (
            model.online_encoder.forward_features(
                sample
            )
        )

        representation = model.encode(
            sample
        )

        projection = (
            model.online_projector(
                representation
            )
        )

        prediction = (
            model.online_predictor(
                projection
            )
        )

    print(
        "Sample input shape:",
        sample.shape,
    )

    print(
        "Temporal feature shape:",
        temporal.shape,
    )

    print(
        "Encoder representation shape:",
        representation.shape,
    )

    print(
        "Projection shape:",
        projection.shape,
    )

    print(
        "Prediction shape:",
        prediction.shape,
    )

    model.train()

    loss_log = []
    global_step = 0

    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(
                device
            )

            x1 = augment_trace(
                batch_x,
                max_shift=max_shift,
                noise_std=noise_std,
            )

            x2 = augment_trace(
                batch_x,
                max_shift=max_shift,
                noise_std=noise_std,
            )

            learning_rate = (
                reference_lr(
                    step=global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    base_learning_rate=(
                        base_learning_rate
                    ),
                    batch_size=batch_size,
                )
            )

            tau = reference_target_ema(
                step=global_step,
                total_steps=total_steps,
                base_ema=base_target_ema,
            )

            for group in (
                optimizer.param_groups
            ):
                group["lr"] = (
                    learning_rate
                )

            loss = model(
                x1,
                x2,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()

            optimizer.step()

            model.update_target_network(
                tau=tau,
            )

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        average_loss = (
            total_loss
            / max(
                num_batches,
                1,
            )
        )

        loss_log.append(
            average_loss
        )

        print(
            f"Epoch #{epoch}: "
            f"byol_loss={average_loss:.6f}, "
            f"lr={learning_rate:.6f}, "
            f"ema={tau:.6f}"
        )

    config = {
        "base_learning_rate": (
            base_learning_rate
        ),
        "weight_decay": weight_decay,
        "base_target_ema": (
            base_target_ema
        ),
        "warmup_epochs": warmup_epochs,
        "lars_momentum": lars_momentum,
        "lars_eta": lars_eta,
    }

    return model, loss_log, config


def encode_representations(
    model,
    X,
    device,
    batch_size: int = 256,
):
    dataset = TensorDataset(
        torch.from_numpy(
            X
        ).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    representations = []

    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(
                device
            )

            representation = (
                model.encode(
                    batch_x
                )
            )

            representations.append(
                representation.cpu().numpy()
            )

    return np.concatenate(
        representations,
        axis=0,
    )


def main():
    ascad_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "ascad"
        / "ASCAD.h5"
    )

    seed = 42

    n_train = 50000
    n_attack = 10000

    n_epochs = 100
    batch_size = 128

    projector_hidden_dim = 4096
    projection_dim = 256
    predictor_hidden_dim = 4096

    warmup_epochs = 10
    lars_momentum = 0.9
    lars_eta = 1e-3

    max_shift = 10
    noise_std = 0.05

    target_byte = 2
    normalize_mode = None

    trace_window = (0, 700)

    window_start, window_end = (
        trace_window
    )

    window_size = (
        window_end
        - window_start
    )

    set_seed(seed)

    backbone_name = (
        "triplet_network_conv_cnn"
    )

    run_name = (
        f"byol_ref_{backbone_name}"
        f"_window{window_start}-{window_end}"
        f"_gap"
        f"_proj{projection_dim}"
        f"_ph{projector_hidden_dim}"
        f"_ep{n_epochs}"
        f"_bs{batch_size}"
        f"_seed{seed}"
    )

    figure_dir = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / run_name
    )

    repr_dir = (
        PROJECT_ROOT
        / "outputs"
        / "representations"
        / run_name
    )

    checkpoint_dir = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / run_name
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repr_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading ASCAD profiling traces..."
    )

    X_profiling, y_profiling = (
        load_ascad_split(
            h5_path=ascad_path,
            split="profiling",
            add_channel=True,
            normalize=normalize_mode,
            load_metadata=False,
            trace_window=trace_window,
        )
    )

    print(
        "Loading ASCAD attack traces "
        "with metadata..."
    )

    (
        X_attack,
        y_attack,
        metadata_attack,
    ) = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=True,
        normalize=normalize_mode,
        load_metadata=True,
        trace_window=trace_window,
    )

    X_train = X_profiling[
        :n_train
    ]

    y_train = y_profiling[
        :n_train
    ]

    X_attack_small = X_attack[
        :n_attack
    ]

    metadata_attack_small = (
        metadata_attack[
            :n_attack
        ]
    )

    print(
        "X_train shape:",
        X_train.shape,
    )

    print(
        "X_attack shape:",
        X_attack_small.shape,
    )

    device = get_device(
        prefer_mps=False,
    )

    print(
        "Using device:",
        device,
    )

    print(
        "Training reference-style BYOL..."
    )

    train_start_time = time.time()

    (
        model,
        loss_log,
        ref_config,
    ) = train_byol(
        X_train=X_train,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
        projector_hidden_dim=(
            projector_hidden_dim
        ),
        projection_dim=projection_dim,
        predictor_hidden_dim=(
            predictor_hidden_dim
        ),
        warmup_epochs=warmup_epochs,
        lars_momentum=lars_momentum,
        lars_eta=lars_eta,
        max_shift=max_shift,
        noise_std=noise_std,
    )

    train_end_time = time.time()

    train_time_sec = (
        train_end_time
        - train_start_time
    )

    print(
        "Training time:",
        f"{train_time_sec:.2f} sec",
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{run_name}_encoder.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(
        "Saved checkpoint to:",
        checkpoint_path,
    )

    print(
        "Encoding profiling representations..."
    )

    repr_train = encode_representations(
        model=model,
        X=X_train,
        device=device,
        batch_size=256,
    )

    print(
        "Encoding attack representations..."
    )

    repr_attack = encode_representations(
        model=model,
        X=X_attack_small,
        device=device,
        batch_size=256,
    )

    print(
        "repr_train shape:",
        repr_train.shape,
    )

    print(
        "repr_attack shape:",
        repr_attack.shape,
    )

    if (
        repr_train.shape[1]
        != model.repr_dim
    ):
        raise ValueError(
            "Unexpected representation "
            f"dimension: expected "
            f"{model.repr_dim}, received "
            f"{repr_train.shape[1]}"
        )

    np.save(
        repr_dir / "repr_train.npy",
        repr_train,
    )

    np.save(
        repr_dir / "repr_attack.npy",
        repr_attack,
    )

    np.save(
        repr_dir / "y_train.npy",
        y_train,
    )

    feature_std = repr_train.std(
        axis=0
    )

    print(
        "Representation overall std:",
        float(
            repr_train.std()
        ),
    )

    print(
        "Representation median feature std:",
        float(
            np.median(
                feature_std
            )
        ),
    )

    print(
        "Representation dead feature ratio:",
        float(
            np.mean(
                feature_std < 1e-6
            )
        ),
    )

    print(
        "Representation mean L2 norm:",
        float(
            np.linalg.norm(
                repr_train,
                axis=1,
            ).mean()
        ),
    )

    print(
        "Training linear classifier "
        "on BYOL encoder representations..."
    )

    classifier = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
    )

    classifier.fit(
        repr_train,
        y_train,
    )

    train_acc = float(
        classifier.score(
            repr_train,
            y_train,
        )
    )

    print(
        "Linear probe train accuracy:",
        train_acc,
    )

    print(
        "Predicting attack probabilities..."
    )

    attack_probas_seen = (
        classifier.predict_proba(
            repr_attack
        )
    )

    attack_probas = expand_proba_to_256(
        attack_probas_seen,
        classes=classifier.classes_,
    )

    ranks = compute_rank_curve(
        probas=attack_probas,
        metadata=metadata_attack_small,
        target_byte=target_byte,
        max_traces=n_attack,
        use_log=True,
    )

    final_rank = int(
        ranks[-1]
    )

    min_rank = int(
        ranks.min()
    )

    rank0_indices = np.where(
        ranks == 0
    )[0]

    rank0_trace = (
        int(
            rank0_indices[0]
            + 1
        )
        if len(
            rank0_indices
        ) > 0
        else -1
    )

    print(
        "Final rank:",
        final_rank,
    )

    print(
        "Minimum rank:",
        min_rank,
    )

    print(
        "Rank-0 trace:",
        rank0_trace,
    )

    rank_path = (
        figure_dir
        / (
            f"{run_name}"
            "_linear_probe_rank.png"
        )
    )

    ranks_path = (
        repr_dir
        / (
            f"{run_name}"
            "_linear_probe_ranks.npy"
        )
    )

    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title=(
            "Reference-Style BYOL "
            "Shared Triplet CNN "
            "Linear Probe Key Rank"
        ),
    )

    np.save(
        ranks_path,
        ranks,
    )

    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / "experiment_summary.csv"
    )

    backbone_params = sum(
        parameter.numel()
        for parameter
        in model.online_encoder.parameters()
        if parameter.requires_grad
    )

    peak_lr = (
        ref_config[
            "base_learning_rate"
        ]
        * batch_size
        / 256.0
    )

    append_experiment_result(
        summary_path,
        {
            "method": (
                "BYOL-reference-style"
            ),
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "seed": seed,
            "n_train": n_train,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "backbone_name": (
                backbone_name
            ),
            "backbone_params": (
                backbone_params
            ),
            "encoder_output_channels": (
                model.online_encoder
                .get_output_channels()
            ),
            "pool_mode": "global_avg",
            "pooled_repr_dim": (
                model.repr_dim
            ),
            "projector_hidden_dim": (
                projector_hidden_dim
            ),
            "proj_dim": projection_dim,
            "predictor_hidden_dim": (
                predictor_hidden_dim
            ),
            "optimizer": "LARS",
            "base_learning_rate": (
                ref_config[
                    "base_learning_rate"
                ]
            ),
            "peak_learning_rate": (
                peak_lr
            ),
            "weight_decay": (
                ref_config[
                    "weight_decay"
                ]
            ),
            "warmup_epochs": (
                warmup_epochs
            ),
            "lr_schedule": (
                "warmup_cosine"
            ),
            "base_target_ema": (
                ref_config[
                    "base_target_ema"
                ]
            ),
            "ema_schedule": (
                "cosine_to_1"
            ),
            "lars_momentum": (
                lars_momentum
            ),
            "lars_eta": (
                lars_eta
            ),
            "max_shift": max_shift,
            "noise_std": noise_std,
            "normalize": (
                normalize_mode
            ),
            "window_start": (
                window_start
            ),
            "window_end": (
                window_end
            ),
            "window_size": (
                window_size
            ),
            "classifier": (
                "LogisticRegression"
            ),
            "linear_probe_train_acc": (
                round(
                    train_acc,
                    6,
                )
            ),
            "target_byte": (
                target_byte
            ),
            "device": str(
                device
            ),
            "train_start_time": (
                train_start_time
            ),
            "train_end_time": (
                train_end_time
            ),
            "train_time_sec": (
                round(
                    train_time_sec,
                    2,
                )
            ),
            "final_rank": (
                final_rank
            ),
            "min_rank": (
                min_rank
            ),
            "rank0_trace": (
                rank0_trace
            ),
            "figure_path": str(
                rank_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "checkpoint_path": str(
                checkpoint_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    )

    print(
        "Saved experiment summary to:",
        summary_path,
    )


if __name__ == "__main__":
    main()
