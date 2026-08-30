from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_zoo import build_backbone
from src.utils.trace_transforms import (
    ensure_trace_matrix,
    make_trace_view,
)


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalConvHead(nn.Module):
    """TCN projection head with short and long temporal receptive fields."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.short_path = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.long_path = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=7,
            dilation=2,
            padding=6,
        )
        self.norm = nn.BatchNorm1d(hidden_dim * 2)
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        features = temporal_features.transpose(1, 2)
        short_features = F.relu(self.short_path(features), inplace=True)
        long_features = F.relu(self.long_path(features), inplace=True)
        combined = self.norm(torch.cat([short_features, long_features], dim=1))
        return self.output(combined.mean(dim=2))


class LHFBootstrap1D(nn.Module):
    """Low/high-frequency feature bootstrapping with a shared CNN encoder."""

    def __init__(
        self,
        backbone_name: str = "shared_cnn_v1",
        pool_mode: str = "mean_max",
        proj_dim: int = 128,
        hidden_dim: int = 512,
        tcn_hidden_dim: int = 160,
        ema_decay: float = 0.996,
        input_length: int = 700,
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.ema_decay = ema_decay

        self.online_encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
        )
        pooled_dim = self.online_encoder.get_output_dim(pool=pool_mode)
        temporal_dim = self.online_encoder.get_temporal_output_dim()

        self.online_mlp = ProjectionMLP(pooled_dim, hidden_dim, proj_dim)
        self.online_tcn = TemporalConvHead(
            temporal_dim,
            tcn_hidden_dim,
            proj_dim,
        )
        self.mlp_predictor = ProjectionMLP(proj_dim, hidden_dim, proj_dim)
        self.tcn_predictor = ProjectionMLP(proj_dim, hidden_dim, proj_dim)

        self.target_encoder = deepcopy(self.online_encoder)
        self.target_mlp = deepcopy(self.online_mlp)
        self.target_tcn = deepcopy(self.online_tcn)
        for module in (self.target_encoder, self.target_mlp, self.target_tcn):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.pooled_repr_dim = pooled_dim

    @staticmethod
    def bootstrap_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=1)
        target = F.normalize(target.detach(), dim=1)
        return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()

    def _online_outputs(self, x: torch.Tensor):
        temporal = self.online_encoder.forward_features(x)
        pooled = self.online_encoder.encode(x, pool=self.pool_mode)
        return (
            self.mlp_predictor(self.online_mlp(pooled)),
            self.tcn_predictor(self.online_tcn(temporal)),
        )

    @torch.no_grad()
    def _target_outputs(self, x: torch.Tensor):
        temporal = self.target_encoder.forward_features(x)
        pooled = self.target_encoder.encode(x, pool=self.pool_mode)
        return self.target_mlp(pooled), self.target_tcn(temporal)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        online_mlp_1, online_tcn_1 = self._online_outputs(x1)
        online_mlp_2, online_tcn_2 = self._online_outputs(x2)
        target_mlp_1, target_tcn_1 = self._target_outputs(x1)
        target_mlp_2, target_tcn_2 = self._target_outputs(x2)

        return 0.25 * (
            self.bootstrap_loss(online_mlp_1, target_mlp_2)
            + self.bootstrap_loss(online_mlp_2, target_mlp_1)
            + self.bootstrap_loss(online_tcn_1, target_tcn_2)
            + self.bootstrap_loss(online_tcn_2, target_tcn_1)
        )

    @torch.no_grad()
    def update_target_network(self) -> None:
        module_pairs = (
            (self.online_encoder, self.target_encoder),
            (self.online_mlp, self.target_mlp),
            (self.online_tcn, self.target_tcn),
        )
        for online_module, target_module in module_pairs:
            for online_parameter, target_parameter in zip(
                online_module.parameters(), target_module.parameters()
            ):
                target_parameter.data.mul_(self.ema_decay).add_(
                    online_parameter.data,
                    alpha=1.0 - self.ema_decay,
                )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder.encode(x, pool=self.pool_mode)


def train_lhf_bootstrap(
    X_train,
    device,
    trace_window,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    proj_dim: int = 128,
    hidden_dim: int = 512,
    tcn_hidden_dim: int = 160,
    ema_decay: float = 0.996,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_shift: int = 5,
    noise_std: float = 0.05,
    denoise_kernel_size: int = 5,
    view_augmentation: str = "random",
    augmentation_family=("random_shift", "denoise", "gaussian_noise"),
    augmentation_probability: float = 0.5,
    input_length: int = 700,
):
    X_train = ensure_trace_matrix(X_train)
    model = LHFBootstrap1D(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        tcn_hidden_dim=tcn_hidden_dim,
        ema_decay=ema_decay,
        input_length=input_length,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        number_of_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            view_1 = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation=view_augmentation,
                augmentation_family=augmentation_family,
                augmentation_probability=augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )
            view_2 = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation=view_augmentation,
                augmentation_family=augmentation_family,
                augmentation_probability=augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )
            loss = model(view_1, view_2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target_network()

            total_loss += loss.item()
            number_of_batches += 1

        average_loss = total_loss / max(number_of_batches, 1)
        loss_log.append(average_loss)
        print(f"Epoch #{epoch}: lhf_bootstrap_loss={average_loss:.6f}")

    return model, loss_log
