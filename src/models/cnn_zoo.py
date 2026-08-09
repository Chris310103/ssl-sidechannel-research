import torch
import torch.nn as nn


class SharedCNN1D(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = 320

        self.net = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                64,
                128,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                128,
                256,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                256,
                320,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
        )

    def _to_channels_first(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input, got {tuple(x.shape)}"
            )

        if x.shape[-1] == self.input_channels:
            return x.transpose(1, 2)

        if x.shape[1] == self.input_channels:
            return x

        raise ValueError(
            f"Unexpected input shape: {tuple(x.shape)}"
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._to_channels_first(x)
        h = self.net(x)

        return h.transpose(1, 2)

    def forward_temporal(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(x)

    def encode(
        self,
        x: torch.Tensor,
        pool: str = "mean_max",
    ) -> torch.Tensor:
        h = self.forward_features(x)

        if pool == "mean":
            return h.mean(dim=1)

        if pool == "max":
            return h.max(dim=1).values

        if pool == "mean_max":
            return torch.cat(
                [
                    h.mean(dim=1),
                    h.max(dim=1).values,
                ],
                dim=1,
            )

        if pool in ("none", None):
            return h

        raise ValueError(
            f"Unsupported pooling mode: {pool}"
        )

    def get_output_dim(
        self,
        pool: str = "mean_max",
    ) -> int:
        if pool in ("mean", "max"):
            return self.output_channels

        if pool == "mean_max":
            return self.output_channels * 2

        if pool in ("none", None):
            raise ValueError(
                "Temporal output does not have one fixed vector dimension."
            )

        raise ValueError(
            f"Unsupported pooling mode: {pool}"
        )

    def get_temporal_output_dim(self) -> int:
        return self.output_channels


def build_cnn_backbone(
    name: str = "shared_cnn_v1",
    input_channels: int = 1,
    input_length: int = 700,
) -> SharedCNN1D:
    del input_length

    if name != "shared_cnn_v1":
        raise ValueError(
            "Only the restored shared_cnn_v1 backbone is enabled. "
            f"Received: {name}"
        )

    return SharedCNN1D(
        input_channels=input_channels,
    )