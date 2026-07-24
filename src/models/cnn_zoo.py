from typing import Literal, Sequence

import torch
import torch.nn as nn


PoolMode = Literal["none", "mean", "max", "mean_max", "last"]


class SharedCNN1D(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        channels: Sequence[int] = (64, 128, 256, 320),
        kernel_sizes: Sequence[int] = (11, 11, 11, 11),
        strides: Sequence[int] = (2, 2, 2, 2),
    ):
        super().__init__()

        if not (
            len(channels)
            == len(kernel_sizes)
            == len(strides)
        ):
            raise ValueError(
                "channels, kernel_sizes, and strides must have equal lengths"
            )

        if len(channels) == 0:
            raise ValueError("At least one CNN layer is required")

        self.input_channels = input_channels
        self.output_channels = channels[-1]

        layers = []
        in_channels = input_channels

        for out_channels, kernel_size, stride in zip(
            channels,
            kernel_sizes,
            strides,
        ):
            if kernel_size <= 0 or stride <= 0:
                raise ValueError(
                    "kernel sizes and strides must be positive"
                )

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )

            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

    def _to_channel_first(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor, received shape {tuple(x.shape)}"
            )

        if x.shape[-1] == self.input_channels:
            return x.transpose(1, 2)

        if x.shape[1] == self.input_channels:
            return x

        raise ValueError(
            "Unable to identify the channel dimension for "
            f"input shape {tuple(x.shape)}"
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._to_channel_first(x)

        features = self.encoder(x)
        features = features.transpose(1, 2)

        return features

    def pool_features(
        self,
        features: torch.Tensor,
        mode: PoolMode = "mean_max",
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "Expected temporal features with shape [B, T, C], "
                f"received {tuple(features.shape)}"
            )

        if mode == "none":
            return features

        if mode == "mean":
            return features.mean(dim=1)

        if mode == "max":
            return features.max(dim=1).values

        if mode == "mean_max":
            mean_features = features.mean(dim=1)
            max_features = features.max(dim=1).values

            return torch.cat(
                [mean_features, max_features],
                dim=1,
            )

        if mode == "last":
            return features[:, -1, :]

        raise ValueError(
            f"Unsupported pooling mode: {mode}"
        )

    def encode(
        self,
        x: torch.Tensor,
        pool: PoolMode = "mean_max",
    ) -> torch.Tensor:
        features = self.forward_features(x)

        return self.pool_features(
            features,
            mode=pool,
        )

    def forward(
        self,
        x: torch.Tensor,
        pool: PoolMode = "mean_max",
    ) -> torch.Tensor:
        return self.encode(
            x,
            pool=pool,
        )

    def get_output_dim(
        self,
        pool: PoolMode = "mean_max",
    ) -> int:
        if pool in ("mean", "max", "last"):
            return self.output_channels

        if pool == "mean_max":
            return self.output_channels * 2

        if pool == "none":
            raise ValueError(
                "pool='none' returns a temporal sequence, "
                "so it does not have one fixed vector dimension"
            )

        raise ValueError(
            f"Unsupported pooling mode: {pool}"
        )


def build_cnn_backbone(
    name: str = "shared_cnn_v1",
    input_channels: int = 1,
) -> SharedCNN1D:
    if name == "shared_cnn_v1":
        return SharedCNN1D(
            input_channels=input_channels,
            channels=(64, 128, 256, 320),
            kernel_sizes=(11, 11, 11, 11),
            strides=(2, 2, 2, 2),
        )

    raise ValueError(
        f"Unsupported backbone name: {name}"
    )