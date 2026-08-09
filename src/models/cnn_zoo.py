from typing import Literal, Sequence

import torch
import torch.nn as nn


PoolMode = Literal["none", "mean", "max", "mean_max", "last"]

import torch
import torch.nn as nn


class TripletCNNBackbone(nn.Module):
    def __init__(
        self,
        input_channels=1,
        input_length=700,
        embedding_dim=256,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv1d(
                128,
                256,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv1d(
                256,
                512,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv1d(
                512,
                512,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2,
            ),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                input_channels,
                input_length,
            )

            dummy_features = self.features(dummy)

            self.temporal_length = dummy_features.shape[-1]

            flatten_dim = dummy_features.flatten(
                start_dim=1
            ).shape[1]

        self.flatten_dim = flatten_dim

        self.embedding_head = nn.Sequential(
            nn.Linear(
                flatten_dim,
                4096,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                4096,
                4096,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                4096,
                embedding_dim,
            ),
            nn.ReLU(inplace=True),
        )

    def _to_channels_first(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input, got shape {tuple(x.shape)}"
            )

        if x.shape[-1] == self.input_channels:
            x = x.transpose(1, 2)

        elif x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input shape [B, L, {self.input_channels}] "
                f"or [B, {self.input_channels}, L], "
                f"got {tuple(x.shape)}"
            )

        return x

    def forward_features(self, x):
        x = self._to_channels_first(x)

        h = self.features(x)

        return h

    def forward_temporal(self, x):
        h = self.forward_features(x)

        return h.transpose(1, 2)

    def forward(self, x):
        h = self.forward_features(x)

        h = torch.flatten(
            h,
            start_dim=1,
        )

        h = self.embedding_head(h)

        return h

    def encode(
        self,
        x,
        pool="identity",
    ):
        if pool not in (
            None,
            "identity",
            "none",
        ):
            raise ValueError(
                "TripletCNNBackbone already produces a "
                "256-D embedding. Use pool='identity'."
            )

        return self.forward(x)

    def get_output_dim(
        self,
        pool="identity",
    ):
        if pool not in (
            None,
            "identity",
            "none",
        ):
            raise ValueError(
                "TripletCNNBackbone only supports "
                "pool='identity'."
            )

        return self.embedding_dim

    def get_temporal_output_dim(self):
        return 512


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
    name,
    input_channels=1,
    input_length=700,
):

    if name == "shared_cnn_v1":
        return SharedCNN1D(
            input_channels=input_channels,
        )

    if name == "triplet_cnn_v1":
        return TripletCNNBackbone(
            input_channels=input_channels,
            input_length=input_length,
            embedding_dim=256,
        )

    raise ValueError(
        f"Unknown CNN backbone: {name}"
    )