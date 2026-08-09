import torch
import torch.nn as nn


class SharedTripletCNNBackbone(nn.Module):
    """
    Shared CNN backbone taken from the convolutional blocks of the
    Triplet Network cnn_best() architecture.

    Input:
        [B, L, 1] or [B, 1, L]

    Output:
        channels-first: [B, 512, 10] for L=700
        temporal:       [B, 10, 512] for L=700

    The original Flatten -> FC4096 -> FC4096 -> task head is intentionally
    excluded. Each method owns its own head.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 700,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.output_channels = 512

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
            out = self.features(dummy)
            self.temporal_length = out.shape[-1]

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
            "Expected input shape "
            f"[B, L, {self.input_channels}] or "
            f"[B, {self.input_channels}, L], "
            f"got {tuple(x.shape)}"
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._to_channels_first(x)
        return self.features(x)

    def forward_temporal(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(x).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_temporal(x)

    def get_output_channels(self) -> int:
        return self.output_channels

    def get_temporal_output_dim(self) -> int:
        return self.output_channels

    def get_temporal_length(self) -> int:
        return self.temporal_length

    def get_readout_dim(
        self,
        mode: str = "mean_max",
    ) -> int:
        if mode == "mean":
            return self.output_channels

        if mode == "max":
            return self.output_channels

        if mode == "mean_max":
            return self.output_channels * 2

        raise ValueError(
            f"Unsupported readout mode: {mode}"
        )


def pool_temporal(
    temporal: torch.Tensor,
    mode: str = "mean_max",
) -> torch.Tensor:
    """
    Parameter-free readout used for a common downstream representation.

    temporal shape:
        [B, T, C]
    """

    if temporal.ndim != 3:
        raise ValueError(
            f"Expected [B, T, C], got {tuple(temporal.shape)}"
        )

    if mode == "mean":
        return temporal.mean(dim=1)

    if mode == "max":
        return temporal.max(dim=1).values

    if mode == "mean_max":
        mean_features = temporal.mean(dim=1)
        max_features = temporal.max(dim=1).values

        return torch.cat(
            [mean_features, max_features],
            dim=1,
        )

    raise ValueError(
        f"Unsupported readout mode: {mode}"
    )


def build_cnn_backbone(
    input_channels: int = 1,
    input_length: int = 700,
) -> SharedTripletCNNBackbone:
    return SharedTripletCNNBackbone(
        input_channels=input_channels,
        input_length=input_length,
    )