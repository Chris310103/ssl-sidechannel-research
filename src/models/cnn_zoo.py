import torch
import torch.nn as nn


class SharedTripletCNNBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 700,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.output_dim = 4096
        self.temporal_output_dim = 512

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

            temporal = self.features(dummy)

            self.temporal_length = temporal.shape[-1]
            self.flatten_dim = temporal.flatten(
                start_dim=1
            ).shape[1]

        self.feature_head = nn.Sequential(
            nn.Linear(
                self.flatten_dim,
                4096,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                4096,
                4096,
            ),
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
        h = self.forward_features(x)
        h = torch.flatten(
            h,
            start_dim=1,
        )
        return self.feature_head(h)

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(x)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_temporal_output_dim(self) -> int:
        return self.temporal_output_dim


def build_cnn_backbone(
    input_channels: int = 1,
    input_length: int = 700,
) -> SharedTripletCNNBackbone:
    return SharedTripletCNNBackbone(
        input_channels=input_channels,
        input_length=input_length,
    )