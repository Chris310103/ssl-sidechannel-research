import torch
import torch.nn as nn
from typing import Optional


class TraceTransformer1D(nn.Module):
    """Patch-based Transformer encoder for one-dimensional power traces."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 700,
        patch_size: int = 10,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_channels = input_channels
        self.input_length = input_length
        self.patch_size = patch_size
        self.output_channels = embed_dim
        self.patch_dim = input_channels * patch_size
        self.num_patches = (input_length + patch_size - 1) // patch_size

        self.patch_projection = nn.Linear(self.patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)

    def _to_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.shape[-1] == self.input_channels:
            return x
        if x.shape[1] == self.input_channels:
            return x.transpose(1, 2)
        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_channels_last(x)
        batch_size, trace_length, channels = x.shape
        padded_length = (
            (trace_length + self.patch_size - 1) // self.patch_size
        ) * self.patch_size

        if padded_length != trace_length:
            padding = x.new_zeros(
                batch_size,
                padded_length - trace_length,
                channels,
            )
            x = torch.cat([x, padding], dim=1)

        return x.reshape(
            batch_size,
            padded_length // self.patch_size,
            self.patch_dim,
        )

    def position_tokens(self, number_of_patches: int) -> torch.Tensor:
        if number_of_patches > self.num_patches:
            raise ValueError(
                "Trace produces more patches than configured: "
                f"{number_of_patches} > {self.num_patches}"
            )
        return self.position_embedding[:, :number_of_patches]

    def embed_patch_values(self, patch_values: torch.Tensor) -> torch.Tensor:
        return self.patch_projection(patch_values)

    def encode_token_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(tokens))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        patch_values = self.patchify(x)
        tokens = self.embed_patch_values(patch_values)
        tokens = tokens + self.position_tokens(tokens.shape[1])
        return self.encode_token_embeddings(tokens)

    def forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def encode(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        features = self.forward_features(x)

        if pool == "mean":
            return features.mean(dim=1)
        if pool == "max":
            return features.max(dim=1).values
        if pool == "mean_max":
            return torch.cat(
                [features.mean(dim=1), features.max(dim=1).values],
                dim=1,
            )
        if pool in ("none", None):
            return features
        raise ValueError(f"Unsupported pooling mode: {pool}")

    def get_output_dim(self, pool: str = "mean") -> int:
        if pool in ("mean", "max"):
            return self.output_channels
        if pool == "mean_max":
            return self.output_channels * 2
        if pool in ("none", None):
            raise ValueError(
                "Temporal output does not have one fixed vector dimension."
            )
        raise ValueError(f"Unsupported pooling mode: {pool}")

    def get_temporal_output_dim(self) -> int:
        return self.output_channels


class CnnBestNorm(nn.Module):
    """
    PyTorch implementation of the 'cnn_best_norm' architecture.
    Based on VGG16 design, adapted for 1D signals.

    Parameters
    ----------
    input_channels : int, default=1
        Number of channels in the input signal.
    emb_size : int, default=256
        Dimensionality of the final output (classification or embedding).
    classification : bool, default=True
        If True, the final layer uses softmax (for classification).
        If False, the final layer uses ReLU (for embedding).
    input_length : int, default=1400
        Length of the input sequence. Must be known because the model uses a
        Flatten operation. Any other length will cause size mismatch.
    """

    def __init__(self, input_channels: int = 1, input_length: int = 1400):
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length

        emb_size = 256
        classification = True
        self.emb_size = emb_size
        self.classification = classification

        # ---------- Convolutional blocks (feature extractor) ----------
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=11, stride=2, padding=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=11, stride=2, padding=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv1d(256, 512, kernel_size=11, stride=2, padding=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv1d(512, 512, kernel_size=11, stride=2, padding=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Compute the number of features after all conv/pool layers.
        # We need this to know the input size of the first FC layer.
        self._conv_output_features = self._calc_conv_output_size(input_length)
        self._num_time_steps = self._calc_output_time_steps(input_length)

        # ---------- Fully connected classifier ----------
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._conv_output_features, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, emb_size),
        )

        # Apply the appropriate final activation
        if classification:
            self.final_act = nn.Softmax(dim=1)
        else:
            self.final_act = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    #   Helper methods to compute output dimensions
    # ------------------------------------------------------------------
    def _calc_output_time_steps(self, length: int) -> int:
        """Compute the number of time steps after the conv blocks."""
        L = length
        for _ in range(5):  # 5 blocks
            # Conv1d stride=2, padding=5, kernel=11
            L = (L + 2 * 5 - 11) // 2 + 1
            # AvgPool1d kernel=2, stride=2
            L = (L + 0 - 2) // 2 + 1  # padding=0 for pool
        return L

    def _calc_conv_output_size(self, length: int) -> int:
        """Total number of features after conv blocks (channels * time steps)."""
        return 512 * self._calc_output_time_steps(length)

    # ------------------------------------------------------------------
    #   Dimension‑helpers (matching SharedCNN1D style)
    # ------------------------------------------------------------------
    @property
    def output_channels(self) -> int:
        """Number of channels in the temporal feature map."""
        return 512

    def get_temporal_output_dim(self) -> int:
        """Dimension of each time step in the feature map."""
        return self.output_channels

    def get_output_dim(self, pool: Optional[str] = None) -> int:
        """Dimension of the final output vector (embedding or class probs)."""
        return self.emb_size

    # ------------------------------------------------------------------
    #   Input format handling (same logic as SharedCNN1D)
    # ------------------------------------------------------------------
    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is (batch, channels, length)."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.shape[-1] == self.input_channels:
            return x.transpose(1, 2)   # (B, L, C) -> (B, C, L)
        if x.shape[1] == self.input_channels:
            return x                   # already (B, C, L)
        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

    # ------------------------------------------------------------------
    #   Core forward methods
    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the temporal feature map from the convolutional backbone.

        Output shape: (batch, time_steps, 512)
        """
        x = self._to_channels_first(x)
        h = self.conv(x)               # (B, 512, L_out)
        return h.transpose(1, 2)       # (B, L_out, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: conv features -> flatten -> FC -> final activation.

        Returns logits (or embeddings) of shape (batch, emb_size).
        """
        feats = self.forward_features(x)   # (B, L_out, 512)
        # Flatten: treat the last two dims as one vector
        flat = feats.reshape(feats.size(0), -1)  # (B, L_out*512)
        x = self.fc[1:](flat)  # skip the Flatten module, we did it manually
        # More elegantly, we could just use self.fc(feats) after transposing back to (B,512,L_out)
        # But let's use the sequential directly on the transposed tensor:
        # feats must be (B, C, L) for Conv1d-like flatten? No, nn.Flatten flattens everything after dim 1.
        # If we give (B, L_out, 512) to self.fc (which starts with Flatten), it will flatten to (B, L_out*512) – correct.
        # So we can simply do:
        # return self.fc(feats)
        # However, our self.fc begins with Flatten, which expects any shape, so we can use it directly.
        # Let's use the cleaner approach:
        return self.final_act(self.fc(feats))

    def encode(
        self,
        x: torch.Tensor,
        pool: Optional[str] = None,  # kept for interface compatibility, not used here
    ) -> torch.Tensor:
        """
        Return the final embedding / classification vector.
        (Same as forward, but explicitly documented for feature extraction.)
        """
        return self.forward(x)
        

class SharedCNN1D(nn.Module):
    def __init__(self, input_channels: int = 1, input_length: int = 700):
        super().__init__()

        self.input_channels = input_channels
        self.input_length = input_length
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

    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_channels_first(x)
        h = self.net(x)

        return h.transpose(1, 2)

    def forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def encode(self, x: torch.Tensor, pool: str = "mean_max") -> torch.Tensor:
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

    def get_output_dim(self, pool: str = "mean_max") -> int:
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


# ----------------------------------------------------------------------
#   Builder functions
# ----------------------------------------------------------------------
def build_cnn_simple_backbone(input_channels: int = 1, input_length: int = 700) -> SharedCNN1D:

    return SharedCNN1D(input_channels=input_channels, input_length=input_length)


def build_cnn_best_norm_backbone(input_channels: int = 1, input_length: int = 1400) -> CnnBestNorm:
    """
    Factory function for the CnnBestNorm model.

    The ``name`` parameter is kept for interface consistency with other
    backbones; currently only ``"cnn_best_norm"`` is supported.
    """
    return CnnBestNorm(input_channels=input_channels, input_length=input_length) 


def build_transformer_backbone(
    input_channels: int = 1,
    input_length: int = 700,
    **kwargs,
) -> TraceTransformer1D:
    return TraceTransformer1D(
        input_channels=input_channels,
        input_length=input_length,
        **kwargs,
    )

def build_backbone(model_name: str = "shared_cnn_v1", input_channels: int = 1, input_length: int = 700, **kwargs):
    """
    Factory for all supported backbones.
    
    Returns a model with:
      - forward_features(x)     -> (B, L_out, C_out)
      - encode(x, pool)         -> (B, rep_dim)  (global pooling or flat)
      - get_output_dim(pool)    -> int
    """
    if model_name == "shared_cnn_v1":
        return SharedCNN1D(input_channels=input_channels, input_length=input_length)
    
    elif model_name == "cnn_best_norm":
        return CnnBestNorm(input_channels=input_channels, input_length=input_length) 
    
    elif model_name == "trace_transformer_v1":
        return build_transformer_backbone(
            input_channels=input_channels,
            input_length=input_length,
            **kwargs,
        )
    
    else:
        raise ValueError(f"Unknown backbone: {model_name}")
