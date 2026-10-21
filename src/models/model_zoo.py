import os
import sys
import pdb
import torch
import torch.nn as nn
from typing import Optional
import src.utils.supporting_functions as supporting_functions


# ---------------------------------------------------------------------------
# Keras-compatibility helpers
# ---------------------------------------------------------------------------
def keras_default_init(module):
    """Keras Dense/Conv default init: glorot_uniform kernel, zero bias."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def he_normal_init(module):
    """Keras kernel_initializer='he_normal'."""
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def keras_batchnorm(num_features):
    """Keras BatchNormalization defaults (momentum=0.99, epsilon=1e-3).

    Keras momentum is the decay of the running stats; PyTorch momentum is its
    complement, hence momentum=0.01.
    """
    return nn.BatchNorm1d(num_features, eps=1e-3, momentum=0.01)


def same_padding(kernel_size):
    """Padding that reproduces Keras padding='same' for odd kernels.

    Verified exactly equal to ceil(L/stride) for kernel_size in {1, 11} and
    stride in {1, 2}, which covers every conv in this file.
    """
    return kernel_size // 2


# ---------------------------------------------------------------------------
#   Model definitions
# ---------------------------------------------------------------------------
class TraceTransformer1D_encoder(nn.Module):
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
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
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
        padded_length = ((trace_length + self.patch_size - 1) // self.patch_size) * self.patch_size

        if padded_length != trace_length:
            padding = x.new_zeros(batch_size, padded_length - trace_length, channels)
            x = torch.cat([x, padding], dim=1)

        return x.reshape(batch_size, padded_length // self.patch_size, self.patch_dim)

    def position_tokens(self, number_of_patches: int) -> torch.Tensor:
        if number_of_patches > self.num_patches:
            raise ValueError("Trace produces more patches than configured: " f"{number_of_patches} > {self.num_patches}")
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
            return torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=1)
        if pool in ("none", None):
            return features
        raise ValueError(f"Unsupported pooling mode: {pool}")

    def get_output_dim(self, pool: str = "mean") -> int:
        if pool in ("mean", "max"):
            return self.output_channels
        if pool == "mean_max":
            return self.output_channels * 2
        if pool in ("none", None):
            raise ValueError("Temporal output does not have one fixed vector dimension.")
        raise ValueError(f"Unsupported pooling mode: {pool}")

    def get_temporal_output_dim(self) -> int:
        return self.output_channels


class TraceTransformer1D_decoder_only(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def encode(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        pass


# ---------------------------------------------------------------------------
# CNN best models (VGG16-inspired 1D)
# ---------------------------------------------------------------------------
class SCAModel(nn.Module):
    """Base class holding the bits Keras used to keep inside model.compile()."""
    #: names of the model outputs, in the order the loss/targets are stacked
    output_names = ["predictions"]
    #: expected trace length (replaces the Keras input_shape sanity check)
    input_dim = None
    #: kwargs needed to rebuild this model from a checkpoint
    init_kwargs = {}
    #: optimizer spec: ("rmsprop"|"adam", lr)
    optimizer_spec = ("adam", 1e-3)

    @property
    def multi_output(self):
        return len(self.output_names) > 1


    def configure_optimizer(self):
        kind, lr = self.optimizer_spec
        if kind == "rmsprop":
            # Keras RMSprop defaults: rho=0.9, epsilon=1e-7
            return torch.optim.RMSprop(self.parameters(), lr=lr, alpha=0.9, eps=1e-7)
        # Keras Adam defaults: beta_1=0.9, beta_2=0.999, epsilon=1e-7
        return torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)


    def _infer_flat_size(self, feature_fn, input_dim, in_channels=1):
        """Run a dummy tensor through the feature extractor to size the head."""
        try:
            device = next(feature_fn.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_dim, device=device)
            flat = feature_fn(dummy).flatten(1).shape[1]
        return flat


class CnnBestNorm(SCAModel):
    """
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

    def __init__(self, input_length=700, emb_size=256, name="cnn_best"):
        super().__init__()
        self.first_stride = 2
        self.input_channels = 1
        self.input_length = input_length
        self.emb_size = emb_size
        self.output_names = ["predictions"]
        self.optimizer_spec = ("rmsprop", 0.00001)
        self.name = name
        self.init_kwargs = dict(input_channels=self.input_channels, input_length=self.input_length,
                                emb_size=self.emb_size, first_stride=self.first_stride, name=self.name)
        self.input_length = input_length

        kernel = 11
        pad_size = same_padding(kernel)
        blocks = []

        in_c = 1
        for i, out_c in enumerate([64, 128, 256, 512, 512]):
            stride = self.first_stride if i == 0 else 1
            blocks += [
                nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=pad_size),
                nn.ReLU(),
                nn.AvgPool1d(2, stride=2),
                nn.BatchNorm1d(out_c),
            ]
            in_c = out_c
        self.features = nn.Sequential(*blocks)

        flat = self._infer_flat_size(self.features, input_length)
        
        self.fc = nn.Sequential(
            nn.Linear(flat, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, self.emb_size),
        )

        self.embedding = nn.Sequential(
            nn.Linear(flat, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 256),
        )
        
        self.apply(keras_default_init)

    def forward(self, x):
        """
        Full forward pass: conv features -> flatten -> FC -> final activation.
        Returns logits (or embeddings) of shape (batch, emb_size).
        """
        x = self.features(x)  # (B, L_out, 512)
        middle = x.flatten(1)
        feats = self.fc(middle)  # skip the Flatten module, we did it inside the self.fc sequential
        return feats

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # kept for interface compatibility, not used here
        """
        Return the final embedding vector, explicitly documented for feature extraction.
        """
        x = self.features(x)
        middle = x.flatten(1)
        feats = self.embedding(middle)
        return feats


# ---------------------------------------------------------------------------
# Save / load / predict
# ---------------------------------------------------------------------------
def get_model_by_name(model_name, **kwargs):
    """Factory function to get a model by name."""
    if model_name == "cnn_best_norm":
        return CnnBestNorm(**kwargs)
    elif model_name == "trace_transformer_encoder":
        return TraceTransformer1D_encoder(**kwargs)
    elif model_name == "trace_transformer_decoder_only":
        return TraceTransformer1D_decoder_only(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def save_sca_model(model, model_file):
    """Checkpoint that carries enough metadata to rebuild the architecture."""
    model_dict ={"model_name": type(model).__name__,
                 "init_kwargs": model.init_kwargs,
                 "state_dict": model.state_dict()
                }
    torch.save(model_dict, model_file)


def load_sca_model(model_file):
    supporting_functions.check_file_exists(model_file)
    device = supporting_functions.get_device()
    try:
        try:
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        except TypeError:  # torch versions without the weights_only kwarg
            checkpoint = torch.load(model_file, map_location=device)
        model_name = checkpoint["model_name"]
        model = get_model_by_name(model_name, **checkpoint["init_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        # In PyTorch, model.eval() puts the model into evaluation mode.
        # Technically, it sets the training attribute of the module and all its submodules to False.
        model.eval()
    except Exception as exc:
        print("Error: can't load PyTorch model file {} ({})".format(model_file, exc))
        sys.exit(-1)
    return model


# ----------------------------------------------------------------------
#   Factory Builder functions
# ----------------------------------------------------------------------
def build_cnn_best_norm(input_length, emb_size=256):
    input_channels = 1  # Assuming single-channel input for classification
    return CnnBestNorm(input_length, emb_size)


def build_transformer_encoder(input_channels, input_length):
    return TraceTransformer1D_encoder(input_channels=input_channels, input_length=input_length)


def build_transformer_decoder_only(input_channels, input_length):
    return TraceTransformer1D_decoder_only(input_channels=input_channels, input_length=input_length)


def build_model(model_name, input_length, emb_size=256):
    """
    Factory for all supported backbones.

    Returns a model with:
      - forward_features(x)     -> (B, L_out, C_out)
      - encode(x, pool)         -> (B, rep_dim)  (global pooling or flat)
      - get_output_dim(pool)    -> int
    """
    input_channels = 1

    if model_name == "cnn_best_norm":
        return build_cnn_best_norm(input_length=input_length, emb_size=emb_size)

    elif model_name == "trace_transformer_encoder":
        return build_transformer_encoder(input_channels=input_channels, input_length=input_length)

    elif model_name == "trace_transformer_decoder_only":
        return build_transformer_decoder_only(input_channels=input_channels, input_length=input_length)

    else:
        raise ValueError(f"Unknown backbone: {model_name}")
