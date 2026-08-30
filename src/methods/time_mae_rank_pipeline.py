from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.methods.mae_rank_pipeline import random_patch_mask
from src.models.model_zoo import build_backbone
from src.utils.trace_transforms import ensure_trace_matrix, prepare_model_input


class TimeMAE1D(nn.Module):
    """TimeMAE masked-representation-regression variant for power traces."""

    def __init__(
        self,
        backbone_name: str = "trace_transformer_v1",
        pool_mode: str = "mean",
        patch_size: int = 10,
        mask_ratio: float = 0.60,
        input_length: int = 700,
        transformer_dim: int = 192,
        transformer_depth: int = 4,
        transformer_heads: int = 6,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        masked_encoder_depth: int = 2,
        target_ema_decay: float = 0.996,
    ):
        super().__init__()

        if backbone_name != "trace_transformer_v1":
            raise ValueError(
                "TimeMAE uses the shared trace_transformer_v1 backbone; "
                f"received {backbone_name}"
            )

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.mask_ratio = mask_ratio
        self.target_ema_decay = target_ema_decay

        self.online_encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
            patch_size=patch_size,
            embed_dim=transformer_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
        )
        self.target_encoder = deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

        self.mask_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        masked_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=int(transformer_dim * transformer_mlp_ratio),
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.masked_region_encoder = nn.TransformerDecoder(
            masked_layer,
            num_layers=masked_encoder_depth,
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim),
        )
        self.pooled_repr_dim = self.online_encoder.get_output_dim(pool=pool_mode)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_parameter.data.mul_(self.target_ema_decay).add_(
                online_parameter.data,
                alpha=1.0 - self.target_ema_decay,
            )

    def forward(self, x: torch.Tensor):
        patch_values = self.online_encoder.patchify(x)
        batch_size, number_of_patches, _ = patch_values.shape
        visible_indices, _, mask = random_patch_mask(
            batch_size=batch_size,
            number_of_patches=number_of_patches,
            mask_ratio=self.mask_ratio,
            device=x.device,
        )
        masked_indices = torch.argsort(mask, dim=1, descending=True)[
            :, : int(mask.sum(dim=1).max().item())
        ]

        positions = self.online_encoder.position_tokens(number_of_patches).expand(
            batch_size, -1, -1
        )
        online_tokens = (
            self.online_encoder.embed_patch_values(patch_values) + positions
        )
        visible_tokens = torch.gather(
            online_tokens,
            dim=1,
            index=visible_indices.unsqueeze(-1).expand(
                -1, -1, online_tokens.shape[-1]
            ),
        )
        visible_context = self.online_encoder.encode_token_embeddings(
            visible_tokens
        )

        masked_positions = torch.gather(
            positions,
            dim=1,
            index=masked_indices.unsqueeze(-1).expand(-1, -1, positions.shape[-1]),
        )
        masked_queries = self.mask_token.expand_as(masked_positions) + masked_positions
        predicted_masked = self.prediction_head(
            self.masked_region_encoder(
                tgt=masked_queries,
                memory=visible_context,
            )
        )

        with torch.no_grad():
            target_tokens = self.target_encoder.forward_features(x)
            target_masked = torch.gather(
                target_tokens,
                dim=1,
                index=masked_indices.unsqueeze(-1).expand(
                    -1, -1, target_tokens.shape[-1]
                ),
            )

        regression_loss = 1.0 - F.cosine_similarity(
            predicted_masked,
            target_masked.detach(),
            dim=-1,
        ).mean()

        return regression_loss, predicted_masked, target_masked, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder.encode(x, pool=self.pool_mode)


def train_time_mae(
    X_train,
    device,
    trace_window,
    backbone_name: str = "trace_transformer_v1",
    pool_mode: str = "mean",
    patch_size: int = 10,
    mask_ratio: float = 0.60,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    input_length: int = 700,
    transformer_dim: int = 192,
    transformer_depth: int = 4,
    transformer_heads: int = 6,
    transformer_mlp_ratio: float = 4.0,
    transformer_dropout: float = 0.1,
    masked_encoder_depth: int = 2,
    target_ema_decay: float = 0.996,
):
    X_train = ensure_trace_matrix(X_train)
    model = TimeMAE1D(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        input_length=input_length,
        transformer_dim=transformer_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        transformer_mlp_ratio=transformer_mlp_ratio,
        transformer_dropout=transformer_dropout,
        masked_encoder_depth=masked_encoder_depth,
        target_ema_decay=target_ema_decay,
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
            batch_x = prepare_model_input(
                batch_x.to(device),
                trace_window=trace_window,
            )
            loss, _, _, _ = model(batch_x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            model.update_target_encoder()

            total_loss += loss.item()
            number_of_batches += 1

        average_loss = total_loss / max(number_of_batches, 1)
        loss_log.append(average_loss)
        print(f"Epoch #{epoch}: time_mae_mrr_loss={average_loss:.6f}")

    return model, loss_log
