import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_zoo import build_backbone
from src.utils.trace_transforms import ensure_trace_matrix, prepare_model_input


def random_patch_mask(
    batch_size: int,
    number_of_patches: int,
    mask_ratio: float,
    device,
):
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between zero and one")

    number_visible = max(1, int(number_of_patches * (1.0 - mask_ratio)))
    noise = torch.rand(batch_size, number_of_patches, device=device)
    shuffled_indices = torch.argsort(noise, dim=1)
    restore_indices = torch.argsort(shuffled_indices, dim=1)
    visible_indices = shuffled_indices[:, :number_visible]

    mask = torch.ones(batch_size, number_of_patches, device=device)
    mask[:, :number_visible] = 0
    mask = torch.gather(mask, dim=1, index=restore_indices)

    return visible_indices, restore_indices, mask


class MAETransformer1D(nn.Module):
    """Generic MAE adapted to non-overlapping one-dimensional trace patches."""

    def __init__(
        self,
        backbone_name: str = "trace_transformer_v1",
        pool_mode: str = "mean",
        patch_size: int = 10,
        mask_ratio: float = 0.75,
        input_length: int = 700,
        transformer_dim: int = 192,
        transformer_depth: int = 4,
        transformer_heads: int = 6,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
    ):
        super().__init__()

        if backbone_name != "trace_transformer_v1":
            raise ValueError(
                "MAE uses the shared trace_transformer_v1 backbone; "
                f"received {backbone_name}"
            )

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.mask_ratio = mask_ratio
        self.input_length = input_length

        self.encoder = build_backbone(
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

        self.decoder_embed = nn.Linear(transformer_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position_embedding = nn.Parameter(
            torch.zeros(1, self.encoder.num_patches, decoder_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_depth,
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.reconstruction_head = nn.Linear(decoder_dim, self.encoder.patch_dim)

        self.pooled_repr_dim = self.encoder.get_output_dim(pool=pool_mode)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_position_embedding, std=0.02)

    def forward(self, x: torch.Tensor):
        patch_values = self.encoder.patchify(x)
        batch_size, number_of_patches, _ = patch_values.shape
        visible_indices, restore_indices, mask = random_patch_mask(
            batch_size=batch_size,
            number_of_patches=number_of_patches,
            mask_ratio=self.mask_ratio,
            device=x.device,
        )

        tokens = self.encoder.embed_patch_values(patch_values)
        positions = self.encoder.position_tokens(number_of_patches).expand(
            batch_size, -1, -1
        )
        tokens = tokens + positions
        visible_tokens = torch.gather(
            tokens,
            dim=1,
            index=visible_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        encoded_visible = self.encoder.encode_token_embeddings(visible_tokens)

        decoded_visible = self.decoder_embed(encoded_visible)
        number_masked = number_of_patches - decoded_visible.shape[1]
        mask_tokens = self.mask_token.expand(batch_size, number_masked, -1)
        decoder_tokens = torch.cat([decoded_visible, mask_tokens], dim=1)
        decoder_tokens = torch.gather(
            decoder_tokens,
            dim=1,
            index=restore_indices.unsqueeze(-1).expand(
                -1, -1, decoder_tokens.shape[-1]
            ),
        )
        decoder_tokens = (
            decoder_tokens
            + self.decoder_position_embedding[:, :number_of_patches]
        )
        reconstructed_patches = self.reconstruction_head(
            self.decoder_norm(self.decoder(decoder_tokens))
        )

        patch_loss = (reconstructed_patches - patch_values).pow(2).mean(dim=-1)
        loss = (patch_loss * mask).sum() / mask.sum().clamp_min(1.0)

        return loss, reconstructed_patches, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x, pool=self.pool_mode)


def train_mae(
    X_train,
    device,
    trace_window,
    backbone_name: str = "trace_transformer_v1",
    pool_mode: str = "mean",
    patch_size: int = 10,
    mask_ratio: float = 0.75,
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
    decoder_dim: int = 128,
    decoder_depth: int = 2,
    decoder_heads: int = 4,
):
    X_train = ensure_trace_matrix(X_train)
    model = MAETransformer1D(
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
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            loss, _, _ = model(batch_x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            number_of_batches += 1

        average_loss = total_loss / max(number_of_batches, 1)
        loss_log.append(average_loss)
        print(f"Epoch #{epoch}: mae_loss={average_loss:.6f}")

    return model, loss_log
