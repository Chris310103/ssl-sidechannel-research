import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_zoo import build_backbone
from src.utils.trace_transforms import ensure_trace_matrix, make_trace_view


def contextual_contrastive_loss(
    context_1: torch.Tensor,
    context_2: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size = context_1.shape[0]
    if batch_size < 2:
        return context_1.new_tensor(0.0)

    representations = F.normalize(
        torch.cat([context_1, context_2], dim=0),
        dim=1,
    )
    similarities = representations @ representations.T / temperature
    diagonal_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=representations.device,
    )
    similarities = similarities.masked_fill(diagonal_mask, float("-inf"))
    positive_indices = (
        torch.arange(2 * batch_size, device=representations.device) + batch_size
    ) % (2 * batch_size)
    return F.cross_entropy(similarities, positive_indices)


class TemporalCrossViewModule(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        context_dim: int = 256,
        prediction_steps: int = 6,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        temperature: float = 0.2,
    ):
        super().__init__()
        if context_dim % transformer_heads != 0:
            raise ValueError("context_dim must be divisible by transformer_heads")

        self.prediction_steps = prediction_steps
        self.temperature = temperature
        self.input_projection = nn.Linear(feature_dim, context_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=transformer_heads,
            dim_feedforward=context_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_depth,
        )
        self.predictors = nn.ModuleList(
            [nn.Linear(context_dim, feature_dim) for _ in range(prediction_steps)]
        )
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.BatchNorm1d(context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, feature_dim // 2),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        sequence_length = source.shape[1]
        if sequence_length <= self.prediction_steps:
            raise ValueError(
                "TS-TCC temporal feature sequence is too short for "
                f"prediction_steps={self.prediction_steps}"
            )

        context_end = int(
            torch.randint(
                low=0,
                high=sequence_length - self.prediction_steps,
                size=(1,),
                device=source.device,
            ).item()
        )
        prefix = self.input_projection(source[:, : context_end + 1])
        context = self.context_encoder(prefix)[:, -1]

        temporal_loss = source.new_tensor(0.0)
        labels = torch.arange(source.shape[0], device=source.device)
        for step, predictor in enumerate(self.predictors, start=1):
            prediction = F.normalize(predictor(context), dim=1)
            future_target = F.normalize(target[:, context_end + step], dim=1)
            logits = future_target @ prediction.T / self.temperature
            temporal_loss = temporal_loss + F.cross_entropy(logits, labels)

        temporal_loss = temporal_loss / self.prediction_steps
        return temporal_loss, self.context_projector(context)


class TSTCC1D(nn.Module):
    """Temporal and contextual contrasting with a shared 1D CNN encoder."""

    def __init__(
        self,
        backbone_name: str = "shared_cnn_v1",
        pool_mode: str = "mean_max",
        input_length: int = 700,
        context_dim: int = 256,
        prediction_steps: int = 6,
        context_depth: int = 2,
        context_heads: int = 4,
        temporal_temperature: float = 0.2,
        contextual_temperature: float = 0.2,
        contextual_weight: float = 0.7,
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.contextual_temperature = contextual_temperature
        self.contextual_weight = contextual_weight
        self.encoder = build_backbone(
            model_name=backbone_name,
            input_channels=1,
            input_length=input_length,
        )
        temporal_dim = self.encoder.get_temporal_output_dim()
        self.temporal_contrast = TemporalCrossViewModule(
            feature_dim=temporal_dim,
            context_dim=context_dim,
            prediction_steps=prediction_steps,
            transformer_depth=context_depth,
            transformer_heads=context_heads,
            temperature=temporal_temperature,
        )
        self.pooled_repr_dim = self.encoder.get_output_dim(pool=pool_mode)

    def forward(self, weak_view: torch.Tensor, strong_view: torch.Tensor):
        weak_features = F.normalize(
            self.encoder.forward_features(weak_view), dim=-1
        )
        strong_features = F.normalize(
            self.encoder.forward_features(strong_view), dim=-1
        )
        temporal_weak, context_weak = self.temporal_contrast(
            weak_features, strong_features
        )
        temporal_strong, context_strong = self.temporal_contrast(
            strong_features, weak_features
        )
        context_loss = contextual_contrastive_loss(
            context_weak,
            context_strong,
            temperature=self.contextual_temperature,
        )
        total_loss = (
            temporal_weak
            + temporal_strong
            + self.contextual_weight * context_loss
        )
        return total_loss, temporal_weak + temporal_strong, context_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x, pool=self.pool_mode)


def train_ts_tcc(
    X_train,
    device,
    trace_window,
    backbone_name: str = "shared_cnn_v1",
    pool_mode: str = "mean_max",
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_shift: int = 5,
    noise_std: float = 0.05,
    denoise_kernel_size: int = 5,
    augmentation_family=("random_shift", "denoise", "gaussian_noise"),
    weak_augmentation_probability: float = 0.3,
    strong_augmentation_probability: float = 0.8,
    context_dim: int = 256,
    prediction_steps: int = 6,
    context_depth: int = 2,
    context_heads: int = 4,
    temporal_temperature: float = 0.2,
    contextual_temperature: float = 0.2,
    contextual_weight: float = 0.7,
    input_length: int = 700,
):
    X_train = ensure_trace_matrix(X_train)
    model = TSTCC1D(
        backbone_name=backbone_name,
        pool_mode=pool_mode,
        input_length=input_length,
        context_dim=context_dim,
        prediction_steps=prediction_steps,
        context_depth=context_depth,
        context_heads=context_heads,
        temporal_temperature=temporal_temperature,
        contextual_temperature=contextual_temperature,
        contextual_weight=contextual_weight,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_log = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        number_of_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            weak_view = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation="random",
                augmentation_family=augmentation_family,
                augmentation_probability=weak_augmentation_probability,
                max_shift=max(1, max_shift // 2),
                noise_std=noise_std * 0.5,
                denoise_kernel_size=denoise_kernel_size,
            )
            strong_view = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation="random",
                augmentation_family=augmentation_family,
                augmentation_probability=strong_augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )
            loss, _, _ = model(weak_view, strong_view)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            number_of_batches += 1

        average_loss = total_loss / max(number_of_batches, 1)
        loss_log.append(average_loss)
        print(f"Epoch #{epoch}: ts_tcc_loss={average_loss:.6f}")

    return model, loss_log
