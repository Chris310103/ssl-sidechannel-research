from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


VALID_AUGMENTATIONS = (
    "identity",
    "gaussian_noise",
    "denoise",
    "random_shift",
)

_AUGMENTATION_ALIASES = {
    "noise": "gaussian_noise",
    "shift": "random_shift",
}


def ensure_trace_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)

    if X.ndim == 2:
        return X

    if X.ndim == 3 and X.shape[-1] == 1:
        return X[..., 0]

    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]

    raise ValueError(
        "Expected traces as a matrix with one trace per row; "
        f"received shape {X.shape}"
    )


def ensure_batch_trace_matrix(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x

    if x.ndim == 3 and x.shape[-1] == 1:
        return x.squeeze(-1)

    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)

    raise ValueError(
        "Expected a trace batch with shape (batch, length), (batch, length, 1), "
        f"or (batch, 1, length); received {tuple(x.shape)}"
    )


def apply_trace_window(x: torch.Tensor, trace_window: Tuple[int, int]) -> torch.Tensor:
    if trace_window is None:
        return x

    window_start, window_end = trace_window

    if window_start < 0:
        raise ValueError("window_start must be non-negative")

    if window_end <= window_start:
        raise ValueError("window_end must be greater than window_start")

    if window_end > x.shape[1]:
        raise ValueError(
            f"window_end={window_end} exceeds trace length={x.shape[1]}"
        )

    return x[:, window_start:window_end]


def add_channel_dimension(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1)


def normalize_augmentation_name(augmentation: str) -> str:
    return _AUGMENTATION_ALIASES.get(augmentation, augmentation)


def parse_augmentation_family(augmentations: str) -> Tuple[str, ...]:
    family = tuple(
        normalize_augmentation_name(augmentation.strip())
        for augmentation in augmentations.split(",")
        if augmentation.strip()
    )

    if not family:
        raise ValueError("At least one augmentation must be provided")

    invalid = [
        augmentation
        for augmentation in family
        if augmentation not in VALID_AUGMENTATIONS
    ]
    if invalid:
        raise ValueError(
            f"Unsupported augmentation(s): {invalid}. "
            f"Supported values: {VALID_AUGMENTATIONS}"
        )

    return family


def random_shift(x: torch.Tensor, max_shift: int = 5) -> torch.Tensor:
    x = ensure_batch_trace_matrix(x)

    if max_shift < 0:
        raise ValueError(f"max_shift must be non-negative, got {max_shift}")

    if max_shift == 0:
        return x

    shifted = torch.zeros_like(x)
    shifts = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(x.shape[0],),
        device=x.device,
    )

    trace_length = x.shape[1]

    for index, shift in enumerate(shifts):
        shift = int(shift.item())

        if shift > 0:
            shifted[index, shift:] = x[index, : trace_length - shift]
        elif shift < 0:
            shifted[index, : trace_length + shift] = x[index, -shift:]
        else:
            shifted[index] = x[index]

    return shifted


def add_gaussian_noise(x: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    if noise_std <= 0:
        return x

    trace_std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(x) * trace_std * noise_std

    return x + noise


def denoise_moving_average(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    x = ensure_batch_trace_matrix(x)

    if kernel_size <= 1:
        return x

    if kernel_size % 2 == 0:
        raise ValueError("denoise_kernel_size must be odd")

    x_channels_first = x.unsqueeze(1)
    denoised = F.avg_pool1d(
        x_channels_first,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    return denoised.squeeze(1)


def apply_trace_augmentation(
    x: torch.Tensor,
    augmentation: str,
    max_shift: int = 5,
    noise_std: float = 0.05,
    denoise_kernel_size: int = 5,
) -> torch.Tensor:
    augmentation = normalize_augmentation_name(augmentation)
    x = ensure_batch_trace_matrix(x)

    if augmentation == "identity":
        return x

    if augmentation == "gaussian_noise":
        return add_gaussian_noise(x, noise_std=noise_std)

    if augmentation == "denoise":
        return denoise_moving_average(
            x,
            kernel_size=denoise_kernel_size,
        )

    if augmentation == "random_shift":
        return random_shift(
            x,
            max_shift=max_shift,
        )

    raise ValueError(f"Unsupported augmentation: {augmentation}")


def make_trace_view(
    x: torch.Tensor,
    trace_window,
    augmentation: str,
    augmentation_family: Iterable[str] = VALID_AUGMENTATIONS,
    augmentation_probability: float = 0.5,
    max_shift: int = 5,
    noise_std: float = 0.05,
    denoise_kernel_size: int = 5,
) -> torch.Tensor:
    x = ensure_batch_trace_matrix(x)
    augmentation = normalize_augmentation_name(augmentation)

    if augmentation == "random":
        augmentation_family = tuple(augmentation_family)
        if not augmentation_family:
            raise ValueError("At least one augmentation must be provided")

        if not 0.0 <= augmentation_probability <= 1.0:
            raise ValueError(
                "augmentation_probability must be in [0, 1], "
                f"got {augmentation_probability}"
            )

        view = x
        apply_mask = torch.rand(
            x.shape[0],
            len(augmentation_family),
            device=x.device,
        ) < augmentation_probability

        no_augmented_samples = ~torch.any(
            apply_mask,
            dim=1,
        )
        if torch.any(no_augmented_samples):
            row_indices = no_augmented_samples.nonzero(as_tuple=True)[0]
            fallback_choices = torch.randint(
                low=0,
                high=len(augmentation_family),
                size=(row_indices.shape[0],),
                device=x.device,
            )
            apply_mask[
                row_indices,
                fallback_choices,
            ] = True

        for choice_index, chosen_augmentation in enumerate(augmentation_family):
            mask = apply_mask[:, choice_index]
            if not torch.any(mask):
                continue

            view = view.clone()
            view[mask] = apply_trace_augmentation(
                view[mask],
                augmentation=chosen_augmentation,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )

        view = apply_trace_window(view, trace_window=trace_window)
        return add_channel_dimension(view)

    view = apply_trace_augmentation(
        x,
        augmentation=augmentation,
        max_shift=max_shift,
        noise_std=noise_std,
        denoise_kernel_size=denoise_kernel_size,
    )
    view = apply_trace_window(view, trace_window=trace_window)

    return add_channel_dimension(view)


def prepare_model_input(x: torch.Tensor, trace_window) -> torch.Tensor:
    x = ensure_batch_trace_matrix(x)
    x = apply_trace_window(x, trace_window=trace_window)
    return add_channel_dimension(x)
