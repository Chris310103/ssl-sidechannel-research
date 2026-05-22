import torch

def get_device(prefer_mps: bool = False):
    if torch.cuda.is_available():
        return "cuda"

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"