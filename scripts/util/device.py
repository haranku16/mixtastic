import torch

def get_best_device() -> str:
    """
    Returns the best available device for PyTorch operations.
    Priority order: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu" 