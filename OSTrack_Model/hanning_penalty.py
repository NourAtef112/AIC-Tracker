import torch
import numpy as np


def create_hanning_window(output_sz: int, device: str = 'cuda') -> torch.Tensor:
    """Pre-compute 2D Hanning window. Call once, reuse across frames."""
    hanning_1d = np.hanning(output_sz)
    hanning_2d = np.outer(hanning_1d, hanning_1d)
    return torch.tensor(hanning_2d, dtype=torch.float32).to(device)


def apply_hanning_penalty(score_map: torch.Tensor,
                          hanning_window: torch.Tensor,
                          window_influence: float = 0.4) -> torch.Tensor:
    """
    Blend raw score_map with Hanning window.
    score_map should be sigmoid-activated before calling.
    window_influence: 0 = no penalty, 1 = full penalty (default 0.4)
    """
    return score_map * (1 - window_influence) + hanning_window * window_influence
