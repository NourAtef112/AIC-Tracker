import torch
import numpy as np

def apply_hanning_penalty(score_map: torch.Tensor,
                           penalty_weight: float = 0.49) -> torch.Tensor:
    """
    Suppresses high-displacement predictions.
    Discourages the tracker from jumping far from the previous position.
    penalty_weight: 0 = no penalty, 1 = full penalty (use 0.49 as default)
    """
    h, w    = score_map.shape[-2:]
    hanning = torch.tensor(
        np.outer(np.hanning(h), np.hanning(w)),
        dtype=score_map.dtype,
        device=score_map.device
    )
    return score_map * (1 - penalty_weight) + hanning * penalty_weight
