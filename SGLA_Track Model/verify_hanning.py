"""Verify Hanning window penalty implementation.

This script tests that the Hanning penalty correctly shifts response map peaks
toward the center of the search region, mitigating distractor responses.

From roadmap Step 5.
"""

import torch
import numpy as np


def build_hanning_window(h: int, w: int) -> torch.Tensor:
    """Create normalized Hanning cosine window.

    Args:
        h: Height
        w: Width

    Returns:
        Tensor of shape (h, w) with values in [0, 1]
    """
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    window = np.outer(win_h, win_w)
    window = window / window.max()  # Normalize to [0, 1]
    return torch.tensor(window, dtype=torch.float32)


def apply_hanning_penalty(score_map: torch.Tensor, lam: float) -> torch.Tensor:
    """Apply Hanning penalty to score map.

    Args:
        score_map: Tensor of shape (1, H, W) or (H, W)
        lam: Penalty weight [0, 1]. 0 = no penalty, 1 = full window multiplication

    Returns:
        Penalized score map (same shape as input)
    """
    H, W = score_map.shape[-2], score_map.shape[-1]
    win = build_hanning_window(H, W)

    # Multiplicative form: suppresses edges proportionally
    penalized = score_map * ((1.0 - lam) + lam * win)
    return penalized


def verify():
    """Test Hanning penalty on synthetic response maps."""
    H, W = 16, 16

    # Synthetic response map with a strong off-center peak (distractor)
    # and a weaker center peak (real target)
    score_map = torch.zeros(1, H, W)
    score_map[0, 2, 3]   = 1.0   # Off-center peak (strong)
    score_map[0, 8, 8]   = 0.7   # Center peak (weaker)

    lam = 0.49

    # Apply penalty
    penalized = apply_hanning_penalty(score_map, lam)

    # Check argmax before/after
    argmax_before = score_map.argmax().item()
    argmax_after = penalized.argmax().item()

    pos_before = divmod(argmax_before, W)
    pos_after = divmod(argmax_after, W)

    print(f"Hanning Penalty Verification (lam={lam})")
    print("=" * 50)
    print(f"Score map shape: {score_map.shape}")
    print(f"Peak positions:")
    print(f"  Before penalty: argmax={argmax_before:3d} --> pos={pos_before} (off-center)")
    print(f"  After  penalty: argmax={argmax_after:3d} --> pos={pos_after}")
    print()

    if pos_after == (8, 8):
        print("[PASS] SUCCESS: Penalty shifted peak to center")
        return True
    else:
        print(f"[FAIL] Peak did not shift to center (8, 8)")
        print(f"  Got: {pos_after}")
        return False


if __name__ == '__main__':
    success = verify()
    exit(0 if success else 1)
