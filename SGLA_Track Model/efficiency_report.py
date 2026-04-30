"""Compute efficiency metrics and S_eff penalty for SGLATrack model.

From roadmap.
"""

import torch
import time
import os
import tempfile
import argparse


def efficiency_report(model, checkpoint_path: str = None):
    """Compute comprehensive efficiency metrics for SGLATrack.

    Args:
        model: SGLATrack PyTorch model (should be in eval mode on GPU)
        checkpoint_path: If provided, save model weights to estimate file size

    Returns:
        Dict with efficiency metrics
    """
    try:
        from thop import profile
    except ImportError:
        print("Error: thop not installed. Install with:")
        print("  pip install thop")
        return None

    # Dummy inputs
    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)

    # Compute MACs (multiply-accumulate operations) and params
    print("Computing FLOPs and params...")
    macs, params = profile(model, inputs=(template, search), verbose=False)
    flops = macs * 2  # MACs to FLOPs

    # Estimate model file size
    print("Estimating model size...")
    if checkpoint_path:
        state_size_mb = os.path.getsize(checkpoint_path) / 1e6
    else:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            state_size_mb = os.path.getsize(tmp.name) / 1e6

    # GPU latency
    print("Benchmarking GPU latency...")
    model_gpu = model.cuda().eval()
    t_gpu = template.cuda()
    s_gpu = search.cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model_gpu(t_gpu, s_gpu)

    # Timing
    times = []
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(200):
            _ = model_gpu(t_gpu, s_gpu)
        gpu_ms = (time.perf_counter() - t0) / 200 * 1000

    # Jetson estimate
    jetson_ms = gpu_ms * 8.0  # Rough 8x slowdown

    # Compute S_eff penalty (from roadmap)
    # S_eff = 0.25*(params/50M) + 0.15*(flops/30G) + 0.35*(latency/30ms) + 0.25*(size/500MB)
    params_score = (params / 1e6) / 50.0
    flops_score  = (flops / 1e9) / 30.0
    latency_score = jetson_ms / 30.0
    size_score   = state_size_mb / 500.0

    s_eff = (0.25 * params_score +
             0.15 * flops_score +
             0.35 * latency_score +
             0.25 * size_score)

    # Score penalty
    score_penalty = 0.2 * s_eff

    print()
    print("=" * 60)
    print("EFFICIENCY REPORT")
    print("=" * 60)
    print(f"Params:         {params/1e6:.2f}M      / 50M cap")
    print(f"FLOPs:          {flops/1e9:.2f}G      / 30G cap")
    print(f"Model size:     {state_size_mb:.1f}MB    / 500MB cap")
    print(f"GPU latency:    {gpu_ms:.2f}ms")
    print(f"Jetson est:     {jetson_ms:.1f}ms    / 30ms cap")
    print()
    print("S_eff Computation:")
    print(f"  params score (0.25×):    {params_score:.4f}")
    print(f"  flops score  (0.15×):    {flops_score:.4f}")
    print(f"  latency score (0.35×):   {latency_score:.4f}")
    print(f"  size score   (0.25×):    {size_score:.4f}")
    print(f"  S_eff:                   {s_eff:.4f}")
    print()
    print(f"Final Score Penalty (0.2 × S_eff): {score_penalty:.4f}")
    print(f"This reduces final score by ~{score_penalty*100:.2f}%")
    print("=" * 60)

    return {
        'params_m': params / 1e6,
        'flops_g': flops / 1e9,
        'size_mb': state_size_mb,
        'gpu_latency_ms': gpu_ms,
        'jetson_latency_ms': jetson_ms,
        's_eff': s_eff,
        'score_penalty': score_penalty,
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path

    _HERE = Path(__file__).parent
    sys.path.insert(0, str(_HERE / 'SGLATrack'))

    from lib.models.sglatrack import build_sglatrack
    from lib.config.sglatrack.config import cfg, update_config_from_file

    parser = argparse.ArgumentParser(description='Compute SGLATrack efficiency metrics')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to SGLATrack checkpoint (for accurate size estimation)')
    parser.add_argument('--config', required=True,
                        help='Path to config YAML')
    args = parser.parse_args()

    update_config_from_file(args.config)
    model = build_sglatrack(cfg, training=False)

    efficiency_report(model, checkpoint_path=args.checkpoint)
