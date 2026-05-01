"""Benchmark ONNX model latency on GPU and estimate Jetson performance.

From roadmap.
"""

import time
import numpy as np
import argparse


def benchmark_latency(onnx_path: str, n_runs: int = 200, warmup: int = 20):
    """Benchmark ONNX model latency.

    Args:
        onnx_path: Path to ONNX model file
        n_runs: Number of forward passes to time
        warmup: Warmup runs before timing

    Returns:
        Tuple: (gpu_latency_ms, jetson_estimate_ms)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Install with:")
        print("  pip install onnxruntime-gpu")
        return None, None

    # Create ONNX session
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # Dummy inputs
    dummy_input = {
        'template': np.random.randn(1, 3, 128, 128).astype(np.float32),
        'search':   np.random.randn(1, 3, 256, 256).astype(np.float32),
    }

    # Warmup
    print(f"Warming up ({warmup} runs)...", end=' ', flush=True)
    for _ in range(warmup):
        sess.run(None, dummy_input)
    print("done")

    # Timing
    print(f"Benchmarking ({n_runs} runs)...", end=' ', flush=True)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, dummy_input)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    print("done")

    # Stats
    gpu_ms = np.mean(times)
    gpu_ms_std = np.std(times)
    gpu_ms_p95 = np.percentile(times, 95)

    # Rough Jetson estimate: Orin Nano is ~8-10x slower than GPU
    jetson_factor = 8.0
    jetson_est_ms = gpu_ms * jetson_factor
    jetson_cap_ms = 30.0

    print()
    print("=" * 50)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model: {onnx_path}")
    print()
    print("GPU Latency (RTX-class):")
    print(f"  Mean:     {gpu_ms:.2f}ms")
    print(f"  Std:      {gpu_ms_std:.2f}ms")
    print(f"  P95:      {gpu_ms_p95:.2f}ms")
    print()
    print(f"Jetson Orin Nano estimate ({jetson_factor}x slowdown):")
    print(f"  Est. latency: {jetson_est_ms:.1f}ms / {jetson_cap_ms:.0f}ms cap")
    if jetson_est_ms < jetson_cap_ms:
        print(f"  ✓ PASS — within cap")
    else:
        print(f"  ✗ BORDERLINE / FAIL — exceeds cap")
        print(f"    Recommendation: INT8 quantization is MANDATORY to hit cap")
    print("=" * 50)

    return gpu_ms, jetson_est_ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark ONNX model latency')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--n_runs', type=int, default=200, help='Number of forward passes')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup runs')
    args = parser.parse_args()

    benchmark_latency(args.model, args.n_runs, args.warmup)
