"""Verify that the exported ONNX model matches the PyTorch model output.

Compares score_map output of both models on identical random inputs.
Max absolute difference must be < threshold (default 0.01) to pass.

Usage:
    python "SGLA_Track Model/verify_onnx.py"
        --onnx      sglatrack_final.onnx
        --checkpoint "SGLA_Track Model/output/checkpoint_best.pth.tar"
        --config    "SGLA_Track Model/SGLATrack/experiments/sglatrack/deit_distilled.yaml"
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / 'SGLATrack'))


def verify(onnx_path: str, checkpoint_path: str, config_path: str,
           template_size: int = 128, search_size: int = 256,
           threshold: float = 0.01) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    from lib.models.sglatrack import build_sglatrack
    from lib.config.sglatrack.config import cfg, update_config_from_file

    print(f"ONNX model:      {onnx_path}")
    print(f"PyTorch ckpt:    {checkpoint_path}")
    print(f"Config:          {config_path}")
    print()

    # Load PyTorch model
    update_config_from_file(config_path)
    model = build_sglatrack(cfg, training=False)

    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict):
        if 'model' in state:
            state = state['model']
        elif 'net' in state:
            state = state['net']
    model.load_state_dict(state)
    model.eval()

    # Load ONNX model (CPU to match PyTorch for fair comparison)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Identical dummy inputs
    torch.manual_seed(42)
    template = torch.randn(1, 3, template_size, template_size)
    search   = torch.randn(1, 3, search_size,   search_size)

    # PyTorch forward
    with torch.no_grad():
        pt_out = model(template, search)

    if isinstance(pt_out, dict):
        pt_score = pt_out['score_map'].numpy()
    elif isinstance(pt_out, (list, tuple)):
        pt_score = pt_out[0].numpy()
    else:
        pt_score = pt_out.numpy()

    # ONNX forward
    onnx_out = sess.run(None, {
        'template': template.numpy(),
        'search':   search.numpy(),
    })
    onnx_score = onnx_out[0]

    diff = float(np.abs(onnx_score - pt_score).max())
    print(f"Max absolute difference (score_map): {diff:.6f}")
    print(f"Threshold: {threshold}")

    if diff < threshold:
        print(f"✅ ONNX verification PASSED")
        return True
    else:
        print(f"❌ ONNX verification FAILED — diff {diff:.6f} >= {threshold}")
        print("   Try re-exporting with export_onnx.py, or check opset version.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify ONNX model matches PyTorch checkpoint')
    parser.add_argument('--onnx',       required=True, help='Path to ONNX model')
    parser.add_argument('--checkpoint', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--config',     required=True, help='Path to config YAML')
    parser.add_argument('--threshold',  type=float, default=0.01,
                        help='Max allowed absolute diff (default 0.01)')
    parser.add_argument('--template_size', type=int, default=128)
    parser.add_argument('--search_size',   type=int, default=256)
    args = parser.parse_args()

    ok = verify(args.onnx, args.checkpoint, args.config,
                args.template_size, args.search_size, args.threshold)
    sys.exit(0 if ok else 1)
