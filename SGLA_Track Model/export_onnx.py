import torch
import os
import argparse


def export_to_onnx(model, checkpoint_path: str, output_path: str,
                   template_size: int = 128, search_size: int = 256):
    """Export SGLATrack model to ONNX format.

    Args:
        model: SGLATrack model instance (PyTorch)
        checkpoint_path: Path to .pth checkpoint
        output_path: Path to save ONNX model
        template_size: Template input size (default 128)
        search_size: Search input size (default 256)
    """
    # Load checkpoint — try multiple key patterns
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if 'model' in state:
            state = state['model']
        elif 'net' in state:
            state = state['net']

    model.load_state_dict(state)
    model.eval()

    # Dummy inputs
    dummy_template = torch.randn(1, 3, template_size, template_size)
    dummy_search   = torch.randn(1, 3, search_size,   search_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_template, dummy_search),
        output_path,
        input_names=["template", "search"],
        output_names=["score_map", "size_map", "offset_map"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'template': {0: 'batch'},
            'search':   {0: 'batch'},
        }
    )

    onnx_size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported → {output_path}")
    print(f"Size: {onnx_size_mb:.1f} MB")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _HERE = Path(__file__).parent
    sys.path.insert(0, str(_HERE / 'SGLATrack'))

    from lib.models.sglatrack import build_sglatrack
    from lib.config.sglatrack.config import cfg, update_config_from_file

    parser = argparse.ArgumentParser(description='Export SGLATrack to ONNX')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to SGLATrack checkpoint (.pth or .pth.tar)')
    parser.add_argument('--config', required=True,
                        help='Path to config YAML (e.g., experiments/sglatrack/deit_distilled.yaml)')
    parser.add_argument('--output', default='sglatrack.onnx',
                        help='Output ONNX model path')
    parser.add_argument('--template_size', type=int, default=128)
    parser.add_argument('--search_size', type=int, default=256)
    args = parser.parse_args()

    update_config_from_file(args.config)
    model = build_sglatrack(cfg, training=False)

    export_to_onnx(model, args.checkpoint, args.output,
                   args.template_size, args.search_size)
