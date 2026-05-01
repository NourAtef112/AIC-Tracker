import os
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(input_path: str, output_path: str):
    """Apply INT8 dynamic quantization to an ONNX model.

    Expected gains:
    - ~40% latency reduction
    - ~75% size reduction (FP32 → INT8)
    """
    orig_size_mb = os.path.getsize(input_path) / 1e6

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    quant_size_mb = os.path.getsize(output_path) / 1e6
    reduction_pct = (1 - quant_size_mb / orig_size_mb) * 100

    print(f"INT8 quantization complete: {output_path}")
    print(f"FP32: {orig_size_mb:.1f}MB → INT8: {quant_size_mb:.1f}MB  ({reduction_pct:.0f}% reduction)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Quantize ONNX model to INT8')
    parser.add_argument("--input", required=True, help='Input ONNX model (FP32)')
    parser.add_argument("--output", required=True, help='Output ONNX model (INT8)')
    args = parser.parse_args()
    quantize_model(args.input, args.output)
