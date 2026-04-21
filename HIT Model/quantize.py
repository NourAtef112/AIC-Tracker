from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path: str, output_path: str):
    """
    Applies INT8 quantization to an ONNX model.
    Expected gains: ~40% latency reduction, ~75% size reduction
    """
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    print(f"INT8 quantization complete: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantize_model(args.input, args.output)
