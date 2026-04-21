import torch

def export_to_onnx(model, checkpoint_path: str, output_path: str,
                   template_size: int = 128, search_size: int = 256):
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    dummy_template = torch.randn(1, 3, template_size, template_size)
    dummy_search   = torch.randn(1, 3, search_size,   search_size)

    torch.onnx.export(
        model,
        (dummy_template, dummy_search),
        output_path,
        input_names=["template", "search"],
        output_names=["pred_boxes", "score"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported → {output_path}")
