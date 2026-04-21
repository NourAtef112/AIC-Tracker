import os
from pathlib import Path

def save_predictions(pred_boxes, seq_key: str, output_dir: str = "predictions"):
    """
    Saves tracker predictions as annotation.txt format.
    pred_boxes: list of (x, y, w, h) tuples, one per frame.
    """
    out_path = Path(output_dir) / (seq_key.replace("/", "_") + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for box in pred_boxes:
            f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
    return str(out_path)
