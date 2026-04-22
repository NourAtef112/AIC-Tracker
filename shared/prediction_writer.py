# prediction_writer.py
# Owner: Yassin
import os
import csv
from pathlib import Path

def save_predictions_txt(pred_boxes, seq_key: str, output_dir: str = "predictions"):
    """
    Saves tracker predictions as annotation.txt format for LOCAL EVALUATION.
    pred_boxes: list of (x, y, w, h) tuples, one per frame.
    Lost/invisible targets should be (0, 0, 0, 0).
    """
    out_path = Path(output_dir) / (seq_key.replace("/", "_") + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        for box in pred_boxes:
            # This forces it to write exactly x,y,w,h without parentheses
            f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
    return str(out_path)    

def generate_submission_csv(all_predictions: dict, output_csv: str = "submission/sample_submission.csv"):
    """
    Generates the final Kaggle-format CSV for the public_lb split.
    all_predictions: dict mapping seq_key -> list of (x, y, w, h)
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        
        for seq_key, boxes in all_predictions.items():
            for frame_idx, box in enumerate(boxes):
                row_id = f"{seq_key}_{frame_idx}"
                writer.writerow([row_id, box[0], box[1], box[2], box[3]])
                
    return str(out_path)