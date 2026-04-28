import json
import sys
from pathlib import Path

# Ensure Python can find your shared utilities
sys.path.append(".")
from shared.annotation_utils import load_annotations
from shared.prediction_writer import generate_submission_csv

def create_kaggle_submission():
    # 1. Paths (You might adjust pred_dir on Day 6 based on where Ahmed saves them)
    manifest_path = "data/metadata/contestant_manifest.json"
    pred_dir = Path("predictions/ostrack/final_public_lb") # Folder with Day 6 .txt files
    output_csv = "submission/team_a_final_submission.csv"
    
    # 2. Load the manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    # We only want the public leaderboard split for Kaggle!
    public_lb_seqs = manifest.get("public_lb", {})
    
    all_predictions = {}
    missing_files = []
    
    # 3. Read every .txt file and add it to the dictionary
    for seq_key in public_lb_seqs.keys():
        txt_filename = seq_key.replace("/", "_") + ".txt"
        txt_filepath = pred_dir / txt_filename
        
        if txt_filepath.exists():
            # Uses Ahmed's utility to turn the .txt into a list of tuples
            boxes = load_annotations(str(txt_filepath))
            all_predictions[seq_key] = boxes
        else:
            missing_files.append(seq_key)
            
    # 4. Generate the final CSV!
    if missing_files:
        print(f"⚠️ WARNING: You are missing .txt files for {len(missing_files)} sequences!")
        print(f"Missing examples: {missing_files[:3]}")
    else:
        final_csv_path = generate_submission_csv(all_predictions, output_csv)
        print(f"✅ SUCCESS: Kaggle submission successfully saved to {final_csv_path}")

if __name__ == "__main__":
    create_kaggle_submission()