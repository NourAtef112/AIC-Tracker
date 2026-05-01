"""Convert competition data to GOT-10k format for OSTrack training.

Reads manifest (dict format: {split: {seq_key: {video_path, annotation_path, ...}}})
and extracts frames from MP4 directly into comp_dir with 1-indexed filenames.

Output structure (matches CompDataset in lib/train/dataset/comp.py):
  out_root/<dataset_name>/<seq_name>/img/00000001.jpg ... 00000NNN.jpg
  out_root/<dataset_name>/<seq_name>/groundtruth.txt
"""

import os
import json
import argparse
import cv2
from pathlib import Path


def convert(comp_root: str, out_root: str, split: str = 'train', skip_small: bool = True):
    manifest_path = os.path.join(comp_root, 'data', 'metadata', 'contestant_manifest.json')
    if not os.path.exists(manifest_path):
        print(f"Error: manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    sequences = manifest.get(split, {})
    print(f"Found {len(sequences)} sequences for split '{split}'")
    os.makedirs(out_root, exist_ok=True)
    converted, skipped = 0, 0

    for seq_key, info in sequences.items():
        dataset_name = info['dataset']   # e.g. "dataset1"
        seq_name = info['seq_name']      # e.g. "Car_video_2"
        video_path = os.path.join(comp_root, 'data', info['video_path'])
        ann_path = os.path.join(comp_root, 'data', info['annotation_path'])

        if not os.path.exists(ann_path):
            print(f"  [skip] No annotation: {seq_key}")
            skipped += 1
            continue

        with open(ann_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < 10 and skip_small:
            print(f"  [skip] Too few frames ({len(lines)}): {seq_key}")
            skipped += 1
            continue

        if not os.path.exists(video_path):
            print(f"  [skip] No video: {video_path}")
            skipped += 1
            continue

        out_seq = os.path.join(out_root, dataset_name, seq_name)
        img_dir = os.path.join(out_seq, 'img')

        # Skip if already extracted
        existing = list(Path(img_dir).glob('*.jpg')) if os.path.exists(img_dir) else []
        if existing:
            print(f"  [skip] Already extracted ({len(existing)} frames): {seq_key}")
            converted += 1
            continue

        os.makedirs(img_dir, exist_ok=True)

        # Extract frames from MP4 — 1-indexed to match comp.py frame_id+1 convention
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [skip] Cannot open video: {video_path}")
            skipped += 1
            continue

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(img_dir, f'{frame_idx:08d}.jpg'), frame)
            frame_idx += 1
        cap.release()

        # Write groundtruth.txt — skip fully-occluded frames (0,0,0,0)
        gt = []
        for line in lines:
            # Handle both comma-separated and space-separated annotation formats
            parts = line.split(',') if ',' in line else line.split()
            vals = list(map(float, parts))
            if len(vals) == 4 and vals[2] > 0 and vals[3] > 0:
                gt.append(f"{vals[0]:.1f},{vals[1]:.1f},{vals[2]:.1f},{vals[3]:.1f}")
            else:
                gt.append(f"0.0,0.0,0.0,0.0")

        with open(os.path.join(out_seq, 'groundtruth.txt'), 'w') as f:
            f.write('\n'.join(gt))

        print(f"  [OK] {seq_key:45s} | {frame_idx-1:4d} frames | {sum(1 for g in gt if g != '0.0,0.0,0.0,0.0'):4d} visible")
        converted += 1

    print(f"\nSummary: {converted} converted/skipped-existing, {skipped} skipped-errors")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert competition data to GOT-10k format')
    parser.add_argument('--comp_root', required=True,
                        help='Competition data root (contains data/ and data/metadata/contestant_manifest.json)')
    parser.add_argument('--out_root', required=True,
                        help='Output root for converted data (comp_dir)')
    parser.add_argument('--split', default='train', choices=['train', 'public_lb'],
                        help='Data split to convert')
    args = parser.parse_args()

    convert(args.comp_root, args.out_root, args.split)
