"""Convert competition data to GOT-10k format for training.

Converts:
  - Competition format: dataset{N}/{seq_id}/video.mp4 + annotation.txt
  - Target format: GOT-10k style with extracted frames in img/ and groundtruth.txt

From roadmap.
"""

import os
import shutil
import json
import argparse
from pathlib import Path


def convert(comp_root: str, out_root: str, split: str = 'train', skip_small: bool = True):
    """Convert competition data to GOT-10k format.

    Args:
        comp_root: Root directory of competition data (contains contestant_manifest.json)
        out_root: Output root directory for converted data
        split: 'train' or 'public_lb'
        skip_small: Skip sequences < 10 frames or missing annotations
    """
    manifest_path = os.path.join(comp_root, 'data', 'metadata', 'contestant_manifest.json')
    if not os.path.exists(manifest_path):
        print(f"Error: manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    seq_list = [e for e in manifest if e.get('split') == split]
    print(f"Found {len(seq_list)} sequences for split '{split}'")

    os.makedirs(out_root, exist_ok=True)
    skipped = 0
    converted = 0

    for entry in seq_list:
        seq_id = entry['seq_id']  # e.g. "dataset1/car_chase3"
        seq_path = os.path.join(comp_root, 'data', seq_id)
        ann_file = os.path.join(seq_path, 'annotation.txt')

        # Check annotation exists
        if not os.path.exists(ann_file):
            print(f"  [skip] No annotation: {seq_id}")
            skipped += 1
            continue

        # Load annotations
        with open(ann_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        # Skip if too few frames
        if len(lines) < 10 and skip_small:
            print(f"  [skip] Too few frames ({len(lines)}): {seq_id}")
            skipped += 1
            continue

        # Output folder: flat name to avoid nested dirs
        flat_name = seq_id.replace('/', '_')
        out_seq = os.path.join(out_root, flat_name)
        os.makedirs(out_seq, exist_ok=True)

        # Copy frames from 'img' subfolder
        img_src = os.path.join(seq_path, 'img')
        if os.path.exists(img_src):
            img_dst = os.path.join(out_seq, 'img')
            if os.path.exists(img_dst):
                shutil.rmtree(img_dst)
            shutil.copytree(img_src, img_dst)
        else:
            # Frames may be directly in seq_path — collect all image files
            img_files = [f for f in os.listdir(seq_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not img_files:
                print(f"  [skip] No frames found: {seq_id}")
                skipped += 1
                continue
            os.makedirs(os.path.join(out_seq, 'img'), exist_ok=True)
            for img_file in img_files:
                shutil.copy(os.path.join(seq_path, img_file),
                           os.path.join(out_seq, 'img', img_file))

        # Write groundtruth.txt — skip invisible frames (0,0,0,0)
        gt = []
        for line in lines:
            vals = list(map(float, line.split(',')))
            if len(vals) != 4:
                print(f"  [warn] Invalid annotation line in {seq_id}: {line}")
                continue
            x, y, w, h = vals
            if w > 0 and h > 0:
                gt.append(f"{x:.1f},{y:.1f},{w:.1f},{h:.1f}")

        with open(os.path.join(out_seq, 'groundtruth.txt'), 'w') as f:
            f.write('\n'.join(gt))

        print(f"  [OK] {seq_id:40s} | {len(lines):3d} frames | {len(gt):3d} visible")
        converted += 1

    print()
    print(f"Summary: {converted} converted, {skipped} skipped")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert competition data to GOT-10k format')
    parser.add_argument('--comp_root', required=True,
                        help='Competition data root (contains data/metadata/contestant_manifest.json)')
    parser.add_argument('--out_root', required=True,
                        help='Output root for converted GOT-10k style data')
    parser.add_argument('--split', default='train', choices=['train', 'public_lb'],
                        help='Data split to convert')
    args = parser.parse_args()

    convert(args.comp_root, args.out_root, args.split)
