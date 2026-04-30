import sys
import types
import argparse
import csv
import os
import cv2
import torch
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / 'SGLATrack'))
sys.path.insert(0, str(_HERE.parent / 'shared'))

try:
    from lib.config.sglatrack.config import cfg, update_config_from_file
except ImportError:
    print("Error: SGLATrack not cloned. Expected at: SGLA_Track Model/SGLATrack/")
    print("Run: git clone https://github.com/GXNU-ZhongLab/SGLATrack.git 'SGLA_Track Model/SGLATrack/'")
    sys.exit(1)

try:
    from sglatrack_ext import SGLATrackExt
except ImportError as e:
    print(f"Import error: {e}. Ensure sglatrack_ext.py exists in SGLA_Track Model/")
    raise

try:
    from manifest_loader import load_manifest
    from annotation_utils import load_annotations, get_init_box
    from prediction_writer import save_predictions
except ImportError as e:
    print(f"Import error: {e}. Check that shared directory contains required modules.")
    raise


def build_tracker(checkpoint_path: str, config_path: str, hanning_weight: float = 0.49):
    """Build SGLATrackExt tracker with configurable Hanning weight.

    Args:
        checkpoint_path: Path to fine-tuned SGLATrack checkpoint (.pth.tar)
        config_path: Path to config YAML
        hanning_weight: Hanning blend weight [0, 1]

    Returns:
        SGLATrackExt instance ready for initialize()/track()
    """
    update_config_from_file(config_path)

    params = types.SimpleNamespace(
        cfg=cfg,
        checkpoint=checkpoint_path,
        search_factor=cfg.TEST.SEARCH_FACTOR,
        template_factor=cfg.TEST.TEMPLATE_FACTOR,
        search_size=cfg.TEST.SEARCH_SIZE,
        template_size=cfg.TEST.TEMPLATE_SIZE,
        save_all_boxes=False,
        debug=0,
        hanning_weight=hanning_weight,
    )
    return SGLATrackExt(params, 'aic')


def run_sequence(tracker, seq_info, frames_root: Path) -> list:
    """Run tracker on a single sequence.

    Args:
        tracker: SGLATrackExt instance
        seq_info: SequenceInfo with key, annotation_path, n_frames
        frames_root: Root directory containing extracted frames

    Returns:
        List of (x, y, w, h) predictions
    """
    seq_frames_dir = frames_root / seq_info.key
    frame_paths = sorted(seq_frames_dir.glob('*.jpg'))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {seq_frames_dir}")

    annotations = load_annotations(seq_info.annotation_path)
    init_frame_idx, init_box = get_init_box(annotations)

    image = cv2.cvtColor(cv2.imread(str(frame_paths[init_frame_idx])), cv2.COLOR_BGR2RGB)
    tracker.initialize(image, {'init_bbox': list(init_box)})

    # Frames before first visible frame get init box as placeholder
    pred_boxes = [init_box] * (init_frame_idx + 1)

    for frame_path in frame_paths[init_frame_idx + 1:]:
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        out = tracker.track(image)
        pred_boxes.append(tuple(out['target_bbox']))

    return pred_boxes


def generate_submission_csv(tracker, manifest_path: str, frames_root: str, output_csv: str):
    """Generate full submission CSV over all public_lb sequences.

    CSV format: id, x, y, w, h  where id = "{seq_key}_{frame_idx}"
    """
    sequences = load_manifest(manifest_path, 'public_lb')
    frames_root_path = Path(frames_root) / 'public_lb'

    rows = []
    failed = []

    print(f"Generating predictions for {len(sequences)} sequences...")
    for i, seq_info in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] {seq_info.key}", end=' ', flush=True)
        try:
            pred_boxes = run_sequence(tracker, seq_info, frames_root_path)
            print(f"OK ({len(pred_boxes)} frames)")
            for frame_idx, box in enumerate(pred_boxes):
                x, y, w, h = box
                rows.append({
                    'id': f"{seq_info.key}_{frame_idx}",
                    'x': round(x, 2), 'y': round(y, 2),
                    'w': round(w, 2), 'h': round(h, 2),
                })
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(seq_info.key)

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x', 'y', 'w', 'h'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written: {output_csv}")
    print(f"Total rows: {len(rows)} (expected: 74,293)")
    print(f"Failed sequences: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description='SGLATrack inference — MTC-AIC4')
    parser.add_argument('--manifest', default='data/metadata/contestant_manifest.json')
    parser.add_argument('--split', default='public_lb', choices=['train', 'public_lb'])
    parser.add_argument('--frames_root', default='frames')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to fine-tuned SGLATrack checkpoint (.pth.tar)')
    parser.add_argument('--config',
                        default='SGLA_Track Model/SGLATrack/experiments/sglatrack/deit_distilled.yaml',
                        help='Path to SGLATrack config YAML')
    parser.add_argument('--hanning_weight', type=float, default=0.49,
                        help='Hanning window blend weight [0, 1] (default 0.49)')
    parser.add_argument('--output_dir', default='predictions/sglatrack')
    parser.add_argument('--output_csv', default=None,
                        help='If provided, generate full submission CSV at this path')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Limit sequences processed (quick sanity check)')
    args = parser.parse_args()

    print(f"Building SGLATrackExt from {args.checkpoint} (hanning_weight={args.hanning_weight})")
    tracker = build_tracker(args.checkpoint, args.config, args.hanning_weight)

    if args.output_csv:
        generate_submission_csv(tracker, args.manifest, args.frames_root, args.output_csv)
    else:
        sequences = load_manifest(args.manifest, args.split)
        if args.max_seqs:
            sequences = sequences[:args.max_seqs]

        frames_root = Path(args.frames_root) / args.split
        print(f"Running on {len(sequences)} sequences | split={args.split}")

        failed = []
        for i, seq_info in enumerate(sequences):
            print(f"  [{i+1}/{len(sequences)}] {seq_info.key}", end=' ', flush=True)
            try:
                pred_boxes = run_sequence(tracker, seq_info, frames_root)
                save_predictions(pred_boxes, seq_info.key, args.output_dir)
                print(f"-> {len(pred_boxes)} predictions saved")
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(seq_info.key)

        print(f"\nDone. {len(sequences) - len(failed)}/{len(sequences)} sequences succeeded.")
        if failed:
            print(f"Failed: {failed}")
        print(f"\nNext step — evaluate:")
        print(f"  python shared/local_eval.py --pred_dir {args.output_dir} --gt_dir frames/{args.split}")


if __name__ == '__main__':
    main()
