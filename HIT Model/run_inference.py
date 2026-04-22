import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / 'LightTrack'))
sys.path.insert(0, str(_HERE.parent / 'shared'))

try:
    from lib.tracker.lighttrack import Lighttrack
except ImportError:
    from lighttrack import Lighttrack

try:
    from manifest_loader import load_manifest
    from annotation_utils import load_annotations, get_init_box
    from prediction_writer import save_predictions
except ImportError as e:
    print(f"Import error: {e}. Check that shared directory contains required modules.")
    raise


def build_model_and_tracker(checkpoint_path: str):
    # LightTrack separates the neural network (model) from the tracker logic
    try:
        from models.lighttrack import LightTrackNet
    except ImportError:
        from lighttrack import LightTrackNet
    model = LightTrackNet()
    state = torch.load(checkpoint_path, map_location='cpu')
    # Checkpoint may be a bare state_dict or wrapped under a key
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'net' in state:
        state = state['net']
    model.load_state_dict(state)
    model.cuda().eval()

    tracker_cfg = {
        'search_factor': 4.0,
        'template_factor': 2.0,
        'search_size': 256,
        'template_size': 128,
        'penalty_k': 0.04,
        'window_influence': 0.44,
        'lr': 0.33,
    }
    tracker = Lighttrack(tracker_cfg, even=0)
    return model, tracker


def run_sequence(model, tracker: Lighttrack, seq_info, frames_root: Path) -> list:
    seq_frames_dir = frames_root / seq_info.key
    frame_paths = sorted(seq_frames_dir.glob('*.jpg'))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {seq_frames_dir}")

    annotations = load_annotations(seq_info.annotation_path)
    init_frame_idx, init_box = get_init_box(annotations)
    x, y, w, h = init_box

    # LightTrack uses center-based coordinates internally
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])

    image = cv2.cvtColor(cv2.imread(str(frame_paths[init_frame_idx])), cv2.COLOR_BGR2RGB)
    state = tracker.init(image, target_pos, target_sz, model)

    # Frames before the first visible frame get the init box as a placeholder
    pred_boxes = [init_box] * (init_frame_idx + 1)

    for frame_path in frame_paths[init_frame_idx + 1:]:
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        state = tracker.track(state, image)
        cx, cy = state['target_pos']
        bw, bh = state['target_sz']
        # Convert center-based → top-left [x, y, w, h]
        pred_boxes.append((cx - bw / 2, cy - bh / 2, bw, bh))

    return pred_boxes


def main():
    parser = argparse.ArgumentParser(description='LightTrack baseline inference — MTC-AIC4 Day 1')
    parser.add_argument('--manifest', default='contestant_manifest.json')
    parser.add_argument('--split', default='train', choices=['train', 'public_lb'])
    parser.add_argument('--frames_root', default='frames')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to LightTrack-Mobile.pth checkpoint')
    parser.add_argument('--output_dir', default='predictions/lighttrack_baseline')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Limit sequences processed (quick sanity check)')
    args = parser.parse_args()

    sequences = load_manifest(args.manifest, args.split)
    if args.max_seqs:
        sequences = sequences[:args.max_seqs]

    frames_root = Path(args.frames_root) / args.split

    print(f"Building LightTrack model from {args.checkpoint}")
    model, tracker = build_model_and_tracker(args.checkpoint)

    print(f"Running on {len(sequences)} sequences | split={args.split}")
    failed = []
    for i, seq_info in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] {seq_info.key} ({seq_info.n_frames} frames)", end=' ', flush=True)
        try:
            pred_boxes = run_sequence(model, tracker, seq_info, frames_root)
            save_predictions(pred_boxes, seq_info.key, args.output_dir)
            print(f"-> {len(pred_boxes)} predictions saved")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(seq_info.key)

    print(f"\nDone. {len(sequences) - len(failed)}/{len(sequences)} sequences succeeded.")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nNext step — evaluate:")
    print(f"  python shared/evaluate.py --pred_dir {args.output_dir} "
        f"--manifest {args.manifest} --split {args.split} "
        f"--params_m <M> --flops_g <G> --latency_ms <ms> --size_gb <GB>")


if __name__ == '__main__':
    main()
