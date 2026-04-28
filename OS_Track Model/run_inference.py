import sys
import types
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / 'OSTrack'))
sys.path.insert(0, str(_HERE.parent / 'shared'))
sys.path.insert(0, str(_HERE))  # for hanning_penalty, dual_template, multi_scale_inference

try:
    from lib.test.tracker.ostrack import OSTrack
    from lib.config.ostrack.config import cfg, update_config_from_file
    from lib.train.data.processing_utils import sample_target
except ImportError as e:
    raise ImportError(
        f"Could not import OSTrack: {e}. "
        "Clone the OSTrack repo into 'OS_Track Model/OSTrack/' first:\n"
        "  git clone https://github.com/botaoye/OSTrack.git 'OS_Track Model/OSTrack'"
    ) from e

try:
    from manifest_loader import load_manifest
    from annotation_utils import load_annotations, get_init_box
    from prediction_writer import save_predictions
except ImportError as e:
    print(f"Import error: {e}. Check that shared directory contains required modules.")
    raise

from dual_template import DualTemplateManager
from multi_scale_inference import multi_scale_track


def build_tracker(args) -> OSTrack:
    update_config_from_file(args.config)
    params = types.SimpleNamespace(
        cfg=cfg,
        checkpoint=args.checkpoint,
        search_factor=4.0,
        template_factor=2.0,
        search_size=256,
        template_size=128,
        save_all_boxes=False,
        debug=0,
        use_hanning=args.use_hanning,
        hanning_weight=args.hanning_weight,
    )
    return OSTrack(params, 'otb')


def run_sequence(tracker: OSTrack, seq_info, frames_root: Path, args) -> list:
    seq_frames_dir = frames_root / seq_info.key
    frame_paths = sorted(seq_frames_dir.glob('*.jpg'))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {seq_frames_dir}")

    annotations = load_annotations(seq_info.annotation_path)
    init_frame_idx, init_box = get_init_box(annotations)

    image = cv2.cvtColor(cv2.imread(str(frame_paths[init_frame_idx])), cv2.COLOR_BGR2RGB)
    tracker.initialize(image, {'init_bbox': list(init_box)})

    template_mgr = None
    if args.use_dual_template:
        template_mgr = DualTemplateManager()
        template_mgr.initialize(tracker.z_dict1.tensors)

    # Frames before the first visible frame get the init box as a placeholder
    pred_boxes = [init_box] * (init_frame_idx + 1)

    for frame_path in frame_paths[init_frame_idx + 1:]:
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)

        if args.use_multi_scale:
            pred_box, confidence = multi_scale_track(tracker, image)
        else:
            out = tracker.track(image)
            pred_box = tuple(out['target_bbox'])
            confidence = out.get('best_score', 0.0)

        if template_mgr is not None:
            z_crop, _, z_amask = sample_target(
                image, list(pred_box),
                tracker.params.template_factor,
                output_sz=tracker.params.template_size,
            )
            new_feat = tracker.preprocessor.process(z_crop, z_amask).tensors
            template_mgr.update(new_feat, confidence)
            tracker.z_dict1.tensors = template_mgr.get_fused()

        pred_boxes.append(pred_box)

    return pred_boxes


def main():
    parser = argparse.ArgumentParser(description='OSTrack inference — MTC-AIC4')
    parser.add_argument('--manifest', default='contestant_manifest.json')
    parser.add_argument('--split', default='train', choices=['train', 'public_lb'])
    parser.add_argument('--frames_root', default='frames')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth.tar checkpoint (key: net)')
    parser.add_argument('--config', required=True,
                        help='Path to OSTrack YAML config, e.g. OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml')
    parser.add_argument('--output_dir', default='predictions/ostrack_baseline')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Limit sequences processed (quick sanity check)')
    # post-processing flags
    parser.add_argument('--use_hanning', action='store_true',
                        help='Replace fixed hann window with configurable weighted blend')
    parser.add_argument('--hanning_weight', type=float, default=0.49,
                        help='Hanning blend weight [0–1]; 0=no penalty, 1=full (default 0.49)')
    parser.add_argument('--use_dual_template', action='store_true',
                        help='Enable static+dynamic template fusion (updates every 5 frames)')
    parser.add_argument('--use_multi_scale', action='store_true',
                        help='Enable multi-scale search at [0.96, 1.0, 1.04]× (Day 5+; adds ~3–5ms/frame)')
    args = parser.parse_args()

    sequences = load_manifest(args.manifest, args.split)
    if args.max_seqs:
        sequences = sequences[:args.max_seqs]

    frames_root = Path(args.frames_root) / args.split

    print(f"Building OSTrack tracker from {args.checkpoint}")
    tracker = build_tracker(args)

    flags = []
    if args.use_hanning:
        flags.append(f"hanning(w={args.hanning_weight})")
    if args.use_dual_template:
        flags.append("dual_template")
    if args.use_multi_scale:
        flags.append("multi_scale")
    flag_str = ", ".join(flags) if flags else "baseline"
    print(f"Post-processing: {flag_str}")
    print(f"Running on {len(sequences)} sequences | split={args.split}")

    failed = []
    for i, seq_info in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] {seq_info.key} ({seq_info.n_frames} frames)", end=' ', flush=True)
        try:
            pred_boxes = run_sequence(tracker, seq_info, frames_root, args)
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
