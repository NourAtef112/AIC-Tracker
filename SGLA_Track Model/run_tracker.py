import sys
import types
import argparse
import csv
import os
import cv2
import numpy as np
import torch
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / 'SGLATrack'))
sys.path.insert(0, str(_HERE.parent / 'shared'))

# Shared utilities — always required
try:
    from manifest_loader import load_manifest
    from annotation_utils import load_annotations, get_init_box
    from prediction_writer import save_predictions
except ImportError as e:
    print(f"Import error: {e}. Check that shared directory contains required modules.")
    raise

# SGLATrack library imports are deferred to build_tracker() so that --onnx mode
# works even when the SGLATrack repo has environment issues (e.g. torch._six).


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_image(image: np.ndarray, cx: float, cy: float, crop_size: float,
                out_size: int) -> np.ndarray:
    """Crop a square region centered at (cx, cy) with side crop_size, resize to out_size."""
    h, w = image.shape[:2]
    x1 = int(round(cx - crop_size / 2))
    y1 = int(round(cy - crop_size / 2))
    x2 = int(round(cx + crop_size / 2))
    y2 = int(round(cy + crop_size / 2))

    # Pad if out of bounds
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if pad_l or pad_t or pad_r or pad_b:
        avg_color = image.mean(axis=(0, 1)).astype(np.uint8)
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=avg_color.tolist())
        x1 += pad_l; x2 += pad_l
        y1 += pad_t; y2 += pad_t

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → normalized float32 NCHW tensor."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


class OnnxTracker:
    """ONNX-runtime tracker with the same initialize()/track() interface as SGLATrackExt.

    Uses standard siamese tracker preprocessing (center-context crop):
      - Template: crop factor 2.0 → 128×128
      - Search:   crop factor 4.0 → 256×256
    Score decoding assumes the head outputs score_map (H×W), size_map (2×H×W),
    offset_map (2×H×W) — standard OSTrack/SGLATrack head convention.
    """

    TEMPLATE_SIZE  = 128
    SEARCH_SIZE    = 256
    TEMPLATE_FACTOR = 2.0
    SEARCH_FACTOR   = 4.0

    def __init__(self, onnx_path: str, hanning_weight: float = 0.49):
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime not installed. Run: pip install onnxruntime")
            sys.exit(1)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.hanning_weight = hanning_weight

        # Build Hanning window for search grid
        h = np.hanning(self.SEARCH_SIZE // 16)  # head output is typically 1/16 of input
        self._hanning = np.outer(h, h)
        self._hanning = self._hanning / self._hanning.max()

        self._template_input = None  # stored at initialize()
        self._state_box = None       # (x, y, w, h) in image coords

    def initialize(self, image: np.ndarray, init_dict: dict):
        """Initialize tracker with first frame and ground-truth box."""
        x, y, w, h = init_dict['init_bbox']
        cx, cy = x + w / 2, y + h / 2

        context = (w + h) / 2
        crop_size = np.sqrt((w + context) * (h + context)) * self.TEMPLATE_FACTOR

        crop = _crop_image(image, cx, cy, crop_size, self.TEMPLATE_SIZE)
        self._template_input = _preprocess(crop)
        self._state_box = (x, y, w, h)

    def track(self, image: np.ndarray) -> dict:
        """Track in next frame. Returns {'target_bbox': [x, y, w, h]}."""
        x, y, w, h = self._state_box
        cx, cy = x + w / 2, y + h / 2

        context = (w + h) / 2
        crop_size = np.sqrt((w + context) * (h + context)) * self.SEARCH_FACTOR

        search_crop = _crop_image(image, cx, cy, crop_size, self.SEARCH_SIZE)
        search_input = _preprocess(search_crop)

        # ONNX forward
        outs = self.sess.run(None, {
            'template': self._template_input,
            'search':   search_input,
        })
        score_map  = outs[0][0, 0]   # (H, W)
        size_map   = outs[1][0]      # (2, H, W) — normalized w/h
        offset_map = outs[2][0]      # (2, H, W) — sub-pixel offset

        # Resize Hanning to match score_map if needed
        grid_h, grid_w = score_map.shape
        if self._hanning.shape != (grid_h, grid_w):
            self._hanning = cv2.resize(
                np.outer(np.hanning(grid_h), np.hanning(grid_w)).astype(np.float32),
                (grid_w, grid_h)
            )
            self._hanning /= self._hanning.max()

        # Apply Hanning penalty
        penalized = score_map * ((1.0 - self.hanning_weight) +
                                  self.hanning_weight * self._hanning)

        # Decode peak location
        flat_idx   = int(penalized.argmax())
        peak_row   = flat_idx // grid_w
        peak_col   = flat_idx % grid_w

        # Offset within grid cell
        off_x = float(offset_map[0, peak_row, peak_col])
        off_y = float(offset_map[1, peak_row, peak_col])
        pred_cx_rel = (peak_col + 0.5 + off_x) / grid_w
        pred_cy_rel = (peak_row + 0.5 + off_y) / grid_h

        # Predicted size (normalized to search crop)
        pred_w_rel = float(size_map[0, peak_row, peak_col])
        pred_h_rel = float(size_map[1, peak_row, peak_col])

        # Map back to image coords
        pred_cx_img = cx + (pred_cx_rel - 0.5) * crop_size
        pred_cy_img = cy + (pred_cy_rel - 0.5) * crop_size
        pred_w_img  = pred_w_rel  * crop_size
        pred_h_img  = pred_h_rel  * crop_size

        new_box = (
            pred_cx_img - pred_w_img / 2,
            pred_cy_img - pred_h_img / 2,
            pred_w_img,
            pred_h_img,
        )
        self._state_box = new_box
        return {'target_bbox': list(new_box)}


def build_tracker(checkpoint_path: str, config_path: str, hanning_weight: float = 0.49):
    """Build SGLATrackExt tracker with configurable Hanning weight.

    Args:
        checkpoint_path: Path to fine-tuned SGLATrack checkpoint (.pth.tar)
        config_path: Path to config YAML
        hanning_weight: Hanning blend weight [0, 1]

    Returns:
        SGLATrackExt instance ready for initialize()/track()
    """
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

    # Model source: either PyTorch checkpoint or ONNX model (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--checkpoint',
                             help='Path to fine-tuned SGLATrack checkpoint (.pth.tar)')
    model_group.add_argument('--onnx',
                             help='Path to exported ONNX model (FP32 or INT8)')

    parser.add_argument('--config',
                        default='SGLA_Track Model/SGLATrack/experiments/sglatrack/deit_distilled.yaml',
                        help='Path to SGLATrack config YAML (only needed with --checkpoint)')
    parser.add_argument('--hanning_weight', type=float, default=0.49,
                        help='Hanning window blend weight [0, 1] (default 0.49)')
    parser.add_argument('--output_dir', default='predictions/sglatrack')
    parser.add_argument('--output_csv', default=None,
                        help='If provided, generate full submission CSV at this path')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Limit sequences processed (quick sanity check)')
    args = parser.parse_args()

    if args.onnx:
        print(f"Building OnnxTracker from {args.onnx} (hanning_weight={args.hanning_weight})")
        tracker = OnnxTracker(args.onnx, args.hanning_weight)
    else:
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
