"""Local evaluation harness — compute AUC + NormPrecision + S_acc.

Simple `--pred_dir` + `--gt_dir` interface (no manifest needed).
Used for quick per-checkpoint evaluation during Day 2 sweeps.

From roadmap.
"""

import os
import argparse
import numpy as np


def compute_iou(boxA, boxB):
    """Compute IoU between two boxes (x, y, w, h)."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compute_auc(preds, gts):
    """Success plot AUC over IoU thresholds [0, 1]."""
    ious = []
    for p, g in zip(preds, gts):
        if g is None or p is None:
            continue
        ious.append(compute_iou(p, g))

    if not ious:
        return 0.0

    thresholds = np.linspace(0, 1, 101)
    rates = [np.mean([io >= t for io in ious]) for t in thresholds]
    return float(np.mean(rates))


def compute_norm_precision(preds, gts, threshold: float = 0.5):
    """Normalized precision: fraction where normalized center error < threshold."""
    scores = []
    for p, g in zip(preds, gts):
        if g is None or p is None:
            continue
        norm = np.sqrt(g[2] * g[3])
        if norm == 0:
            continue
        cx_p = p[0] + p[2] / 2
        cy_p = p[1] + p[3] / 2
        cx_g = g[0] + g[2] / 2
        cy_g = g[1] + g[3] / 2
        dist = np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2)
        scores.append(1.0 if (dist / norm) < threshold else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def load_gt(ann_path):
    """Load annotation.txt (x,y,w,h format). Returns list of (x,y,w,h) or None."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, w, h = map(float, line.split(','))
            boxes.append(None if (w == 0 and h == 0) else (x, y, w, h))
    return boxes


def load_preds(pred_path):
    """Load prediction file (x,y,w,h format). Returns list of (x,y,w,h)."""
    preds = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(tuple(map(float, line.split(','))))
    return preds


def evaluate(pred_dir: str, gt_dir: str, seq_list=None):
    """Evaluate predictions against ground truth.

    Args:
        pred_dir: Directory of prediction files (seq_name.txt)
        gt_dir: Directory of GT sequences (each with annotation.txt)
        seq_list: List of sequence names to evaluate. If None, auto-discover.
    """
    if seq_list is None:
        seq_list = [d for d in os.listdir(gt_dir)
                    if os.path.isdir(os.path.join(gt_dir, d))]

    results = []
    for seq in sorted(seq_list):
        ann_path = os.path.join(gt_dir, seq, 'annotation.txt')
        pred_path = os.path.join(pred_dir, f'{seq}.txt')

        if not os.path.exists(ann_path):
            print(f"  [skip] No GT: {seq}")
            continue
        if not os.path.exists(pred_path):
            print(f"  [skip] No pred: {seq}")
            continue

        gt = load_gt(ann_path)
        preds = load_preds(pred_path)

        # Frame 0 is init — skip in evaluation
        gt_eval = gt[1:]
        preds_eval = preds[1:] if len(preds) > len(gt_eval) else preds

        # Align lengths
        min_len = min(len(gt_eval), len(preds_eval))
        gt_eval = gt_eval[:min_len]
        preds_eval = preds_eval[:min_len]

        auc = compute_auc(preds_eval, gt_eval)
        np_ = compute_norm_precision(preds_eval, gt_eval)
        results.append({'seq': seq, 'auc': auc, 'np': np_})

    if not results:
        print("No results found.")
        return None, None, None

    auc_mean = np.mean([r['auc'] for r in results])
    np_mean = np.mean([r['np'] for r in results])
    s_acc = 0.6 * auc_mean + 0.4 * np_mean

    print()
    print("=" * 50)
    print(f"Sequences evaluated: {len(results)}")
    print(f"AUC:           {auc_mean:.4f}")
    print(f"NormPrecision: {np_mean:.4f}")
    print(f"S_acc:         {s_acc:.4f}  ← your accuracy score")
    print("=" * 50)

    return auc_mean, np_mean, s_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local evaluation — AUC + NormPrecision + S_acc')
    parser.add_argument('--pred_dir', required=True,
                        help='Directory of prediction files (seq_name.txt)')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory of GT sequences (each with annotation.txt)')
    args = parser.parse_args()

    evaluate(args.pred_dir, args.gt_dir)
