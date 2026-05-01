import numpy as np
import argparse
import json
from pathlib import Path
from annotation_utils import load_annotations, is_visible


def compute_iou(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    ix1 = max(px, gx);         iy1 = max(py, gy)
    ix2 = min(px+pw, gx+gw);   iy2 = min(py+ph, gy+gh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = pw*ph + gw*gh - inter
    return inter / union if union > 0 else 0.0


def compute_center_error(pred, gt):
    cx_p = pred[0] + pred[2]/2;  cy_p = pred[1] + pred[3]/2
    cx_g = gt[0]  + gt[2]/2;    cy_g = gt[1]  + gt[3]/2
    return np.sqrt((cx_p - cx_g)**2 + (cy_p - cy_g)**2)


def compute_norm_error(pred, gt):
    err  = compute_center_error(pred, gt)
    diag = np.sqrt(gt[2]**2 + gt[3]**2)
    return err / diag if diag > 0 else 0.0


def evaluate_sequence(pred_boxes, gt_boxes):
    ious, errors, norm_errors = [], [], []
    for pred, gt in zip(pred_boxes, gt_boxes):
        if not is_visible(gt):
            continue
        ious.append(compute_iou(pred, gt))
        errors.append(compute_center_error(pred, gt))
        norm_errors.append(compute_norm_error(pred, gt))
    return np.array(ious), np.array(errors), np.array(norm_errors)


def compute_auc(all_ious):
    thresholds = np.linspace(0, 1, 101)
    return float(np.mean([np.mean(all_ious >= t) for t in thresholds]))


def accuracy_score(auc, norm_precision):
    return 0.6 * auc + 0.4 * norm_precision


def efficiency_score(params_m, flops_g, latency_ms, size_gb):
    e_p = max(0.0, 1 - params_m  / 50.0)
    e_f = max(0.0, 1 - flops_g   / 30.0)
    e_l = max(0.0, 1 - latency_ms / 30.0)
    e_s = max(0.0, 1 - size_gb   / 0.5)
    return 0.25*e_p + 0.15*e_f + 0.35*e_l + 0.25*e_s


def final_score(acc, eff):
    return 0.8 * acc + 0.2 * eff


def main():
    parser = argparse.ArgumentParser(description="MTC-AIC4 Evaluator")
    parser.add_argument("--pred_dir",   required=True)
    parser.add_argument("--manifest",   default="contestant_manifest.json")
    parser.add_argument("--split",      default="train")
    parser.add_argument("--params_m",   type=float, required=True)
    parser.add_argument("--flops_g",    type=float, required=True)
    parser.add_argument("--latency_ms", type=float, required=True)
    parser.add_argument("--size_gb",    type=float, required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    sequences = manifest.get(args.split, {})

    all_ious, all_errors, all_norm_errors = [], [], []
    missing = []

    for seq_key, info in sequences.items():
        if not info.get("annotation_path"):
            continue
        pred_file = Path(args.pred_dir) / (seq_key.replace("/","_") + ".txt")
        if not pred_file.exists():
            missing.append(seq_key)
            continue
        gt_boxes   = load_annotations(str(Path("data") / info["annotation_path"]))
        pred_boxes = load_annotations(str(pred_file))
        n = min(len(pred_boxes), len(gt_boxes))
        ious, errors, norm_errors = evaluate_sequence(pred_boxes[:n], gt_boxes[:n])
        all_ious.extend(ious)
        all_errors.extend(errors)
        all_norm_errors.extend(norm_errors)

    all_ious        = np.array(all_ious)
    all_errors      = np.array(all_errors)
    all_norm_errors = np.array(all_norm_errors)

    auc       = compute_auc(all_ious)
    prec      = float(np.mean(all_errors < 20.0))
    norm_prec = float(np.mean(all_norm_errors < 0.1))
    acc       = accuracy_score(auc, norm_prec)
    eff       = efficiency_score(args.params_m, args.flops_g,args.latency_ms, args.size_gb)
    final     = final_score(acc, eff)

    print("\n" + "█"*52)
    print("   MTC-AIC4 EVALUATION REPORT")
    print("█"*52)
    print(f"  Sequences evaluated : {len(sequences) - len(missing)}")
    print(f"  Missing predictions : {len(missing)}")
    print("█"*52)
    print(f"  AUC (Success Rate)  : {auc:.4f}")
    print(f"  Precision @20px     : {prec:.4f}")
    print(f"  Norm Precision      : {norm_prec:.4f}")
    print(f"  ├─ AccuracyScore    : {acc:.4f}   (0.6×AUC + 0.4×NP)")
    print("█"*52)
    print(f"  Params  {args.params_m:.1f}M  / 50M   → {1-(args.params_m/50):.4f}")
    print(f"  FLOPs   {args.flops_g:.1f}G   / 30G   → {1-(args.flops_g/30):.4f}")
    print(f"  Latency {args.latency_ms:.1f}ms / 30ms → {1-(args.latency_ms/30):.4f}")
    print(f"  Size    {args.size_gb:.3f}GB / 0.5GB → {1-(args.size_gb/0.5):.4f}")
    print(f"  ├─ EfficiencyScore  : {eff:.4f}")
    print("█"*52)
    print(f"  ⭐ FINAL SCORE      : {final:.4f}")
    print("█"*52 + "\n")
    if missing:
        print(f"  ⚠️  Missing: {missing[:3]}{'...' if len(missing)>3 else ''}\n")


if __name__ == "__main__":
    main()
