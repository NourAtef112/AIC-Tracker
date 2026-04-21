# MTC-AIC4 — Efficient Aerial Single-Object Tracking
## Full 6-Day Competition Plan

---

## Competition At a Glance

| Field | Detail |
|-------|--------|
| **Competition** | MTC-AIC4 — Efficient Aerial Single-Object Tracking |
| **Organizer** | Military Technical College × Applied Innovation Center, Egypt |
| **Task** | Single-Object Tracking (SOT) on UAV aerial video sequences |
| **Deadline** | Day 6 = Submission Day |
| **Round** | Online Qualification Round |

### Final Score Formula
```
AccuracyScore   = 0.6 × AUC + 0.4 × NormPrecision
EfficiencyScore = 0.25×(1 - params/50)  +  0.15×(1 - flops/30)
                + 0.35×(1 - latency/30) +  0.25×(1 - size/0.5)
FinalScore      = 0.8 × AccuracyScore + 0.2 × EfficiencyScore
```

### Efficiency Hard Budgets

| Metric | Maximum Allowed |
|--------|----------------|
| FLOPs | 30 GFLOPs |
| Parameters | 50 Million |
| Inference Latency | 30 ms |
| Model Size | 0.5 GB |

> **Key Insight:** Accuracy = 80% of score. Latency = most weighted efficiency metric (w=0.35).
> Optimize in that priority order.

---

## Hardware Overview

| Member | GPU | VRAM | Training Capable |
|--------|-----|------|-----------------|
| Ahmed | RTX 5060 | ~12GB | ✅ Best — main trainer |
| Nour | RTX 3060 | 12GB | ✅ Strong — experiments |
| Yassin | RTX 4050 | 6GB | ⚠️ Inference + eval only |
| Leil | RTX 3050 | 4–6GB | ⚠️ Light fine-tuning only |
| Barawy | Quadro T500 | 4GB | ❌ CPU tasks only |

---

## Final Team Structure

### 🔵 Team A — OSTrack-256 (Accuracy-Focused)

| Member | Role | Exact Responsibilities |
|--------|------|------------------------|
| **Ahmed** | Lead Trainer | All OSTrack training runs. Sets hyperparameters. Owns every training decision. |
| **Nour** | Post-processing Engineer | Dual template strategy. Hanning window penalty. Secondary experiments. |
| **Yassin** | Evaluation Engineer | Prediction writer. Full eval runs. Logs all numbers to shared sheet. |

### 🟠 Team B — LightTrack (Efficiency-Focused)

| Member | Role | Exact Responsibilities |
|--------|------|------------------------|
| **Leil** | Trainer | Fine-tunes LightTrack. LightTrack is light enough for RTX 3050. |
| **Barawy** | Efficiency Engineer | Data prep. ONNX export. INT8 quantization. Measures all 4 efficiency metrics. |

---

## Dataset Format Reference

```
<dataset>/<sequence_name>/
    <sequence_name>.mp4        ← video at native FPS
    annotation.txt             ← one bounding box per line (train only)
```

**Annotation format** (one line per frame):
```
x,y,w,h        ← top-left corner + dimensions in pixels
0,0,0,0        ← target not visible in this frame
```

**Splits:**
- `train` — annotated, use for training + local evaluation
- `public_lb` — no annotations, use for final submission only

---

## Shared Infrastructure
> **Owner: Ahmed (Day 1 morning) — everything depends on this**

### 1. Video → Frames Extractor

```python
# extract_frames.py
import cv2, os, json
from pathlib import Path

def extract_sequence(video_path: str, out_dir: str, force: bool = False):
    out_dir = Path(out_dir)
    if out_dir.exists() and not force:
        existing = list(out_dir.glob("*.jpg"))
        if existing:
            print(f"  [SKIP] {out_dir} — {len(existing)} frames already extracted")
            return len(existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{idx:08d}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"  [DONE] {video_path} → {idx} frames")
    return idx

def extract_all(manifest_path: str, split: str = "train", frames_root: str = "frames"):
    with open(manifest_path) as f:
        manifest = json.load(f)
    sequences = manifest.get(split, {})
    print(f"Extracting {len(sequences)} sequences | split={split}")
    for seq_key, info in sequences.items():
        extract_sequence(info["video_path"], str(Path(frames_root) / seq_key))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="contestant_manifest.json")
    p.add_argument("--split", default="train", choices=["train", "public_lb"])
    p.add_argument("--output", default="frames")
    args = p.parse_args()
    extract_all(args.manifest, args.split, args.output)
```

```bash
python extract_frames.py --split train --output frames/train
```

---

### 2. Annotation Utilities

```python
# annotation_utils.py

def load_annotations(ann_path: str):
    """Returns list of (x, y, w, h) tuples. 0,0,0,0 = not visible."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            boxes.append(tuple(map(float, line.split(','))))
    return boxes

def is_visible(box) -> bool:
    return not (box[0] == 0 and box[1] == 0 and
                box[2] == 0 and box[3] == 0)

def get_init_box(boxes):
    """Returns (frame_idx, box) of first visible frame."""
    for i, box in enumerate(boxes):
        if is_visible(box):
            return i, box
    raise ValueError("No visible frame found in sequence.")
```

---

### 3. Manifest Loader

```python
# manifest_loader.py
import json
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SequenceInfo:
    key: str
    dataset: str
    seq_name: str
    n_frames: int
    native_fps: int
    video_path: str
    annotation_path: Optional[str]

def load_manifest(path: str, split: str) -> List[SequenceInfo]:
    with open(path) as f:
        manifest = json.load(f)
    return [
        SequenceInfo(
            key=k,
            dataset=v["dataset"],
            seq_name=v["seq_name"],
            n_frames=v["n_frames"],
            native_fps=v["native_fps"],
            video_path=v["video_path"],
            annotation_path=v.get("annotation_path")
        )
        for k, v in manifest.get(split, {}).items()
    ]
```

---

### 4. Prediction Writer

```python
# prediction_writer.py
# Owner: Yassin
import os
from pathlib import Path

def save_predictions(pred_boxes, seq_key: str, output_dir: str = "predictions"):
    """
    Saves tracker predictions as annotation.txt format.
    pred_boxes: list of (x, y, w, h) tuples, one per frame.
    """
    out_path = Path(output_dir) / (seq_key.replace("/", "_") + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for box in pred_boxes:
            f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
    return str(out_path)
```

---

### 5. Complete Shared Evaluation Script

```python
# evaluate.py
# Owner: Yassin — used by EVERYONE, never modify without team agreement
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
        gt_boxes   = load_annotations(info["annotation_path"])
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
    eff       = efficiency_score(args.params_m, args.flops_g,
                                  args.latency_ms, args.size_gb)
    final     = final_score(acc, eff)

    print("\n" + "═"*52)
    print("   MTC-AIC4 EVALUATION REPORT")
    print("═"*52)
    print(f"  Sequences evaluated : {len(sequences) - len(missing)}")
    print(f"  Missing predictions : {len(missing)}")
    print("─"*52)
    print(f"  AUC (Success Rate)  : {auc:.4f}")
    print(f"  Precision @20px     : {prec:.4f}")
    print(f"  Norm Precision      : {norm_prec:.4f}")
    print(f"  ── AccuracyScore    : {acc:.4f}   (0.6×AUC + 0.4×NP)")
    print("─"*52)
    print(f"  Params  {args.params_m:.1f}M  / 50M   → {1-(args.params_m/50):.4f}")
    print(f"  FLOPs   {args.flops_g:.1f}G   / 30G   → {1-(args.flops_g/30):.4f}")
    print(f"  Latency {args.latency_ms:.1f}ms / 30ms → {1-(args.latency_ms/30):.4f}")
    print(f"  Size    {args.size_gb:.3f}GB / 0.5GB → {1-(args.size_gb/0.5):.4f}")
    print(f"  ── EfficiencyScore  : {eff:.4f}")
    print("─"*52)
    print(f"  ★  FINAL SCORE      : {final:.4f}")
    print("═"*52 + "\n")
    if missing:
        print(f"  ⚠️  Missing: {missing[:3]}{'...' if len(missing)>3 else ''}\n")


if __name__ == "__main__":
    main()
```

**Run example:**
```bash
python evaluate.py \
  --pred_dir ./predictions/ostrack \
  --manifest contestant_manifest.json \
  --split train \
  --params_m 28.5 \
  --flops_g 17.2 \
  --latency_ms 14.3 \
  --size_gb 0.11
```

---

### 6. Efficiency Measurement Script

```python
# measure_efficiency.py
# Owner: Barawy
import torch, time, os
from thop import profile  # pip install thop

def measure_all(model, template_size=128, search_size=256, n_runs=200):
    model.eval().cuda()
    t = torch.randn(1, 3, template_size, template_size).cuda()
    s = torch.randn(1, 3, search_size,   search_size).cuda()

    # FLOPs & Params
    flops, params = profile(model, inputs=(t, s), verbose=False)
    print(f"FLOPs  : {flops/1e9:.2f} GFLOPs  (budget: 30)")
    print(f"Params : {params/1e6:.2f} M        (budget: 50M)")

    # Latency — warm up first
    for _ in range(20):
        with torch.no_grad():
            model(t, s)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(t, s)
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) / n_runs * 1000
    print(f"Latency: {latency_ms:.2f} ms       (budget: 30ms)")

    # Model size
    torch.save(model.state_dict(), "_tmp.pth")
    size_gb = os.path.getsize("_tmp.pth") / 1e9
    os.remove("_tmp.pth")
    print(f"Size   : {size_gb:.4f} GB       (budget: 0.5GB)")

    return flops/1e9, params/1e6, latency_ms, size_gb
```

---

### 7. Google Sheet Columns

| Run | Model | AUC | NormPrec | AccScore | Params(M) | FLOPs(G) | Lat(ms) | Size(GB) | EffScore | **FinalScore** | Notes |
|-----|-------|-----|----------|----------|-----------|----------|---------|----------|----------|----------------|-------|
| A-base | OSTrack-256 | | | | | | | | | | No fine-tune |
| B-base | LightTrack | | | | | | | | | | No fine-tune |
| A-v1 | OSTrack-256 | | | | | | | | | | Fine-tuned |
| A-v2 | OSTrack-256 | | | | | | | | | | + Dual template |
| A-v3 | OSTrack-256 | | | | | | | | | | + Hanning |
| B-v1 | LightTrack | | | | | | | | | | Fine-tuned |
| B-v2 | LightTrack | | | | | | | | | | + INT8 |

---

## Team A — OSTrack-256 Full Plan

### Setup (Ahmed — Day 1)

```bash
# Clone
git clone https://github.com/botaoye/OSTrack.git
cd OSTrack

# Environment
conda create -n ostrack python=3.8 -y
conda activate ostrack
pip install torch==1.13.1+cu117 torchvision==0.14.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python tracking/create_default_local_file.py \
    --workspace_dir . --data_dir ./data --save_dir ./output

# Download pretrained checkpoint
mkdir pretrained
wget https://github.com/botaoye/OSTrack/releases/download/main/OSTrack_ep0300.pth.tar \
     -O pretrained/ostrack256_ep300.pth.tar
```

### Custom Dataset Adapter

```python
# lib/train/dataset/mtc_dataset.py
import os, json
import numpy as np
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from annotation_utils import load_annotations, is_visible

class MTCDataset(BaseVideoDataset):
    def __init__(self, root, manifest_path, split="train",
                 image_loader=jpeg4py_loader):
        super().__init__("MTC", root, image_loader)
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.sequences = []
        for key, info in manifest.get(split, {}).items():
            if not info.get("annotation_path"):
                continue
            boxes   = load_annotations(info["annotation_path"])
            visible = [i for i, b in enumerate(boxes) if is_visible(b)]
            if len(visible) < 2:
                continue
            self.sequences.append({
                "key":        key,
                "frames_dir": os.path.join("frames", split, key),
                "boxes":      boxes,
                "visible":    visible,
            })

    def get_num_sequences(self):
        return len(self.sequences)

    def get_sequence_info(self, seq_id):
        seq   = self.sequences[seq_id]
        boxes = np.array(seq["boxes"], dtype=np.float32)
        valid = np.array([is_visible(b) for b in seq["boxes"]], dtype=bool)
        return {"bbox": boxes, "valid": valid, "visible": valid}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq    = self.sequences[seq_id]
        frames = [self.image_loader(
                    os.path.join(seq["frames_dir"], f"{fid:08d}.jpg"))
                  for fid in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        return frames, anno, {"dataset": "MTC"}
```

### Fine-tuning Config

```yaml
# experiments/ostrack/mtc_finetune.yaml
TRAIN:
  LR: 4e-5            # Lower than default — fine-tuning not training from scratch
  WEIGHT_DECAY: 1e-4
  EPOCH: 30           # ⚠️ If 1 epoch > 5hrs → cut to 15 immediately
  BATCH_SIZE: 16      # Ahmed: increase to 32 if VRAM allows
  AMP: True           # Mixed precision — required for RTX cards
  GRAD_CLIP_NORM: 0.1
  SCHEDULER:
    TYPE: step
    DECAY_RATE: 0.1
    MILESTONES: [20]

DATA:
  SEARCH:
    SIZE: 256
    FACTOR: 4.0       # Search region = 4× target size
    CENTER_JITTER: 4.5
    SCALE_JITTER: 0.5
  TEMPLATE:
    SIZE: 128
    FACTOR: 2.0
  TRAIN:
    DATASETS_NAME: ["MTC"]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
```

```bash
# Ahmed launches this on RTX 5060
python tracking/train.py \
  --script ostrack \
  --config mtc_finetune \
  --save_dir output/mtc_run1 \
  --use_wandb 0 \
  --pretrained_path pretrained/ostrack256_ep300.pth.tar
```

### Dual Template Strategy (Nour implements)

```python
# dual_template.py
import torch

class DualTemplateManager:
    """
    Maintains two templates:
      - static:  initialized once, never updated (anchors identity)
      - dynamic: updated every `interval` frames when confidence is high
    Final template = 0.5 * static + 0.5 * dynamic
    """
    def __init__(self, conf_threshold: float = 0.65, interval: int = 5):
        self.conf_threshold = conf_threshold
        self.interval       = interval
        self.static         = None
        self.dynamic        = None
        self.frame_count    = 0

    def initialize(self, template_feat: torch.Tensor):
        self.static      = template_feat.clone()
        self.dynamic     = template_feat.clone()
        self.frame_count = 0

    def get_fused(self) -> torch.Tensor:
        return 0.5 * self.static + 0.5 * self.dynamic

    def update(self, new_feat: torch.Tensor, confidence: float):
        self.frame_count += 1
        if (confidence > self.conf_threshold and
                self.frame_count % self.interval == 0):
            self.dynamic = new_feat.clone()
```

### Hanning Window Penalty (Nour implements)

```python
# hanning_penalty.py
import torch
import numpy as np

def apply_hanning_penalty(score_map: torch.Tensor,
                           penalty_weight: float = 0.49) -> torch.Tensor:
    """
    Suppresses high-displacement predictions.
    Discourages the tracker from jumping far from the previous position.
    penalty_weight: 0 = no penalty, 1 = full penalty (use 0.49 as default)
    """
    h, w    = score_map.shape[-2:]
    hanning = torch.tensor(
        np.outer(np.hanning(h), np.hanning(w)),
        dtype=score_map.dtype,
        device=score_map.device
    )
    return score_map * (1 - penalty_weight) + hanning * penalty_weight
```

### Multi-scale Inference (Nour — only if time left on Day 5)

```python
# multi_scale_inference.py
def multi_scale_track(model, frame, template, scales=[0.96, 1.0, 1.04]):
    """
    Runs inference at multiple search region scales.
    Returns prediction from the scale with highest confidence.
    Adds ~3–5ms latency — check budget before enabling.
    """
    best_box, best_conf = None, -1
    for scale in scales:
        pred_box, conf = model.track(frame, template, scale_factor=scale)
        if conf > best_conf:
            best_conf = conf
            best_box  = pred_box
    return best_box, best_conf
```

---

## Team B — LightTrack Full Plan

### Setup (Leil — Day 1)

```bash
# Clone
git clone https://github.com/researchmm/LightTrack.git
cd LightTrack

# Environment
conda create -n lighttrack python=3.7 -y
conda activate lighttrack
pip install torch==1.9.0+cu111 torchvision==0.10.0 \
    --extra-index-url https://download.pytorch.org/whl/cu111
pip install -r requirements.txt

# Download pretrained checkpoint
mkdir pretrained
wget https://github.com/researchmm/LightTrack/releases/download/weights/LightTrack-Mobile.pth \
     -O pretrained/lighttrack_mobile.pth
```

### Fine-tuning Config (Leil)

```yaml
TRAIN:
  LR: 1e-4
  EPOCH: 20
  BATCH_SIZE: 16      # Safe for RTX 3050 with 4-6GB VRAM
  AMP: True
  SCHEDULER:
    TYPE: step
    DECAY_RATE: 0.1
    MILESTONES: [14]

DATA:
  SEARCH_SIZE: 256
  TEMPLATE_SIZE: 128
  SEARCH_FACTOR: 4.0
```

```bash
# Leil launches this on RTX 3050
python tracking/train.py \
  --model lighttrack_mobile \
  --pretrained pretrained/lighttrack_mobile.pth \
  --lr 1e-4 --epochs 20 --batch 16 --amp
```

### ONNX Export (Barawy)

```python
# export_onnx.py
import torch

def export_to_onnx(model, checkpoint_path: str, output_path: str,
                   template_size: int = 128, search_size: int = 256):
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    dummy_template = torch.randn(1, 3, template_size, template_size)
    dummy_search   = torch.randn(1, 3, search_size,   search_size)

    torch.onnx.export(
        model,
        (dummy_template, dummy_search),
        output_path,
        input_names=["template", "search"],
        output_names=["pred_boxes", "score"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported → {output_path}")

# Run: python export_onnx.py
```

### INT8 Quantization (Barawy)

```python
# quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="lighttrack.onnx",
    model_output="lighttrack_int8.onnx",
    weight_type=QuantType.QInt8
)
print("INT8 quantization complete.")
# Expected gains: ~40% latency reduction, ~75% size reduction
```

---

## 6-Day Schedule

### Day 1 — Setup & Baseline
> Goal: Both models running. Baseline FinalScores in Google Sheet.

| Member | Morning | Afternoon |
|--------|---------|-----------|
| **Ahmed** | Build all shared scripts (extract_frames, annotation_utils, manifest_loader, evaluate.py, prediction_writer). Push to shared repo. | Clone OSTrack. Install environment. Run baseline inference on 5 val sequences. |
| **Nour** | Pull shared repo. Study OSTrack inference loop. Understand template extraction. | Read dual_template.py skeleton. Plan implementation for Day 2. |
| **Yassin** | Pull shared repo. Set up evaluate.py. Test on 5 sequences. | Run baseline eval → log baseline row to Google Sheet. |
| **Leil** | Clone LightTrack. Install environment. | Run baseline inference on 5 val sequences. |
| **Barawy** | Pull shared repo. Run extract_frames.py (CPU task — no GPU needed). Extract ALL train sequences. | Run measure_efficiency.py on LightTrack baseline. Log efficiency metrics. |

🔴 **CRITICAL:** If any environment fails to install by 2pm → switch to Google Colab immediately.

---

### Day 2 — Fine-tuning Starts
> Goal: Both training runs launched and stable overnight.

| Member | Tasks |
|--------|-------|
| **Ahmed** | Build MTCDataset adapter for OSTrack. Launch fine-tuning run with config above. Monitor first epoch loss — if diverging, halve LR immediately. |
| **Nour** | Implement DualTemplateManager on top of OSTrack inference. Test it on baseline model (no fine-tuned weights needed yet). |
| **Yassin** | Build prediction_writer.py. Run it on baseline model for all val sequences. Verify evaluate.py produces clean output for all sequences. |
| **Leil** | Build MTCDataset adapter for LightTrack (copy pattern from Ahmed's). Launch fine-tuning. ⚠️ If OOM → reduce batch to 8. |
| **Barawy** | Validate ALL annotation.txt files: check corrupt lines, mismatched frame counts. Set up Google Sheet with all column headers. Monitor Leil's training loss every hour. |

⚠️ **TIME RISK:** If 1 epoch > 5 hours → cut epochs from 30 to 15 immediately.

---

### Day 3 — Optimize & Post-process
> Goal: Fine-tuned scores + efficiency scores for both models in sheet.

| Member | Tasks |
|--------|-------|
| **Ahmed** | Evaluate best fine-tuned checkpoint. If AUC improved → integrate DualTemplate from Nour. Run 2nd eval with dual template. Log delta. |
| **Nour** | Plug DualTemplateManager into OSTrack inference loop. Implement Hanning window penalty. Test both together on Ahmed's checkpoint. |
| **Yassin** | Run full eval on fine-tuned OSTrack (all val sequences). Compute and log: AUC, NormPrecision, AccuracyScore. Compare vs baseline. |
| **Leil** | Evaluate fine-tuned LightTrack on 5 val sequences. If AUC > baseline → run full val eval. |
| **Barawy** | Export LightTrack to ONNX. Apply INT8 quantization. Run measure_efficiency.py on INT8 model. Calculate EfficiencyScore. Log everything. |

⚠️ **TIME RISK:** Quantization may drop accuracy. Always measure AUC before and after INT8.

---

### Day 4 — Decision Day
> Goal: One model chosen by noon. Everyone on it by afternoon.

**Morning (both teams):**
- Run FULL evaluation on complete val split
- Barawy compiles final comparison table by 11am

**11:00am — Decision Meeting (all 5 members, 45 minutes max):**

```
Compute FinalScore for both models.
gap = OSTrack.FinalScore - LightTrack.FinalScore

gap > +0.02  → Pick OSTrack
gap < -0.02  → Pick LightTrack
|gap| ≤ 0.02 → Too close:
    OSTrack.AccuracyScore > LightTrack.AccuracyScore + 0.03 → OSTrack
    else → LightTrack

Model failed to converge (AUC < 0.30) → abandon immediately.
No debates. The spreadsheet decides.
```

**Afternoon (all 5 on winning model):**

| Ablation | What to test | Time |
|----------|-------------|------|
| Search factor | 3.5 / 4.0 / 4.5 | 1hr |
| Template update interval | 3 / 5 / 10 frames | 1hr |
| Confidence threshold | 0.55 / 0.65 / 0.75 | 30min |
| Hanning penalty weight | 0.35 / 0.49 / 0.60 | 30min |

---

### Day 5 — Polish Only
> Goal: Final model locked. All metrics confirmed clean.

| Member | Tasks |
|--------|-------|
| **Ahmed** | Final training run with best ablation config (if needed). |
| **Nour** | Add multi-scale inference if latency budget allows (test 224×224 vs 256×256). |
| **Yassin** | Run complete final eval. Confirm metrics. Flag any regressions. |
| **Leil** | Test inference on her machine (RTX 3050) — confirm it runs under 30ms. |
| **Barawy** | Verify ALL 4 metrics under budget. If any fails → apply quantization to fix. |

**Budget check (Barawy must sign off on all 4):**
```
✅ FLOPs    < 30 GFLOPs
✅ Params   < 50 Million
✅ Latency  < 30 ms
✅ Size     < 0.5 GB
```

🔴 **CRITICAL:** Do NOT start new experiments on Day 5. Polish only.

---

### Day 6 — Submit
> Goal: Submitted by 3pm. 3-hour buffer for issues.

| Time | Action | Owner |
|------|--------|-------|
| 9am | Final clean eval run on public_lb split | Ahmed + Yassin |
| 10am | Package submission files | Barawy |
| 11am | Verify prediction format matches submission spec | Leil |
| 12pm | Double-check all efficiency metrics one last time | Barawy |
| 1pm | Submit | Ahmed |
| 1–4pm | Buffer for fixes | Everyone on standby |

---

## Risk Mitigation

| Risk | Prevention |
|------|------------|
| 🔴 Barawy can't train | Confirmed — CPU tasks only. Ahmed owns all Team B training. |
| 🔴 Leil OOMs during fine-tuning | Batch=16 → reduce to 8 if needed. Leil does NO training if it fails. |
| 🔴 Environment fails Day 1 | Ahmed installs OSTrack env tonight (Day 0). Test with one inference call before Day 1 starts. |
| ⚠️ Training diverges | Use pretrained checkpoints always. LR ≤ 4e-5 (OSTrack), ≤ 1e-4 (LightTrack). Never from scratch. |
| ⚠️ Decision paralysis Day 4 | Rule is written above. 45-minute timer. Spreadsheet decides. No exceptions. |

---

## Quick Wins (Under 2 Hours Each)

| Win | Impact | Who | Time |
|-----|--------|-----|------|
| Increase search factor 3.5 → 4.5 | Catches fast UAV motion | Ahmed | 20min |
| Confidence-gated template update | Prevents template drift | Nour | 1hr |
| Hanning window penalty | Smoother predictions on fast sequences | Nour | 30min |
| INT8 quantization (LightTrack) | ~40% latency cut, ~75% size cut | Barawy | 1.5hr |
| Skip invisible frames in training | Cleaner gradient signal | Ahmed | 30min |

---

## Team Quick Reference Card

```
═══════════════════════════════════════════
  TEAM A — OSTrack-256
  Ahmed  → All training (RTX 5060)
  Nour   → Dual template + Hanning + experiments (RTX 3060)
  Yassin → Evaluation + prediction writer + logging (RTX 4050)

  TEAM B — LightTrack
  Leil   → Fine-tuning (RTX 3050)
  Barawy → Data prep + ONNX + INT8 + efficiency metrics (Quadro)
═══════════════════════════════════════════
  Day 1  → Setup + baseline scores
  Day 2  → Fine-tuning launched
  Day 3  → Optimize + measure efficiency
  Day 4  → NOON: pick ONE model, all 5 join it
  Day 5  → Polish only, no new experiments
  Day 6  → Submit by 3pm
═══════════════════════════════════════════
```

---

## Project File Structure

```
MTC-AIC4/
│
├── 📁 data/                                  ← Raw competition dataset (read-only)
│   ├── contestant_manifest.json              ← Split metadata (train / public_lb)
│   ├── dataset2/
│   │   └── basketball_player1/
│   │       ├── basketball_player1.mp4
│   │       └── annotation.txt
│   ├── dataset5/
│   │   └── bike1/
│   │       ├── bike1.mp4
│   │       └── annotation.txt
│   └── ...                                   ← All other sequences
│
├── 📁 frames/                                ← Extracted frames (built by Ahmed Day 1)
│   ├── train/
│   │   ├── dataset2/basketball_player1/
│   │   │   ├── 00000000.jpg
│   │   │   ├── 00000001.jpg
│   │   │   └── ...
│   │   └── dataset5/bike1/
│   │       ├── 00000000.jpg
│   │       └── ...
│   └── public_lb/
│       └── ...                               ← Extracted on Day 6 for final submission
│
├── 📁 shared/                                ← Shared infrastructure (Ahmed, Day 1)
│   ├── extract_frames.py                     ← MP4 → JPEG frames extractor
│   ├── annotation_utils.py                   ← load_annotations(), is_visible()
│   ├── manifest_loader.py                    ← SequenceInfo dataclass + loader
│   ├── prediction_writer.py                  ← Saves tracker output as txt (Yassin)
│   ├── evaluate.py                           ← Official competition scoring script
│   └── measure_efficiency.py                 ← FLOPs / params / latency / size (Barawy)
│
├── 📁 team_a/                                ← Team A workspace (Ahmed + Nour + Yassin)
│   ├── OSTrack/                              ← Cloned from github.com/botaoye/OSTrack
│   │   ├── lib/
│   │   │   ├── train/
│   │   │   │   └── dataset/
│   │   │   │       └── mtc_dataset.py        ← Custom dataset adapter (Ahmed)
│   │   │   └── ...
│   │   ├── experiments/
│   │   │   └── ostrack/
│   │   │       └── mtc_finetune.yaml         ← Fine-tuning config (Ahmed)
│   │   ├── tracking/
│   │   │   └── train.py
│   │   └── ...
│   ├── pretrained/
│   │   └── ostrack256_ep300.pth.tar          ← Downloaded pretrained checkpoint
│   ├── output/
│   │   ├── mtc_run1/                         ← Training run 1 checkpoints
│   │   │   ├── checkpoints/
│   │   │   │   └── ostrack_ep0030.pth.tar
│   │   │   └── logs/
│   │   └── mtc_run2/                         ← Training run 2 (if needed)
│   ├── dual_template.py                      ← DualTemplateManager (Nour)
│   ├── hanning_penalty.py                    ← Hanning window implementation (Nour)
│   ├── multi_scale_inference.py              ← Multi-scale search (Nour, Day 5 only)
│   └── run_inference.py                      ← Full inference pipeline (Nour + Yassin)
│
├── 📁 team_b/                                ← Team B workspace (Leil + Barawy)
│   ├── LightTrack/                           ← Cloned from github.com/researchmm/LightTrack
│   │   ├── tracking/
│   │   │   └── train.py
│   │   └── ...
│   ├── pretrained/
│   │   └── lighttrack_mobile.pth             ← Downloaded pretrained checkpoint
│   ├── output/
│   │   ├── lighttrack_run1/                  ← Fine-tuned checkpoints (Leil)
│   │   │   └── checkpoints/
│   │   └── lighttrack_run2/
│   ├── export_onnx.py                        ← ONNX export script (Barawy)
│   ├── quantize.py                           ← INT8 quantization script (Barawy)
│   ├── lighttrack.onnx                       ← Exported ONNX model
│   ├── lighttrack_int8.onnx                  ← INT8 quantized model
│   └── run_inference.py                      ← Full inference pipeline (Leil)
│
├── 📁 predictions/                           ← Tracker output txts (Yassin owns this)
│   ├── ostrack/
│   │   ├── baseline/
│   │   │   └── dataset5_bike1.txt            ← One txt per sequence
│   │   ├── v1_finetuned/
│   │   │   └── dataset5_bike1.txt
│   │   ├── v2_dual_template/
│   │   │   └── dataset5_bike1.txt
│   │   └── v3_hanning/
│   │       └── dataset5_bike1.txt
│   └── lighttrack/
│       ├── baseline/
│       ├── v1_finetuned/
│       └── v2_int8/
│
├── 📁 evaluation/                            ← Eval results (Yassin + Barawy)
│   ├── results_ostrack_baseline.txt
│   ├── results_ostrack_v1.txt
│   ├── results_ostrack_v2.txt
│   ├── results_lighttrack_baseline.txt
│   ├── results_lighttrack_v1.txt
│   └── results_lighttrack_v2_int8.txt
│
├── 📁 submission/                            ← Final submission package (Barawy)
│   ├── predictions/                          ← Final prediction txts for public_lb
│   │   └── <seq_key>.txt
│   ├── model/
│   │   └── final_model.pth / .onnx
│   └── efficiency_report.txt                 ← FLOPs, params, latency, size
│
└── 📄 scores.xlsx                            ← Shared Google Sheet export (Barawy)
                                              ← All runs, all metrics, all FinalScores
```

---

### File Ownership Summary

| File / Folder | Owner | Day Created |
|---------------|-------|-------------|
| `shared/extract_frames.py` | Ahmed | Day 1 AM |
| `shared/annotation_utils.py` | Ahmed | Day 1 AM |
| `shared/manifest_loader.py` | Ahmed | Day 1 AM |
| `shared/evaluate.py` | Ahmed | Day 1 AM |
| `shared/prediction_writer.py` | Yassin | Day 2 |
| `shared/measure_efficiency.py` | Barawy | Day 1 PM |
| `team_a/OSTrack/lib/train/dataset/mtc_dataset.py` | Ahmed | Day 2 |
| `team_a/OSTrack/experiments/ostrack/mtc_finetune.yaml` | Ahmed | Day 2 |
| `team_a/dual_template.py` | Nour | Day 2–3 |
| `team_a/hanning_penalty.py` | Nour | Day 3 |
| `team_a/multi_scale_inference.py` | Nour | Day 5 only |
| `team_b/export_onnx.py` | Barawy | Day 3 |
| `team_b/quantize.py` | Barawy | Day 3 |
| `predictions/` | Yassin | Day 1 onward |
| `evaluation/` | Yassin + Barawy | Day 1 onward |
| `submission/` | Barawy | Day 6 |
| `scores.xlsx` | Barawy | Day 1 onward |

> **Rule:** Never edit another member's files without telling them.
> All shared scripts go in `shared/` — never duplicate them into team folders.

---

*MTC-AIC4 Competition Plan — Generated for 5-member team*
*OSTrack repo: https://github.com/botaoye/OSTrack*
*LightTrack repo: https://github.com/researchmm/LightTrack*
