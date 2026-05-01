# MTC-AIC4 — Full 2-Day Execution Roadmap

> **Submission model:** SGLATrack (DeiT-tiny) — fine-tuned on competition data + Hanning penalty + INT8  
> **Teacher model:** OSTrack (ViT-B) — distillation only, never submitted (exceeds 50M param cap)  
> **Final score:** 0.3 × PublicLB + 0.7 × HiddenScore  
> **Scoring:** `S_acc = 0.6×AUC + 0.4×NormPrecision` | `FinalScore = S_acc − 0.2×S_eff`

---

## Team & Machine Split

| Machine | People | Responsibility |
|---|---|---|
| **Machine 1** | Ahmed + Leil | SGLATrack: setup → fine-tune → ONNX → INT8 → CSV |
| **Machine 2** | Nour + Yassin | Hanning (first) → OSTrack fine-tune → local eval → distillation |

---

## Hard Caps (Violations = Disqualification)

| Metric | Cap | SGLATrack Estimate | Risk |
|---|---|---|---|
| Parameters | ≤ 50M | ~5–6M | ✅ Safe |
| FLOPs | ≤ 30 GFLOPs | ~3–5G | ✅ Safe |
| Model size | ≤ 500MB | ~25–40MB | ✅ Safe |
| Latency | ≤ 30ms on Jetson Orin Nano | ~22–45ms estimated | ⚠️ Verify |

> **Latency is the only real risk.** SGLATrack runs 225 FPS on RTX 2080Ti = ~4.4ms GPU.
> Jetson is roughly 8–10× slower → estimated 35–45ms FP32.
> INT8 typically halves latency → ~18–22ms. INT8 is **mandatory**, not optional.

---

## Fallback Hierarchy — Always Have a CSV Ready

| Priority | Model | State | Submittable by |
|---|---|---|---|
| 1 | SGLATrack fine-tuned | + Hanning + INT8 | Day 2 noon |
| 2 | SGLATrack fine-tuned | + Hanning, FP32 | Day 2 morning |
| 3 | SGLATrack pretrained | + Hanning | Day 1 evening |
| 4 | SGLATrack pretrained | no Hanning | Day 1 afternoon |

**Never enter Day 2 without fallback #3 already generated.**

---

---

# PRE-FLIGHT (Do Before Day 1 Clock Starts)

Both machines run these in parallel.

---

## Machine 1 — SGLATrack Setup (Ahmed + Leil)

```bash
# 1. Clone and install
git clone https://github.com/GXNU-ZhongLab/SGLATrack.git
cd SGLATrack
conda env create -f environment_sgla1.yml
conda activate sglatrack

# 2. Download pretrained DeiT-tiny weights (from SGLATrack README)
mkdir -p pretrained_models
# download deit_tiny_distilled_patch16_224.pth into pretrained_models/

# 3. Verify param count — MUST be < 50M before anything else
python - <<'EOF'
import torch, sys
# Adjust import to match SGLATrack's actual model builder
from lib.models.sglatrack import build_sglatrack
from lib.config.sglatrack.config import cfg, update_config_from_file
update_config_from_file('experiments/sglatrack/deit_distilled.yaml')
model = build_sglatrack(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M")
if n > 50e6:
    print("FAIL — exceeds 50M cap")
    sys.exit(1)
print("PASS")
EOF

# 4. Single forward pass smoke test
python - <<'EOF'
import torch
template = torch.randn(1, 3, 128, 128).cuda()
search   = torch.randn(1, 3, 256, 256).cuda()
model    = model.cuda().eval()
with torch.no_grad():
    out = model(template, search)
print("Forward pass OK. Output keys:", out.keys() if isinstance(out, dict) else type(out))
EOF

# 5. Set paths
python tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ./data \
    --save_dir ./output
```

---

## Machine 2 — OSTrack + Eval Setup (Nour + Yassin)

```bash
# 1. Clone and install
git clone https://github.com/botaoye/OSTrack.git
cd OSTrack
pip install -r requirements.txt

# 2. Verify param count (expect ~86M — confirms it's teacher-only)
python - <<'EOF'
from lib.models.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file
update_config_from_file('experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml')
model = build_ostrack(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M — teacher only, NOT for submission")
EOF

# 3. Verify annotation.txt format on one public_lb sequence
python - <<'EOF'
import os
SEQ = "/path/to/public_lb/sequence_name"
with open(os.path.join(SEQ, "annotation.txt")) as f:
    lines = f.read().strip().split('\n')
print(f"Total frames: {len(lines)}")
print(f"Frame 0 (init box): {lines[0]}")
print(f"Frame 1:            {lines[1]}")
# Expected format: x,y,w,h  (0,0,0,0 for invisible)
EOF
```

---

## Shared: Dataset Preparation Script

Run on **both machines** pointing to competition data:

```python
# convert_dataset.py
# Converts competition annotation format → GOT-10k style for SGLATrack/OSTrack training

import os, shutil, json, argparse

def convert(comp_root, out_root, split='train'):
    manifest_path = os.path.join(comp_root, 'contestant_manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    seq_list = [e for e in manifest if e.get('split') == split]
    skipped = 0

    for entry in seq_list:
        seq_id   = entry['seq_id']             # e.g. "dataset1/car_chase3"
        seq_path = os.path.join(comp_root, seq_id)
        ann_file = os.path.join(seq_path, 'annotation.txt')

        if not os.path.exists(ann_file):
            skipped += 1
            continue

        with open(ann_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < 10:
            skipped += 1
            continue

        # Output folder: flat name to avoid nested dirs
        flat_name = seq_id.replace('/', '_')
        out_seq   = os.path.join(out_root, flat_name)
        os.makedirs(out_seq, exist_ok=True)

        # Copy frames — adjust 'img' to actual frame subfolder name
        img_src = os.path.join(seq_path, 'img')
        if os.path.exists(img_src):
            shutil.copytree(img_src, os.path.join(out_seq, 'img'), dirs_exist_ok=True)
        else:
            # frames may be directly in seq_path — handle both
            frames = sorted([f for f in os.listdir(seq_path)
                             if f.endswith(('.jpg','.png'))])
            os.makedirs(os.path.join(out_seq, 'img'), exist_ok=True)
            for fr in frames:
                shutil.copy(os.path.join(seq_path, fr),
                            os.path.join(out_seq, 'img', fr))

        # Write groundtruth.txt — skip invisible frames (0,0,0,0)
        gt = []
        for line in lines:
            vals = list(map(float, line.split(',')))
            x, y, w, h = vals
            if w > 0 and h > 0:
                gt.append(f"{x},{y},{w},{h}")

        with open(os.path.join(out_seq, 'groundtruth.txt'), 'w') as f:
            f.write('\n'.join(gt))

    print(f"Converted: {len(seq_list)-skipped} sequences | Skipped: {skipped}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comp_root', required=True)
    parser.add_argument('--out_root',  required=True)
    args = parser.parse_args()
    convert(args.comp_root, args.out_root)
```

```bash
# Machine 1
python convert_dataset.py \
    --comp_root /path/to/competition_data \
    --out_root  ./data/comp_train

# Machine 2 (same command, same data if shared drive; else rsync)
rsync -av /path/to/comp_train/ machine2:/path/to/OSTrack/data/comp_train/
```

---

---

# DAY 1 — Build Everything

---

## Hour 0–3 | Nour: Hanning Window (Machine 2, First Priority)

**This must happen before OSTrack training starts. The Hanning code goes into SGLATrack.**

### Step 1: Find the response map

```bash
cd SGLATrack
grep -rn "score_map\|cls_score\|heatmap\|response\|argmax" lib/ --include="*.py" | grep -v ".pyc"
# Look for where the tracker extracts the peak location from the network output
# Typically in: lib/tracker/sglatrack.py or lib/test/tracker/sglatrack.py
```

### Step 2: Implement the window

```python
# In SGLATrack's test-time tracker file (e.g. lib/tracker/sglatrack.py)
# Add to the class:

import numpy as np
import torch

def _build_hanning_window(self, h: int, w: int, device) -> torch.Tensor:
    """Cosine window of shape (H, W), normalized to [0, 1]."""
    if hasattr(self, '_hanning_cache') and self._hanning_cache.shape == (h, w):
        return self._hanning_cache
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    window = np.outer(win_h, win_w)
    window = window / window.max()
    self._hanning_cache = torch.tensor(window, dtype=torch.float32, device=device)
    return self._hanning_cache

def _apply_hanning(self, score_map: torch.Tensor, lam: float) -> torch.Tensor:
    """
    score_map: (1, H, W) or (H, W) — raw classification response
    lam: penalty weight. 0.0 = no penalty, 1.0 = full window
    """
    H, W = score_map.shape[-2], score_map.shape[-1]
    win = self._build_hanning_window(H, W, score_map.device)
    # Multiplicative form: suppresses edges proportionally
    penalized = score_map * ((1.0 - lam) + lam * win)
    return penalized
```

### Step 3: Inject into the tracking step

```python
# In the track() method, find where score_map peak is extracted:
# BEFORE:
#   peak_idx = score_map.argmax()
#
# AFTER:
    score_map = self._apply_hanning(score_map, lam=self.hanning_lambda)
    peak_idx  = score_map.argmax()
```

### Step 4: Make λ configurable

```python
# In tracking/test.py — add argument:
parser.add_argument('--hanning_lambda', type=float, default=0.49,
                    help='Hanning window penalty weight [0, 1]')

# Pass through to tracker init:
tracker = SGLATracker(..., hanning_lambda=args.hanning_lambda)
```

### Step 5: Quick sanity check

```python
# verify_hanning.py
import torch
import numpy as np

# Simulate a response map with a strong off-center peak (distractor case)
H, W = 16, 16
score_map = torch.zeros(1, H, W)
score_map[0, 2, 3]   = 1.0   # strong off-center peak (distractor)
score_map[0, 8, 8]   = 0.7   # weaker center peak (real target)

lam = 0.49
win_h = torch.tensor(np.hanning(H), dtype=torch.float32)
win_w = torch.tensor(np.hanning(W), dtype=torch.float32)
win   = torch.outer(win_h, win_w)
win   = win / win.max()

penalized = score_map * ((1 - lam) + lam * win)

print("Before penalty — argmax:", score_map.argmax().item(),
      "→ position:", divmod(score_map.argmax().item(), W))
print("After  penalty — argmax:", penalized.argmax().item(),
      "→ position:", divmod(penalized.argmax().item(), W))
# Expected: after penalty, peak should shift toward center (8,8)
```

### Step 6: Hand off to Machine 1

```bash
# Create a patch file
cd SGLATrack
git diff > hanning_patch.diff

# Send to Ahmed/Leil
scp hanning_patch.diff machine1:/path/to/SGLATrack/
# On Machine 1:
git apply hanning_patch.diff
```

**After this, Nour switches fully to OSTrack on Machine 2.**

---

## Hour 0–3 | Ahmed + Leil: Dataset Prep + Config (Machine 1, parallel)

While Nour implements Hanning, run dataset conversion and verify the training config.

```bash
# Convert competition training data
python convert_dataset.py \
    --comp_root /path/to/competition_data \
    --out_root  ./data/comp_train

# Count sequences and verify
python - <<'EOF'
import os
seqs = os.listdir('./data/comp_train')
print(f"Training sequences ready: {len(seqs)}")
# Spot-check one
import random
s = random.choice(seqs)
gt = open(f"./data/comp_train/{s}/groundtruth.txt").readlines()
imgs = os.listdir(f"./data/comp_train/{s}/img")
print(f"Sample seq '{s}': {len(gt)} gt lines, {len(imgs)} frames")
EOF
```

Edit the training config before launching:

```yaml
# experiments/sglatrack/deit_distilled.yaml — verify these values:
TRAIN:
  BATCH_SIZE: 16         # lower to 8 if you see OOM
  NUM_EPOCH: 20          # 20 is enough for fine-tuning; don't do 40+
  LR: 0.0001             # 1e-4 for fine-tuning from pretrained
  LR_DROP_EPOCH: 15      # drop at epoch 15
  GRAD_CLIP_NORM: 0.1
  PRETRAINED_PATH: './pretrained_models/deit_tiny_distilled_patch16_224.pth'

DATA:
  TRAIN:
    DATASETS_NAME: ['COMP']   # your competition dataset
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
```

Also register the competition dataset in the data loader:

```python
# lib/train/dataset/comp.py — create this file based on GOT10k dataset class
# The simplest approach: symlink or subclass GOT10k since you're using the same format

# In lib/train/base_functions.py, add:
from lib.train.dataset.comp import Comp

def build_dataloaders(cfg, settings):
    ...
    if 'COMP' in cfg.DATA.TRAIN.DATASETS_NAME:
        comp_train = Comp(settings.env.comp_dir, split='train')
        dataset_train = sampler([comp_train], ...)
```

---

## Hour 3: Hanning Patch Lands on Machine 1

Ahmed pulls the patch from Nour, applies it, verifies sanity check passes:

```bash
git apply hanning_patch.diff
python verify_hanning.py
# Must print: after penalty peak shifts toward center
# If it doesn't — call Nour before training starts
```

---

## Hour 3–4 | Both: Training Launches

**Machine 1 — SGLATrack (Ahmed + Leil):**

```bash
python tracking/train.py \
    --script sglatrack \
    --config deit_distilled \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 1 \
    --use_wandb 0 \
    2>&1 | tee logs/sgla_train.log &

# Monitor: loss should decrease by epoch 3
watch -n 30 "tail -5 logs/sgla_train.log"
```

**Machine 2 — OSTrack (Nour + Yassin):**

```bash
cd OSTrack
python tracking/train.py \
    --script ostrack \
    --config vitb_256_mae_ce_32x4_ep300 \
    --save_dir ./output_ostrack \
    --mode multiple \
    --nproc_per_node 1 \
    2>&1 | tee logs/ostrack_train.log &

watch -n 30 "tail -5 logs/ostrack_train.log"
```

Both jobs run overnight. Don't stop them unless loss diverges (goes up after epoch 5).

---

## Hour 4–8 | Ahmed: ONNX + INT8 Pipeline (Machine 1)

Training is running in background. Build the export pipeline now on any checkpoint.
You'll swap in final weights on Day 2.

### ONNX Export

```python
# export_onnx.py
import torch, os, argparse

def export(checkpoint_path, output_path, cfg):
    from lib.models.sglatrack import build_sglatrack
    
    model = build_sglatrack(cfg)
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    
    # Try different key names
    state = ckpt.get('net', ckpt.get('model', ckpt))
    model.load_state_dict(state)
    model.eval()

    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)

    # Dry run first
    with torch.no_grad():
        out = model(template, search)
    print("Dry run OK:", type(out))

    torch.onnx.export(
        model,
        (template, search),
        output_path,
        input_names=['template', 'search'],
        output_names=['score_map', 'size_map', 'offset_map'],
        opset_version=17,
        dynamic_axes={
            'template': {0: 'batch'},
            'search':   {0: 'batch'},
        }
    )
    print(f"Exported: {output_path}")
    print(f"Size: {os.path.getsize(output_path)/1e6:.1f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='sglatrack.onnx')
    args = parser.parse_args()
    export(args.checkpoint, args.output, cfg)
```

### ONNX Verification

```python
# verify_onnx.py
import torch, numpy as np
import onnxruntime as ort

def verify(onnx_path, pt_model):
    sess = ort.InferenceSession(onnx_path,
               providers=['CPUExecutionProvider'])
    
    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)
    
    # ONNX output
    onnx_out = sess.run(None, {
        'template': template.numpy(),
        'search':   search.numpy()
    })
    
    # PyTorch output
    with torch.no_grad():
        pt_out = pt_model(template, search)
    
    # Compare first output (score_map)
    pt_arr = pt_out['score_map'].numpy() if isinstance(pt_out, dict) else pt_out[0].numpy()
    diff   = np.abs(onnx_out[0] - pt_arr).max()
    print(f"Max output diff (PT vs ONNX): {diff:.6f}")
    assert diff < 0.01, f"FAIL: diff too large ({diff})"
    print("ONNX verification PASSED")
```

### INT8 Quantization

```python
# quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType
import os, argparse

def quantize(input_path, output_path):
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8
    )
    orig_mb = os.path.getsize(input_path)  / 1e6
    q_mb    = os.path.getsize(output_path) / 1e6
    print(f"FP32: {orig_mb:.1f}MB → INT8: {q_mb:.1f}MB  ({q_mb/orig_mb*100:.0f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    quantize(args.input, args.output)
```

### Latency Benchmark

```python
# benchmark_latency.py
import time, numpy as np
import onnxruntime as ort

def benchmark(onnx_path, n_runs=200, warmup=20):
    sess = ort.InferenceSession(onnx_path,
               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    dummy = {
        'template': np.random.randn(1, 3, 128, 128).astype(np.float32),
        'search':   np.random.randn(1, 3, 256, 256).astype(np.float32),
    }
    
    for _ in range(warmup):
        sess.run(None, dummy)
    
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, dummy)
        times.append((time.perf_counter() - t0) * 1000)
    
    gpu_ms  = np.mean(times)
    jet_est = gpu_ms * 8  # rough Jetson Orin Nano 4GB factor
    
    print(f"GPU latency  (mean): {gpu_ms:.2f}ms")
    print(f"Jetson est   (8×):   {jet_est:.1f}ms  (cap: 30ms)")
    print(f"Status: {'✅ PASS' if jet_est < 30 else '⚠️ BORDERLINE — INT8 required'}")
    return gpu_ms, jet_est

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    args = p.parse_args()
    benchmark(args.model)
```

Run this on both FP32 and INT8 models. Report numbers to team.

---

## Hour 4–8 | Yassin: Local Eval Harness (Machine 2)

This unblocks all future decisions. Build it completely today.

```python
# local_eval.py
import os, json, glob, argparse
import numpy as np

# ── Metrics ────────────────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx);       iy = max(ay, by)
    iw = min(ax+aw, bx+bw) - ix
    ih = min(ay+ah, by+bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def compute_auc(preds, gts):
    """Success plot AUC over IoU thresholds 0→1."""
    ious = []
    for p, g in zip(preds, gts):
        if g is None:    continue   # skip invisible
        if p is None:    continue
        ious.append(compute_iou(p, g))
    if not ious:
        return 0.0
    thresholds = np.linspace(0, 1, 101)
    rates = [np.mean([io >= t for io in ious]) for t in thresholds]
    return float(np.mean(rates))

def compute_norm_precision(preds, gts, threshold=0.5):
    """Normalized precision: fraction of frames where normalized center error < threshold."""
    scores = []
    for p, g in zip(preds, gts):
        if g is None or p is None: continue
        norm = np.sqrt(g[2] * g[3])
        if norm == 0: continue
        cx_p = p[0] + p[2]/2;  cy_p = p[1] + p[3]/2
        cx_g = g[0] + g[2]/2;  cy_g = g[1] + g[3]/2
        dist = np.sqrt((cx_p-cx_g)**2 + (cy_p-cy_g)**2)
        scores.append(1.0 if (dist/norm) < threshold else 0.0)
    return float(np.mean(scores)) if scores else 0.0

# ── Data loading ────────────────────────────────────────────────────────────

def load_gt(ann_path):
    """Returns list of (x,y,w,h) or None per frame."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            x, y, w, h = map(float, line.split(','))
            boxes.append(None if (w == 0 and h == 0) else (x, y, w, h))
    return boxes

def load_preds(pred_path):
    """Prediction file: one x,y,w,h per line."""
    preds = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            preds.append(tuple(map(float, line.split(','))))
    return preds

# ── Eval loop ───────────────────────────────────────────────────────────────

def evaluate(pred_dir, gt_dir, seq_list=None):
    if seq_list is None:
        seq_list = [d for d in os.listdir(gt_dir)
                    if os.path.isdir(os.path.join(gt_dir, d))]

    results = []
    for seq in sorted(seq_list):
        ann_path  = os.path.join(gt_dir, seq, 'annotation.txt')
        pred_path = os.path.join(pred_dir, f'{seq}.txt')

        if not os.path.exists(ann_path):
            print(f"  [skip] No GT: {seq}")
            continue
        if not os.path.exists(pred_path):
            print(f"  [skip] No pred: {seq}")
            continue

        gt    = load_gt(ann_path)
        preds = load_preds(pred_path)

        # Frame 0 is init — skip in evaluation
        gt_eval    = gt[1:]
        preds_eval = preds[1:] if len(preds) > len(gt_eval) else preds

        # Align lengths
        min_len = min(len(gt_eval), len(preds_eval))
        gt_eval    = gt_eval[:min_len]
        preds_eval = preds_eval[:min_len]

        auc = compute_auc(preds_eval, gt_eval)
        np_ = compute_norm_precision(preds_eval, gt_eval)
        results.append({'seq': seq, 'auc': auc, 'np': np_})

    if not results:
        print("No results found.")
        return

    auc_mean = np.mean([r['auc'] for r in results])
    np_mean  = np.mean([r['np']  for r in results])
    s_acc    = 0.6 * auc_mean + 0.4 * np_mean

    print(f"\n{'─'*50}")
    print(f"Sequences evaluated: {len(results)}")
    print(f"AUC:           {auc_mean:.4f}")
    print(f"NormPrecision: {np_mean:.4f}")
    print(f"S_acc:         {s_acc:.4f}  ← your accuracy score")
    print(f"{'─'*50}")
    return auc_mean, np_mean, s_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, help='Folder of seq_name.txt prediction files')
    parser.add_argument('--gt_dir',   required=True, help='Folder of public_lb sequences')
    args = parser.parse_args()
    evaluate(args.pred_dir, args.gt_dir)
```

### Generate Fallback CSV (Yassin — do this today)

Run pretrained SGLATrack on all 89 public sequences. This is your insurance:

```python
# run_tracker.py
import os, json, cv2, argparse
import numpy as np

def run_sequence(tracker, seq_path, init_box, out_txt):
    """Run tracker on a sequence, write predictions to out_txt."""
    frame_dir = os.path.join(seq_path, 'img')
    frames    = sorted([f for f in os.listdir(frame_dir)
                        if f.endswith(('.jpg', '.png', '.jpeg'))])

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    lines = []

    # Frame 0: use init box directly
    lines.append(f"{init_box[0]},{init_box[1]},{init_box[2]},{init_box[3]}")
    tracker.initialize(os.path.join(frame_dir, frames[0]), init_box)

    for frame_file in frames[1:]:
        frame_path = os.path.join(frame_dir, frame_file)
        box = tracker.track(frame_path)   # returns (x,y,w,h)
        lines.append(f"{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}")

    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))

    return lines

def generate_submission_csv(tracker, manifest_path, data_root, pred_root, output_csv):
    import csv
    with open(manifest_path) as f:
        manifest = json.load(f)

    pub_entries = [e for e in manifest if e.get('split') == 'public_lb']

    rows = []
    for entry in pub_entries:
        seq_id   = entry['seq_id']
        seq_path = os.path.join(data_root, seq_id)
        ann_path = os.path.join(seq_path, 'annotation.txt')

        with open(ann_path) as f:
            first_line = f.readline().strip()
        init_box = tuple(map(float, first_line.split(',')))

        flat_name = seq_id.replace('/', '_')
        out_txt   = os.path.join(pred_root, f'{flat_name}.txt')

        print(f"Running: {seq_id} ...", end=' ', flush=True)
        preds = run_sequence(tracker, seq_path, init_box, out_txt)
        print(f"{len(preds)} frames")

        for frame_idx, line in enumerate(preds):
            vals = line.split(',')
            x, y, w, h = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
            rows.append({
                'id':  f"{seq_id}_{frame_idx}",
                'x':   round(x, 2),
                'y':   round(y, 2),
                'w':   round(w, 2),
                'h':   round(h, 2),
            })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id','x','y','w','h'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written: {output_csv}  ({len(rows)} rows)")
    assert len(rows) == 74293, f"Row mismatch: {len(rows)}"
    return output_csv
```

### Validate CSV

```python
# validate_submission.py
import pandas as pd, sys

def validate(submission_path, sample_path):
    sub    = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)

    errors = []

    if len(sub) != len(sample):
        errors.append(f"Row count: {len(sub)} (expected {len(sample)})")

    sub_ids    = set(sub['id'])
    sample_ids = set(sample['id'])
    missing    = sample_ids - sub_ids
    extra      = sub_ids - sample_ids
    if missing: errors.append(f"Missing IDs: {len(missing)} (e.g. {list(missing)[:3]})")
    if extra:   errors.append(f"Extra IDs:   {len(extra)}")

    if sub.isnull().sum().sum() > 0:
        errors.append("NaN values found")

    neg_wh = (sub[['w','h']] < 0).any().any()
    if neg_wh:
        errors.append("Negative w or h found")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors: print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("✅ Submission valid")
        print(f"   Rows:    {len(sub)}")
        print(f"   Columns: {list(sub.columns)}")
        print(sub[['x','y','w','h']].describe().to_string())
```

---

## Hour 8–12 | Efficiency Report (Ahmed — Machine 1)

Compute and log these numbers. Send to the group chat before everyone sleeps.

```python
# efficiency_report.py
import torch, time, os, numpy as np
from thop import profile  # pip install thop

def report(model, onnx_fp32_path, onnx_int8_path):
    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)

    macs, params = profile(model, inputs=(template, search), verbose=False)

    # Model size from checkpoint
    torch.save(model.state_dict(), '/tmp/_tmp_weights.pth')
    size_mb = os.path.getsize('/tmp/_tmp_weights.pth') / 1e6

    # GPU latency
    model_gpu = model.cuda().eval()
    t_gpu, s_gpu = template.cuda(), search.cuda()
    with torch.no_grad():
        for _ in range(20): model_gpu(t_gpu, s_gpu)
        t0 = time.perf_counter()
        for _ in range(200): model_gpu(t_gpu, s_gpu)
        gpu_ms = (time.perf_counter() - t0) / 200 * 1000

    jet_est = gpu_ms * 8

    # S_eff
    s_eff = (0.25 * params/1e6 / 50 +
             0.15 * (macs*2/1e9) / 30 +
             0.35 * jet_est / 30 +
             0.25 * size_mb / 500)

    print("=" * 45)
    print("EFFICIENCY REPORT")
    print("=" * 45)
    print(f"Params:         {params/1e6:.2f}M   / 50M cap")
    print(f"FLOPs:          {macs*2/1e9:.2f}G   / 30G cap")
    print(f"Model size:     {size_mb:.1f}MB / 500MB cap")
    print(f"GPU latency:    {gpu_ms:.2f}ms")
    print(f"Jetson est:     {jet_est:.1f}ms  / 30ms cap")
    print(f"S_eff penalty:  {s_eff:.4f}")
    print(f"Score lost:     {0.2 * s_eff:.4f}  (out of total score)")
    print("=" * 45)
    if jet_est > 30:
        print("⚠️  INT8 is MANDATORY to clear latency cap")
    else:
        print("✅  Latency clears cap even without INT8")
```

---

## Hour 12: Day 1 Sync — 6 Questions

Answer these before anyone sleeps:

| # | Question | Owner | Needed |
|---|---|---|---|
| 1 | Is SGLATrack loss decreasing by epoch 5? | Leil | ✅ yes / ❌ crash |
| 2 | Is OSTrack loss decreasing? | Yassin | ✅ yes / ❌ crash |
| 3 | Does Hanning shift response peak toward center? | Nour | ✅ yes / ❌ broken |
| 4 | Does ONNX export work? | Ahmed | ✅ yes / ❌ error |
| 5 | What is Jetson latency estimate? | Ahmed | number in ms |
| 6 | Is fallback CSV generated and validated? | Yassin | ✅ yes / ❌ missing |

**If #6 is no → Yassin stays up to fix it. All other gaps can be fixed morning of Day 2.**

---

---

# DAY 2 — Lock and Ship

---

## Hours 0–3: Pull Best Checkpoints

Training finished overnight. First action: find best epoch before doing anything else.

**Yassin — sweep SGLATrack checkpoints** (on Machine 2, using Machine 1's checkpoint files):

```bash
for epoch in 5 10 15 20; do
    python tracking/test.py \
        --tracker_name sglatrack \
        --tracker_param deit_distilled \
        --checkpoint /path/to/SGLATrack/output/checkpoints/epoch_${epoch}.pth \
        --dataset comp_val \
        --hanning_lambda 0.49 \
        --output_dir ./results/sgla_ep${epoch}

    python local_eval.py \
        --pred_dir ./results/sgla_ep${epoch} \
        --gt_dir   /path/to/public_lb
done
```

Pick the epoch with highest S_acc. **That checkpoint is now frozen.**

**Nour — λ sweep on frozen SGLATrack checkpoint:**

```bash
BEST_CKPT=/path/to/SGLATrack/output/checkpoints/epoch_XX.pth

for lam in 0.3 0.40 0.49 0.60 0.70; do
    python tracking/test.py \
        --checkpoint $BEST_CKPT \
        --hanning_lambda $lam \
        --dataset comp_val \
        --output_dir ./results/lam_${lam}

    python local_eval.py \
        --pred_dir ./results/lam_${lam} \
        --gt_dir   /path/to/public_lb
done

# Lock in best λ. Record it. Never change it again.
```

**Yassin — evaluate OSTrack checkpoint** for distillation decision:

```bash
python tracking/test.py \
    --tracker_name ostrack \
    --checkpoint ./output_ostrack/checkpoints/best.pth \
    --dataset comp_val \
    --output_dir ./results/ostrack_best

python local_eval.py \
    --pred_dir ./results/ostrack_best \
    --gt_dir   /path/to/public_lb
```

---

## Hour 2: Distillation Decision Gate

Look at the numbers Yassin just produced:

| Condition | Decision |
|---|---|
| SGLATrack S_acc ≥ OSTrack S_acc | Skip distillation. SGLATrack fine-tuned is final. |
| OSTrack − SGLATrack gap < 2% | Skip distillation. Not worth the risk. |
| Gap ≥ 2% AND time < hour 3 | Attempt distillation: 5 epochs max (Nour sets up) |
| Gap ≥ 2% BUT time ≥ hour 3 | Skip distillation. Ship fine-tuned SGLATrack. |

**If distilling (Nour):**

```python
# distill.py — response-map KL distillation (5 epochs only)
import torch
import torch.nn.functional as F

def distillation_loss(teacher_out, student_out, gt_labels, alpha=0.5, temp=4.0):
    # Task loss: student's own supervised loss
    task_loss = compute_task_loss(student_out, gt_labels)

    # Distillation: match teacher's response map distribution
    t_resp = teacher_out['score_map'].view(teacher_out['score_map'].size(0), -1)
    s_resp = student_out['score_map'].view(student_out['score_map'].size(0), -1)

    t_soft = F.softmax(t_resp / temp, dim=-1)
    s_log  = F.log_softmax(s_resp / temp, dim=-1)
    kl_loss = F.kl_div(s_log, t_soft, reduction='batchmean') * (temp ** 2)

    return (1 - alpha) * task_loss + alpha * kl_loss
```

Run 5 epochs. Evaluate immediately. If distilled S_acc > fine-tuned S_acc → swap checkpoint. Otherwise → discard.

---

## Hours 3–5: Final Model Build (Ahmed + Leil)

One final checkpoint is now selected. Build the submission artifact.

```bash
# Lock in the final checkpoint
FINAL_CKPT=./output/sglatrack/best_final.pth
BEST_LAM=0.49   # whatever Nour's sweep returned

# Step 1: Export ONNX
python export_onnx.py \
    --checkpoint $FINAL_CKPT \
    --output     sglatrack_final.onnx

# Step 2: Verify ONNX
python verify_onnx.py \
    --onnx      sglatrack_final.onnx \
    --checkpoint $FINAL_CKPT
# Max diff must be < 0.01

# Step 3: INT8 quantize
python quantize.py \
    --input  sglatrack_final.onnx \
    --output sglatrack_int8.onnx

# Step 4: Benchmark both — confirm INT8 clears latency cap
python benchmark_latency.py --model sglatrack_final.onnx
python benchmark_latency.py --model sglatrack_int8.onnx

# Step 5: Run local eval on INT8 to confirm accuracy drop < 2%
python tracking/test.py \
    --onnx           sglatrack_int8.onnx \
    --hanning_lambda $BEST_LAM \
    --dataset        comp_val \
    --output_dir     ./results/int8_final

python local_eval.py \
    --pred_dir ./results/int8_final \
    --gt_dir   /path/to/public_lb
# If S_acc drops > 2% vs FP32 → submit FP32 version instead
```

---

## Hours 5–7: Generate Final Submission CSV (Yassin)

```bash
python run_tracker.py \
    --onnx           sglatrack_int8.onnx \
    --hanning_lambda $BEST_LAM \
    --manifest       contestant_manifest.json \
    --data_root      /path/to/competition_data \
    --pred_root      ./final_predictions \
    --output_csv     submission_final.csv

python validate_submission.py \
    --submission submission_final.csv \
    --sample     sample_submission.csv
```

**Expected output:**
```
✅ Submission valid
   Rows:    74293
   Columns: ['id', 'x', 'y', 'w', 'h']
```

If validation fails:
- Row count wrong → check manifest parsing, missing sequences
- ID format wrong → check `f"{seq_id}_{frame_idx}"` matches sample exactly
- NaN values → tracker crashed on a frame, add fallback to last known box

---

## Hours 7–8: Buffer

**Do not touch model weights, λ, or any code after hour 7.**

Use this time for:
- Cross-checking 5 random rows in submission CSV against expected format
- Uploading to Kaggle
- Confirming public LB score appears (sanity check — don't obsess over it)
- Writing down final efficiency numbers for the submission form if required

---

---

# Final Numbers to Record

Fill this in during Day 2 before submitting:

| Metric | Value | Cap | Status |
|---|---|---|---|
| Parameters | __ M | 50M | |
| FLOPs | __ G | 30G | |
| Model size | __ MB | 500MB | |
| GPU latency | __ ms | — | |
| Jetson estimate | __ ms | 30ms | |
| S_eff | __ | — | |
| AUC (local eval) | __ | — | |
| NormPrecision (local) | __ | — | |
| S_acc (local eval) | __ | — | |

---

# Critical Rules — Read Before Day 2

1. **Never submit without local eval first.** Yassin's harness exists for this reason.
2. **Never tune λ after it's locked.** One sweep, one decision, done.
3. **Never run distillation past hour 3 on Day 2.** Fine-tuned baseline ships if distillation takes too long.
4. **Never trust public LB as ground truth.** 70% of score is hidden. Generalization wins.
5. **Always have a CSV.** Fallback hierarchy must be maintained at all times.
6. **INT8 accuracy drop > 2% AUC → ship FP32.** The latency penalty is small; an accuracy drop is not.
