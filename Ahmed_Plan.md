# Ahmed's Plan — OSTrack Fine-tuning & Baseline Run

> **Status of Nour's work:** All post-processing modules are complete and wired into the inference pipeline. You can run inference immediately with the pretrained weights.

---

## Which Model Should You Use?

**Use: `vitb_256_mae_ce_32x4_ep300`**

| Variant | GFLOPs | Budget OK? | Accuracy | Decision |
|---------|--------|------------|----------|----------|
| vitb_256_mae_32x4_ep300 | 17.2G | ✅ | Lower (no CE) | Skip |
| **vitb_256_mae_ce_32x4_ep300** | **17.2G** | **✅** | **Best within budget** | **USE THIS** |
| vitb_256_mae_ce_32x4_got10k_ep100 | 17.2G | ✅ | Lower (single dataset) | Skip |
| vitb_384_mae_ce_32x4_ep300 | ~58G | ❌ Exceeds limit | Best overall | Disqualified |

**Why:** The CE (Candidate Elimination) variant trained on all 4 datasets (LaSOT + GOT-10k + COCO + TrackingNet) gives the best accuracy while staying under the 30 GFLOPs hard budget. The 384px version is more accurate but blows the budget at 58 GFLOPs.

**Paths (already downloaded):**
- Checkpoint: `OS_Track Model/pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar`
- Config: `OS_Track Model/OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml`

---

## How Long Will Training Take?

| Setting | Value |
|---------|-------|
| Your GPU | RTX 5060 (~12GB VRAM, ~22 TFLOPS FP16) |
| Epochs | 30 |
| Samples/epoch | 60,000 |
| Batch size | 16 (try 32 with AMP — faster) |
| Iterations/epoch | 3,750 |

| Batch Size | Time per Epoch | Total (30 epochs) |
|------------|---------------|-------------------|
| 16 + AMP | ~15–20 min | **~7–10 hours** |
| 32 + AMP | ~8–13 min | **~4–7 hours** |

**Launch it before you go to sleep.** Check in the morning.

> ⚠️ Rule: If 1 epoch takes more than 5 hours → immediately cut `EPOCH: 30` to `EPOCH: 15` in the config.

---

## Step-by-Step Tasks

### Step 1 — Extract Remaining Frames

Only `dataset1` has been extracted so far. Run this to extract datasets 2–5 (it will skip dataset1 automatically):

```bash
cd C:/Projects/AIC-Tracker/AIC-Tracker

python shared/extract_frames.py \
  --manifest data/metadata/contestant_manifest.json \
  --split train \
  --output frames
```

---

### Step 2 — Run Baseline Inference (Pretrained Weights, 5 Sequences)

Test that everything works end-to-end before committing to a full run:

```bash
cd C:/Projects/AIC-Tracker/AIC-Tracker

python "OS_Track Model/run_inference.py" \
  --manifest data/metadata/contestant_manifest.json \
  --split train \
  --frames_root frames \
  --checkpoint "OS_Track Model/pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar" \
  --config "OS_Track Model/OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml" \
  --output_dir predictions/ostrack/baseline \
  --max_seqs 5
```

Then score it:

```bash
python shared/evaluate.py \
  --pred_dir predictions/ostrack/baseline \
  --manifest data/metadata/contestant_manifest.json \
  --split train \
  --params_m 28.5 --flops_g 17.2 --latency_ms 14.3 --size_gb 0.11
```

If the scores look reasonable → run the full baseline (remove `--max_seqs 5`).

---

### Step 3 — Create the MTCDataset Adapter

Create this file: `OS_Track Model/OSTrack/lib/train/dataset/mtc_dataset.py`

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

Then register it — add this line to `OS_Track Model/OSTrack/lib/train/dataset/__init__.py`:

```python
from .mtc_dataset import MTCDataset
```

---

### Step 4 — Create the Fine-tuning Config

Create this file: `OS_Track Model/OSTrack/experiments/ostrack/mtc_finetune.yaml`

```yaml
TRAIN:
  LR: 4e-5
  WEIGHT_DECAY: 1e-4
  EPOCH: 30           # Cut to 15 if 1 epoch > 5 hours
  BATCH_SIZE: 16      # Increase to 32 if VRAM allows
  AMP: True
  GRAD_CLIP_NORM: 0.1
  SCHEDULER:
    TYPE: step
    DECAY_RATE: 0.1
    MILESTONES: [20]

DATA:
  SEARCH:
    SIZE: 256
    FACTOR: 4.0
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

---

### Step 5 — Launch Fine-tuning Overnight

```bash
cd "C:/Projects/AIC-Tracker/AIC-Tracker/OS_Track Model/OSTrack"

python tracking/train.py \
  --script ostrack \
  --config mtc_finetune \
  --save_dir output/mtc_run1 \
  --use_wandb 0 \
  --pretrained_path ../pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
```

Monitor the first epoch loss. If it's diverging (loss going up instead of down) → halve the LR to `2e-5` and relaunch.

---

### Step 6 — Evaluate Fine-tuned Checkpoint (Next Day)

```bash
cd C:/Projects/AIC-Tracker/AIC-Tracker

python "OS_Track Model/run_inference.py" \
  --manifest data/metadata/contestant_manifest.json \
  --split train \
  --frames_root frames \
  --checkpoint "OS_Track Model/output/mtc_run1/checkpoints/ostrack_ep0030.pth.tar" \
  --config "OS_Track Model/OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml" \
  --output_dir predictions/ostrack/v1_finetuned

python shared/evaluate.py \
  --pred_dir predictions/ostrack/v1_finetuned \
  --manifest data/metadata/contestant_manifest.json \
  --split train \
  --params_m 28.5 --flops_g 17.2 --latency_ms 14.3 --size_gb 0.11
```

Log the AUC and FinalScore to `scores.xlsx` (row: A-v1).

---

## Post-processing Flags Available (from Nour)

Once you have fine-tuned weights, try these flags on `run_inference.py`:

| Flag | What it does | When to use |
|------|-------------|-------------|
| `--use_hanning --hanning_weight 0.49` | Suppresses jumpy predictions | Day 3+ ablation |
| `--use_dual_template` | Static + dynamic template fusion | Day 3+ ablation |
| `--use_multi_scale` | 3-scale search (adds ~3–5ms) | Day 5 only if latency budget allows |

Example with all post-processing:
```bash
python "OS_Track Model/run_inference.py" \
  --checkpoint "OS_Track Model/output/mtc_run1/checkpoints/ostrack_ep0030.pth.tar" \
  --config "OS_Track Model/OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml" \
  --manifest data/metadata/contestant_manifest.json \
  --split train \
  --frames_root frames \
  --output_dir predictions/ostrack/v3_hanning \
  --use_hanning --hanning_weight 0.49 \
  --use_dual_template
```

---

## Summary Checklist

- [ ] Extract all frames (datasets 2–5)
- [ ] Run 5-sequence baseline sanity check
- [ ] Run full baseline inference → log A-base row in scores.xlsx
- [ ] Create `mtc_dataset.py` and register it
- [ ] Create `mtc_finetune.yaml`
- [ ] Launch fine-tuning overnight
- [ ] Evaluate fine-tuned checkpoint → log A-v1 row
- [ ] Ablate post-processing (Hanning + dual template) → log A-v2, A-v3
