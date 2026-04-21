# MTC-AIC4: Efficient Aerial Single-Object Tracking

Competition organized by the **Military Technical College × Applied Innovation Center, Egypt**.

## Quick Overview

| Field | Value |
|-------|-------|
| **Task** | Single-Object Tracking (SOT) on UAV aerial videos |
| **Scoring** | 80% accuracy (AUC + normalized precision) + 20% efficiency (FLOPs, latency, params, size) |
| **Team Size** | 5 members across 2 models |
| **Timeline** | 6 days (Day 1 setup → Day 6 submit) |

## Final Score Formula

```
FinalScore = 0.8 × AccuracyScore + 0.2 × EfficiencyScore

where:
  AccuracyScore   = 0.6 × AUC + 0.4 × NormPrecision
  EfficiencyScore = 0.25×(1-p/50) + 0.15×(1-f/30) + 0.35×(1-l/30) + 0.25×(1-s/0.5)
  p = params (M), f = FLOPs (G), l = latency (ms), s = size (GB)
```

## Efficiency Hard Budgets

| Metric | Maximum |
|--------|---------|
| FLOPs | 30 GFLOPs |
| Parameters | 50 Million |
| Inference Latency | 30 ms |
| Model Size | 0.5 GB |

## Team Structure

### Team A – OSTrack-256 (Accuracy-Focused)
- **Ahmed** (RTX 5060) – Lead trainer, all training runs
- **Nour** (RTX 3060) – Post-processing engineer, dual template, Hanning window, ablations
- **Yassin** (RTX 4050) – Evaluation engineer, prediction writing, metric logging

### Team B – LightTrack (Efficiency-Focused)
- **Leil** (RTX 3050) – Fine-tuning on lightweight model
- **Barawy** (Quadro T500) – Data prep, ONNX export, INT8 quantization, efficiency metrics

## Project Structure

```
AIC-Tracker/
├── shared/                    ← Shared infrastructure (Ahmed owns Day 1)
│   ├── extract_frames.py      ← MP4 to JPEG frames
│   ├── annotation_utils.py    ← Load/validate annotations
│   ├── manifest_loader.py     ← Dataset metadata parser
│   ├── prediction_writer.py   ← Save tracker output (Yassin)
│   ├── evaluate.py            ← Official scoring script
│   └── measure_efficiency.py  ← FLOPs, latency, size measurement (Barawy)
│
├── OS_Track Model/                    ← OSTrack workspace
│   ├── OSTrack/               ← (cloned from GitHub, Day 1)
│   ├── pretrained/            ← Pretrained checkpoint
│   ├── output/                ← Fine-tuned checkpoints
│   ├── dual_template.py       ← Static + dynamic template fusion (Nour)
│   ├── hanning_penalty.py     ← Suppress high-displacement (Nour)
│   ├── multi_scale_inference.py ← Multi-scale search (Nour, Day 5 only)
│   └── run_inference.py       ← Full inference pipeline (stub)
│
├── HIT Model/                    ← LightTrack workspace
│   ├── LightTrack/            ← (cloned from GitHub, Day 1)
│   ├── pretrained/            ← Pretrained checkpoint
│   ├── output/                ← Fine-tuned checkpoints
│   ├── export_onnx.py         ← ONNX export (Barawy)
│   ├── quantize.py            ← INT8 quantization (Barawy)
│   └── run_inference.py       ← Full inference pipeline (stub)
│
├── frames/                    ← Extracted video frames
│   ├── train/                 ← Training split (built by Ahmed)
│   └── public_lb/             ← Public leaderboard (extracted Day 6)
│
├── data/                      ← Raw competition dataset (read-only)
│   └── contestant_manifest.json
│
├── predictions/               ← Tracker outputs (Yassin owns)
│   ├── ostrack/
│   │   ├── baseline/
│   │   ├── v1_finetuned/
│   │   ├── v2_dual_template/
│   │   └── v3_hanning/
│   └── lighttrack/
│       ├── baseline/
│       ├── v1_finetuned/
│       └── v2_int8/
│
├── evaluation/                ← Eval results and metrics
├── submission/                ← Final submission package
│   ├── predictions/           ← Final txts for public_lb
│   └── model/                 ← Final model weights
│
└── scores.xlsx                ← Shared Google Sheet (all runs/metrics)
```

## Getting Started

### Day 1 Morning (Ahmed)
```bash
cd team_a
git clone https://github.com/botaoye/OSTrack.git
cd OSTrack
# Create environment and download pretrained checkpoint
```

### Day 1 Morning (Barawy)
```bash
python shared/extract_frames.py --split train --output frames/train
# Validates all annotation.txt files
```

### Day 1 Afternoon (Yassin)
```bash
python shared/evaluate.py \
  --pred_dir predictions/ostrack/baseline \
  --manifest data/contestant_manifest.json \
  --split train \
  --params_m 28.5 --flops_g 17.2 --latency_ms 14.3 --size_gb 0.11
```

## Key Dates

| Day | Milestone | Critical Task |
|-----|-----------|---------------|
| 1 | Setup complete | Both models running baseline |
| 2 | Fine-tuning started | Both training runs launched overnight |
| 3 | Optimize + ablate | Dual template, Hanning, INT8 integration |
| 4 @ 11am | Decision meeting | Pick one model (spreadsheet decides) |
| 5 | Polish only | No new experiments, budget verification |
| 6 @ 3pm | Submit | All prediction txts packaged and uploaded |

## Shared Rules

1. **Never modify another member's code without asking first**
2. **All shared scripts in `shared/` – never duplicate**
3. **Update `scores.xlsx` after every major run** (AUC, metrics, FinalScore)
4. **If 1 epoch > 5 hours → cut training epochs from 30 to 15 immediately**
5. **Day 5 is polish only – no new experiments after Day 4 noon**

## References

- **OSTrack**: https://github.com/botaoye/OSTrack
- **LightTrack**: https://github.com/researchmm/LightTrack
- **Full Competition Plan**: See `MTC-AIC4_Full_Plan.md` in root