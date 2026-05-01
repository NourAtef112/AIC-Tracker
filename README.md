# MTC-AIC4: Efficient Aerial Single-Object Tracking

Competition organized by the **Military Technical College × Applied Innovation Center, Egypt**.

## Quick Overview

| Field | Value |
|-------|-------|
| **Task** | Single-Object Tracking (SOT) on UAV aerial videos |
| **Scoring** | 80% accuracy (AUC + normalized precision) + 20% efficiency (FLOPs, latency, params, size) |
| **Team Size** | 5 members across 2 machines |
| **Timeline** | 2 days (Day 1 build → Day 2 submit) |

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

> **Submission model:** SGLATrack (DeiT-tiny) — fine-tuned + Hanning + INT8
> **Teacher model:** OSTrack (ViT-B) — distillation only, never submitted (exceeds 50M param cap)

### Machine 1 – SGLATrack (Ahmed + Leil)
- **Ahmed** (RTX 5060) – Lead trainer, fine-tune + ONNX + INT8 pipeline
- **Leil** (RTX 3050) – Dataset prep, config setup, training support

### Machine 2 – OSTrack + Eval (Nour + Yassin)
- **Nour** (RTX 3060) – Hanning window, λ sweep, distillation (if triggered)
- **Yassin** (RTX 4050) – Local eval harness, checkpoint sweep, submission CSV

## Project Structure

```
AIC-Tracker/
├── shared/                      ← Shared infrastructure
│   ├── extract_frames.py        ← MP4 to JPEG frames
│   ├── annotation_utils.py      ← Load/validate annotations
│   ├── manifest_loader.py       ← Dataset metadata parser
│   ├── prediction_writer.py     ← Save tracker output
│   ├── evaluate.py              ← Official scoring script
│   ├── measure_efficiency.py    ← FLOPs, latency, size measurement
│   ├── convert_dataset.py       ← Convert competition data format
│   ├── local_eval.py            ← Quick local evaluation shortcut
│   └── validate_submission.py   ← Validate CSV before submitting
│
├── SGLA_Track Model/            ← SGLATrack workspace (submission model)
│   ├── SGLATrack/               ← (cloned from GitHub)
│   ├── pretrained/              ← DeiT-tiny pretrained weights
│   ├── output/                  ← Fine-tuned checkpoints
│   ├── export_onnx.py           ← ONNX export (Ahmed)
│   ├── quantize.py              ← INT8 quantization (Ahmed)
│   ├── benchmark_latency.py     ← Latency benchmarking (Ahmed)
│   ├── efficiency_report.py     ← Full efficiency metrics report
│   └── run_tracker.py           ← Full inference pipeline
│
├── OS_Track Model/              ← OSTrack workspace (teacher model only)
│   ├── OSTrack/                 ← (cloned from GitHub)
│   ├── pretrained/              ← ViT-B pretrained checkpoint
│   ├── output/                  ← Fine-tuned checkpoints
│   ├── dual_template.py         ← Static + dynamic template fusion (Nour)
│   ├── hanning_penalty.py       ← Suppress high-displacement (Nour)
│   ├── multi_scale_inference.py ← Multi-scale search (Nour)
│   ├── distill.py               ← Knowledge distillation (Nour, Day 2)
│   └── run_inference.py         ← Full inference pipeline
│
├── frames/                      ← Extracted video frames
│   ├── train/                   ← Training split
│   └── public_lb/               ← Public leaderboard sequences
│
├── data/                        ← Raw competition dataset (read-only)
│   └── contestant_manifest.json
│
├── predictions/                 ← Tracker outputs
│   ├── ostrack/
│   └── lighttrack/
│
├── evaluation/                  ← Eval results and metrics
├── submission/                  ← Final submission package
│   ├── predictions/             ← Final CSVs for public_lb
│   └── model/                   ← Final model weights
│
└── scores.xlsx                  ← Shared metrics tracker (all runs)
```

## Getting Started

### Pre-flight: Machine 1 (Ahmed + Leil) — SGLATrack Setup
```bash
git clone https://github.com/GXNU-ZhongLab/SGLATrack.git "SGLA_Track Model/SGLATrack"
cd "SGLA_Track Model/SGLATrack"
conda env create -f environment_sgla1.yml
conda activate sglatrack
# Download deit_tiny_distilled_patch16_224.pth into pretrained/
# Verify params < 50M before anything else
```

### Pre-flight: Machine 2 (Nour + Yassin) — OSTrack Setup
```bash
git clone https://github.com/botaoye/OSTrack.git "OS_Track Model/OSTrack"
cd "OS_Track Model/OSTrack"
pip install -r requirements.txt
# Download vitb_256_mae_ce_32x4_ep300 checkpoint into pretrained/
```

### Day 1 Morning — Extract Frames + Baseline
```bash
python shared/extract_frames.py --manifest data/contestant_manifest.json --split train --output frames/train
python shared/convert_dataset.py --comp_root data/ --out_root data/comp_train
```

### Day 2 — Validate Before Submitting
```bash
python shared/validate_submission.py --submission submission/predictions/final.csv --sample data/sample_submission.csv
```

## Key Milestones

| Checkpoint | Milestone | Critical Task |
|------------|-----------|---------------|
| Pre-flight | Both machines set up | SGLATrack param count verified (< 50M) |
| Day 1, Hour 0–3 | Hanning live + training launched | SGLATrack loss decreasing by epoch 5 |
| Day 1, Hour 4–8 | ONNX pipeline built | Fallback CSV generated (pretrained SGLATrack) |
| Day 1, Hour 12 | Day 1 sync | All 6 checklist questions answered |
| Day 2, Hour 0–3 | Best checkpoint locked | λ sweep done; checkpoint frozen |
| Day 2, Hour 2 | Distillation decision | Skip or run (≤ 5 epochs, ≤ hour 3) |
| Day 2, Hour 3–5 | Final model built | INT8 ONNX exported and benchmarked |
| Day 2, Hour 5–7 | Submission CSV ready | Validated and row count confirmed |
| Day 2, Hour 7 | Hard stop | No code changes; upload + confirm LB score |

## Fallback Hierarchy

| Priority | Model | State | Ready by |
|----------|-------|-------|----------|
| 1 | SGLATrack fine-tuned | + Hanning + INT8 | Day 2 noon |
| 2 | SGLATrack fine-tuned | + Hanning, FP32 | Day 2 morning |
| 3 | SGLATrack pretrained | + Hanning | Day 1 evening |
| 4 | SGLATrack pretrained | no Hanning | Day 1 afternoon |

**Never enter Day 2 without fallback #3 already generated.**

## Shared Rules

1. **Never modify another member's code without asking first**
2. **All shared scripts in `shared/` – never duplicate**
3. **Update `scores.xlsx` after every major run** (AUC, metrics, FinalScore)
4. **INT8 accuracy drop > 2% AUC → ship FP32 instead**
5. **No code changes after Day 2, Hour 7**

## References

- **SGLATrack**: https://github.com/GXNU-ZhongLab/SGLATrack
- **OSTrack**: https://github.com/botaoye/OSTrack
- **Full Execution Plan**: See `Ideas/(C) MTC-AIC4 2-Day Execution Roadmap.md`
