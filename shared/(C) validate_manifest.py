#!/usr/bin/env python3
"""Validate manifest against actual data files."""

import json
import os
from pathlib import Path
from collections import defaultdict

def main():
    manifest_path = Path("data/metadata/contestant_manifest.json")
    data_root = manifest_path.parent.parent
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    splits = manifest.keys()
    stats = defaultdict(lambda: {"total": 0, "missing": 0, "ok": 0})
    missing_seqs = []
    
    for split in splits:
        sequences = manifest[split]
        for seq_key, seq_info in sequences.items():
            stats[split]["total"] += 1
            ann_path = data_root / seq_info["annotation_path"]
            
            if not ann_path.exists():
                stats[split]["missing"] += 1
                missing_seqs.append((split, seq_key, str(ann_path)))
                print(f"❌ MISSING: {split}/{seq_key}")
                print(f"   Expected: {ann_path}")
            else:
                stats[split]["ok"] += 1
    
    print("\n" + "="*60)
    print("MANIFEST VALIDATION REPORT")
    print("="*60)
    for split in splits:
        s = stats[split]
        pct = 100 * s["ok"] / s["total"] if s["total"] > 0 else 0
        print(f"\n{split.upper()}: {s['ok']}/{s['total']} valid ({pct:.1f}%)")
        print(f"  Missing files: {s['missing']}")
    
    if missing_seqs:
        print("\n" + "="*60)
        print(f"MISSING SEQUENCES ({len(missing_seqs)} total):")
        print("="*60)
        for split, seq_key, path in missing_seqs[:20]:  # Show first 20
            print(f"  {split}: {seq_key}")
        if len(missing_seqs) > 20:
            print(f"  ... and {len(missing_seqs) - 20} more")

if __name__ == "__main__":
    main()
