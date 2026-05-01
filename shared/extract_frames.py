import cv2, os, json
from pathlib import Path

def extract_sequence(video_path: str, out_dir: str, force: bool = False):
    out_dir = Path(out_dir)
    if out_dir.exists() and not force:
        existing = list(out_dir.glob("*.jpg"))
        if existing:
            print(f"  [SKIP] {out_dir} -> {len(existing)} frames already extracted")
            return len(existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        out_dir.rmdir() if out_dir.exists() and not any(out_dir.iterdir()) else None
        print(f"  [MISSING] {video_path}")
        return 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{idx:08d}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"  [DONE] {video_path} -> {idx} frames")
    return idx

def extract_all(manifest_path: str, split: str = "train", frames_root: str = "frames", data_root: str = "data"):
    with open(manifest_path) as f:
        manifest = json.load(f)
    sequences = manifest.get(split, {})
    print(f"Extracting {len(sequences)} sequences | split={split}")
    for seq_key, info in sequences.items():
        video_path = str(Path(data_root) / info["video_path"])
        extract_sequence(video_path, str(Path(frames_root) / seq_key))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="contestant_manifest.json")
    p.add_argument("--split", default="train", choices=["train", "public_lb"])
    p.add_argument("--output", default="frames")
    args = p.parse_args()
    extract_all(args.manifest, args.split, args.output)
