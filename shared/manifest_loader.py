import json
from dataclasses import dataclass
from pathlib import Path
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
    manifest_path = Path(path)
    # Paths in the manifest are relative to the data root (one level above metadata/)
    data_root = manifest_path.parent.parent
    with open(manifest_path) as f:
        manifest = json.load(f)

    def resolve(p: Optional[str]) -> Optional[str]:
        return str(data_root / p) if p else None

    return [
        SequenceInfo(
            key=k,
            dataset=v["dataset"],
            seq_name=v["seq_name"],
            n_frames=v["n_frames"],
            native_fps=v["native_fps"],
            video_path=resolve(v["video_path"]),
            annotation_path=resolve(v.get("annotation_path"))
        )
        for k, v in manifest.get(split, {}).items()
    ]
