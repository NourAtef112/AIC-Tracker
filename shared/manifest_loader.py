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
