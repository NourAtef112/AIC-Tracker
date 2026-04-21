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
