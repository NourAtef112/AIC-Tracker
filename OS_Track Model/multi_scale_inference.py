def multi_scale_track(tracker, image, scales=[0.96, 1.0, 1.04]):
    """
    Runs inference at multiple search region scales by temporarily adjusting
    tracker.params.search_factor. Returns the prediction with highest confidence.
    Adds ~3–5ms latency — check efficiency budget before enabling (Day 5+).
    """
    best_box, best_conf = None, -1.0
    original_sf = tracker.params.search_factor
    saved_state = list(tracker.state)
    saved_frame_id = tracker.frame_id

    for scale in scales:
        tracker.params.search_factor = original_sf * scale
        tracker.state = list(saved_state)
        tracker.frame_id = saved_frame_id
        out = tracker.track(image)
        conf = out.get('best_score', 0.0)
        if conf > best_conf:
            best_conf = conf
            best_box = tuple(out['target_bbox'])

    tracker.params.search_factor = original_sf
    tracker.state = list(best_box)
    tracker.frame_id = saved_frame_id + 1
    return best_box, best_conf
