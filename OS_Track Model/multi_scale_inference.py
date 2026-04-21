def multi_scale_track(model, frame, template, scales=[0.96, 1.0, 1.04]):
    """
    Runs inference at multiple search region scales.
    Returns prediction from the scale with highest confidence.
    Adds ~3–5ms latency – check budget before enabling.
    """
    best_box, best_conf = None, -1
    for scale in scales:
        pred_box, conf = model.track(frame, template, scale_factor=scale)
        if conf > best_conf:
            best_conf = conf
            best_box  = pred_box
    return best_box, best_conf
