import torch

class DualTemplateManager:
    """
    Maintains two templates:
      - static:  initialized once, never updated (anchors identity)
      - dynamic: updated every `interval` frames when confidence is high
    Final template = 0.5 * static + 0.5 * dynamic
    """
    def __init__(self, conf_threshold: float = 0.65, interval: int = 5):
        self.conf_threshold = conf_threshold
        self.interval       = interval
        self.static         = None
        self.dynamic        = None
        self.frame_count    = 0

    def initialize(self, template_feat: torch.Tensor):
        self.static      = template_feat.clone()
        self.dynamic     = template_feat.clone()
        self.frame_count = 0

    def get_fused(self) -> torch.Tensor:
        return 0.5 * self.static + 0.5 * self.dynamic

    def update(self, new_feat: torch.Tensor, confidence: float):
        self.frame_count += 1
        if (confidence > self.conf_threshold and
                self.frame_count % self.interval == 0):
            self.dynamic = new_feat.clone()
