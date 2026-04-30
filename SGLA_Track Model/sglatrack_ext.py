import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / 'SGLATrack'))

import torch
from lib.test.tracker.sglatrack import sglatrack as SGLATrackBase
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class SGLATrackExt(SGLATrackBase):
    """SGLATrack with configurable Hanning blend weight and best_score output.

    Changes vs vanilla sglatrack.track():
      - hanning_weight param controls the blend: score*(1-w) + window*w
        (0 = no penalty, 1 = full window multiplication)
      - Returns 'best_score' in output dict (max response value),
        consumed by DualTemplateManager if used.
    """

    def __init__(self, params, dataset_name='aic'):
        super().__init__(params, dataset_name)
        self.hanning_weight = getattr(params, 'hanning_weight', 0.49)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            out_dict = self.network(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z,
            )

        pred_score_map = out_dict['score_map']
        hw = self.hanning_weight
        response = pred_score_map * (1 - hw) + self.output_window * hw

        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        best_score = float(response.max())
        return {"target_bbox": self.state, "best_score": best_score}
