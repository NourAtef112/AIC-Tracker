"""
OSTrackExt — subclass of OSTrack with competition post-processing hooks.

Changes vs vanilla OSTrack.track():
  - Configurable hanning blend (params.use_hanning + params.hanning_weight)
    instead of fixed element-wise multiplication with output_window.
  - Returns 'best_score' in every output dict (max response value),
    consumed by DualTemplateManager and multi_scale_track.
"""
import torch

from lib.test.tracker.ostrack import OSTrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class OSTrackExt(OSTrack):
    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z,
            )

        pred_score_map = out_dict['score_map']
        if getattr(self.params, 'use_hanning', False):
            hw = getattr(self.params, 'hanning_weight', 0.49)
            response = pred_score_map * (1 - hw) + self.output_window * hw
        else:
            response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        best_score = float(response.max())
        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor, resize_factor)
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes.view(-1).tolist(),
                    "best_score": best_score}
        return {"target_bbox": self.state, "best_score": best_score}
