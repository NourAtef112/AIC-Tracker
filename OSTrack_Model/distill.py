"""Knowledge distillation: OSTrack (teacher) → SGLATrack (student).

Teacher: OSTrack (vitb_256_mae_ce_32x4_ep300) — full model, excellent accuracy
Student: SGLATrack — lightweight submission model

Only runs if OSTrack-vs-SGLATrack gap ≥ 2% before Day 2 hour 3.
5 epochs maximum (time budget is tight).

From roadmap.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / 'OSTrack'))

from lib.utils.box_ops import giou_loss, box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.focal_loss import FocalLoss


def distillation_loss(teacher_out, student_out, gt_bbox, gt_gaussian_maps, loss_weight: dict,
                      alpha: float = 0.5, temp: float = 4.0):
    """Compute distillation loss (task + KL divergence on response maps).

    Task loss = GIoU + L1 + Focal (matches OSTrack actor)
    Distillation = KL divergence on teacher score_map

    Args:
        teacher_out: Teacher output dict with 'score_map'
        student_out: Student output dict with 'pred_boxes', 'score_map', 'size_map', 'offset_map'
        gt_bbox: Ground truth boxes (N, 4) in xywh format
        gt_gaussian_maps: Ground truth Gaussian heatmaps for focal loss
        loss_weight: Dict with 'giou' and 'l1' weights (from config)
        alpha: Balance between task loss and distillation. 0.5 = equal weight
        temp: Temperature for softmax (higher = softer targets)

    Returns:
        Tuple: (combined_loss, task_loss, kl_loss)
    """
    # Task loss: GIoU + L1 + Focal (matches OSTrack's CENTER head loss)
    pred_boxes_vec = box_cxcywh_to_xyxy(student_out['pred_boxes']).view(-1, 4)
    gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)
    if gt_boxes_vec.dim() == 2 and pred_boxes_vec.size(0) > gt_boxes_vec.size(0):
        gt_boxes_vec = gt_boxes_vec[:, None, :].expand(-1, pred_boxes_vec.size(0) // gt_boxes_vec.size(0), -1).reshape(-1, 4)

    giou, _ = giou_loss(pred_boxes_vec, gt_boxes_vec)
    l1 = F.l1_loss(pred_boxes_vec, gt_boxes_vec)

    focal = FocalLoss()(student_out['score_map'], gt_gaussian_maps)

    task_loss = (loss_weight['giou'] * giou +
                 loss_weight['l1'] * l1 +
                 focal)

    # Distillation: match teacher's response map distribution via KL divergence
    t_resp = teacher_out['score_map'].view(teacher_out['score_map'].size(0), -1)
    s_resp = student_out['score_map'].view(student_out['score_map'].size(0), -1)

    t_soft = F.softmax(t_resp / temp, dim=-1)
    s_log = F.log_softmax(s_resp / temp, dim=-1)
    kl_loss = F.kl_div(s_log, t_soft, reduction='batchmean') * (temp ** 2)

    # Combined loss
    combined_loss = (1 - alpha) * task_loss + alpha * kl_loss

    return combined_loss, task_loss, kl_loss


class DistillationTrainer:
    """Wrapper for distillation training."""

    def __init__(self, teacher_model, student_model, device: str = 'cuda'):
        """
        Args:
            teacher_model: OSTrack teacher (frozen)
            student_model: SGLATrack student (trainable)
            device: 'cuda' or 'cpu'
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device).train()
        self.device = device

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_epoch(self, train_loader, optimizer, loss_weight: dict, alpha: float = 0.5, temp: float = 4.0):
        """Train one epoch with distillation loss.

        Args:
            train_loader: DataLoader yielding (template, search, gt_boxes, gt_gaussian_maps)
            optimizer: Optimizer for student parameters
            loss_weight: Dict with 'giou' and 'l1' weights
            alpha: Task/distillation balance
            temp: Temperature for softmax

        Returns:
            Avg loss, avg task_loss, avg kl_loss for the epoch
        """
        total_loss = 0.0
        total_task = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch_idx, (template, search, gt_boxes, gt_gaussian_maps) in enumerate(train_loader):
            template = template.to(self.device)
            search = search.to(self.device)
            gt_boxes = gt_boxes.to(self.device)
            gt_gaussian_maps = gt_gaussian_maps.to(self.device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(template, search)

            # Student forward
            student_out = self.student(template, search)

            # Compute loss
            loss, task_loss, kl_loss = distillation_loss(
                teacher_out, student_out, gt_boxes, gt_gaussian_maps,
                loss_weight=loss_weight, alpha=alpha, temp=temp
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=0.1)
            optimizer.step()

            # Accumulate
            total_loss += loss.item()
            total_task += task_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}: loss={loss.item():.4f} "
                      f"(task={task_loss.item():.4f}, kl={kl_loss.item():.4f})")

        return (total_loss / n_batches,
                total_task / n_batches,
                total_kl / n_batches)

    def save_checkpoint(self, path: str, epoch: int):
        """Save student checkpoint."""
        torch.save({
            'epoch': epoch,
            'model': self.student.state_dict(),
            'student_state': self.student.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")


def main_distill(teacher_model, student_model, train_loader, cfg, num_epochs: int = 5):
    """Main distillation training loop (5 epochs max for time budget).

    Args:
        teacher_model: OSTrack (frozen)
        student_model: SGLATrack (trainable)
        train_loader: DataLoader with competition training data
        cfg: Config object with TRAIN settings
        num_epochs: Number of epochs (cap at 5 for time budget)
    """
    num_epochs = min(num_epochs, 5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = DistillationTrainer(teacher_model, student_model, device=device)

    optimizer = optim.AdamW(trainer.student.parameters(), lr=cfg.TRAIN.LR,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LR_DROP_EPOCH)

    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT,
        'l1': cfg.TRAIN.L1_WEIGHT,
    }

    print(f"Starting distillation training ({num_epochs} epochs, max)")
    print(f"LR={cfg.TRAIN.LR}, Weight_decay={cfg.TRAIN.WEIGHT_DECAY}")
    print(f"Loss weights: GIoU={loss_weight['giou']}, L1={loss_weight['l1']}")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss, avg_task, avg_kl = trainer.train_epoch(train_loader, optimizer,
                                                         loss_weight=loss_weight)
        scheduler.step()

        print(f"Epoch {epoch + 1} summary:")
        print(f"  Loss: {avg_loss:.4f} (task={avg_task:.4f}, kl={avg_kl:.4f})")

        # Save checkpoint
        trainer.save_checkpoint(f"distill_ckpt_ep{epoch+1}.pth", epoch + 1)

        # Early stopping if task loss plateaus
        if epoch > 2 and avg_task > 0.5:
            print("Early stop: task loss plateauing")
            break

    print()
    print("Distillation training complete.")
    print("Next: evaluate final checkpoint vs SGLATrack fine-tuned baseline.")
    print("      If distilled S_acc > fine-tuned, use distilled model.")
    print("      Else: stick with fine-tuned SGLATrack (safer choice).")


if __name__ == '__main__':
    print("Distillation training script")
    print("Usage: instantiate teacher (OSTrack) + student (SGLATrack),")
    print("       load config, create train_loader, call main_distill()")
