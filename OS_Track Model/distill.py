"""Knowledge distillation: OSTrack (teacher) → SGLATrack (student).

Teacher: OSTrack (vitb_256_mae_ce_32x4_ep300) — full model, excellent accuracy
Student: SGLATrack — lightweight submission model

Only runs if OSTrack-vs-SGLATrack gap ≥ 2% before Day 2 hour 3.
5 epochs maximum (time budget is tight).

From roadmap.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim


def distillation_loss(teacher_out, student_out, gt_labels, alpha: float = 0.5, temp: float = 4.0):
    """Compute distillation loss (task + KL divergence on response maps).

    Args:
        teacher_out: Teacher output dict (should contain 'score_map')
        student_out: Student output dict (should contain 'score_map')
        gt_labels: Ground truth labels for task loss
        alpha: Balance between task loss and distillation. 0.5 = equal weight
        temp: Temperature for softmax (higher = softer targets)

    Returns:
        Combined loss tensor
    """
    # Task loss: student's own supervised loss (e.g., BCE or focal)
    # This is model-specific; shown as placeholder
    task_loss = F.binary_cross_entropy_with_logits(
        student_out['score_map'].view(student_out['score_map'].size(0), -1),
        gt_labels.view(gt_labels.size(0), -1)
    )

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

    def train_epoch(self, train_loader, optimizer, alpha: float = 0.5, temp: float = 4.0):
        """Train one epoch with distillation loss.

        Args:
            train_loader: DataLoader yielding (template, search, gt_boxes)
            optimizer: Optimizer for student parameters
            alpha: Task/distillation balance
            temp: Temperature for softmax

        Returns:
            Avg loss, avg task_loss, avg kl_loss for the epoch
        """
        total_loss = 0.0
        total_task = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch_idx, (template, search, gt_labels) in enumerate(train_loader):
            template = template.to(self.device)
            search = search.to(self.device)
            gt_labels = gt_labels.to(self.device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(template, search)

            # Student forward
            student_out = self.student(template, search)

            # Compute loss
            loss, task_loss, kl_loss = distillation_loss(
                teacher_out, student_out, gt_labels,
                alpha=alpha, temp=temp
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


def main_distill(teacher_model, student_model, train_loader, num_epochs: int = 5):
    """Main distillation training loop (5 epochs max for time budget).

    Args:
        teacher_model: OSTrack (frozen)
        student_model: SGLATrack (trainable)
        train_loader: DataLoader with competition training data
        num_epochs: Number of epochs (cap at 5 for time budget)
    """
    num_epochs = min(num_epochs, 5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = DistillationTrainer(teacher_model, student_model, device=device)

    optimizer = optim.Adam(trainer.student.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print(f"Starting distillation training ({num_epochs} epochs, max)")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss, avg_task, avg_kl = trainer.train_epoch(train_loader, optimizer)
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
    print("       create train_loader, call main_distill()")
