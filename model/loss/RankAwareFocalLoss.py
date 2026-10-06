import torch
import torch.nn as nn
import torch.nn.functional as F


class RankAwareFocalLoss(nn.Module):
    def __init__(self, num_classes: int = 5, gamma: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("rank_values", torch.arange(num_classes).float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (((1 - pt) ** self.gamma) * ce_loss).mean()
        probs = F.softmax(logits, dim=-1)
        expected_ranks = torch.sum(probs * self.rank_values.to(logits.device), dim=-1)
        rank_loss = F.mse_loss(expected_ranks, targets.float())
        return focal_loss + self.alpha * rank_loss
