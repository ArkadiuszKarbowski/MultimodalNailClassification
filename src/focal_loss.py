import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_softmax = F.log_softmax(logits, dim=1)
        log_pt = log_softmax.gather(1, targets.view(-1, 1)).squeeze()
        pt = log_pt.exp()

        ce_loss = -log_pt
        modulating_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * modulating_factor * ce_loss
        else:
            focal_loss = modulating_factor * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
