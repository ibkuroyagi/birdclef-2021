import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, gamma=1.5):
        super(self.__class__, self).__init__()
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        bce_loss = self.loss_fct(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(
            targets >= 0.5,
            self.alpha * (1.0 - probas) ** self.gamma * bce_loss,
            probas ** self.gamma * bce_loss,
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
