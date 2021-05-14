import torch.nn as nn


class BCE2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.weights = weights

    def forward(self, logit, frame_logit, target):
        target = target.float()
        clipwise_output_with_max, _ = frame_logit.max(dim=1)

        loss = self.bce(logit, target)
        aux_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class BCEMasked(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=pos_weight
        )

    def forward(self, inputs, targets, mask=None):
        bce_loss = self.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        if mask is not None:
            bce_loss = bce_loss[mask > 0]
        return bce_loss.mean()
