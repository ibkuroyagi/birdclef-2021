import torch.nn as nn


class BCE2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], pos_weights=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weights=pos_weights)
        self.weights = weights

    def forward(self, logit, frame_logit, target):
        target = target.float()
        clipwise_output_with_max, _ = frame_logit.max(dim=1)

        loss = self.bce(logit, target)
        aux_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
