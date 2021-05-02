import torch
import torch.nn as nn


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1.0e-4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds).clamp(self.epsilon, 1 - self.epsilon)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas ** self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, logit, frame_logit, target):
        target = target.float()
        clipwise_output_with_max, _ = frame_logit.max(dim=1)

        loss = self.focal(logit, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
