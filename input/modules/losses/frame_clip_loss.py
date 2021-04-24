# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

"""Frame and Clip Loss modules."""

import torch.nn as nn


class FrameClipLoss(nn.Module):
    """Frame-wise and Clip-wise loss."""

    def __init__(self, clip_ratio=0.5, reduction="mean"):
        super(self.__class__, self).__init__()
        self.clip_ratio = clip_ratio
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, frame_preds, frame_true, clip_preds, clip_true):
        """Calculate Frame-wise and Clip-wise loss.

        Args:
            frame_preds (tensor): (B, T, n_class)
            frame_true (tensor): (B, T, n_class)
            clip_preds (tensor): (B, n_class)
            clip_true (tensor): (B, n_class)

        Returns:
            tensor: loss
        """
        clip_loss = self.bce(clip_preds, clip_true)
        frame_loss = self.bce(frame_preds, frame_true)
        loss = clip_loss * self.clip_ratio + frame_loss * (1 - self.clip_ratio)
        return loss
