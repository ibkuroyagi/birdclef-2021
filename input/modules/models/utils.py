import numpy as np
import torch
import torch.nn as nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def mixup_for_sed(X: torch.tensor, Y: torch.tensor, alpha=0.2):
    """MixUp for SED.

    Args:
        X (torch.tensor): (B, ...)
        Y (torch.tensor): (B, ...)
        alpha (float, optional): parameter for beta distribution. Defaults to 0.2.
    Return:
        mixed_X (torch.tensor): (B, ...)
        mixed_Y (torch.tensor): (B, ...)
    """
    with torch.no_grad():
        batch_size = X.size(0)
        perm = torch.randperm(batch_size).to(X.device)
        if alpha == 1.0:
            mixed_X = X + X[perm]
            mixed_Y = Y.to(torch.int64) | Y[perm].to(torch.int64)
        else:
            lam = torch.tensor(
                np.random.beta(alpha, alpha, batch_size), dtype=torch.float32
            ).to(X.device)[:, None]
            mixed_X = lam * X + (1 - lam) * X[perm]
            mixed_Y = lam * Y + (1 - lam) * Y[perm]
    return mixed_X.float(), mixed_Y.float()
