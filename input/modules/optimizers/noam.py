# -*- coding: utf-8 -*-

# Created by Ibuki Kuroyanagi

"""Noam optimizer modules."""

import torch


class Noam(object):
    """Noam optimizer, a.k.a., Adam + Warmup learning rate.

    This code is modified from https://github.com/espnet/espnet.

    """

    def __init__(
        self,
        params,
        model_size,
        base_lr,
        warmup,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        """Construct an NoamOpt object."""
        self.optimizer = torch.optim.Adam(
            params, lr=0, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self._step = 0
        self.warmup = warmup
        self.base_lr = base_lr
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self._update_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def _update_rate(self, step=None):
        if step is None:
            step = self._step
        return (
            self.base_lr
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "base_lr": self.base_lr,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)
