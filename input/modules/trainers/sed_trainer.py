import logging
import os
import random
import sys
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter

sys.path.append("../../")
sys.path.append("../input/modules")
import losses  # noqa: E402
import optimizers  # noqa: E402
from losses import CenterLoss  # noqa: E402
from utils import lwlrap  # noqa: E402
from utils import original_mixup  # noqa: E402


class SEDTrainer(object):
    """Customized trainer module for SED training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        train_sampler=None,
        device=torch.device("cpu"),
        train=False,
        use_center_loss=False,
        use_dializer=False,
        save_name="",
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            train_sampler (sampler): sampler. If you use multi-gpu you need to define.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (torch.nn): It must contrain "stft" and "mse" criterions.
            optimizer (object): Optimizers.
            scheduler (object): Schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
            train (bool): Select mode of trainer.
            use_center_loss(bool): Select whether to use center loss.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.train_sampler = train_sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.mixup_alpha = config.get("mixup_alpha", None)
        self.train = train
        if train:
            self.writer = SummaryWriter(config["outdir"])
        self.use_center_loss = use_center_loss
        self.use_dializer = use_dializer
        self.save_name = save_name
        if use_center_loss:
            self.center_loss = CenterLoss(device=device, **config["center_loss_params"])
            optimizer_class = getattr(
                optimizers,
                config.get("center_optimizer_type", "Adam"),
            )
            self.optimizer_centloss = optimizer_class(
                [
                    {
                        "params": self.center_loss.parameters(),
                        **config["center_loss_optimizer_params"],
                    }
                ]
            )
            self.train_label_epoch = np.empty(0)
            self.dev_label_epoch = np.empty(0)
            self.train_embedding_epoch = np.empty(
                (0, config["center_loss_params"]["feat_dim"])
            )
            self.dev_embedding_epoch = np.empty(
                (0, config["center_loss_params"]["feat_dim"])
            )
            self.tsne = TSNE(**config["tsne_params"])
        if use_dializer:
            loss_class = getattr(
                losses,
                config.get("dializer_loss_type", "BCEWithLogitsLoss"),
            )
            self.dializer_loss = loss_class(**config["dializer_loss_params"])
            self.train_pred_frame_mask_epoch = torch.empty(
                (0, config["l_target"], 1)
            ).to(device)
            self.train_y_frame_mask_epoch = torch.empty((0, config["l_target"], 1)).to(
                device
            )
            self.dev_pred_frame_mask_epoch = torch.empty((0, config["l_target"], 1)).to(
                device
            )
            self.dev_y_frame_mask_epoch = torch.empty((0, config["l_target"], 1)).to(
                device
            )

        self.finish_train = False
        self.best_score = 0
        self.n_target = 26 if config.get("use_song_type", False) else 24
        self.epoch_train_loss = defaultdict(float)
        self.epoch_eval_loss = defaultdict(float)
        self.eval_metric = defaultdict(float)
        self.train_pred_epoch = np.empty((0, self.n_target))
        self.train_pred_frame_epoch = torch.empty(
            (0, config["l_target"], config["n_class"])
        ).to(device)
        self.train_y_epoch = np.empty((0, self.n_target))
        self.train_y_frame_epoch = torch.empty(
            (0, config["l_target"], config["n_class"])
        ).to(device)
        self.dev_pred_epoch = np.empty((0, self.n_target))
        self.dev_pred_frame_epoch = torch.empty(
            (0, config["l_target"], config["n_class"])
        ).to(device)
        self.dev_y_epoch = np.empty((0, self.n_target))
        self.dev_y_frame_epoch = torch.empty(
            (0, config["l_target"], config["n_class"])
        ).to(device)
        self.n_eval_split = config["n_eval_split"]
        self.last_checkpoint = ""
        self.forward_count = 0

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()
            self._eval_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path, save_model_only=True):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            save_model_only (bool): Whether to save model parameters only.
        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
            "best_score": self.best_score,
        }
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()
        if not save_model_only:
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict["scheduler"] = self.scheduler.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, checkpoint_path)
        self.last_checkpoint = checkpoint_path

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.best_score = state_dict.get("best_score", 0)
            logging.info(
                f"Steps:{self.steps}, Epochs:{self.epochs}, BEST score:{self.best_score}"
            )
            if (self.optimizer is not None) and (
                state_dict.get("optimizer", None) is not None
            ):
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if (self.scheduler is not None) and (
                state_dict.get("scheduler", None) is not None
            ):
                self.scheduler.load_state_dict(state_dict["scheduler"])

    def _train_step(self, batch):
        """Train model one step."""
        if self.mixup_alpha is not None:
            batch = original_mixup(batch, self.mixup_alpha)  # (B*2,...) -> (B,...)
        x = batch["X"].to(self.device)  # (B, mel, T')
        y_frame = batch["y_frame"].to(self.device)
        y_clip = batch["y_clip"].to(self.device)
        if self.config["model_type"] in [
            "TransformerEncoderDecoder",
            "ConformerEncoderDecoder",
        ]:
            if not self.config["model_params"].get("require_prep", False):
                # Add waek label frame and transpose (B, mel, T') to (B, 1+T', mel).
                x = torch.cat(
                    [
                        torch.ones((x.shape[0], x.shape[1], 1), dtype=torch.float32).to(
                            self.device
                        ),
                        x,
                    ],
                    axis=2,
                ).transpose(2, 1)
        logging.debug(
            f"y_frame,{y_frame.shape}:{y_frame[0]}, y_clip,{y_clip.shape}:{y_clip[0]}"
        )
        y_ = self.model(x)  # {y_frame: (B, T', n_class), y_clip: (B, n_class)}
        logging.debug(f"y_frame_:{y_['y_frame'][0]}, y_clip_:{y_['y_clip'][0]}")
        if self.config["loss_type"] == "FrameClipLoss":
            loss = self.criterion(
                y_["y_frame"][:, :, : self.n_target],
                y_frame[:, :, : self.n_target],
                y_["y_clip"][:, : self.n_target],
                y_clip[:, : self.n_target],
            )
        elif self.config["loss_type"] in ["BCEWithLogitsLoss", "FocalLoss"]:
            loss = self.criterion(
                y_["y_clip"][:, : self.n_target], y_clip[:, : self.n_target]
            )

        if self.use_center_loss:
            center_loss_label = self._get_center_loss_label(y_clip[:, : self.n_target])
            logging.debug(f"center_loss_label:{center_loss_label}")
            center_loss = (
                self.center_loss(y_["embedding"], center_loss_label)
                * self.config["center_loss_alpha"]
            )
            loss += center_loss
            logging.debug(
                f"loss:{loss.item()}, center:{center_loss.item()}, clip:{loss.item()-center_loss.item()}"
            )
            self.optimizer_centloss.zero_grad()
        if self.use_dializer:
            frame_mask = batch["frame_mask"].to(self.device)
            loss += (
                self.dializer_loss(y_["frame_mask"], frame_mask)
                * self.config["dializer_loss_alpha"]
            )
        if not torch.isnan(loss):
            loss = loss / self.config["accum_grads"]
            loss.backward()
            if self.use_center_loss:
                # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                for param in self.center_loss.parameters():
                    param.grad.data *= 1.0 / self.config["center_loss_alpha"]
                self.optimizer_centloss.step()
            self.forward_count += 1
            if self.forward_count == self.config["accum_grads"]:
                # update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.forward_count = 0

                # update scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # update counts
                self.steps += 1
                self.tqdm.update(1)
                self._check_train_finish()
        else:
            logging.warn("Loss contain NaN. Don't back-poropagated.")

        if self.config["model_type"] in [
            "Cnn14_DecisionLevelAtt",
            "ResNext50",
            "TransformerEncoderDecoder",
            "ConformerEncoderDecoder",
            "EfficientNet_simple",
            "EfficientNet_b",
            "MobileNetV2",
            "MobileNetV2_simple",
        ]:
            self.train_pred_frame_epoch = torch.cat(
                [self.train_pred_frame_epoch, y_["y_frame"][:, :, : self.n_target]],
                dim=0,
            )
            self.train_y_frame_epoch = torch.cat(
                [self.train_y_frame_epoch, y_frame[:, :, : self.n_target]], dim=0
            )
            self.train_pred_epoch = np.concatenate(
                [
                    self.train_pred_epoch,
                    y_["y_clip"][:, : self.n_target].detach().cpu().numpy(),
                ],
                axis=0,
            )
            self.train_y_epoch = np.concatenate(
                [self.train_y_epoch, y_clip[:, : self.n_target].detach().cpu().numpy()],
                axis=0,
            )
        if self.use_center_loss:
            self.train_label_epoch = np.concatenate(
                [
                    self.train_label_epoch,
                    center_loss_label.detach().cpu().numpy().astype(np.float32),
                ],
            )
            self.train_embedding_epoch = np.concatenate(
                [
                    self.train_embedding_epoch,
                    y_["embedding"].detach().cpu().numpy().astype(np.float32),
                ]
            )
        if self.use_dializer:
            # logging.info(
            #     f"{self.train_pred_frame_mask_epoch.shape}, {y_['frame_mask'].shape}"
            # )
            self.train_pred_frame_mask_epoch = torch.cat(
                [self.train_pred_frame_mask_epoch, y_["frame_mask"]],
                dim=0,
            )
            self.train_y_frame_mask_epoch = torch.cat(
                [self.train_y_frame_mask_epoch, frame_mask],
                dim=0,
            )

    def _train_epoch(self):
        """Train model one epoch."""
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return
        try:
            logging.debug(
                f"Epoch train pred frame:{self.train_pred_frame_epoch.shape}{self.train_pred_frame_epoch.sum():.4f}\n"
                f"Epoch train    y frame:{self.train_y_frame_epoch.shape}{self.train_y_frame_epoch.sum()}\n"
                f"Epoch train  pred clip:{self.train_pred_epoch.shape}{self.train_pred_epoch.sum():.4f}\n"
                f"Epoch train     y clip:{self.train_y_epoch.shape}{self.train_y_epoch.sum()}\n"
            )
            if self.config["loss_type"] == "FrameClipLoss":
                self.epoch_train_loss["train/epoch_main_loss"] = self.criterion(
                    self.train_pred_frame_epoch,
                    self.train_y_frame_epoch,
                    torch.tensor(self.train_pred_epoch).to(self.device),
                    torch.tensor(self.train_y_epoch).to(self.device),
                ).item()
            elif self.config["loss_type"] in ["BCEWithLogitsLoss", "FocalLoss"]:
                self.epoch_train_loss["train/epoch_main_loss"] = self.criterion(
                    torch.tensor(self.train_pred_epoch).to(self.device),
                    torch.tensor(self.train_y_epoch).to(self.device),
                ).item()
            self.epoch_train_loss["train/epoch_loss"] = self.epoch_train_loss[
                "train/epoch_main_loss"
            ]
            if self.use_center_loss:
                self.epoch_train_loss["train/epoch_center_loss"] = (
                    self.center_loss(
                        torch.tensor(self.train_embedding_epoch)
                        .float()
                        .to(self.device),
                        torch.tensor(self.train_label_epoch).float().to(self.device),
                    ).item()
                    * self.config["center_loss_alpha"]
                )
                self.epoch_train_loss["train/epoch_loss"] += self.epoch_train_loss[
                    "train/epoch_center_loss"
                ]
            if self.use_dializer:
                self.epoch_train_loss["train/epoch_dializer_loss"] = (
                    self.dializer_loss(
                        self.train_pred_frame_mask_epoch, self.train_y_frame_mask_epoch
                    ).item()
                    * self.config["dializer_loss_alpha"]
                )
                self.epoch_train_loss["train/epoch_loss"] += self.epoch_train_loss[
                    "train/epoch_dializer_loss"
                ]
                self.train_pred_frame_epoch *= torch.sigmoid(
                    self.train_y_frame_mask_epoch
                )
            if self.config.get("use_song_type", False):
                self.train_pred_epoch = self._fix_class_data(
                    self.train_pred_epoch, mode="np_clip"
                )
                self.train_y_epoch = self._fix_class_data(
                    self.train_y_epoch, mode="np_clip"
                )
                self.train_pred_frame_epoch = self._fix_class_data(
                    self.train_pred_frame_epoch, mode="frame"
                )
            self.epoch_train_loss["train/epoch_lwlrap_clip"] = lwlrap(
                self.train_y_epoch[:, :24], self.train_pred_epoch[:, :24]
            )
            self.epoch_train_loss["train/epoch_lwlrap_frame"] = lwlrap(
                self.train_y_epoch[:, :24],
                self.train_pred_frame_epoch.detach().cpu().numpy().max(axis=1)[:, :24],
            )
            self.epoch_train_loss["train/epoch_lwlrap"] = (
                self.epoch_train_loss["train/epoch_lwlrap_clip"]
                + self.epoch_train_loss["train/epoch_lwlrap_frame"]
            ) / 2.0
            self.epoch_train_loss["train/lr"] = self.optimizer.param_groups[0]["lr"]
        except ValueError:
            logging.warning("Raise ValueError: May be contain NaN in y_pred.")
            pass
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({train_steps_per_epoch} steps per epoch)."
        )
        for key in self.epoch_train_loss.keys():
            logging.info(
                f"(Epoch: {self.epochs}) {key} = {self.epoch_train_loss[key]:.6f}."
            )
        self._write_to_tensorboard(self.epoch_train_loss)
        if self.use_center_loss and (self.epochs % 10 == 0):
            self.plot_embedding(
                self.train_embedding_epoch, self.train_label_epoch, name="train"
            )
        # update
        self.train_steps_per_epoch = train_steps_per_epoch
        self.epochs += 1
        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.train_sampler.set_epoch(self.epochs)
        # reset
        self.train_pred_frame_epoch = torch.empty(
            (0, self.config["l_target"], self.config["n_class"])
        ).to(self.device)
        self.train_y_frame_epoch = torch.empty(
            (0, self.config["l_target"], self.config["n_class"])
        ).to(self.device)
        self.train_y_epoch = np.empty((0, self.n_target))
        self.train_pred_epoch = np.empty((0, self.n_target))
        self.epoch_train_loss = defaultdict(float)
        if self.use_center_loss:
            self.train_embedding_epoch = np.empty(
                (0, self.config["center_loss_params"]["feat_dim"])
            )
            self.train_label_epoch = np.empty(0)
        if self.use_dializer:
            self.train_pred_frame_mask_epoch = torch.empty(
                (0, self.config["l_target"], 1)
            ).to(self.device)
            self.train_y_frame_mask_epoch = torch.empty(
                (0, self.config["l_target"], 1)
            ).to(self.device)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        x = batch["X"].to(self.device)
        if self.config["model_type"] in [
            "TransformerEncoderDecoder",
            "ConformerEncoderDecoder",
        ]:
            if not self.config["model_params"].get("require_prep", False):
                # Add waek label frame and transpose (B, mel, T') to (B, 1+T', mel).
                x = torch.cat(
                    [
                        torch.ones((x.shape[0], x.shape[1], 1), dtype=torch.float32).to(
                            self.device
                        ),
                        x,
                    ],
                    axis=2,
                ).transpose(2, 1)
        y_frame = batch["y_frame"].to(self.device)
        y_clip = batch["y_clip"].to(self.device)
        y_ = self.model(x)
        if self.config["loss_type"] == "FrameClipLoss":
            loss = self.criterion(
                y_["y_frame"][:, :, : self.n_target],
                y_frame[:, :, : self.n_target],
                y_["y_clip"][:, : self.n_target],
                y_clip[:, : self.n_target],
            )
        elif self.config["loss_type"] in ["BCEWithLogitsLoss", "FocalLoss"]:
            loss = self.criterion(
                y_["y_clip"][:, : self.n_target], y_clip[:, : self.n_target]
            )
        if self.use_center_loss:
            center_loss_label = self._get_center_loss_label(y_clip[:, : self.n_target])
            loss += (
                self.center_loss(y_["embedding"], center_loss_label)
                * self.config["center_loss_alpha"]
            )
        if self.use_dializer:
            frame_mask = batch["frame_mask"].to(self.device)
            loss += (
                self.dializer_loss(y_["frame_mask"], frame_mask)
                * self.config["dializer_loss_alpha"]
            )
        # add to total eval loss
        if self.config["model_type"] in [
            "Cnn14_DecisionLevelAtt",
            "ResNext50",
            "TransformerEncoderDecoder",
            "ConformerEncoderDecoder",
            "EfficientNet_simple",
            "EfficientNet_b",
            "MobileNetV2",
            "MobileNetV2_simple",
        ]:
            self.dev_pred_frame_epoch = torch.cat(
                [self.dev_pred_frame_epoch, y_["y_frame"][:, :, : self.n_target]], dim=0
            )
            self.dev_y_frame_epoch = torch.cat(
                [self.dev_y_frame_epoch, y_frame[:, :, : self.n_target]], dim=0
            )
            self.dev_pred_epoch = np.concatenate(
                [
                    self.dev_pred_epoch,
                    y_["y_clip"][:, : self.n_target]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                ],
                axis=0,
            )
            self.dev_y_epoch = np.concatenate(
                [
                    self.dev_y_epoch,
                    y_clip[:, : self.n_target]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                ],
                axis=0,
            )
        if self.use_center_loss:
            self.dev_label_epoch = np.concatenate(
                [self.dev_label_epoch, center_loss_label.detach().cpu().numpy()]
            )
            self.dev_embedding_epoch = np.concatenate(
                [self.dev_embedding_epoch, y_["embedding"].detach().cpu().numpy()]
            )
        if self.use_dializer:
            self.dev_pred_frame_mask_epoch = torch.cat(
                [self.dev_pred_frame_mask_epoch, y_["frame_mask"]],
                dim=0,
            )
            self.dev_y_frame_mask_epoch = torch.cat(
                [self.dev_y_frame_mask_epoch, frame_mask],
                dim=0,
            )

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start dev data's evaluation.")
        # change mode
        self.model.training = False
        self.model.eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[dev]"), 1
        ):
            # eval one step
            self._eval_step(batch)
        try:
            logging.debug(
                f"Epoch dev pred_frame:{self.dev_pred_frame_epoch.sum():.4f}\n"
                f"Epoch dev    y_frame:{self.dev_y_frame_epoch.sum():.4f}\n"
                f"Epoch dev pred_clip:{self.dev_pred_epoch.sum()}\n"
                f"Epoch dev    y_clip:{self.dev_y_epoch.sum()}\n"
            )
            if self.config["loss_type"] == "FrameClipLoss":
                self.epoch_eval_loss["dev/epoch_main_loss"] = self.criterion(
                    self.dev_pred_frame_epoch,
                    self.dev_y_frame_epoch,
                    torch.tensor(self.dev_pred_epoch).to(self.device),
                    torch.tensor(self.dev_y_epoch).to(self.device),
                ).item()
            elif self.config["loss_type"] in ["BCEWithLogitsLoss", "FocalLoss"]:
                self.epoch_eval_loss["dev/epoch_main_loss"] = self.criterion(
                    torch.tensor(self.dev_pred_epoch).to(self.device),
                    torch.tensor(self.dev_y_epoch).to(self.device),
                ).item()
            self.epoch_eval_loss["dev/epoch_loss"] = self.epoch_eval_loss[
                "dev/epoch_main_loss"
            ]
            if self.use_center_loss:
                self.epoch_eval_loss["dev/epoch_center_loss"] = (
                    self.center_loss(
                        torch.tensor(self.dev_embedding_epoch).float().to(self.device),
                        torch.tensor(self.dev_label_epoch).float().to(self.device),
                    ).item()
                    * self.config["center_loss_alpha"]
                )
                self.epoch_eval_loss["dev/epoch_loss"] += self.epoch_eval_loss[
                    "dev/epoch_center_loss"
                ]
            if self.use_dializer:
                self.epoch_eval_loss["dev/epoch_dializer_loss"] = (
                    self.dializer_loss(
                        self.dev_pred_frame_mask_epoch, self.dev_y_frame_mask_epoch
                    ).item()
                    * self.config["dializer_loss_alpha"]
                )
                self.epoch_eval_loss["dev/epoch_loss"] += self.epoch_eval_loss[
                    "dev/epoch_dializer_loss"
                ]
                self.dev_pred_frame_epoch *= torch.sigmoid(self.dev_y_frame_mask_epoch)
            if self.config.get("use_song_type", False):
                self.dev_pred_epoch = self._fix_class_data(
                    self.dev_pred_epoch, mode="np_clip"
                )
                self.dev_y_epoch = self._fix_class_data(
                    self.dev_y_epoch, mode="np_clip"
                )
                self.dev_pred_frame_epoch = self._fix_class_data(
                    self.dev_pred_frame_epoch, mode="frame"
                )
            self.epoch_eval_loss["dev/epoch_lwlrap_clip"] = lwlrap(
                self.dev_y_epoch[:, :24], self.dev_pred_epoch[:, :24]
            )
            self.epoch_eval_loss["dev/epoch_lwlrap_frame"] = lwlrap(
                self.dev_y_epoch[:, :24],
                self.dev_pred_frame_epoch.detach().cpu().numpy().max(axis=1)[:, :24],
            )
            self.epoch_eval_loss["dev/epoch_lwlrap"] = (
                self.epoch_eval_loss["dev/epoch_lwlrap_clip"]
                + self.epoch_eval_loss["dev/epoch_lwlrap_frame"]
            ) / 2.0
        except ValueError:
            logging.warning("Raise ValueError: May be contain NaN in y_pred.")
            pass
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished dev data's evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )
        for key in self.epoch_eval_loss.keys():
            logging.info(
                f"(Epoch: {self.epochs}) {key} = {self.epoch_eval_loss[key]:.6f}."
            )
        if self.epoch_eval_loss["dev/epoch_lwlrap"] > self.best_score:
            self.best_score = self.epoch_eval_loss["dev/epoch_lwlrap"]
            logging.info(
                f"Epochs: {self.epochs}, BEST score was updated {self.best_score:.6f}."
            )
            save_path = os.path.join(
                self.config["outdir"],
                "best_score",
                f"best_score{self.save_name}.pkl",
            )
            self.save_checkpoint(save_path, save_model_only=False)
            logging.info(
                f"Best model was updated @ {self.steps} steps." f"Saved at {save_path}"
            )

        logging.info(f"(Steps: {self.steps}) Start eval data's evaluation.")
        if self.epochs % self.config["eval_interval_epochs"] == 0:
            items = self.inference(mode="valid")
            logging.info(
                f"Inference (Epochs: {self.epochs}) lwlrap: {items['score']:.6f}"
            )
            self._write_to_tensorboard(self.eval_metric)
            self.eval_metric = defaultdict(float)
        # record
        self._write_to_tensorboard(self.epoch_eval_loss)
        if self.use_center_loss and (
            self.epochs % self.config["eval_interval_epochs"] == 0
        ):
            self.plot_embedding(
                self.dev_embedding_epoch, self.dev_label_epoch, name="dev"
            )

        # reset
        self.epoch_eval_loss = defaultdict(float)
        self.dev_pred_frame_epoch = torch.empty(
            (0, self.config["l_target"], self.config["n_class"])
        ).to(self.device)
        self.dev_y_frame_epoch = torch.empty(
            (0, self.config["l_target"], self.config["n_class"])
        ).to(self.device)
        self.dev_pred_epoch = np.empty((0, self.n_target))
        self.dev_y_epoch = np.empty((0, self.n_target))
        if self.use_center_loss:
            self.dev_embedding_epoch = np.empty(
                (0, self.config["center_loss_params"]["feat_dim"])
            )
            self.dev_label_epoch = np.empty(0)
        if self.use_dializer:
            self.dev_pred_frame_mask_epoch = torch.empty(
                (0, self.config["l_target"], 1)
            ).to(self.device)
            self.dev_y_frame_mask_epoch = torch.empty(
                (0, self.config["l_target"], 1)
            ).to(self.device)
        # restore mode
        self.model.training = True
        self.model.train()

    def inference(self, mode="test"):
        """Evaluate and save intermediate result."""
        # evaluate
        keys_list = [f"X{i}" for i in range(self.n_eval_split)]
        y_clip = [
            torch.empty((0, 24)).to(self.device) for _ in range(self.n_eval_split)
        ]
        y_frame = [
            torch.empty((0, self.config["l_target"], 24)).to(self.device)
            for _ in range(self.n_eval_split)
        ]
        y_clip_true = torch.empty((0, 24))
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data_loader["eval"]):
                if mode == "valid":
                    if self.config.get("use_song_type", False):
                        batch["y_clip"] = self._fix_class_data(
                            batch["y_clip"], mode="clip"
                        )
                    y_clip_true = torch.cat(
                        [y_clip_true, batch["y_clip"][:, :24]], dim=0
                    )
                x_batchs = [batch[key].to(self.device) for key in keys_list]
                for i in range(self.n_eval_split):
                    if self.config["model_type"] in [
                        "TransformerEncoderDecoder",
                        "ConformerEncoderDecoder",
                    ]:
                        if not self.config["model_params"].get("require_prep", False):
                            # Add waek label frame and transpose (B, mel, T') to (B, 1+T', mel).
                            x_batchs[i] = torch.cat(
                                [
                                    torch.ones(
                                        (x_batchs[i].shape[0], x_batchs[i].shape[1], 1),
                                        dtype=torch.float32,
                                    ).to(self.device),
                                    x_batchs[i],
                                ],
                                axis=2,
                            ).transpose(2, 1)
                    y_batch_ = self.model(x_batchs[i])
                    if self.config.get("use_song_type", False):
                        y_batch_["y_clip"] = self._fix_class_data(
                            y_batch_["y_clip"], mode="clip"
                        )
                        y_batch_["y_frame"] = self._fix_class_data(
                            y_batch_["y_frame"], mode="frame"
                        )
                        # logging.info(
                        #     f'fix shape:{y_batch_["y_clip"].shape}, {y_batch_["y_frame"].shape}'
                        # )
                    y_clip[i] = torch.cat(
                        [y_clip[i], y_batch_["y_clip"][:, :24]], dim=0
                    )
                    if self.use_dializer:
                        y_batch_["y_frame"] *= torch.sigmoid(y_batch_["frame_mask"])
                    y_frame[i] = torch.cat(
                        [y_frame[i], y_batch_["y_frame"][:, :, :24]], dim=0
                    )
        # (B, n_eval_split, n_target)
        y_clip = (
            torch.sigmoid(torch.stack(y_clip, dim=0)).cpu().numpy().transpose(1, 0, 2)
        )
        # (B, n_eval_split, T, n_class)
        y_frame = (
            torch.sigmoid(torch.stack(y_frame, dim=0))
            .cpu()
            .numpy()
            .transpose(1, 0, 2, 3)
        )
        if mode == "valid":
            y_clip_true = y_clip_true.numpy()
            clip_score = lwlrap(y_clip_true[:, :24], y_clip.max(axis=1)[:, :24])
            self.eval_metric["eval_metric/lwlrap_clip"] = clip_score
            frame_score = lwlrap(
                y_clip_true[:, :24], y_frame.max(axis=1).max(axis=1)[:, :24]
            )
            self.eval_metric["eval_metric/lwlrap_frame"] = frame_score
            score = (clip_score + frame_score) / 2.0
            self.eval_metric["eval_metric/lwlrap"] = score
            return {
                "y_clip": y_clip,
                "y_frame": y_frame,
                "score": score,
            }
        return {"y_clip": y_clip, "y_frame": y_frame}

    def plot_embedding(self, embedding, label, name="", dirname=""):
        """Plot distribution of embedding layer.

        Args:
            embedding (ndarray): (B, 2048)
            label (ndarray): (B,)
        """
        import matplotlib.pyplot as plt

        X_embedded = self.tsne.fit_transform(embedding)
        label_names = np.unique(label)
        plt.figure()
        for i, label_name in enumerate(label_names):
            tmp = X_embedded[label == label_name]
            plt.scatter(tmp[:, 0], tmp[:, 1], alpha=0.3, label=label_name)
        plt.legend()
        plt.title(f"Distribution of {name} embedding layer at {self.epochs}.")
        if len(dirname) == 0:
            dirname = os.path.join(self.config["outdir"], "predictions", "embedding")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.tight_layout()
        plt.savefig(os.path.join(dirname, f"{name}epoch{self.epochs}.png"))

    def _get_center_loss_label(self, y_clip):
        """Get center loss label.

        Args:
            y_clip (torch.tensor): (B, n_target)
        Returns:
            label: (torch.tensor): (B,)
        """
        batch_size = len(y_clip)
        label = torch.zeros(batch_size).to(self.device)
        for i in range(batch_size):
            called_idx = torch.where(y_clip[i][:24] != 0)[0]
            if len(called_idx) == 0:
                label[i] = 24
            else:
                label[i] = called_idx[random.randint(0, len(called_idx) - 1)]
        return label

    def _fix_class_data(self, class_data, mode="clip"):
        if mode == "clip":
            class_data[:, 17] = class_data[:, [17, 24]].max(dim=1)[0]
            class_data[:, 23] = class_data[:, [23, 25]].max(dim=1)[0]
        elif mode == "frame":
            class_data[:, :, 17] = class_data[:, :, [17, 24]].max(dim=2)[0]
            class_data[:, :, 23] = class_data[:, :, [23, 25]].max(dim=2)[0]
        elif mode == "np_clip":
            class_data[:, 17] = class_data[:, [17, 24]].max(axis=1)
            class_data[:, 23] = class_data[:, [23, 25]].max(axis=1)
        elif mode == "np_frame":
            class_data[:, :, 17] = class_data[:, :, [17, 24]].max(axis=2)
            class_data[:, :, 23] = class_data[:, :, [23, 25]].max(axis=2)
        return class_data

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        if self.train:
            for key, value in loss.items():
                self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if (self.steps % self.config["save_interval_steps"] == 0) and (self.steps != 0):
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"],
                    f"checkpoint-{self.steps}",
                    f"checkpoint-{self.steps}{self.save_name}.pkl",
                ),
                save_model_only=False,
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True
