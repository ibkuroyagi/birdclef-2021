import gc
import os
import json
import argparse
import sys
import logging
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.utils.data as torchdata
import yaml
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
from sklearn import metrics

sys.path.append("../input/modules")
import datasets  # noqa: E402
import losses  # noqa: E402
import optimizers  # noqa: E402
from models import TimmSED  # noqa: E402
from models import mixup_for_sed  # noqa: E402
from utils import target_columns  # noqa: E402
from utils import set_seed  # noqa: E402
from utils import sigmoid  # noqa: E402
from utils import mixup_apply_rate  # noqa: E402
from utils import pos_weight  # noqa: E402

BATCH_SIZE = 64

# ## Config
parser = argparse.ArgumentParser(
    description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
)
parser.add_argument("--outdir", type=str, required=True, help="name of outdir.")
parser.add_argument("--save_name", type=str, default="", help="name of save file.")
parser.add_argument(
    "--resume",
    default=[],
    type=str,
    nargs="*",
    help="checkpoint file path to resume training. (default=[])",
)
parser.add_argument(
    "--n_gpus", default=1, type=int, help="The number of gpu. (default=1)",
)
parser.add_argument(
    "--verbose",
    default=1,
    type=int,
    help="logging level. higher is more logging. (default=1)",
)
parser.add_argument(
    "--fold", default=0, type=int, help="Fold. (default=0)",
)
parser.add_argument(
    "--rank",
    "--local_rank",
    default=0,
    type=int,
    help="rank for distributed training. no need to explictly specify.",
)
args = parser.parse_args()
args.distributed = False
if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    torch.cuda.set_device(args.rank)
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
# suppress logging for distributed training
if args.rank != 0:
    sys.stdout = open(os.devnull, "w")
# set logger
if args.verbose > 1:
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.getLogger("matplotlib.font_manager").disabled = True
elif args.verbose > 0:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
else:
    logging.basicConfig(
        level=logging.WARN,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.warning("Skip DEBUG/INFO messages")
config = {
    ######################
    # Globals #
    ######################
    "seed": 1213,
    "epochs": 30,
    "train": True,
    "folds": [args.fold],
    "img_size": 128,
    "n_frame": 128,
    ######################
    # Interval setting #
    ######################
    "save_interval_epochs": 5,
    ######################
    # Dataset #
    ######################
    "transforms": {
        "train": {
            "Normalize": {},
            "VolumeControl": {
                "always_apply": False,
                "p": 0.8,
                "db_limit": 10,
                "mode": "uniform",
            },
        },
        "valid": {"Normalize": {}},
    },
    "period": 20,
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "sample_rate": 32000,
    "melspectrogram_parameters": {"n_mels": 128, "fmin": 20, "fmax": 16000},
    "accum_grads": 1,
    ######################
    # Mixup #
    ######################
    "mixup_alpha": 0.2,  # if you don't use mixup, please input 0.
    "mode": "cos",
    "max_rate": 1.0,
    "min_rate": 0.0,
    ######################
    # Loaders #
    ######################
    "loader_params": {
        "train": {"batch_size": BATCH_SIZE, "num_workers": 2},
        "valid": {"batch_size": BATCH_SIZE * 2, "num_workers": 2},
    },
    ######################
    # Model #
    ######################
    "base_model_name": "tf_efficientnet_b0_ns",
    "pooling": "max",
    "pretrained": True,
    "n_target": 397,
    "in_channels": 1,
    ######################
    # Criterion #
    ######################
    "loss_type": "BCE2WayLoss",
    # "loss_params": {"pos_weight": None},  # pos_weight
    "loss_params": {"pos_weight": pos_weight},
    ######################
    # Optimizer #
    ######################
    "optimizer_type": "Adam",
    "optimizer_params": {"lr": 2.0e-3},
    ######################
    # Scheduler #
    ######################
    "scheduler_type": "CosineAnnealingLR",
    "scheduler_params": {"T_max": 15, "eta_min": 5.0e-4},
}
config.update(vars(args))
train_short_audio_df = pd.read_csv("dump/relabel20sec/b0_mixup2/relabel.csv")
train_short_audio_df = train_short_audio_df[train_short_audio_df["birds"] != "nocall"]
soundscape = pd.read_csv("exp/arai_infer_tf_efficientnet_b0_ns/no_aug/bce/train_y.csv")
soundscape = soundscape[
    (soundscape["birds"] != "nocall") & (soundscape["dataset"] == "soundscape")
]
df = pd.concat([train_short_audio_df, soundscape], axis=0).reset_index(drop=True)
steps_per_epoch = len(df[df["fold"] != config["folds"][0]]) // (
    BATCH_SIZE * config["n_gpus"] * config["accum_grads"]
)
config["log_interval_steps"] = steps_per_epoch // 3
config["train_max_steps"] = config["epochs"] * steps_per_epoch
save_name = f"fold{config['folds'][0]}{args.save_name}"
if not os.path.exists(os.path.join(config["outdir"], save_name)):
    os.makedirs(os.path.join(config["outdir"], save_name), exist_ok=True)
with open(os.path.join(args.outdir, save_name, "config.yml"), "w") as f:
    yaml.dump(config, f, Dumper=yaml.Dumper)

for key, value in config.items():
    logging.info(f"{key} = {value}")
set_seed(config["seed"])
# check distributed training
if args.distributed:
    logging.info(f"device:{device}")
    logging.info(f"args.rank:{args.rank}")
    logging.info(f"os.environ:{os.environ}")
    logging.info(f"args.world_size:{args.world_size}")
    logging.info(f"args.distributed:{args.distributed}")


class WaveformDataset(torchdata.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label: np.array,
        waveform_transforms=None,
        period=20,
        validation=False,
    ):
        self.df = df
        self.label = label
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.df.loc[idx, "path"]
        x, sr = sf.read(path)
        len_x = len(x)
        effective_length = sr * self.period
        if len_x < effective_length:
            new_x = np.zeros(effective_length, dtype=x.dtype)
            if not self.validation:
                start = np.random.randint(effective_length - len_x)
            else:
                start = 0
            new_x[start : start + len_x] = x
            x = new_x.astype(np.float32)
        elif len_x > effective_length:
            if not self.validation:
                start = np.random.randint(len_x - effective_length)
            else:
                start = 0
            x = x[start : start + effective_length].astype(np.float32)
        else:
            x = x.astype(np.float32)
        if self.waveform_transforms:
            x = self.waveform_transforms(x)
        return {"X": np.nan_to_num(x), "y": self.label[idx]}


def get_transforms(phase: str):
    transforms = config["transforms"]
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for key, params in transforms[phase].items():
            trns_cls = getattr(datasets, key)
            trns_list.append(trns_cls(**params))

        if len(trns_list) > 0:
            return datasets.Compose(trns_list)
        else:
            return None


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
        self.train = train
        if train:
            if not os.path.exists(os.path.join(config["outdir"], save_name)):
                os.makedirs(os.path.join(config["outdir"], save_name), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(config["outdir"], save_name))
        self.save_name = save_name

        self.finish_train = False
        self.best_score = 0
        self.n_target = 397
        self.total_train_loss = defaultdict(float)
        self.epoch_train_loss = defaultdict(float)
        self.epoch_valid_loss = defaultdict(float)
        self.valid_metric = defaultdict(float)
        self.train_pred_epoch = np.empty((0, self.n_target))
        self.train_pred_logit_epoch = np.empty((0, config["n_target"]))
        self.train_pred_logitframe_epoch = np.empty(
            (0, config["n_frame"], config["n_target"])
        )
        self.train_y_epoch = np.empty((0, self.n_target))
        self.valid_pred_epoch = np.empty((0, self.n_target))
        self.valid_pred_logit_epoch = np.empty((0, config["n_target"]))
        self.valid_pred_logitframe_epoch = np.empty(
            (0, config["n_frame"], config["n_target"])
        )
        self.valid_y_epoch = np.empty((0, self.n_target))
        self.last_checkpoint = ""
        self.forward_count = 0
        self.log_dict = {}

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()
            # self._valid_epoch()

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
        x = batch["X"].to(self.device)  # (B, mel, T')
        y_clip = batch["y"].to(self.device)
        if self.config.get("mixup_alpha", 0) > 0:
            if np.random.rand() < mixup_apply_rate(
                max_step=self.config["train_max_steps"],
                step=self.steps,
                max_rate=self.config.get("max_rate", 1.0),
                min_rate=self.config.get("min_rate", 0.0),
                mode=self.config.get("mode", "cos"),
            ):
                x, y_clip = mixup_for_sed(x, y_clip, alpha=self.config["mixup_alpha"])
        logging.debug(f"y_clip,{y_clip.shape}:{y_clip[0]}")
        y_ = self.model(x)  # {y_frame: (B, T', n_target), y_clip: (B, n_target)}
        for key, val in y_.items():
            logging.debug(f"{key}:{val.shape},{val[0]}")
        if self.config["loss_type"] in [
            "BCEWithLogitsLoss",
            "BCEFocalLoss",
        ]:
            loss = self.criterion(y_["clipwise_output"], y_clip)
        elif self.config["loss_type"] in ["BCEFocal2WayLoss", "BCE2WayLoss"]:
            loss = self.criterion(y_["logit"], y_["framewise_logit"], y_clip)
        if not torch.isnan(loss):
            self.forward_count += 1
            # if (self.config["accum_grads"] > self.forward_count) and self.config[
            #     "distributed"
            # ]:
            #     with self.model.no_sync():
            #         loss = loss / self.config["accum_grads"]
            #         loss.backward()
            # else:
            loss = loss / self.config["accum_grads"]
            loss.backward()

            if self.forward_count == self.config["accum_grads"]:
                self.total_train_loss["train/loss"] += loss.item()
                logging.debug(
                    f'{y_clip.cpu().numpy()},{y_["clipwise_output"].detach().cpu().numpy() > 0.5}'
                )
                self.total_train_loss["train/f1_01"] += metrics.f1_score(
                    y_clip.cpu().numpy() > 0,
                    y_["clipwise_output"].detach().cpu().numpy() > 0.1,
                    average="samples",
                    zero_division=0,
                )
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
            logging.warn("Loss contain NaN. Don't back-propagated.")

        self.train_pred_logit_epoch = np.concatenate(
            [self.train_pred_logit_epoch, y_["logit"].detach().cpu().numpy()], axis=0,
        )
        self.train_pred_logitframe_epoch = np.concatenate(
            [
                self.train_pred_logitframe_epoch,
                y_["framewise_logit"].detach().cpu().numpy(),
            ],
            axis=0,
        )
        self.train_pred_epoch = np.concatenate(
            [self.train_pred_epoch, y_["clipwise_output"].detach().cpu().numpy()],
            axis=0,
        )
        self.train_y_epoch = np.concatenate(
            [self.train_y_epoch, y_clip.detach().cpu().numpy()], axis=0,
        )

    def _train_epoch(self):
        """Train model one epoch."""
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()

            # check whether training is finished
            if self.finish_train:
                return
        try:
            logging.debug(
                f"Epoch train  pred clip:{self.train_pred_epoch.shape}{self.train_pred_epoch.sum():.4f}\n"
                f"Epoch train     y clip:{self.train_y_epoch.shape}{self.train_y_epoch.sum()}\n"
            )
            self.epoch_train_loss["train/epoch_f1_02_clip"] = metrics.f1_score(
                self.train_y_epoch > 0,
                self.train_pred_epoch > 0.2,
                average="samples",
                zero_division=0,
            )
            self.epoch_train_loss["train/epoch_f1_015_clip"] = metrics.f1_score(
                self.train_y_epoch > 0,
                self.train_pred_epoch > 0.15,
                average="samples",
                zero_division=0,
            )
            self.epoch_train_loss["train/epoch_f1_01_clip"] = metrics.f1_score(
                self.train_y_epoch > 0,
                self.train_pred_epoch > 0.1,
                average="samples",
                zero_division=0,
            )
            self.epoch_train_loss["train/epoch_f1_02_frame"] = metrics.f1_score(
                self.train_y_epoch > 0,
                sigmoid(self.train_pred_logitframe_epoch.max(axis=1)) > 0.2,
                average="samples",
                zero_division=0,
            )
            self.epoch_train_loss["train/epoch_f1_01_frame"] = metrics.f1_score(
                self.train_y_epoch > 0,
                sigmoid(self.train_pred_logitframe_epoch.max(axis=1)) > 0.1,
                average="samples",
                zero_division=0,
            )
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
        # update
        self.train_steps_per_epoch = train_steps_per_epoch
        self.epochs += 1
        self._check_save_interval()
        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.train_sampler.set_epoch(self.epochs)
        # reset
        self.train_y_epoch = np.empty((0, self.n_target))
        self.train_pred_epoch = np.empty((0, self.n_target))
        self.train_pred_logit_epoch = np.empty((0, self.config["n_target"]))
        self.train_pred_logitframe_epoch = np.empty(
            (0, self.config["n_frame"], self.config["n_target"])
        )
        self.epoch_train_loss = defaultdict(float)

    @torch.no_grad()
    def _valid_step(self, batch):
        """Evaluate model one step."""
        x = batch["X"].to(self.device)
        y_clip = batch["y"].to(self.device)
        y_ = self.model(x)
        self.valid_pred_logit_epoch = np.concatenate(
            [self.valid_pred_logit_epoch, y_["logit"].detach().cpu().numpy()], axis=0,
        )
        self.valid_pred_logitframe_epoch = np.concatenate(
            [
                self.valid_pred_logitframe_epoch,
                y_["framewise_logit"].detach().cpu().numpy(),
            ],
            axis=0,
        )
        self.valid_pred_epoch = np.concatenate(
            [
                self.valid_pred_epoch,
                y_["clipwise_output"].detach().cpu().numpy().astype(np.float32),
            ],
            axis=0,
        )
        self.valid_y_epoch = np.concatenate(
            [self.valid_y_epoch, y_clip.detach().cpu().numpy().astype(np.float32)],
            axis=0,
        )

    def _valid_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start dev data's validation.")
        # change mode
        self.model.eval()
        self.model.training = False
        # calculate loss for each batch
        for valid_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["valid"], desc="[valid]"), 1
        ):
            # valid one step
            self._valid_step(batch)
        try:
            logging.debug(
                f"Epoch valid pred_clip:{self.valid_pred_epoch.sum()}\n"
                f"Epoch valid    y_clip:{self.valid_y_epoch.sum()}\n"
            )
            if self.config["loss_type"] in [
                "BCEWithLogitsLoss",
                "BCEFocalLoss",
            ]:
                self.epoch_valid_loss["valid/epoch_main_loss"] = self.criterion(
                    torch.tensor(self.valid_pred_epoch).to(self.device),
                    torch.tensor(self.valid_y_epoch).to(self.device),
                ).item()
            elif self.config["loss_type"] in ["BCEFocal2WayLoss", "BCE2WayLoss"]:
                self.epoch_valid_loss["valid/epoch_main_loss"] = self.criterion(
                    torch.tensor(self.valid_pred_logit_epoch).to(self.device),
                    torch.tensor(self.valid_pred_logitframe_epoch).to(self.device),
                    torch.tensor(self.valid_y_epoch).to(self.device),
                ).item()
            self.epoch_valid_loss["valid/epoch_loss"] = self.epoch_valid_loss[
                "valid/epoch_main_loss"
            ]
            self.epoch_valid_loss["valid/epoch_f1_02_clip"] = metrics.f1_score(
                self.valid_y_epoch > 0,
                self.valid_pred_epoch > 0.2,
                average="samples",
                zero_division=0,
            )
            self.epoch_valid_loss["valid/epoch_f1_015_clip"] = metrics.f1_score(
                self.valid_y_epoch > 0,
                self.valid_pred_epoch > 0.15,
                average="samples",
                zero_division=0,
            )
            self.epoch_valid_loss["valid/epoch_f1_01_clip"] = metrics.f1_score(
                self.valid_y_epoch > 0,
                self.valid_pred_epoch > 0.1,
                average="samples",
                zero_division=0,
            )
            self.epoch_valid_loss["valid/epoch_f1_02_frame"] = metrics.f1_score(
                self.valid_y_epoch > 0,
                sigmoid(self.valid_pred_logitframe_epoch.max(axis=1)) > 0.2,
                average="samples",
                zero_division=0,
            )
            self.epoch_valid_loss["valid/epoch_f1_01_frame"] = metrics.f1_score(
                self.valid_y_epoch > 0,
                sigmoid(self.valid_pred_logitframe_epoch.max(axis=1)) > 0.1,
                average="samples",
                zero_division=0,
            )
        except ValueError:
            logging.warning("Raise ValueError: May be contain NaN in y_pred.")
            pass
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished valid data's validation "
            f"({valid_steps_per_epoch} steps per epoch)."
        )
        for key in self.epoch_valid_loss.keys():
            logging.info(
                f"(Epoch: {self.epochs}) {key} = {self.epoch_valid_loss[key]:.6f}."
            )
        if self.epoch_valid_loss["valid/epoch_f1_01_clip"] > self.best_score:
            self.best_score = self.epoch_valid_loss["valid/epoch_f1_01_clip"]
            logging.info(
                f"Epochs: {self.epochs}, BEST score was updated {self.best_score:.6f}."
            )
            save_path = os.path.join(
                self.config["outdir"], "best_score", f"best_score{self.save_name}.pkl",
            )
            self.save_checkpoint(save_path, save_model_only=False)
            logging.info(
                f"Best model was updated @ {self.steps} steps." f"Saved at {save_path}"
            )

        logging.info(f"(Steps: {self.steps}) Start valid data's validation.")
        # record
        self._write_to_tensorboard(self.epoch_valid_loss)

        # reset
        self.epoch_valid_loss = defaultdict(float)
        self.valid_pred_epoch = np.empty((0, self.n_target))
        self.valid_y_epoch = np.empty((0, self.n_target))
        self.valid_pred_logit_epoch = np.empty((0, self.config["n_target"]))
        self.valid_pred_logitframe_epoch = np.empty(
            (0, self.config["n_frame"], self.config["n_target"])
        )
        # restore mode
        self.model.training = True
        self.model.train()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        if self.train:
            self.log_dict[self.steps] = {}
            for key, value in loss.items():
                self.writer.add_scalar(key, value, self.steps)
                self.log_dict[self.steps][key] = value
            with open(
                os.path.join(self.config["outdir"], f"metric{self.save_name}.json"), "w"
            ) as f:
                json.dump(self.log_dict, f, indent=4)

    def _check_save_interval(self):
        if (self.epochs % self.config["save_interval_epochs"] == 0) and (
            self.epochs != 0
        ):
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"],
                    f"checkpoint-{self.epochs}",
                    f"checkpoint-{self.epochs}{self.save_name}.pkl",
                ),
                save_model_only=False,
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            self._valid_epoch()
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


# data
y = np.zeros((len(df), 397))
for i in range(len(df)):
    for bird in df.loc[i, "birds"].split(" "):
        y[i, np.array(target_columns) == bird] = 1.0
# main loop
for i in range(5):
    if i not in config["folds"]:
        continue
    logging.info("=" * 120)
    logging.info(f"Fold {i} Training")
    logging.info("=" * 120)
    trn_idx = df["fold"] != i
    val_idx = df["fold"] == i
    trn_df = df[trn_idx].reset_index(drop=True)
    val_df = df[val_idx].reset_index(drop=True)
    data_loader = {}
    for phase, df_, label in zip(
        ["valid", "train"], [val_df, trn_df], [y[val_idx], y[trn_idx]]
    ):
        dataset = WaveformDataset(
            df_,
            label=label,
            waveform_transforms=get_transforms(phase),
            period=config["period"],
            validation=(phase == "valid"),
        )
        logging.info(f"{phase}:{len(dataset)} samples.")
        sampler = None
        if args.distributed:
            logging.info("Use multi gpu.")
            # setup sampler for distributed training
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=(phase == "train"),
            )
            data_loader[phase] = torchdata.DataLoader(
                dataset, sampler=sampler, **config["loader_params"][phase],
            )
        else:
            data_loader[phase] = torchdata.DataLoader(
                dataset,
                sampler=sampler,
                shuffle=(phase == "train"),
                **config["loader_params"][phase],
            )
    model = TimmSED(
        base_model_name=config["base_model_name"],
        pretrained=config["pretrained"],
        num_classes=config["n_target"],
        in_channels=config["in_channels"],
        n_mels=config["n_mels"],
    )
    if args.distributed:
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "Apex is not installed. Please check https://github.com/NVIDIA/apex."
            )
        # NOTE(ibkuroyagi): Needed to place the model on GPU
        model = DistributedDataParallel(model.to(device))
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model.to(device), device_ids=[args.rank], output_device=args.rank,
        # )
    if i == 0:
        logging.info(model)
    loss_class = getattr(losses, config.get("loss_type", "BCEWithLogitsLoss"),)
    if config["loss_params"].get("pos_weight", None) is not None:
        weight = config["loss_params"]["pos_weight"]
        config["loss_params"]["pos_weight"] = torch.tensor(
            weight, dtype=torch.float
        ).to(device)
    criterion = loss_class(**config["loss_params"]).to(device)
    optimizer_class = getattr(optimizers, config["optimizer_type"])
    if config["optimizer_type"] == "SAM":
        base_optimizer = getattr(optimizers, config["base_optimizer"])
        optimizer = optimizer_class(
            model.parameters(), base_optimizer, **config["optimizer_params"]
        )
    else:
        optimizer = optimizer_class(model.parameters(), **config["optimizer_params"])
    scheduler = None
    if config.get("scheduler_type", None) is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["scheduler_type"],)
        scheduler = scheduler_class(optimizer=optimizer, **config["scheduler_params"])
    trainer = SEDTrainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        train_sampler=sampler,
        model=model.to(device),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        train=True,
        save_name=save_name,
    )
    # resume from checkpoint
    if len(args.resume) != 0:
        if args.resume[0] != "no_model":
            trainer.load_checkpoint(args.resume[0], load_only_params=False)
            logging.info(f"Successfully resumed from {args.resume[0]}.")
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(
                config["outdir"], f"checkpoint-{trainer.steps}stepsfold{i}.pkl"
            )
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
    logging.info(f"Finish runner {i} fold.")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
