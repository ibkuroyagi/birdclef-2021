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
import losses  # noqa: E402
import optimizers  # noqa: E402
from models import TimmSED  # noqa: E402
from utils import target_columns  # noqa: E402
from utils import set_seed  # noqa: E402

sys.path.append("../input/iterative-stratification-master")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: E402

BATCH_SIZE = 4

# ## Config
parser = argparse.ArgumentParser(
    description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
)
parser.add_argument("--outdir", type=str, required=True, help="name of outdir.")
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
    "epochs": 2,
    "train": True,
    "folds": [0],
    "img_size": 128,
    "n_frame": 128,
    ######################
    # Interval setting #
    ######################
    "save_interval_epochs": 2,
    ######################
    # Data #
    ######################
    "train_datadir": "../input/birdclef-2021/train_short_audio",
    "train_csv": "../input/birdclef-2021/train_metadata.csv",
    "train_soundscape": "../input/birdclef-2021/train_soundscape_labels.csv",
    ######################
    # Dataset #
    ######################
    "transforms": {"train": [{"name": "Normalize"}], "valid": [{"name": "Normalize"}]},
    "period": 20,
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "sample_rate": 32000,
    "melspectrogram_parameters": {"n_mels": 128, "fmin": 20, "fmax": 16000},
    "accum_grads": 4,
    ######################
    # Loaders #
    ######################
    "loader_params": {
        "train": {"batch_size": BATCH_SIZE, "num_workers": 2},
        "valid": {"batch_size": BATCH_SIZE * 2, "num_workers": 2},
    },
    ######################
    # Split #
    ######################
    "split": "StratifiedKFold",
    "split_params": {"n_splits": 5, "shuffle": True, "random_state": 1213},
    ######################
    # Model #
    ######################
    "base_model_name": "tf_efficientnet_b7_ns",
    "pooling": "max",
    "pretrained": True,
    "n_target": 397,
    "in_channels": 1,
    ######################
    # Criterion #
    ######################
    "loss_type": "BCEFocal2WayLoss",
    "loss_params": {},
    ######################
    # Optimizer #
    ######################
    "optimizer_type": "Adam",
    "optimizer_params": {"lr": 2.0e-3},
    # For SAM optimizer
    "base_optimizer": "Adam",
    ######################
    # Scheduler #
    ######################
    "scheduler_type": "CosineAnnealingLR",
    "scheduler_params": {"T_max": 10},
}
config.update(vars(args))

# this notebook is by default run on debug mode (only train one epoch).
# If you'd like to get the results on par with that of inference notebook, you'll need to train the model around 30 epochs

train_meta_df = pd.read_csv(config["train_csv"])
train_soundscape = pd.read_csv(config["train_soundscape"])
train_meta_df = train_meta_df.rename(columns={"primary_label": "birds"})
train_meta_df["path"] = (
    config["train_datadir"]
    + "/"
    + train_meta_df["birds"]
    + "/"
    + train_meta_df["filename"]
)

soundscape = pd.read_csv(f"dump/train_{config['period']}sec.csv")
soundscape = soundscape[soundscape["birds"] != "nocall"]
df = pd.concat(
    [soundscape[["path", "birds"]], train_meta_df[["path", "birds"]]], axis=0
).reset_index(drop=True)
ALL_DATA = len(df)
DEBUG = False
if DEBUG:
    config["epochs"] = 1
steps_per_epoch = ALL_DATA // (BATCH_SIZE * config["n_gpus"] * config["accum_grads"])
config["log_interval_steps"] = steps_per_epoch // 5
config["train_max_steps"] = config["epochs"] * steps_per_epoch
with open(os.path.join(args.outdir, "config.yml"), "w") as f:
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

# In this section, I define dataset that crops 20 second chunk. The output of this dataset is a pair of waveform and corresponding label.


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
        path, bird = self.df.iloc[idx]
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
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None


class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


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
            self.writer = SummaryWriter(config["outdir"])
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
            self._valid_epoch()

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
        logging.debug(f"y_clip,{y_clip.shape}:{y_clip[0]}")
        y_ = self.model(x)  # {y_frame: (B, T', n_target), y_clip: (B, n_target)}
        for key, val in y_.items():
            logging.debug(f"{key}:{val.shape},{val[0]}")
        if self.config["loss_type"] in [
            "BCEWithLogitsLoss",
            "BCEFocalLoss",
        ]:
            loss = self.criterion(y_["clipwise_output"], y_clip)
        elif self.config["loss_type"] in ["BCEFocal2WayLoss"]:
            loss = self.criterion(y_["logit"], y_["framewise_logit"], y_clip)
        if not torch.isnan(loss):
            loss = loss / self.config["accum_grads"]
            loss.backward()
            self.forward_count += 1
            self.total_train_loss["train/loss"] += loss.item()
            logging.debug(
                f'{y_clip.cpu().numpy()},{y_["clipwise_output"].detach().cpu().numpy() > 0.5}'
            )
            self.total_train_loss["train/f1_02"] += metrics.f1_score(
                y_clip.cpu().numpy(),
                y_["clipwise_output"].detach().cpu().numpy() > 0.2,
                average="samples",
                zero_division=0,
            )
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
            with torch.no_grad():
                if self.config["loss_type"] in [
                    "BCEWithLogitsLoss",
                    "BCEFocalLoss",
                ]:
                    self.epoch_train_loss["train/epoch_main_loss"] = self.criterion(
                        torch.tensor(self.train_pred_epoch).to(self.device),
                        torch.tensor(self.train_y_epoch).to(self.device),
                    ).item()
                elif self.config["loss_type"] in ["BCEFocal2WayLoss"]:
                    self.epoch_train_loss["train/epoch_main_loss"] = self.criterion(
                        torch.tensor(self.train_pred_logit_epoch).to(self.device),
                        torch.tensor(self.train_pred_logitframe_epoch).to(self.device),
                        torch.tensor(self.train_y_epoch).to(self.device),
                    ).item()
            self.epoch_train_loss["train/epoch_loss"] = self.epoch_train_loss[
                "train/epoch_main_loss"
            ]

            self.epoch_train_loss["train/epoch_f1_03_clip"] = metrics.f1_score(
                self.train_y_epoch,
                self.train_pred_epoch > 0.3,
                average="samples",
                zero_division=0,
            )
            self.epoch_train_loss["train/epoch_f1_02_clip"] = metrics.f1_score(
                self.train_y_epoch,
                self.train_pred_epoch > 0.2,
                average="samples",
                zero_division=0,
            )
            self.epoch_train_loss["train/epoch_f1_01_clip"] = metrics.f1_score(
                self.train_y_epoch,
                self.train_pred_epoch > 0.1,
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
        # if self.config["loss_type"] in [
        #     "BCEWithLogitsLoss",
        #     "BCEFocalLoss",
        # ]:
        #     loss = self.criterion(y_["clipwise_output"], y_clip)
        # elif self.config["loss_type"] in ["BCEFocal2WayLoss"]:
        #     loss = self.criterion(y_["logit"], y_["framewise_logit"], y_clip)
        # add to total valid loss
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
            elif self.config["loss_type"] in ["BCEFocal2WayLoss"]:
                self.epoch_valid_loss["valid/epoch_main_loss"] = self.criterion(
                    torch.tensor(self.valid_pred_logit_epoch).to(self.device),
                    torch.tensor(self.valid_pred_logitframe_epoch).to(self.device),
                    torch.tensor(self.valid_y_epoch).to(self.device),
                ).item()
            self.epoch_valid_loss["valid/epoch_loss"] = self.epoch_valid_loss[
                "valid/epoch_main_loss"
            ]
            self.epoch_valid_loss["valid/epoch_f1_03_clip"] = metrics.f1_score(
                self.valid_y_epoch,
                self.valid_pred_epoch > 0.3,
                average="samples",
                zero_division=0,
            )
            self.epoch_valid_loss["valid/epoch_f1_02_clip"] = metrics.f1_score(
                self.valid_y_epoch,
                self.valid_pred_epoch > 0.2,
                average="samples",
                zero_division=0,
            )
            self.epoch_valid_loss["valid/epoch_f1_01_clip"] = metrics.f1_score(
                self.valid_y_epoch,
                self.valid_pred_epoch > 0.1,
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
        if self.epoch_valid_loss["valid/epoch_f1_clip"] > self.best_score:
            self.best_score = self.epoch_valid_loss["valid/epoch_f1_clip"]
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


# Training!

# validation
splitter = MultilabelStratifiedKFold(**config["split_params"])

# data
y = np.zeros((len(df), 397))
for i in range(len(df)):
    for bird in df.loc[i, "birds"].split(" "):
        y[i, np.array(target_columns) == bird] = 1.0
# main loop
for i, (trn_idx, val_idx) in enumerate(splitter.split(df, y=y)):
    if i not in config["folds"]:
        continue
    logging.info("=" * 120)
    logging.info(f"Fold {i} Training")
    logging.info("=" * 120)

    trn_df = df.loc[trn_idx, :].reset_index(drop=True)
    val_df = df.loc[val_idx, :].reset_index(drop=True)
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
    if i == 0:
        logging.info(model)
    loss_class = getattr(losses, config.get("loss_type", "BCEWithLogitsLoss"),)
    if config["loss_params"].get("pos_weight", None) is not None:
        pos_weight = config["loss_params"]["pos_weight"]
        config["loss_params"]["pos_weight"] = torch.tensor(
            pos_weight, dtype=torch.float
        ).to(device)
    criterion = loss_class(**config["loss_params"]).to(device)
    optimizer_class = getattr(optimizers, config.get("optimizer_type", "Adam"),)
    optimizer = optimizer_class(model.parameters(), **config["optimizer_params"])
    scheduler = None
    if config.get("scheduler_type", None) is not None:
        scheduler_class = getattr(
            torch.optim.lr_scheduler, config.get("scheduler_type", "StepLR"),
        )
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
        train=i == 0,
        save_name=f"fold{i}",
    )
    # resume from checkpoint
    if len(args.resume) != 0:
        if args.resume[i] != "no_model":
            trainer.load_checkpoint(args.resume[i], load_only_params=False)
            logging.info(f"Successfully resumed from {args.resume[i]}.")
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
