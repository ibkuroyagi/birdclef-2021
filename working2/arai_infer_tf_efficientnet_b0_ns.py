import logging
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.utils.data as torchdata

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from sklearn import metrics

sys.path.append("../input/modules")
import datasets  # noqa: E402
from models import TimmSED  # noqa: E402
from utils import target_columns  # noqa: E402
from utils import set_seed  # noqa: E402
from utils import get_logger  # noqa: E402

sys.path.append("../input/iterative-stratification-master")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: E402

BATCH_SIZE = 32

# Config
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
    "--verbose",
    default=1,
    type=int,
    help="logging level. higher is more logging. (default=1)",
)
args = parser.parse_args()
if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


config = {
    ######################
    # Globals #
    ######################
    "seed": 1213,
    "epochs": 20,
    "train": True,
    "folds": [0],
    "img_size": 128,
    "n_frame": 128,
    ######################
    # Interval setting #
    ######################
    "save_interval_epochs": 5,
    ######################
    # Data #
    ######################
    "train_datadir": "../input/birdclef-2021/train_short_audio",
    "train_csv": "../input/birdclef-2021/train_metadata.csv",
    "train_soundscape": "../input/birdclef-2021/train_soundscape_labels.csv",
    ######################
    # Dataset #
    ######################
    "transforms": {"train": {"Normalize": {}}, "valid": {"Normalize": {}},},
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
    "mixup_alpha": 1,  # if you don't use mixup, please input 0.
    "mode": "const",
    "max_rate": 0.8,
    "min_rate": 0.0,
    ######################
    # Loaders #
    ######################
    "loader_params": {"valid": {"batch_size": BATCH_SIZE * 2, "num_workers": 2}},
    ######################
    # Split #
    ######################
    "split": "StratifiedKFold",
    "split_params": {"n_splits": 5, "shuffle": True, "random_state": 1213},
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
save_name = f"{config['save_name']}"
if not os.path.exists(os.path.join(config["outdir"], save_name)):
    os.makedirs(os.path.join(config["outdir"], save_name), exist_ok=True)
set_seed(config["seed"])
logger = get_logger(os.path.join(config["outdir"], save_name, "infer.log"))

TEST = len(list(Path("../input/birdclef-2021/test_soundscapes/").glob("*.ogg"))) != 0
if TEST:
    DATADIR = Path("../input/birdclef-2021/test_soundscapes/")
else:
    DATADIR = Path("../input/birdclef-2021/train_soundscapes/")

# set dataset
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
splitter = MultilabelStratifiedKFold(**config["split_params"])
y = np.zeros((len(df), 397))
for i in range(len(df)):
    for bird in df.loc[i, "birds"].split(" "):
        y[i, np.array(target_columns) == bird] = 1.0
df["dataset"] = "train_soundscape"
df.loc[
    ["train_short_audio" in path for path in df["path"].values], "dataset"
] = "train_short_audio"
df["fold"] = 0
for i, (trn_idx, val_idx) in enumerate(splitter.split(df, y=y)):
    df.loc[val_idx, "fold"] = i
df.to_csv(os.path.join(config["outdir"], save_name, "train_y.csv"), index=False)
# %%

path_list = config["resume"]
pred_y_clip = np.zeros((len(df), config["n_target"]))
pred_y_frame = np.zeros((len(df), config["n_target"]))


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


def load_checkpoint(model, checkpoint_path, load_only_params=False, distributed=False):
    """Load checkpoint.

    Args:
        checkpoint_path (str): Checkpoint path to be loaded.
        load_only_params (bool): Whether to load only model parameters.

    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if distributed:
        model.module.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict["model"])
    if not load_only_params:
        steps = state_dict["steps"]
        epochs = state_dict["epochs"]
        best_score = state_dict.get("best_score", 0)
        logging.info(f"Steps:{steps}, Epochs:{epochs}, BEST score:{best_score}")
        # print(f"Steps:{steps}, Epochs:{epochs}, BEST score:{best_score}")
    return model.eval()


# predict oof
for i, path in enumerate(path_list):
    logger.info(f"Start fold {i}")
    valid_idx = df["fold"] == i
    val_df = df[valid_idx].reset_index(drop=True)
    logger.info(f"fold {i}, {valid_idx.sum()}, {val_df.shape}")
    label = y[valid_idx]
    model = TimmSED(
        base_model_name=config["base_model_name"],
        pretrained=False,
        num_classes=config["n_target"],
        in_channels=config["in_channels"],
    )
    model.training = False
    model = load_checkpoint(model, path, load_only_params=False, distributed=False).to(
        device
    )
    data_loader = torchdata.DataLoader(
        WaveformDataset(
            val_df,
            label=label,
            waveform_transforms=get_transforms("valid"),
            period=config["period"],
            validation=True,
        ),
        shuffle=False,
        **config["loader_params"]["valid"],
    )
    pred_y_clip_fold = np.empty((0, config["n_target"]))
    pred_y_frame_fold = np.empty((0, config["n_target"]))
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch["X"].to(device)
            y_ = model(x)
            pred_y_clip_fold = np.concatenate(
                [pred_y_clip_fold, torch.sigmoid(y_["logit"]).cpu().numpy()], axis=0
            )
            pred_y_frame_fold = np.concatenate(
                [
                    pred_y_frame_fold,
                    torch.sigmoid(y_["framewise_logit"]).cpu().numpy().max(axis=1),
                ],
                axis=0,
            )
    pred_y_clip[valid_idx] = pred_y_clip_fold.copy()
    pred_y_frame[valid_idx] = pred_y_frame_fold.copy()
    clip_f1 = metrics.f1_score(
        label, pred_y_clip[valid_idx] > 0.1, average="samples", zero_division=0,
    )
    frame_f1 = metrics.f1_score(
        label, pred_y_frame[valid_idx] > 0.1, average="samples", zero_division=0,
    )
    logger.info(f"fold {i} clip f1 0.1:{clip_f1:.4f}, frame f1:{frame_f1:.4f}")
    logger.info(f"Finish fold {i}")
    # np.save(
    #     os.path.join(config["outdir"], save_name, f"pred_y_clip{i}.npy"),
    #     pred_y_clip_fold,
    # )
    # np.save(
    #     os.path.join(config["outdir"], save_name, f"pred_y_frame{i}.npy"),
    #     pred_y_frame_fold,
    # )
np.save(os.path.join(config["outdir"], save_name, "train_y.npy"), y)
np.save(os.path.join(config["outdir"], save_name, "pred_y_clip.npy"), pred_y_clip)
np.save(os.path.join(config["outdir"], save_name, "pred_y_frame.npy"), pred_y_frame)
# calculate oof f1 score
best_clip_f1 = 0
best_frame_f1 = 0
best_clip_thred = 0.01
best_frame_thred = 0.01
for threshold in np.arange(0.01, 0.5, 0.01):
    clip_f1 = metrics.f1_score(
        y, pred_y_clip > threshold, average="samples", zero_division=0,
    )
    frame_f1 = metrics.f1_score(
        y, pred_y_frame > threshold, average="samples", zero_division=0,
    )
    if clip_f1 > best_clip_f1:
        best_clip_f1 = clip_f1
        best_clip_thred = threshold
    if frame_f1 > best_frame_f1:
        best_frame_f1 = frame_f1
        best_frame_thred = threshold
    logger.info(
        f"threshold:{threshold:.2f}, clip f1:{clip_f1:.4f}, frame f1:{frame_f1:.4f}"
    )
logger.info(
    f"best clip threshold:{best_clip_thred:.2f}, best_clip_f1:{best_clip_f1:.4f}"
)
logger.info(
    f"best frame threshold:{best_frame_thred:.2f}, best_frame_f1:{best_frame_f1:.4f}"
)
