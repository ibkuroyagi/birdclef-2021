# %%
import logging
import json
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.utils.data as torchdata

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm

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
args.distributed = False
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
    "transforms": {
        "train": [{"name": "Normalize"}],
        "valid": [{"name": "Normalize"}],
        "test": [{"name": "Normalize"}],
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
    "mixup_alpha": 1,  # if you don't use mixup, please input 0.
    "mode": "const",
    "max_rate": 0.8,
    "min_rate": 0.0,
    ######################
    # Loaders #
    ######################
    "loader_params": {
        "train": {"batch_size": BATCH_SIZE, "num_workers": 2},
        "valid": {"batch_size": BATCH_SIZE, "num_workers": 2},
        "test": {"batch_size": BATCH_SIZE, "num_workers": 2},
    },
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
set_seed(config["seed"])
logger = get_logger("main.log")
TEST = len(list(Path("../input/birdclef-2021/test_soundscapes/").glob("*.ogg"))) != 0
if TEST:
    DATADIR = Path("../input/birdclef-2021/test_soundscapes/")
    save
else:
    DATADIR = Path("../input/birdclef-2021/train_soundscapes/")


# %%
all_audios = list(DATADIR.glob("*.ogg"))
all_audio_ids = ["_".join(audio_id.name.split("_")[:2]) for audio_id in all_audios]
submission_df = pd.DataFrame({"row_id": all_audio_ids})
submission_df


# %%
class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray, waveform_transforms=None):
        self.df = df
        self.clip = clip
        self.waveform_transforms = waveform_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        start_index = SR * start_seconds
        end_index = SR * end_seconds

        y = self.clip[start_index:end_index].astype(np.float32)

        y = np.nan_to_num(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = np.nan_to_num(y)

        return y, row_id


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


# %%
def prepare_model_for_inference(model, path: Path):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_checkpoint(
    self, model, checkpoint_path, load_only_params=False, distributed=False
):
    """Load checkpoint.

    Args:
        checkpoint_path (str): Checkpoint path to be loaded.
        load_only_params (bool): Whether to load only model parameters.

    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if config["distributed"]:
        model.module.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict["model"])
    if not load_only_params:
        steps = state_dict["steps"]
        epochs = state_dict["epochs"]
        best_score = state_dict.get("best_score", 0)
        logging.info(f"Steps:{steps}, Epochs:{epochs}, BEST score:{best_score}")
    return model


# %%
def prediction_for_clip(test_df: pd.DataFrame, clip: np.ndarray, model, threshold=0.5):

    dataset = TestDataset(
        df=test_df, clip=clip, waveform_transforms=get_transforms(phase="test")
    )
    loader = torchdata.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    prediction_dict = {}
    for image, row_id in tqdm(loader):
        row_id = row_id[0]
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            proba = prediction["clipwise_output"].detach().cpu().numpy().reshape(-1)

        events = proba >= threshold
        labels = np.argwhere(events).reshape(-1).tolist()

        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: target_columns[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
    return prediction_dict


# %%
def prediction(test_audios, weights_path: Path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmSED(
        base_model_name=config["base_model_name"],
        pretrained=False,
        num_classes=config["num_classes"],
        in_channels=config["in_channels"],
    )
    model = prepare_model_for_inference(model, weights_path).to(device)

    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_path in test_audios:
        with timer(f"Loading {str(audio_path)}", logger):
            clip, _ = sf.read(audio_path)

        seconds = []
        row_ids = []
        for second in range(5, 605, 5):
            row_id = "_".join(audio_path.name.split("_")[:2]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)

        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        with timer(f"Prediction on {audio_path}", logger):
            prediction_dict = prediction_for_clip(
                test_df, clip=clip, model=model, threshold=threshold
            )
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({"row_id": row_id, "birds": birds})
        prediction_dfs.append(prediction_df)

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df


# %%
weights_path = Path("../input/birdclef2021-effnetb0-starter-weight/best.pth")
submission = prediction(
    test_audios=all_audios, weights_path=weights_path, threshold=0.5
)
submission.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv")
