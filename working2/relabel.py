# %%
import os
import sys
import soundfile as sf
import numpy as np
import pandas as pd
import torch
import torch.utils.data as torchdata
from tqdm import tqdm

sys.path.append("../input/modules")
import datasets  # noqa: E402
from models import TimmSED  # noqa: E402
from utils import target_columns  # noqa: E402
from utils import best_th  # noqa: E402

BATCH_SIZE = 32
split_sec = 20
outdir = f"dump/relabel{split_sec}sec"
save_name = "b0_no_aug"
if not os.path.exists(os.path.join(outdir, save_name)):
    os.makedirs(os.path.join(outdir, save_name), exist_ok=True)
train_short_audio_df = pd.read_csv(f"dump/train_short_audio_{split_sec}sec.csv")
# data
y = np.zeros((len(train_short_audio_df), 397))
for i in range(len(train_short_audio_df)):
    for bird in train_short_audio_df.loc[i, "birds"].split(" "):
        y[i, np.array(target_columns) == bird] = 1.0
checkpoint_list = [
    f"exp/arai_train_tf_efficientnet_b0_ns_mgpu/no_aug/best_score/best_scorefold{fold}bce.pkl"
    for fold in range(5)
]
if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
# device = "cpu"
config = {
    ######################
    # Globals #
    ######################
    "seed": 1213,
    "epochs": 40,
    "train": True,
    "img_size": 128,
    "n_frame": 128,
    "train_soundscape": "../input/birdclef-2021/train_soundscape_labels.csv",
    ######################
    # Dataset #
    ######################
    "transforms": {"valid": {"Normalize": {}}},
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
    # Loaders #
    ######################
    "loader_params": {"valid": {"batch_size": BATCH_SIZE * 2, "num_workers": 2}},
    ######################
    # Model #
    ######################
    "base_model_name": "tf_efficientnet_b0_ns",
    "pooling": "max",
    "pretrained": True,
    "n_target": 397,
    "in_channels": 1,
}


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
        # logging.info(f"Steps:{steps}, Epochs:{epochs}, BEST score:{best_score}")
        print(f"Steps:{steps}, Epochs:{epochs}, BEST score:{best_score}")
    return model.eval()


pred_y_clip = np.zeros((len(train_short_audio_df), config["n_target"]))
pred_y_frame = np.zeros((len(train_short_audio_df), config["n_target"]))
for fold in range(5):
    print(f"Fold {fold}")
    valid_idx = train_short_audio_df["fold"] == fold
    val_df = train_short_audio_df[valid_idx].reset_index(drop=True)
    label = y[valid_idx]
    model = TimmSED(
        base_model_name=config["base_model_name"],
        pretrained=config["pretrained"],
        num_classes=config["n_target"],
        in_channels=config["in_channels"],
        n_mels=config["n_mels"],
    )
    model.training = False
    model = load_checkpoint(
        model, checkpoint_list[fold], load_only_params=False, distributed=False
    ).to(device)
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
np.save(os.path.join(outdir, save_name, "pred_y_clip.npy"), pred_y_clip)
print("Successfully saved pred_y_clip.npy")
np.save(os.path.join(outdir, save_name, "pred_y_frame.npy"), pred_y_frame)
print("Successfully saved pred_y_frame.npy")
new_y = (y > best_th).astype(np.int64)

nocall_idx = np.zeros(len(new_y)).astype(bool)
for i, bird in enumerate(target_columns):
    tmp_idx = (train_short_audio_df["birds"] == bird) & (
        pred_y_frame[:, i] < best_th[i]
    )
    nocall_idx |= tmp_idx
print(f"N nocall: {nocall_idx.sum()} / {len(nocall_idx)}")
new_train_short_audio_df = train_short_audio_df.copy()
new_train_short_audio_df.loc[nocall_idx, "birds"] = "nocall"
new_train_short_audio_df.to_csv(os.path.join(outdir, save_name, "relabel.csv"))
print(f"Successfully saved {os.path.join(outdir, save_name, 'relabel.csv')}")
# %%
