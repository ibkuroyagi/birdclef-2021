import os
import sys

import soundfile as sf
import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.append("../input/modules")
from utils import target_columns  # noqa: E402

BATCH_SIZE = 32
input_dir = "../input/birdclef-2021"
split_sec = 5
output_dir = f"dump/train_short_audio_{split_sec}sec"
train_soundscape = pd.read_csv(os.path.join(input_dir, "train_soundscape_labels.csv"))
train_y_df = pd.read_csv("exp/arai_infer_tf_efficientnet_b0_ns/no_aug/bce/train_y.csv")
sr = 32000

# dumpディレクトリに20secに分割したtrain_short_audioとそれに対応するdf["path","birds", "dataset", "fold"]を保存

effective_length = sr * split_sec
all_list = []
new_df = pd.DataFrame(columns=train_y_df.columns)
for i, bird in enumerate(tqdm(target_columns)):
    indir_name = os.path.join(input_dir, "train_short_audio", bird)
    outdir_name = os.path.join(output_dir, bird)
    if not os.path.isdir(outdir_name):
        os.makedirs(outdir_name)

    for ogg_name in sorted(os.listdir(indir_name)):
        items = {}
        ogg_path = os.path.join(indir_name, ogg_name)
        train_y_df_idx = train_y_df["path"] == ogg_path
        items["birds"] = bird
        items["fold"] = train_y_df.loc[train_y_df_idx, "fold"].values[0]
        items["dataset"] = "train_short_audio"
        x, sr = sf.read(ogg_path)
        len_x = len(x)
        if len_x <= effective_length:
            new_x = np.zeros(effective_length)
            new_x[:len_x] = x.copy()
            save_path = os.path.join(
                outdir_name, ogg_name.split(".")[0] + f"_0_{split_sec}.ogg"
            )
            items["path"] = save_path
            sf.write(save_path, new_x, sr)
            new_df = new_df.append(items, ignore_index=True)
        for j in range(len_x // effective_length):
            start = effective_length * j
            end = effective_length * (j + 1)
            start_sec = j * split_sec
            end_sec = (j + 1) * split_sec
            items["path"] = os.path.join(
                outdir_name, ogg_name.split(".")[0] + f"_{start_sec}_{end_sec}.ogg",
            )
            new_df = new_df.append(items, ignore_index=True)
            sf.write(items["path"], x[start:end], sr)
        print(new_df, flush=True)
new_df.to_csv(output_dir + ".csv", index=False)
print(f"Successfully saved {output_dir}.csv")
