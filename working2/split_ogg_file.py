import glob
import os
import soundfile as sf
import pandas as pd
from tqdm import tqdm

input_dir = "../input/birdclef-2021"
split_sec = 5
output_dir = f"dump/train_{split_sec}sec"
train_soundscape = pd.read_csv(os.path.join(input_dir, "train_soundscape_labels.csv"))
# 1. 何秒で切り出すか指定
# 2. 指定した秒数でのバンドを作成し、row_id,seconds, birdsを更新
# 3. 対象ファイルをogg形式で保存

shift_idx = split_sec // 5
audio_ids = train_soundscape["audio_id"].unique()
all_list = []
# new_df = pd.DataFrame([], columns=train_soundscape.columns)
for audio_id in audio_ids:
    tmp_df = train_soundscape[train_soundscape["audio_id"] == audio_id]
    for start in range(len(tmp_df)):
        tmp_df = tmp_df.reset_index(drop=True)
        end = int(start + shift_idx - 1)
        if end > len(tmp_df) - 1:
            continue
        items = {}
        items["site"] = tmp_df.loc[start, "site"]
        items["audio_id"] = str(audio_id)
        items["seconds"] = str(tmp_df.loc[end, "seconds"])
        items["row_id"] = (
            items["audio_id"] + "_" + items["site"] + "_" + items["seconds"]
        )
        birds = ""
        for bird in tmp_df.loc[start:end, "birds"].values:
            birds += f"{bird} "
        birds = set(birds.split(" "))
        birds.remove("")
        if (len(birds) > 1) and ("nocall" in birds):
            birds.remove("nocall")
        items["birds"] = " ".join(list(birds))
        all_list.append(items)
        # print(len(birds), birds)
new_df = pd.DataFrame.from_dict(all_list)[train_soundscape.columns]


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
new_df["path"] = output_dir + "/" + new_df["row_id"] + ".ogg"
new_df.to_csv(output_dir + ".csv", index=False)
ogg_list = glob.glob(os.path.join(input_dir, "train_soundscapes", "*.ogg"))
for fname in tqdm(ogg_list):
    y, sr = sf.read(fname)
    audio_id = str(fname.split("/")[-1].split("_")[0])
    tmp_df = new_df[new_df["audio_id"] == audio_id]
    for i in range(len(tmp_df)):
        tmp_df = tmp_df.reset_index(drop=True)
        end = int(tmp_df.loc[i, "seconds"]) * sr
        start = (int(tmp_df.loc[i, "seconds"]) - split_sec) * sr
        sf.write(new_df.loc[i, "path"], y[start:end], sr)
print("Successfully split ogg files.")
