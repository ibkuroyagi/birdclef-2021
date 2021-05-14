# %%
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../input/modules")
from utils import target_columns  # noqa: E402

input_dir = "exp/infer_b0_relabel/mixup2/bce"
p_y = np.load(os.path.join(input_dir, "pred_y_frame.npy"))
# t_y = np.load(
#     "exp/arai_infer_tf_efficientnet_b0_ns/arai_train_tf_efficientnet_b0_ns_mgpu_mixup_new/bce_/train_y.npy"
# )
target_columns.append("nocall")
df = pd.read_csv("exp/infer_b0_relabel/mixup2/bce/train_y.csv")
soundscape_idx = df["dataset"] == "train_soundscape"
t_y = np.zeros((len(df), len(target_columns)))
for i in range(len(df)):
    for bird in df.loc[i, "birds"].split(" "):
        t_y[i, np.array(target_columns) == bird] = 1.0
t_y = t_y[soundscape_idx]
p_y = p_y[soundscape_idx]


def lb_f1(y_pred, y_true):
    epsilon = 1e-7
    tp = (y_true * y_pred).sum(1)
    fp = ((1 - y_true) * y_pred).sum(1)
    fn = (y_true * (1 - y_pred)).sum(1)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean()


bin_pred = np.zeros_like(t_y)
bin_pred[:, :-1] = p_y > 0.1
bin_pred[:, -1] = ~(p_y > 0.1).any(axis=1)
print("beforemagic", lb_f1(bin_pred.astype(int), t_y), flush=True)

best_thresholds = np.zeros(397) + 0.1
for i in tqdm(range(397)):
    best_score = 0
    th = best_thresholds.copy()
    for t in np.linspace(0.01, 0.90, 90):
        th[i] *= 0
        th[i] += t
        bin_pred = np.zeros_like(t_y)
        bin_pred[:, :-1] = p_y > th
        bin_pred[:, -1] = ~(p_y > th).any(axis=1)
        score = lb_f1(bin_pred, t_y)
        if score > best_score:
            best_score = score
            print(f"i:{i},t:{t:.2f},best score:{best_score:.4f}")
            best_t = t
    best_thresholds[i] *= 0
    best_thresholds[i] += best_t
    # save_path = os.path.join(input_dir, "best_thresholds.npy")
    # np.save(save_path, best_thresholds)
save_path = os.path.join(input_dir, "best_thresholds.npy")
np.save(save_path, best_thresholds)
bin_pred = np.zeros_like(t_y)
bin_pred[:, :-1] = p_y > best_thresholds
bin_pred[:, -1] = ~(p_y > best_thresholds).any(axis=1)
print("aftermagic", lb_f1(bin_pred.astype(int), t_y))
print(f"Saved magiced file at {save_path}.")

# %%
