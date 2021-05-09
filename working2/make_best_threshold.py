import os
import numpy as np
from tqdm import tqdm

input_dir = "exp/arai_infer_tf_efficientnet_b0_ns/arai_train_tf_efficientnet_b0_ns_mgpu_mixup_new/bce_"
p_y = np.load(os.path.join(input_dir, "pred_y_frame.npy"))
t_y = np.load(os.path.join(input_dir, "train_y.npy"))


def lb_f1(y_pred, y_true):
    epsilon = 1e-7
    tp = (y_true * y_pred).sum(1)
    tn = ((1 - y_true) * (1 - y_pred)).sum(1)
    fp = ((1 - y_true) * y_pred).sum(1)
    fn = (y_true * (1 - y_pred)).sum(1)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean()


print("beforemagic", lb_f1((p_y > 0.1).astype(int), t_y))


best_thresholds = np.zeros(397) + 0.1
for i in tqdm(range(397)):
    print(i)
    best_score = 0
    th = best_thresholds.copy()
    for t in np.linspace(0.01, 0.49, 49):
        th[i] *= 0
        th[i] += t
        score = lb_f1((p_y > th), t_y)
        if score > best_score:
            best_score = score
            best_t = t
    best_thresholds[i] *= 0
    best_thresholds[i] += best_t
print("aftermagic", lb_f1((p_y > best_thresholds).astype(int), t_y))
save_path = os.path.join(input_dir, "best_thresholds.npy")
np.save(save_path, best_thresholds)
print(f"Saved magiced file at {save_path}.")
