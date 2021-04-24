import logging
import random
import numpy as np
from torch.utils.data.sampler import BatchSampler


class MultiLabelBalancedBatchSampler(BatchSampler):
    """BatchSampler - s0:s1:..:s23 = 1:1:..:1

    Returns batches of size batch_size
    """

    def __init__(self, dataset, batch_size=64, shuffle=False, n_class=24):
        self.n_class = n_class
        self.label_to_indices = {label: [] for label in range(n_class)}
        for i, use_time_df in enumerate(dataset.use_time_list):
            for j in range(len(use_time_df)):
                self.label_to_indices[use_time_df[j, 2]].append(i)
        self.used_label_indices_count = {label: 0 for label in range(n_class)}
        self.count = 0
        self.batch_size = batch_size
        self.n_samples = max(batch_size // n_class, 1)
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        self.count = 0
        label_to_indices = {}
        if self.shuffle:
            for key, value in self.label_to_indices.items():
                label_to_indices[key] = random.sample(value, len(value))
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(
                np.arange(self.n_class), self.n_class, replace=False
            )
            indices = []
            for class_ in classes:
                indices.extend(
                    label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    label_to_indices[class_]
                ):
                    label_to_indices[class_] = random.sample(
                        label_to_indices[class_], len(label_to_indices[class_])
                    )
                    self.used_label_indices_count[class_] = 0
                if self.batch_size <= len(indices):
                    break
            if self.shuffle:
                random.shuffle(indices)
            logging.debug(f"indices:{indices}")
            yield indices
            self.count += self.n_class * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
