# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

import logging
import random
import sys
import h5py
import numpy as np
from multiprocessing import Manager
from torch.utils.data import Dataset

sys.path.append("../../")
sys.path.append("../input/modules")
import datasets  # noqa: E402
from utils import find_files  # noqa: E402
from utils import logmelfilterbank  # noqa: E402


class RainForestDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dirs=[],
        files=None,
        train_tp=None,
        train_fp=None,
        keys=["feats"],
        mode="tp",
        is_normalize=False,
        allow_cache=False,
        use_on_the_fly=False,
        config={},
        seed=None,
    ):
        """Initialize dataset.

        Args:
            root_dirs (list): List of root directories for dumped files.
            train_tp (DataFrame): train_tp (default: None)
            train_fp (DataFrame): train_fp (default: None)
            keys: (list): List of key of dataset.
            mode (list): Mode of dataset. [tp, all, test]
            allow_cache (bool): Whether to allow cache of the loaded files.
            use_on_the_fly (bool): Whether to use on the fly proprocess(don't use collater_fc).
            config (dict): Setting dict for requir_prep=True.
            seed (int): seed
        """
        # if seed is not None:
        #     self.seed = seed
        #     np.random.seed(seed)
        # find all of the mel files
        if (files is None) and (len(root_dirs) != 0):
            files = []
            for root_dir in root_dirs:
                files += sorted(find_files(root_dir, "*.h5"))
        use_file_keys = []
        use_file_list = []
        use_time_list = []
        if mode == "tp":
            tp_list = train_tp["recording_id"].unique()
            for file in files:
                recording_id = file.split("/")[-1].split(".")[0]
                if recording_id in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
                    # logging.debug(f"{facter}: {file}")
                    use_time_list.append(
                        train_tp[train_tp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max", "species_id", "songtype_id"]]
                        .values
                    )
        elif mode == "all":
            tp_list = train_tp["recording_id"].unique()
            fp_list = train_fp["recording_id"].unique()
            for file in files:
                recording_id = file.split("/")[-1].split(".")[0]
                if recording_id in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
                    # logging.debug(f"{facter}: {file}")
                    use_time_list.append(
                        train_tp[train_tp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max", "species_id", "songtype_id"]]
                        .values
                    )
                if recording_id in fp_list:
                    use_file_keys.append(keys + ["matrix_fp"])
                    use_file_list.append(file)
                    use_time_list.append(
                        train_fp[train_fp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max", "species_id", "songtype_id"]]
                        .values
                    )
        elif mode == "valid":
            tp_list = train_tp["recording_id"].unique()
            for file in files:
                recording_id = file.split("/")[-1].split(".")[0]
                if recording_id in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
                    use_time_list.append(
                        train_tp[train_tp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max", "species_id", "songtype_id"]]
                        .values
                    )
        elif mode == "test":
            for file in files:
                use_file_keys.append(keys)
                use_file_list.append(file)
        self.use_file_keys = use_file_keys
        self.use_file_list = use_file_list
        self.use_time_list = use_time_list
        self.keys = keys
        self.mode = mode
        self.allow_cache = allow_cache
        self.use_on_the_fly = use_on_the_fly
        self.config = config
        self.transform = None
        if ("wave" in use_file_keys) and (
            config.get("augmentation_params", None) is not None
        ):
            compose_list = []
            for key in self.config["augmentation_params"].keys():
                aug_class = getattr(
                    datasets,
                    key,
                )
                compose_list.append(
                    aug_class(**self.config["augmentation_params"][key])
                )
                logging.debug(f"{key}")
            self.transform = datasets.Compose(compose_list)
        # NOTE(ibuki): Manager is need to share memory in dataloader with num_workers > 0
        self.manager = Manager()
        self.caches = self.manager.list()
        self.caches += [() for _ in range(len(use_file_list))]
        self.wave_caches = self.manager.list()
        self.wave_caches += [() for _ in range(len(use_file_list))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            items: Dict
                wave: (ndarray) Wave (T, ).
                feats: (ndarray) Feature (T', C).
                matrix_tp: (ndrray) Matrix of ground truth.
                time_list: (ndrray) (n_recoding_id, t_max, t_min).
        """
        if (
            self.allow_cache
            and (len(self.caches[idx]) != 0)
            and ("wave" not in self.keys)
        ):
            return self.caches[idx]

        items = {}
        if len(self.wave_caches[idx]) == 0:
            # feats is dumped files
            hdf5_file = h5py.File(self.use_file_list[idx], "r")
            for key in self.use_file_keys[idx]:
                items[key] = hdf5_file[key][()]
                if key == "wave":
                    original_wave = items[key]
            hdf5_file.close()
        else:
            items = self.wave_caches[idx]
        if self.use_on_the_fly:
            # Make specrorgram on Dataset.
            return self._on_the_fly(items["wave"], self.use_time_list[idx], split=8)
        if self.transform is not None:
            # Make specrorgram on Model.
            items["wave"] = self.transform(items["wave"])
        if (self.mode == "all") or (self.mode == "tp") or (self.mode == "valid"):
            items["time_list"] = self.use_time_list[idx]
        if self.allow_cache:
            if ("wave" in self.keys) and (len(self.wave_caches[idx]) == 0):
                self.wave_caches[idx] = items
                self.wave_caches[idx]["wave"] = original_wave
            else:
                self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.
        """
        return len(self.use_file_list)

    def wave2spec(self, wave: np.ndarray):
        """Transfom wave to log mel sprctrogram and apply augmentation.

        Args:
            wave (ndarray): Original wave form (T,)
        Returns:
            feats (ndarray): Augmented log mel spectrogram(T', mel).
        """
        if self.transform is not None:
            wave = self.transform(wave)
        feats = logmelfilterbank(
            wave,
            sampling_rate=self.config["sr"],
            hop_size=self.config["hop_size"],
            fft_size=self.config["fft_size"],
            window=self.config["window"],
            num_mels=self.config["num_mels"],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
        )
        return feats

    def _on_the_fly(self, wave: np.ndarray, time_list: np.ndarray, split=8):
        """Return mel-spectrogram and clip-level label.

        Args:
            wave (np.ndarray): wave form data(y,)
            time_list (np.ndarray): The number of time to call(n_sample, 2)
            split (int): split ratio.
        Returns:
            item: (dict):
                feats: (ndarray) Feature (mel, T').
                y_clip: (ndarray) Clip level targte(T'', n_class).
                y_frame: (ndarray) Frame level target(n_class,).
        """
        l_wave = len(wave)
        wave_frames = int(self.config["sec"] * self.config["sr"])
        idx = random.randrange(len(time_list))
        time_start = int(l_wave * time_list[idx][0] / 60)
        time_end = int(l_wave * time_list[idx][1] / 60)
        center = np.round((time_start + time_end) / 2)
        quarter = ((time_end - time_start) // split) * (split // 2 - 1)
        beginning = center - wave_frames - quarter
        if beginning < 0:
            beginning = 0
        beginning = random.randrange(beginning, center + quarter)
        ending = beginning + wave_frames
        if ending > l_wave:
            ending = l_wave
        beginning = ending - wave_frames
        feat = self.wave2spec(wave[beginning:ending])
        t_begging = beginning / l_wave * 60
        t_ending = ending / l_wave * 60
        y_clip = np.zeros(self.config["n_class"])
        y_frame = np.zeros((self.config["l_target"], self.config["n_class"]))
        for i in range(len(time_list)):
            if time_list[i][0] - self.config["sec"] <= t_begging <= time_list[i][1]:
                y_clip[int(time_list[i][2])] = 1.0
                checker = np.linspace(t_begging, t_ending, self.config["l_target"])
                call_idx = (checker > time_list[i][0]) & (checker < time_list[i][1])
                y_frame[call_idx, int(time_list[i][2])] = 1.0
        items = {}
        items["X"] = feat.T.astype(np.float32)
        items["y_clip"] = y_clip.astype(np.float32)
        items["y_frame"] = y_frame.astype(np.float32)
        if self.config.get("use_dializer", False):
            items["frame_mask"] = y_frame.any(axis=1).reshape(-1, 1).astype(np.float32)
        return items
