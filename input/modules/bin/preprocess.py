#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanai
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""
import argparse
import logging
import os
import pickle
import sys
import librosa
import numpy as np
import pandas as pd
import yaml
import gc
from tqdm import tqdm

sys.path.append("../../")
sys.path.append("../input/modules")
from utils import write_hdf5  # noqa: E402
from utils import logmelfilterbank  # noqa: E402


def make_utt_matrix(train_df, recording_id: str, l_spec=5626, n_class=24):
    """Make ground truth matrix.

    Args:
        train_df (DataFrame): train_tp or train_fp
        recording_id (str): recording_id
        l_spec (int, optional): Length of mel-spectrogram. Defaults to 5626.
        n_class (int, optional): The number of class. Defaults to 24.

    Returns:
        matrix (ndarray): Ground truth matrix. (l_spec, n_class)
    """
    matrix = np.zeros((l_spec, n_class))
    tmp = train_df[train_df["recording_id"] == recording_id].reset_index(drop=True)
    for i in range(len(tmp)):
        t_start = int(l_spec * (tmp.loc[i, "t_min"] / 60.0))
        t_end = int(l_spec * (tmp.loc[i, "t_max"] / 60.0))
        matrix[t_start:t_end, tmp.loc[i, "species_id"]] = 1.0
    return matrix


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py)."
    )
    parser.add_argument(
        "--datadir",
        required=True,
        type=str,
        help="directory including flac files.",
    )
    parser.add_argument(
        "--dumpdir", type=str, required=True, help="directory to dump feature files."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--statistic_path", type=str, default="", help="wave statistic in pkl file."
    )
    parser.add_argument(
        "--cal_type", type=int, default=1, help="whether calculate statistics."
    )
    parser.add_argument(
        "--facter", type=float, default=1.0, help="The facter of speed."
    )
    parser.add_argument("--type", type=str, default="wave", help="Type of preprocess.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    sr = config["sr"]
    train_dir = os.path.join(args.datadir, "train")
    train_path_list = [
        os.path.join(train_dir, fname) for fname in os.listdir(train_dir)
    ]
    test_dir = os.path.join(args.datadir, "test")
    test_path_list = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)]
    all_path_list = train_path_list + test_path_list
    train_tp = pd.read_csv(os.path.join(args.datadir, "train_tp.csv"))
    tp_list = train_tp["recording_id"].unique()
    train_fp = pd.read_csv(os.path.join(args.datadir, "train_fp.csv"))
    fp_list = train_fp["recording_id"].unique()
    # get dataset
    if (args.datadir is not None) and args.cal_type == 1:
        tmp = np.zeros((len(all_path_list), 2880000))
        for i, path in enumerate(tqdm(all_path_list)):
            tmp[i], _ = librosa.load(path, sr=sr)
        statistic = {}
        statistic["mean"] = tmp.mean()
        statistic["std"] = tmp.std()
        with open(args.statistic_path, "wb") as f:
            pickle.dump(statistic, f)
        logging.info(f"Successfully saved statistic to {args.statistic_path}.")
        del tmp
        gc.collect()
    else:
        with open(args.statistic_path, "rb") as f:
            statistic = pickle.load(f)
        logging.info(f"Successfully loaded statistic from {args.statistic_path}.")
    logging.info(
        f"Statistic mean: {statistic['mean']:.4f}, std: {statistic['std']:.4f}"
    )
    # process each data
    modes = ["train", "test"]
    for i, path_list in enumerate([train_path_list, test_path_list]):
        # check directly existence
        outdir = os.path.join(args.dumpdir, args.type, modes[i])
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        for path in tqdm(path_list):
            x, _ = librosa.load(path=path, sr=sr)
            x = (x - statistic["mean"]) / statistic["std"]
            if args.facter != 1.0:
                x = librosa.effects.time_stretch(x, args.facter)
            # extract feature
            mel = logmelfilterbank(
                x,
                sampling_rate=sr,
                hop_size=config["hop_size"],
                fft_size=config["fft_size"],
                window=config["window"],
                num_mels=config["num_mels"],
                fmin=config["fmin"],
                fmax=config["fmax"],
            )
            wave_id = path.split("/")[-1][:-5]
            # save
            if (wave_id in tp_list) and (i == 0):
                matrix_tp = make_utt_matrix(
                    train_tp, wave_id, l_spec=len(mel), n_class=config["n_class"]
                )
                write_hdf5(
                    os.path.join(outdir, f"{wave_id}.h5"),
                    "matrix_tp",
                    matrix_tp.astype(np.int64),
                )
            if (wave_id in fp_list) and (i == 0):
                matrix_fp = make_utt_matrix(
                    train_fp, wave_id, l_spec=len(mel), n_class=config["n_class"]
                )
                write_hdf5(
                    os.path.join(outdir, f"{wave_id}.h5"),
                    "matrix_fp",
                    matrix_fp.astype(np.int64),
                )
            write_hdf5(
                os.path.join(outdir, f"{wave_id}.h5"),
                "wave",
                x.astype(np.float32),
            )
            write_hdf5(
                os.path.join(outdir, f"{wave_id}.h5"),
                "feats",
                mel.astype(np.float32),
            )


if __name__ == "__main__":
    main()
