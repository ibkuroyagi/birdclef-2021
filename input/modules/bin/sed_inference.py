#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

"""Train Sound Event Detection model."""

import argparse
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

sys.path.append("../../")
sys.path.append("../input/modules")
import models  # noqa: E402
from datasets import RainForestDataset  # noqa: E402
from trainers import SEDTrainer  # noqa: E402
from utils import write_hdf5  # noqa: E402
from utils import lwlrap  # noqa: E402

sys.path.append("../input/iterative-stratification-master")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: E402

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def plot_distribution(ground_truth, pred_df, save_path="dist.png", mode="oof"):
    n_col = 3
    n_raw = 24 // n_col
    plt.figure(figsize=(16, 24))
    for i in range(24):
        plt.subplot(n_raw, n_col, i + 1)
        if mode == "oof":
            label_idx = ground_truth[f"s{i}"] == 1
            weight = np.ones(int(label_idx.sum())) / label_idx.sum()
            plt.hist(
                pred_df.loc[label_idx, f"s{i}"].values,
                alpha=0.5,
                label="pos",
                bins=50,
                weights=weight,
            )
            weight = np.ones(int((~label_idx).sum())) / (~label_idx).sum()
            plt.hist(
                pred_df.loc[~label_idx, f"s{i}"].values,
                alpha=0.5,
                label="neg",
                bins=50,
                weights=weight,
            )
            logloss = log_loss(
                ground_truth.loc[:, f"s{i}"].values, pred_df.loc[:, f"s{i}"].values
            )
            plt.title(
                f"s{i}, BCE:{logloss:.4f} ratio:{label_idx.sum()/len(label_idx):.4f}, count:{label_idx.sum()}"
            )
        else:
            prob = pred_df.loc[:, f"s{i}"].values
            weight = np.ones(len(prob)) / len(prob)
            plt.hist(
                prob,
                alpha=0.5,
                label="test",
                bins=50,
                weights=weight,
            )
            plt.title(f"TEST: s{i} ratio:{sum(prob>=0.5)/len(prob):.4f}")
        plt.legend()
        plt.xlim([0, 1])
        plt.xlabel("Probability")
        plt.ylim([0, 1])
        plt.ylabel("count")
        plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outliter exposure model (See detail in parallel_wavegan/bin/train.py)."
    )
    parser.add_argument(
        "--datadir",
        default=None,
        type=str,
        help="root data directory.",
    )
    parser.add_argument(
        "--dumpdirs",
        default=[],
        type=str,
        nargs="+",
        help="root dump directory.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument("--seed", type=int, default=1, help="seed.")
    parser.add_argument(
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="List of checkpoint file path to resume training. (default=[])",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()
    # check distributed training
    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.rank)
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["n_TTA"] = len(args.dumpdirs)
    config.update(vars(args))
    config["n_target"] = 24
    config["trained_model_fold"] = []
    for i, checkpoint in enumerate(args.checkpoints):
        if checkpoint != "no_model":
            config["trained_model_fold"].append(i)
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    train_tp = pd.read_csv(os.path.join(args.datadir, "train_tp.csv"))
    # get dataset
    tp_list = train_tp["recording_id"].unique()
    columns = ["recording_id"] + [f"s{i}" for i in range(config["n_target"])]
    ground_truth = pd.DataFrame(
        np.zeros((len(tp_list), config["n_target"] + 1)), columns=columns
    )
    ground_truth["recording_id"] = tp_list
    for i, recording_id in enumerate(train_tp["recording_id"].values):
        ground_truth.iloc[
            ground_truth["recording_id"] == recording_id,
            train_tp.loc[i, "species_id"] + 1,
        ] = 1.0
    ground_truth_path = os.path.join(args.datadir, "ground_truth.csv")
    if not os.path.isfile(ground_truth_path):
        ground_truth.to_csv(ground_truth_path, index=False)
    kfold = MultilabelStratifiedKFold(
        n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]
    )
    y = ground_truth.iloc[:, 1:].values
    # Initialize out of fold
    oof_clip = np.zeros((len(ground_truth), config["n_eval_split"], config["n_target"]))
    oof_frame = np.zeros(
        (
            len(ground_truth),
            config["n_eval_split"],
            config["l_target"],
            config["n_target"],
        )
    )
    scores = []
    # Initialize each fold prediction.
    sub = pd.read_csv(os.path.join(args.datadir, "sample_submission.csv"))
    pred_clip = np.zeros(
        (
            len(config["trained_model_fold"]),
            len(sub),
            config["n_eval_split"],
            config["n_target"],
        )
    )
    pred_frame = np.zeros(
        (
            len(config["trained_model_fold"]),
            len(sub),
            config["n_eval_split"],
            config["l_target"],
            config["n_target"],
        )
    )
    eval_key = (
        ["wave"] if config["model_params"].get("require_prep", False) else ["feats"]
    )
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(y, y)):
        logging.info(f"Start training fold {fold}.")
        # define models and optimizers
        if fold not in config["trained_model_fold"]:
            logging.info(f"Skip fold {fold}. Due to not found trained model.")
            continue
        model_class = getattr(
            models,
            # keep compatibility
            config.get("model_type", "Cnn14_DecisionLevelAtt"),
        )
        model = model_class(training=False, **config["model_params"]).to(device)
        if config["model_type"] in ["ResNext50", "Cnn14_DecisionLevelAtt"]:
            from models import AttBlock

            model.bn0 = nn.BatchNorm2d(config["num_mels"])
            model.att_block = AttBlock(**config["att_block"])
            if config["model_params"].get("use_dializer", False):
                model.dialize_layer = nn.Linear(config["n_class"], 1, bias=True)
            if config["model_params"].get("require_prep", False):
                from torchlibrosa.stft import LogmelFilterBank
                from torchlibrosa.stft import Spectrogram

                model.spectrogram_extractor = Spectrogram(
                    n_fft=config["fft_size"],
                    hop_length=config["hop_size"],
                    win_length=config["fft_size"],
                    window=config["window"],
                    center=True,
                    pad_mode="reflect",
                )
                model.logmel_extractor = LogmelFilterBank(
                    sr=config["sr"],
                    n_fft=config["fft_size"],
                    n_mels=config["num_mels"],
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                    ref=1.0,
                    amin=1e-6,
                    top_db=None,
                    freeze_parameters=True,
                )
            logging.info("Successfully initialize custom weight.")

        if fold == 0:
            logging.info(model)
        # train_y = ground_truth.iloc[train_idx]
        valid_y = ground_truth.iloc[valid_idx]
        train_tp["use_train"] = train_tp["recording_id"].map(
            lambda x: x not in valid_y["recording_id"].values
        )
        # get data loader
        if config["model_params"].get("require_prep", False):
            from datasets import WaveEvalCollater

            dev_collater = WaveEvalCollater(
                sr=config.get("sr", 48000),
                sec=config.get("sec", 10.0),
                n_split=config.get("n_eval_split", 6),
                is_label=True,
            )
        else:
            from datasets import FeatEvalCollater

            dev_collater = FeatEvalCollater(
                max_frames=config.get("max_frames", 512),
                n_split=config.get("n_eval_split", 20),
                is_label=True,
                use_song_type=config.get("use_song_type", False),
            )
        tta_oof_clip = np.zeros(
            (
                config["n_TTA"],
                len(valid_idx),
                config["n_eval_split"],
                config["n_target"],
            )
        )
        tta_oof_frame = np.zeros(
            (
                config["n_TTA"],
                len(valid_idx),
                config["n_eval_split"],
                config["l_target"],
                config["n_target"],
            )
        )
        tta_scores = np.zeros(config["n_TTA"])
        # Initialize each fold prediction.
        tta_pred_clip = np.zeros(
            (
                config["n_TTA"],
                len(sub),
                config["n_eval_split"],
                config["n_target"],
            )
        )
        tta_pred_frame = np.zeros(
            (
                config["n_TTA"],
                len(sub),
                config["n_eval_split"],
                config["l_target"],
                config["n_target"],
            )
        )
        for i, dumpdir in enumerate(args.dumpdirs):
            valid_dataset = RainForestDataset(
                files=[
                    os.path.join(dumpdir, "train", f"{recording_id}.h5")
                    for recording_id in tp_list[valid_idx]
                ],
                keys=eval_key,
                train_tp=train_tp[~train_tp["use_train"]],
                mode="valid",
                is_normalize=config.get("is_normalize", False),
                allow_cache=False,
                seed=None,
            )
            logging.info(f"The number of validation files = {len(valid_dataset)}.")

            data_loader = {
                "eval": DataLoader(
                    valid_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    collate_fn=dev_collater,
                    num_workers=config["num_workers"],
                    pin_memory=config["pin_memory"],
                ),
            }
            # define valid trainer
            trainer = SEDTrainer(
                steps=0,
                epochs=0,
                data_loader=data_loader,
                model=model.to(device),
                criterion=None,
                optimizer=None,
                scheduler=None,
                config=config,
                device=device,
                train=False,
                use_center_loss=config.get("use_center_loss", False),
                use_dializer=config.get("use_dializer", False),
                save_name=f"fold{fold}",
            )
            trainer.load_checkpoint(args.checkpoints[fold], load_only_params=False)
            logging.info(
                f"Successfully resumed from {args.checkpoints[fold]}.(Epochs:{trainer.epochs}, Steps:{trainer.steps})"
            )
            # inference validation data
            oof_dict = trainer.inference(mode="valid")
            tta_oof_clip[i] = oof_dict["y_clip"][:, :, : config["n_target"]]
            tta_oof_frame[i] = oof_dict["y_frame"]
            tta_scores[i] = oof_dict["score"]
            logging.info(f"Fold:{fold},TTA:{i} lwlrap:{tta_scores[i]:.6f}")
            # initialize test data
            test_dataset = RainForestDataset(
                root_dirs=[os.path.join(dumpdir, "test")],
                keys=eval_key,
                mode="test",
                is_normalize=config.get("is_normalize", False),
                allow_cache=False,
                seed=None,
            )
            logging.info(f"The number of test files = {len(test_dataset)}.")
            if config["model_params"].get("require_prep", False):
                eval_collater = WaveEvalCollater(
                    sr=config.get("sr", 48000),
                    sec=config.get("sec", 10.0),
                    n_split=config.get("n_eval_split", 6),
                    is_label=False,
                )
            else:
                eval_collater = FeatEvalCollater(
                    max_frames=config.get("max_frames", 512),
                    n_split=config.get("n_eval_split", 20),
                    is_label=False,
                )
            data_loader = {
                "eval": DataLoader(
                    test_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    collate_fn=eval_collater,
                    num_workers=config["num_workers"],
                    pin_memory=config["pin_memory"],
                ),
            }
            # define valid trainer
            trainer = SEDTrainer(
                steps=0,
                epochs=0,
                data_loader=data_loader,
                model=model.to(device),
                criterion=None,
                optimizer=None,
                scheduler=None,
                config=config,
                device=device,
                train=False,
                use_center_loss=config.get("use_center_loss", False),
                use_dializer=config.get("use_dializer", False),
                save_name=f"fold{fold}",
            )
            trainer.load_checkpoint(args.checkpoints[fold], load_only_params=False)
            logging.info(
                f"Successfully resumed from {args.checkpoints[fold]}.(Epochs:{trainer.epochs}, Steps:{trainer.steps})"
            )
            # inference test data
            pred_dict = trainer.inference(mode="test")
            tta_pred_clip[i] = pred_dict["y_clip"][:, :, : config["n_target"]]
            tta_pred_frame[i] = pred_dict["y_frame"]

            logging.info(f"Fold:{fold},TTA:{i} Successfully inference test data.")
        oof_clip[valid_idx] = tta_oof_clip.mean(axis=0)
        oof_frame[valid_idx] = tta_oof_frame.mean(axis=0)
        scores.append(tta_scores.mean())
        logging.info(f"Fold:{fold}, lwlrap:{scores[-1]:.6f}")
        pred_clip[fold] = tta_pred_clip.mean(axis=0)
        pred_frame[fold] = tta_pred_frame.mean(axis=0)

    # save inference results
    write_hdf5(
        os.path.join(args.outdir, "oof.h5"),
        "y_clip",
        oof_clip.astype(np.float32),
    )
    write_hdf5(
        os.path.join(args.outdir, "oof.h5"),
        "y_frame",
        oof_frame.astype(np.float32),
    )
    logging.info(f"Successfully saved oof at {os.path.join(args.outdir, 'oof.h5')}.")
    pred_clip_mean = pred_clip.mean(axis=0)
    pred_frame_mean = pred_frame.mean(axis=0)
    write_hdf5(
        os.path.join(args.outdir, "pred.h5"),
        "y_clip",
        pred_clip_mean.astype(np.float32),
    )
    write_hdf5(
        os.path.join(args.outdir, "pred.h5"),
        "y_frame",
        pred_frame_mean.astype(np.float32),
    )
    logging.info(f"Successfully saved pred at {os.path.join(args.outdir, 'pred.h5')}.")

    # modify submission shape
    if not os.path.exists(os.path.join(args.outdir, "clip")):
        os.makedirs(os.path.join(args.outdir, "clip"))
    if not os.path.exists(os.path.join(args.outdir, "frame")):
        os.makedirs(os.path.join(args.outdir, "frame"))
    oof_sub = ground_truth.copy()
    oof_sub.iloc[:, 1:] = oof_clip.max(axis=1)
    clip_oof_path = os.path.join(args.outdir, "clip", "oof.csv")
    oof_sub.to_csv(clip_oof_path, index=False)
    logging.info(f"Successfully saved oof at {clip_oof_path}.")
    oof_score = lwlrap(ground_truth.iloc[:, 1:].values, oof_sub.iloc[:, 1:].values)
    for i, score in enumerate(scores):
        logging.info(f"Fold:{i} oof score is {score:.6f}")
    logging.info(f"Average oof score is {np.array(scores).mean():.6f}")
    logging.info(f"All clip oof score is {oof_score:.6f}")
    plot_distribution(
        ground_truth,
        oof_sub,
        save_path=os.path.join(args.outdir, "clip", "oof_dist.png"),
    )
    oof_sub.iloc[:, 1:] = oof_frame.max(axis=1).max(axis=1)
    frame_oof_path = os.path.join(args.outdir, "frame", "oof.csv")
    oof_sub.to_csv(frame_oof_path, index=False)
    logging.info(f"Successfully saved oof at {frame_oof_path}.")
    oof_score = lwlrap(ground_truth.iloc[:, 1:].values, oof_sub.iloc[:, 1:].values)
    logging.info(f"All frame oof score is {oof_score:.6f}")
    plot_distribution(
        ground_truth,
        oof_sub,
        save_path=os.path.join(args.outdir, "frame", "oof_dist.png"),
    )

    # test inference
    sub.iloc[:, 1:] = pred_clip_mean.max(axis=1)
    clip_sub_path = os.path.join(args.outdir, "clip", "submission.csv")
    sub.to_csv(clip_sub_path, index=False)
    logging.info(f"Successfully saved clip submission at {clip_sub_path}.")
    plot_distribution(
        ground_truth,
        sub,
        save_path=os.path.join(args.outdir, "clip", "dist.png"),
        mode="test",
    )
    sub.iloc[:, 1:] = pred_frame_mean.max(axis=1).max(axis=1)
    frame_sub_path = os.path.join(args.outdir, "frame", "submission.csv")
    sub.to_csv(frame_sub_path, index=False)
    logging.info(f"Successfully saved frame submission at {frame_sub_path}.")
    plot_distribution(
        ground_truth,
        sub,
        save_path=os.path.join(args.outdir, "frame", "dist.png"),
        mode="test",
    )


if __name__ == "__main__":
    main()
