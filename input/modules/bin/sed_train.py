#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

"""Train Sound Event Detection model."""

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader

sys.path.append("../../")
sys.path.append("../input/modules")
import losses  # noqa: E402
import models  # noqa: E402
import optimizers  # noqa: E402
from datasets import RainForestDataset  # noqa: E402

sys.path.append("../input/iterative-stratification-master")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: E402

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


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
        "--cache_path",
        type=str,
        default="",
        help="Paht of official pretrained model's weight.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument("--seed", type=int, default=1, help="seed.")
    parser.add_argument(
        "--resume",
        default=[],
        type=str,
        nargs="*",
        help="checkpoint file path to resume training. (default=[])",
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

    # check distributed training
    if args.distributed:
        logging.info(f"device:{device}")
        logging.info(f"args.rank:{args.rank}")
        logging.info(f"os.environ:{os.environ}")
        logging.info(f"args.world_size:{args.world_size}")
        logging.info(f"args.distributed:{args.distributed}")
    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    if config["model_params"].get("require_prep", False):
        train_keys = ["wave"]
        eval_keys = ["wave"]
    elif config.get("use_on_the_fly", False):
        train_keys = ["wave"]
        eval_keys = ["feats"]
    else:
        train_keys = ["feats"]
        eval_keys = ["feats"]
    logging.info(f"train key:{train_keys}, eval key: {eval_keys}")
    train_tp = pd.read_csv(os.path.join(args.datadir, "train_tp.csv"))
    if config.get("train_dataset_mode", "tp") == "all":
        train_fp = pd.read_csv(os.path.join(args.datadir, "train_fp.csv"))
    else:
        train_fp = None
    # get dataset
    tp_list = train_tp["recording_id"].unique()
    columns = ["recording_id"] + [f"s{i}" for i in range(24)]
    ground_truth = pd.DataFrame(np.zeros((len(tp_list), 25)), columns=columns)
    ground_truth["recording_id"] = tp_list
    for i, recording_id in enumerate(train_tp["recording_id"].values):
        ground_truth.iloc[
            ground_truth["recording_id"] == recording_id,
            train_tp.loc[i, "species_id"] + 1,
        ] = 1.0
    kfold = MultilabelStratifiedKFold(
        n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]
    )
    y = ground_truth.iloc[:, 1:].values
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(y, y)):
        logging.info(f"Start training fold {fold}.")
        # train_y = ground_truth.iloc[train_idx]
        valid_y = ground_truth.iloc[valid_idx]
        train_tp["use_train"] = train_tp["recording_id"].map(
            lambda x: x not in valid_y["recording_id"].values
        )
        train_dataset = RainForestDataset(
            root_dirs=[os.path.join(dumpdir, "train") for dumpdir in args.dumpdirs],
            train_tp=train_tp[train_tp["use_train"]],
            train_fp=train_fp,
            keys=train_keys,
            mode=config.get("train_dataset_mode", "tp"),
            is_normalize=config.get("is_normalize", False),
            allow_cache=config.get("allow_cache", False),
            seed=None,
            config=config,
            use_on_the_fly=config.get("use_on_the_fly", False),
        )
        logging.info(f"The number of training files = {len(train_dataset)}.")
        dev_dataset = RainForestDataset(
            root_dirs=[os.path.join(args.dumpdirs[0], "train")],
            train_tp=train_tp[~train_tp["use_train"]],
            train_fp=train_fp,
            keys=train_keys,
            mode=config.get("train_dataset_mode", "tp"),
            is_normalize=config.get("is_normalize", False),
            allow_cache=not config.get("use_on_the_fly", False),
            seed=None,
            config=config,
            use_on_the_fly=config.get("use_on_the_fly", False),
        )
        logging.info(f"The number of development files = {len(dev_dataset)}.")
        eval_dataset = RainForestDataset(
            files=[
                os.path.join(args.dumpdirs[0], "train", f"{recording_id}.h5")
                for recording_id in tp_list[valid_idx]
            ],
            keys=eval_keys,
            train_tp=train_tp[~train_tp["use_train"]],
            mode="valid",
            is_normalize=config.get("is_normalize", False),
            allow_cache=True,  # keep compatibility
            seed=None,
        )
        logging.info(f"The number of evaluation files = {len(eval_dataset)}.")
        train_sampler, dev_sampler, eval_sampler = None, None, None
        if args.distributed:
            logging.info("Use multi gpu.")
            # setup sampler for distributed training
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
            )
            dev_sampler = DistributedSampler(
                dataset=dev_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
            )
            eval_sampler = DistributedSampler(
                dataset=eval_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
            )
        # get batch sampler
        if config.get("batch_sampler_type", None) == "MultiLabelBalancedBatchSampler":
            from datasets import MultiLabelBalancedBatchSampler

            if args.distributed:
                raise NotImplementedError(
                    "If you use BERTSUMDynamicBatchSampler, you can use single gpu only."
                )
            train_batch_sampler = MultiLabelBalancedBatchSampler(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                n_class=config["n_class"],
            )
        else:
            train_batch_sampler = None
        # get data loader
        if config["model_params"].get("require_prep", False):
            from datasets import WaveEvalCollater

            eval_collater = WaveEvalCollater(
                sr=config.get("sr", 48000),
                sec=config.get("sec", 10.0),
                n_split=config.get("n_eval_split", 6),
                is_label=True,
            )
            from datasets import WaveTrainCollater

            train_collater = WaveTrainCollater(
                sr=config.get("sr", 48000),
                sec=config.get("sec", 10.0),
                l_target=config.get("l_target", 32),
                mode=config.get("mode", "binary"),
                random=config.get("random", False),
                use_dializer=config.get("use_dializer", False),
            )
        else:
            from datasets import FeatEvalCollater

            eval_collater = FeatEvalCollater(
                max_frames=config.get("max_frames", 512),
                n_split=config.get("n_eval_split", 20),
                is_label=True,
                use_song_type=config.get("use_song_type", False),
            )
            from datasets import FeatTrainCollater

            if config.get("use_on_the_fly", False):
                train_collater = None
            else:
                train_collater = FeatTrainCollater(
                    max_frames=config.get("max_frames", 512),
                    l_target=config.get("l_target", 16),
                    mode=config.get("collater_mode", "sum"),
                    random=config.get("random", False),
                    use_dializer=config.get("use_dializer", False),
                    hop_size=config.get("hop_size", 512),
                    use_song_type=config.get("use_song_type", False),
                )
        data_loader = {
            "dev": DataLoader(
                dataset=dev_dataset,
                collate_fn=train_collater,
                sampler=dev_sampler,
                shuffle=False,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            ),
            "eval": DataLoader(
                eval_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=eval_sampler,
                collate_fn=eval_collater,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            ),
        }
        if config.get("batch_sampler_type", None) is not None:
            data_loader["train"] = DataLoader(
                dataset=train_dataset,
                collate_fn=train_collater,
                batch_sampler=train_batch_sampler,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            )
        else:
            data_loader["train"] = DataLoader(
                dataset=train_dataset,
                collate_fn=train_collater,
                batch_size=config["batch_size"],
                shuffle=False if args.distributed else True,
                sampler=train_sampler,
                drop_last=True,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            )
        # define models and optimizers
        model_class = getattr(
            models,
            # keep compatibility
            config.get("model_type", "Cnn14_DecisionLevelAtt"),
        )
        if config["model_type"] == "Cnn14_DecisionLevelAtt":
            model = model_class(training=True, **config["model_params"]).to(device)
            if len(args.cache_path) != 0:
                weights = torch.load(args.cache_path)
                model.load_state_dict(weights["model"])
                logging.info(f"Successfully load weight from {args.cache_path}")
            model.bn0 = nn.BatchNorm2d(config["num_mels"])
            model.att_block = models.AttBlock(**config["att_block"])
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
            conv_params = []
            fc_param = []
            for name, param in model.named_parameters():
                if name.startswith(("fc1", "att_block")):
                    fc_param.append(param)
                elif name.startswith(("conv_block")):
                    conv_params.append(param)
        elif config["model_type"] == "ResNext50":
            model = model_class(training=True, **config["model_params"]).to(device)
            model.bn0 = nn.BatchNorm2d(config["num_mels"])
            model.att_block = models.AttBlock(**config["att_block"])
            nn.init.xavier_uniform_(model.att_block.att.weight)
            nn.init.xavier_uniform_(model.att_block.cla.weight)
            logging.info("Successfully initialize custom weight.")
            conv_params = []
            fc_param = []
            for name, param in model.named_parameters():
                if name.startswith(("resnext50.fc", "fc1", "att_block")):
                    fc_param.append(param)
                else:
                    conv_params.append(param)
        else:
            model = model_class(training=True, **config["model_params"]).to(device)
        # wrap model for distributed training
        if args.distributed:
            try:
                from apex.parallel import DistributedDataParallel
            except ImportError:
                raise ImportError(
                    "Apex is not installed. Please check https://github.com/NVIDIA/apex."
                )
            # NOTE(ibkuroyagi): Needed to place the model on GPU
            model = DistributedDataParallel(model.to(device))
        loss_class = getattr(
            losses,
            # keep compatibility
            config.get("loss_type", "BCEWithLogitsLoss"),
        )
        if config["loss_params"].get("pos_weight", None) is not None:
            pos_weight = config["loss_params"]["pos_weight"]
            config["loss_params"]["pos_weight"] = torch.tensor(
                pos_weight, dtype=torch.float
            ).to(device)
        criterion = loss_class(**config["loss_params"]).to(device)
        optimizer_class = getattr(
            optimizers,
            # keep compatibility
            config.get("optimizer_type", "Adam"),
        )
        if config["model_type"] in ["Cnn14_DecisionLevelAtt", "ResNext50"]:
            optimizer = optimizer_class(
                [
                    {
                        "params": conv_params,
                        "lr": config["optimizer_params"]["conv_lr"],
                    },
                    {"params": fc_param, "lr": config["optimizer_params"]["fc_lr"]},
                ]
            )
        else:
            optimizer = optimizer_class(
                model.parameters(), **config["optimizer_params"]
            )
        scheduler = None
        if config.get("scheduler_type", None) is not None:
            scheduler_class = getattr(
                torch.optim.lr_scheduler,
                # keep compatibility
                config.get("scheduler_type", "StepLR"),
            )
            scheduler = scheduler_class(
                optimizer=optimizer, **config["scheduler_params"]
            )
        if fold == 0:
            logging.info(model)
        # define trainer
        from trainers import SEDTrainer

        trainer = SEDTrainer(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            train_sampler=train_sampler,
            model=model.to(device),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            train=fold == 0,
            use_center_loss=config.get("use_center_loss", False),
            use_dializer=config.get("use_dializer", False),
            save_name=f"fold{fold}",
        )
        # resume from checkpoint
        if len(args.resume) != 0:
            if args.resume[fold] != "no_model":
                trainer.load_checkpoint(args.resume[fold], load_only_params=False)
                logging.info(f"Successfully resumed from {args.resume[fold]}.")
        # run training loop
        try:
            trainer.run()
        except KeyboardInterrupt:
            trainer.save_checkpoint(
                os.path.join(
                    config["outdir"], f"checkpoint-{trainer.steps}stepsfold{fold}.pkl"
                )
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
        ##############################################
        #   Add no augmentation process 500 steps.   #
        ##############################################
        last_checkpoint = os.path.join(
            config["outdir"],
            f"best_score/best_scorefold{fold}.pkl",
        )
        config["train_max_steps"] += config.get("additional_steps", 1000)
        config["augmentation_params"] = None
        if config.get("mixup_alpha", None) is not None:
            config["batch_size"] //= 2
            config["mixup_alpha"] = None
        train_dataset = RainForestDataset(
            root_dirs=[os.path.join(args.dumpdirs[0], "train")],
            train_tp=train_tp[train_tp["use_train"]],
            train_fp=train_fp,
            keys=train_keys,
            mode=config.get("train_dataset_mode", "tp"),
            is_normalize=config.get("is_normalize", False),
            allow_cache=config.get("allow_cache", False),
            seed=None,
            config=config,
            use_on_the_fly=config.get("use_on_the_fly", False),
        )
        logging.info(f"The number of training files = {len(train_dataset)}.")
        dev_dataset = RainForestDataset(
            root_dirs=[os.path.join(args.dumpdirs[0], "train")],
            train_tp=train_tp[~train_tp["use_train"]],
            train_fp=train_fp,
            keys=train_keys,
            mode=config.get("train_dataset_mode", "tp"),
            is_normalize=config.get("is_normalize", False),
            allow_cache=not config.get("use_on_the_fly", False),
            seed=None,
            config=config,
            use_on_the_fly=config.get("use_on_the_fly", False),
        )
        logging.info(f"The number of development files = {len(dev_dataset)}.")
        eval_dataset = RainForestDataset(
            files=[
                os.path.join(args.dumpdirs[0], "train", f"{recording_id}.h5")
                for recording_id in tp_list[valid_idx]
            ],
            keys=eval_keys,
            train_tp=train_tp[~train_tp["use_train"]],
            mode="valid",
            is_normalize=config.get("is_normalize", False),
            allow_cache=True,  # keep compatibility
            seed=None,
        )
        logging.info(f"The number of evaluation files = {len(eval_dataset)}.")
        if args.distributed:
            # setup sampler for distributed training
            train_sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
            )
            dev_sampler = DistributedSampler(
                dataset=dev_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
            )
            eval_sampler = DistributedSampler(
                dataset=eval_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
            )
        # get batch sampler
        if config.get("batch_sampler_type", None) == "MultiLabelBalancedBatchSampler":
            train_batch_sampler = MultiLabelBalancedBatchSampler(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                n_class=config["n_class"],
            )
        else:
            train_batch_sampler = None
        # get data loader
        if config["model_params"].get("require_prep", False):
            eval_collater = WaveEvalCollater(
                sr=config.get("sr", 48000),
                sec=config.get("sec", 10.0),
                n_split=config.get("n_eval_split", 6),
                is_label=True,
            )
            train_collater = WaveTrainCollater(
                sr=config.get("sr", 48000),
                sec=config.get("sec", 10.0),
                l_target=config.get("l_target", 32),
                mode=config.get("mode", "binary"),
                random=config.get("random", False),
                use_dializer=config.get("use_dializer", False),
            )
        else:
            eval_collater = FeatEvalCollater(
                max_frames=config.get("max_frames", 512),
                n_split=config.get("n_eval_split", 20),
                is_label=True,
                use_song_type=config.get("use_song_type", False),
            )
            if config.get("use_on_the_fly", False):
                train_collater = None
            else:
                train_collater = FeatTrainCollater(
                    max_frames=config.get("max_frames", 512),
                    l_target=config.get("l_target", 16),
                    mode=config.get("collater_mode", "sum"),
                    random=config.get("random", False),
                    use_dializer=config.get("use_dializer", False),
                    hop_size=config.get("hop_size", 512),
                    use_song_type=config.get("use_song_type", False),
                )
        data_loader = {
            "dev": DataLoader(
                dataset=dev_dataset,
                collate_fn=train_collater,
                shuffle=False,
                sampler=dev_sampler,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            ),
            "eval": DataLoader(
                eval_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=eval_sampler,
                collate_fn=eval_collater,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            ),
        }
        if config.get("batch_sampler_type", None) is not None:
            data_loader["train"] = DataLoader(
                dataset=train_dataset,
                collate_fn=train_collater,
                batch_sampler=train_batch_sampler,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            )
        else:
            data_loader["train"] = DataLoader(
                dataset=train_dataset,
                collate_fn=train_collater,
                batch_size=config["batch_size"],
                sampler=train_sampler,
                shuffle=False if args.distributed else True,
                drop_last=True,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            )
        if args.distributed:
            model_class = getattr(
                models,
                # keep compatibility
                config.get("model_type", "Cnn14_DecisionLevelAtt"),
            )
            if config["model_type"] == "Cnn14_DecisionLevelAtt":
                model = model_class(training=True, **config["model_params"])
                model.bn0 = nn.BatchNorm2d(config["num_mels"])
                model.att_block = models.AttBlock(**config["att_block"])
                if config["model_params"].get("use_dializer", False):
                    model.dialize_layer = nn.Linear(config["n_class"], 1, bias=True)
                if config["model_params"].get("require_prep", False):
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
            elif config["model_type"] == "ResNext50":
                model = model_class(training=True, **config["model_params"]).to(device)
                model.bn0 = nn.BatchNorm2d(config["num_mels"])
                model.att_block = models.AttBlock(**config["att_block"])
            else:
                model = model_class(training=True, **config["model_params"]).to(device)
            model = DistributedDataParallel(model.to(device))
        trainer = SEDTrainer(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            train_sampler=train_sampler,
            model=model.to(device),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            train=fold == 0,
            use_center_loss=config.get("use_center_loss", False),
            use_dializer=config.get("use_dializer", False),
            save_name=f"fold{fold}",
        )
        # resume from checkpoint
        trainer.load_checkpoint(last_checkpoint, load_only_params=False)
        logging.info(f"Successfully resumed from {last_checkpoint}.")
        # run training loop
        try:
            trainer.run()
        except KeyboardInterrupt:
            trainer.save_checkpoint(
                os.path.join(
                    config["outdir"], f"checkpoint-{trainer.steps}stepsfold{fold}.pkl"
                )
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
        config["train_max_steps"] -= config.get("additional_steps", 1000)


if __name__ == "__main__":
    main()
