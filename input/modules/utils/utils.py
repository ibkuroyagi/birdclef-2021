# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import logging
import os
import sys

import h5py
import librosa
import numpy as np
import torch


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-5,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = retrieved_cumulative_hits[class_rankings[pos_class_indices]] / (
        1 + class_rankings[pos_class_indices].astype(np.float)
    )
    return pos_class_indices, precision_at_hits


def lwlrap(truth, scores):
    """Calculate LWLRAP

    Args:
        truth (ndarray): Ground truth.(B, n_class)
        scores (ndarray): Predicted score.(B, n_class)

    Returns:
        per_class_lwlrap (ndarray): (n_class)
        weight_per_class (ndarray): (n_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(
            scores[sample_num, :], truth[sample_num, :]
        )
        precisions_for_samples_by_classes[
            sample_num, pos_class_indices
        ] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(
        1, labels_per_class
    )
    score = (per_class_lwlrap * weight_per_class).sum()
    return score


def down_sampler(source, l_target=16, mode="sum"):
    """Down sampling function.

    Args:
        source (ndarray): (T, n_class)
        l_target (int, optional): Target length. Defaults to 16.
        mode (str, optional): ["sum" or "binary"]. Defaults to "sum".

    Returns:
        new_source: down sampled tensor. (l_target, n_class)
    """
    l_source, n_class = source.shape
    new_source = np.zeros((l_target, n_class))
    l_effect = l_source // l_target
    for i in range(l_target):
        t_start = i * l_effect
        t_end = (i + 1) * l_effect
        if i == l_target - 1:
            t_end = l_source
        if mode == "sum":
            new_source[i] = source[t_start:t_end].sum(axis=0) / l_effect
        elif mode == "binary":
            new_source[i] = (source[t_start:t_end] > 0).any(axis=0).astype(np.int64)
    return new_source


def get_down_sample_matrix(
    matrix, l_target=16, max_frames=512, n_eval_split=20, mode="binary"
):
    """Get down-sampled ground truth data.

    Args:
        matrix (ndarray): Gound truth matrix(l_original, n_class).
        l_target (int, optional): Target length. Defaults to 16.
        max_frames (int, optional): Max frame of model's input. Defaults to 512.
        n_eval_split (int, optional): The number of split eval data. Defaults to 20.
        mode (str, optional): Mode of down sampling. Defaults to binary.

    Returns:
        ground_truth_frame(ndarray): (n_eval_split, l_target, n_class)
    """
    l_original, n_class = matrix.shape
    ground_truth_frame = np.zeros((n_eval_split, l_target, n_class))
    for i in range(n_eval_split):
        if i == n_eval_split - 1:
            beginning = l_original - max_frames
            endding = l_original
        else:
            beginning = int(i * ((l_original - max_frames) // (n_eval_split - 1)))
            endding = beginning + max_frames
        tmp = matrix[beginning:endding]
        ground_truth_frame[i] = down_sampler(tmp, l_target=l_target, mode=mode)
    return ground_truth_frame


def get_concat_down_frame(y_frame, l_original=5626, max_frames=512):
    """Get concatenated down samplied frame data.

    Args:
        y_frame (ndarray): Splited down sampled frame data(n_eval_split, l_target, n_class).
        l_original (int, optional): Length od spectrogam. Defaults to 5626.
        max_frames (int, optional): Max frame of model's input. Defaults to 512.

    Returns:
        down_concat_frame (ndarray): Concatenated down sampled frames(l_down_target, n_class).
    """
    n_eval_split, l_target, n_class = y_frame.shape
    if n_eval_split == 1:
        return y_frame.squeeze(1)
    else:
        shift = np.round((l_original / n_eval_split) * (l_target / max_frames)).astype(
            np.int64
        )
        l_down_target = shift * (n_eval_split - 1) + l_target
    down_concat_frame = np.zeros((l_down_target, n_class))
    avg_frame = np.zeros((l_down_target, 1))
    for i in range(n_eval_split):
        beginning = int(i * shift)
        endding = beginning + l_target
        # print(i, l_down_target, beginning, endding)
        down_concat_frame[beginning:endding] += y_frame[i]
        avg_frame[beginning:endding] += 1
    down_concat_frame = down_concat_frame / avg_frame
    return down_concat_frame


def original_mixup(batch: torch.tensor, mixup_alpha=0.2, device="cpu"):
    """Original mixup function

    Args:
        batch (torch.tensor): batch {X:(B*2, ...), y_frame:(B*2, ...), y_clip:(B*2, ...)}
        mixup_alpha (float, optional): Parameter of beta distribution. Defaults to 0.2.
        device (str, optional): Cuda device. Defaults to "cup".
    Returns:
        batch (torch.tensor): batch {X:(B, ...), y_frame:(B, ...), y_clip:(B, ...)}
    """
    for i, (key, value) in enumerate(batch.items()):
        if i == 0:
            batch_size = value.shape[0] // 2
            lambda_arr = torch.tensor(
                np.random.beta(mixup_alpha, mixup_alpha, batch_size).astype(np.float32)
            ).to(device)
        batch[key] = (
            value[:batch_size].transpose(0, -1) * lambda_arr
            + value[batch_size : batch_size * 2].transpose(0, -1) * (1 - lambda_arr)
        ).transpose(0, -1)
    return batch


if __name__ == "__main__":
    y_true = np.array([[1, 0, 0], [0, 0, 1]])
    y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])

    score = lwlrap(y_true, y_score)
    print(f"score:{score:.4f}")
