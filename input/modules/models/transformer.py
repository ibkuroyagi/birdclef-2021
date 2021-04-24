# -*- coding: utf-8 -*-

# Copyright 2020 Human Dataware Lab. Co., Ltd.
# Created by Tomoki Hayashi

"""Transformer-based auto encoder modules."""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank
from torchlibrosa.stft import Spectrogram


class RandomCutCollater(object):
    """Collater to cut the sequence randomly."""

    def __init__(self, min_length, max_length=None, return_mask=True):
        """Initialize RandomCutCollater module."""
        self.min_length = min_length
        self.max_length = max_length
        self.return_mask = return_mask

    def __call__(self, batch):
        """Cut the length of each item in batch randomly."""
        if self.max_length is None:
            self.max_length = batch[0].shape[0]
        assert self.max_length <= batch[0].shape[0]
        assert self.min_length <= batch[0].shape[0]
        lengths = sorted(
            np.random.randint(self.min_length, self.max_length, len(batch))
        )
        new_batch = []
        for x, length in zip(batch, lengths):
            start_idx = np.random.randint(0, x.shape[0] - length)
            new_batch += [torch.from_numpy(x[start_idx : start_idx + length]).float()]
        new_batch = self._pad_list_of_tensors(new_batch)

        if not self.return_mask:
            return new_batch
        else:
            mask = self._make_mask_from_list_of_lengths(lengths)
            return new_batch, mask

    @staticmethod
    def _make_mask_from_list_of_lengths(list_of_lengths):
        """Make mask from the list of lengths.
        Examples:
            >>> lengths = [1, 2, 3, 4]
            >>> _make_mask_from_list_of_lengths(lengths)
            tensor([[True, False, False, False],
                    [True, True,  False, False],
                    [True, True,  True,  False],
                    [True, True,  True,  True]])
        """
        batch_size = int(len(list_of_lengths))
        maxlen = int(max(list_of_lengths))
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
        seq_length_expand = seq_range_expand.new(list_of_lengths).unsqueeze(-1)
        return seq_range_expand < seq_length_expand

    @staticmethod
    def _pad_list_of_tensors(list_of_tensors, pad_value=0.0):
        """Perform padding for the list of tensors.
        Examples:
            >>> list_of_tensors = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> list_of_tensors
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> _pad_list_of_tensors(list_of_tensors, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])
        """
        batch_size = len(list_of_tensors)
        maxlen = max(x.size(0) for x in list_of_tensors)
        padded_tensor = (
            list_of_tensors[0]
            .new(batch_size, maxlen, *list_of_tensors[0].size()[1:])
            .fill_(pad_value)
        )
        for i in range(batch_size):
            padded_tensor[i, : list_of_tensors[i].size(0)] = list_of_tensors[i]
        return padded_tensor


class TransformerEncoderDecoder(nn.Module):
    """Transformer-based Encoder Decoder module."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        sequence_length: int,
        input_layer: str = "linear",
        num_blocks: int = 4,
        num_heads: int = 4,
        num_hidden_units: int = 64,
        num_feedforward_units: int = 128,
        num_latent_units: int = 8,
        num_embeddings=0,
        embedding_dim=0,
        concat_embedding=False,
        activation: str = "relu",
        use_position_encode: bool = False,
        max_position_encode_length: int = 512,
        dropout: float = 0.1,
        use_reconstruct: bool = False,
        use_dializer=False,
        is_spec_augmenter=False,
        training=False,
        require_prep=False,
        sample_rate=48000,
        window_size=2048,
        hop_size=512,
        fmin=30,
        fmax=16000,
    ):
        super().__init__()
        self.use_embedding = embedding_dim > 0
        num_features = num_features - 1 if self.use_embedding else num_features
        self.num_features = num_features
        self.concat_embedding = concat_embedding

        self.use_position_encode = use_position_encode
        self.num_classes = num_classes
        self.use_reconstruct = use_reconstruct
        self.use_dializer = use_dializer
        self.training = training
        self.is_spec_augmenter = is_spec_augmenter
        self.require_prep = require_prep
        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-6
        top_db = None
        if require_prep:
            self.spectrogram_extractor = Spectrogram(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=True,
            )

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(
                sr=sample_rate,
                n_fft=window_size,
                n_mels=num_features,
                fmin=fmin,
                fmax=fmax,
                ref=ref,
                amin=amin,
                top_db=top_db,
                freeze_parameters=True,
            )
        if is_spec_augmenter:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=40,
                time_stripes_num=2,
                freq_drop_width=10,
                freq_stripes_num=2,
            )

        # Build encoder.
        if input_layer == "linear":
            self.input_layer = nn.Linear(num_features, num_hidden_units)
        else:
            raise NotImplementedError(f"{input_layer} is not supported.")
        if use_position_encode:
            self.position_encode = PositionalEncoding(
                d_model=num_hidden_units,
                dropout=dropout,
                maxlen=max_position_encode_length,
            )
        self.norm = nn.LayerNorm(
            (sequence_length + int(not require_prep), num_hidden_units)
        )
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_hidden_units,
            nhead=num_heads,
            dim_feedforward=num_feedforward_units,
            dropout=dropout,
            activation=activation,
        )
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_blocks,
        )
        self.encoder_projection = nn.Sequential(
            nn.Linear(num_hidden_units, num_latent_units),
            nn.ReLU(inplace=True),
        )

        # Build decoder.
        if self.use_embedding:
            self.embed_layer = nn.Embedding(num_embeddings, embedding_dim)
            if self.concat_embedding:
                decoder_input_units = num_latent_units + embedding_dim
            else:
                if num_latent_units != embedding_dim:
                    logging.warning(
                        f"embedding_dim is modified from {embedding_dim} to {num_latent_units}."
                    )
                    embedding_dim = num_latent_units
                decoder_input_units = num_latent_units
        else:
            decoder_input_units = num_latent_units
        self.decoder_projection = nn.Linear(decoder_input_units, num_hidden_units)
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=num_hidden_units,
            nhead=num_heads,
            dim_feedforward=num_feedforward_units,
            dropout=dropout,
            activation=activation,
        )
        self.decoder_transformer = nn.TransformerDecoder(
            decoder_layer=transformer_decoder_layer,
            num_layers=num_blocks,
        )
        self.frame_layer = nn.Linear(num_hidden_units, num_classes)
        if use_reconstruct:
            self.reconstruct_layer = nn.Linear(num_hidden_units, num_features)
        if use_dializer:
            self.dialize_layer = nn.Linear(num_hidden_units, 1)

    def forward(self, x, mask=None):
        """Calcualte forward propagation.
        Args:
            x (Tensor): Input tensor (batch_size, 1+sequence_length, num_features).
                If you use waek label, you have to input weak label idx 0.
            mask (Tensor): Mask tensor (batch_size, sequence_length).
        Returns:
            Tensor: y_clip (batch_size, num_class).
            Tensor: y_frame (batch_size, sequence_length, num_class).
            Tensor: frame_mask (batch_size, sequence_length, 1).
            Tensor: reconstructed inputs (batch_size, sequence_length, num_features).
        """
        if self.require_prep:
            if len(x.shape) < 2:
                x = x.unsqueeze(0)
            x = self.spectrogram_extractor(x)  # (batch_size, 1, T, mel_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, T, mel_bins)
            x = x.squeeze(1)  # (B, T, mels)
        if self.use_embedding:
            id_ = x[:, :, -1].long()
            x = x[:, :, :-1]
        if self.training and self.is_spec_augmenter:
            x = x.unsqueeze(1)
            x[:, :, 1:, :] = self.spec_augmenter(x[:, :, 1:, :])  # (B, 1, T', mel_bins)
            x = x.squeeze(1)  # (B, T', mels)
        enc = self.input_layer(x)
        enc = self.norm(enc)
        if self.use_position_encode:
            enc = self.position_encode(enc)
        enc = self.encoder_transformer(
            enc.transpose(0, 1),
            src_key_padding_mask=~mask if mask is not None else None,
        )
        enc = self.encoder_projection(enc.transpose(0, 1))
        if self.use_embedding:
            id_emb = self.embed_layer(id_)
            if self.concat_embedding:
                enc = torch.cat((enc, id_emb), dim=2)
            else:
                enc = enc + id_emb
        dec = self.decoder_projection(enc)
        dec = self.decoder_transformer(
            tgt=dec.transpose(0, 1), memory=dec.transpose(0, 1)
        )
        out = self.frame_layer(dec.transpose(0, 1))
        y_ = {}
        if self.require_prep:
            if self.use_dializer:
                y_["frame_mask"] = self.dialize_layer(dec.transpose(0, 1))
            y_["y_clip"] = F.max_pool2d(out, kernel_size=out.size()[2:]).squeeze(2)
            y_["y_frame"] = out
        else:
            if self.use_dializer:
                frame_mask = self.dialize_layer(dec.transpose(0, 1))
                y_["frame_mask"] = frame_mask[:, 1:]
            y_["y_clip"] = out[:, 0, :]
            y_["y_frame"] = out[:, 1:, :]
        if self.use_reconstruct:
            y_["reconstructed"] = self.reconstruct_layer(dec.transpose(0, 1))
        return y_


class PositionalEncoding(torch.nn.Module):
    """Positional encoding module."""

    def __init__(self, d_model, dropout=0.0, maxlen=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)
        self.maxlen = maxlen
        self.xscale = math.sqrt(self.d_model)
        self._initialize_positional_encoding()

    def _initialize_positional_encoding(self):
        pe = torch.zeros(self.maxlen, self.d_model)
        position = torch.arange(0, self.maxlen, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (Tensor): Input tensor (B, T, `*`).
        Returns:
            Tensor: Encoded tensor (B, T, `*`).
        """
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)