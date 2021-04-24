# -*- coding: utf-8 -*-

# Created by Ibuki Kuroyanagi

"""Conformer algorithm."""

import logging

import torch
from torchlibrosa.augmentation import SpecAugmentation


class ConformerEncoderDecoder(torch.nn.Module):
    """Conformer-based encoder decoder."""

    def __init__(
        self,
        num_features=256,
        num_classes=25,
        num_blocks=8,
        num_channels=144,
        kernel_size=31,
        num_heads=4,
        num_latent_units=32,
        num_embeddings=0,
        embedding_dim=0,
        concat_embedding=False,
        dropout=0.1,
        bias=True,
        use_bottleneck=True,
        use_reconstruct=False,
        use_dializer=False,
        use_mask=False,
        is_spec_augmenter=False,
        training=False,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.use_embedding = embedding_dim > 0
        num_features = num_features - 1 if self.use_embedding else num_features
        self.concat_embedding = concat_embedding
        self.use_reconstruct = use_reconstruct
        self.use_dializer = use_dializer
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.training = training
        self.is_spec_augmenter = is_spec_augmenter
        if is_spec_augmenter:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=80,
                time_stripes_num=2,
                freq_drop_width=20,
                freq_stripes_num=2,
            )

        self.input_layer = torch.nn.Conv1d(num_features, num_channels, 1, bias=bias)
        encoder_blocks = []
        for _ in range(num_blocks):
            encoder_blocks += [
                ConformerBlock(
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                )
            ]
        self.encoder_block = torch.nn.Sequential(*encoder_blocks)

        if use_bottleneck:
            self.encoder_projection = torch.nn.Sequential(
                torch.nn.Conv1d(num_channels, num_latent_units, 1, bias=bias),
                torch.nn.ReLU(inplace=True),
            )

            if self.use_embedding:
                if concat_embedding:
                    decoder_input_units = num_latent_units + embedding_dim
                else:
                    if num_latent_units != embedding_dim:
                        logging.warning(
                            f"embedding_dim is modified from {embedding_dim} to {num_latent_units}."
                        )
                        embedding_dim = num_latent_units
                    decoder_input_units = num_latent_units
                self.embed_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
            else:
                decoder_input_units = num_latent_units

            self.decoder_projection = torch.nn.Conv1d(
                decoder_input_units, num_channels, 1, bias=bias
            )
            decoder_blocks = []
            for _ in range(num_blocks):
                decoder_blocks += [
                    ConformerBlock(
                        num_channels=num_channels,
                        kernel_size=kernel_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        bias=bias,
                    )
                ]
            self.decoder_block = torch.nn.Sequential(*decoder_blocks)
        self.frame_layer = torch.nn.Conv1d(num_channels, num_classes, 1, bias=bias)
        if use_reconstruct:
            self.reconstruct_layer = torch.nn.Conv1d(
                num_channels, num_features, 1, bias=bias
            )
        if use_dializer:
            self.dialize_layer = torch.nn.Conv1d(num_channels, 1, 1, bias=bias)

    def forward(self, x):
        """Calcualte forward propagation.
        Args:
            x (Tensor): Input tensor (batch_size, 1+sequence_length, num_features).
        Returns:
            Tensor: y_clip (batch_size, num_class).
            Tensor: y_frame (batch_size, sequence_length, num_class).
            Tensor: frame_mask (batch_size, sequence_length, 1).
            Tensor: reconstructed inputs (batch_size, sequence_length, num_features).
        """
        if self.use_embedding:
            id_ = x[:, :, -1].long()
            x = x[:, :, :-1]
        if self.training and self.is_spec_augmenter:
            x = x.unsqueeze(1)
            x[:, :, 1:, :] = self.spec_augmenter(x[:, :, 1:, :])  # (B, 1, T', mels)
            x = x.squeeze(1)  # (B, T', mels)
        enc = self.input_layer(x.transpose(1, 2))
        enc = self.encoder_block(enc)
        if self.use_bottleneck:
            enc = self.encoder_projection(enc)
            if self.use_embedding:
                id_emb = self.embed_layer(id_).transpose(1, 2)
                if self.concat_embedding:
                    enc = torch.cat((enc, id_emb), dim=1)
                else:
                    enc = enc + id_emb
            dec = self.decoder_projection(enc)
            dec = self.decoder_block(dec)
        else:
            dec = enc
        out = self.frame_layer(dec).transpose(1, 2)
        y_ = {}
        if self.use_dializer:
            # frame_mask = self.dialize_layer(dec).transpose(1, 2)[:, 1:, :]
            # out[:, 1:, :] = out[:, 1:, :] * torch.sigmoid(frame_mask)
            # out[:, 0, :] = out[:, 0, :] + out[:, 1:, :].max(dim=1)[0]
            frame_mask = self.dialize_layer(dec).transpose(1, 2)
            if self.use_mask:
                out = out * torch.sigmoid(frame_mask)
            y_["frame_mask"] = frame_mask[:, 1:]
        y_["y_clip"] = out[:, 0, :]
        y_["y_frame"] = out[:, 1:, :]
        if self.use_reconstruct:
            y_["reconstructed"] = self.reconstruct_layer(dec).transpose(1, 2)

        return y_


class ConformerBlock(torch.nn.Module):
    """Conformer block in Conformer."""

    def __init__(
        self,
        num_channels=144,
        kernel_size=31,
        num_heads=4,
        dropout=0.1,
        bias=True,
    ):
        super(ConformerBlock, self).__init__()
        self.ff_module_1 = FeedForwardModule(num_channels, dropout, bias)
        self.mhsa_module = MultiHeadedSelfAttentionModule(
            num_channels, num_heads, dropout, bias
        )
        self.conv_module = ConvolutionModule(num_channels, kernel_size, dropout, bias)
        self.ff_module_2 = FeedForwardModule(num_channels, dropout, bias)
        self.layer_norm = torch.nn.LayerNorm(num_channels)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, num_channels, T).
        Returns:
            Tensor: Output tensor (B, num_channels, T).
        """
        x = x + 0.5 * self.ff_module_1(x)
        x = x + self.mhsa_module(x)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_module_2(x)
        return self.layer_norm(x.transpose(1, 2)).transpose(1, 2)


class ConvolutionModule(torch.nn.Module):
    """Convolution module in Conformer."""

    def __init__(
        self,
        num_channels=144,
        kernel_size=31,
        dropout=0.1,
        bias=True,
    ):
        super(ConvolutionModule, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd number."
        padding = (kernel_size - 1) // 2
        self.layer_norm = torch.nn.LayerNorm(num_channels)
        layers = []
        layers += [torch.nn.Conv1d(num_channels, num_channels * 2, 1, bias=bias)]
        layers += [torch.nn.GLU(dim=1)]
        layers += [
            DepthwiseConv1d(num_channels, kernel_size, padding=padding, bias=bias)
        ]
        layers += [torch.nn.BatchNorm1d(num_channels)]
        layers += [SwishActivation()]
        layers += [torch.nn.Conv1d(num_channels, num_channels, 1, bias=bias)]
        layers += [torch.nn.Dropout(dropout)]
        self.conv_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, num_channels, T).
        Returns:
            Tensor: Output tensor (B, num_channels, T).
        """
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.conv_module(x)


class FeedForwardModule(torch.nn.Module):
    """Feed-forward module in Conformer."""

    def __init__(self, num_channels, dropout=0.1, bias=True):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(num_channels)
        layers = []
        layers += [torch.nn.Conv1d(num_channels, num_channels * 4, 1, bias=bias)]
        layers += [SwishActivation()]
        layers += [torch.nn.Dropout(dropout)]
        layers += [torch.nn.Conv1d(num_channels * 4, num_channels, 1, bias=bias)]
        layers += [torch.nn.Dropout(dropout)]
        self.ff_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, num_channels, T).
        Returns:
            Tensor: Output tensor (B, num_channels, T).
        """
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.ff_module(x)


class MultiHeadedSelfAttentionModule(torch.nn.Module):
    """Multi-headed self attention module in Conformer."""

    def __init__(self, d_model=144, num_heads=4, dropout=0.1, bias=True):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(d_model)
        # TODO(kan-bayashi): Use relative positional encoding
        self.self_attn = torch.nn.MultiheadAttention(d_model, num_heads, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, num_channels, T).
        Returns:
            Tensor: Output tensor (B, num_channels, T).
        """
        x = self.layer_norm(x.transpose(1, 2)).transpose(0, 1)  # (T, B, C)
        # TODO(kan-bayashi): Consider masking
        x, _ = self.self_attn(x, x, x)
        return self.dropout(x.permute(1, 2, 0))  # (B, C, T)


class SwishActivation(torch.nn.Module):
    """Swish activation function."""

    def __init__(self):
        super(SwishActivation, self).__init__()

    def forward(self, x):
        """Calculate forward propagation."""
        return x * torch.sigmoid(x)


class DepthwiseConv1d(torch.nn.Conv1d):
    """DepthwiseConv1d module."""

    def __init__(self, num_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DepthwiseConv1d, self).__init__(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=num_channels,
        )
