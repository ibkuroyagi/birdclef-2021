import sys
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank
from torchlibrosa.stft import Spectrogram

sys.path.append("../../")
sys.path.append("../input/modules")

from models.utils import init_bn  # noqa: E402
from models.utils import init_layer  # noqa: E402


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation="linear", temperature=1.0):
        super(self.__class__, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.max(norm_att * cla, dim=2)[0]
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class Cnn14_DecisionLevelAtt(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        window_size=1024,
        hop_size=256,
        mel_bins=128,
        fmin=50,
        fmax=8000,
        classes_num=2,
        training=False,
        require_prep=False,
        is_spec_augmenter=False,
        use_dializer=False,
    ):

        super(self.__class__, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio

        # Spectrogram extractor
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
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=80,
            time_stripes_num=2,
            freq_drop_width=20,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.dropout = nn.Dropout(p=0.2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.att_block = AttBlock(2048, classes_num, activation="sigmoid")

        self.init_weight()
        self.training = training
        self.require_prep = require_prep
        self.is_spec_augmenter = is_spec_augmenter
        self.use_dializer = use_dializer
        if use_dializer:
            self.dialize_layer = nn.Linear(classes_num, 1, bias=True)

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input):
        """Input: (batch_size, data_length)"""

        if self.require_prep:
            x = self.spectrogram_extractor(
                input
            )  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)
        else:
            x = input.unsqueeze(3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.is_spec_augmenter:
            x = self.spec_augmenter(x)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block2(self.dropout(x), pool_size=(2, 2), pool_type="avg")
        x = self.conv_block3(self.dropout(x), pool_size=(2, 2), pool_type="avg")
        x = self.conv_block4(self.dropout(x), pool_size=(2, 2), pool_type="avg")
        x = self.conv_block5(self.dropout(x), pool_size=(2, 2), pool_type="avg")
        x = self.conv_block6(self.dropout(x), pool_size=(1, 1), pool_type="avg")
        # print(f"feature_map:{x.shape}")
        x = torch.mean(x, dim=3)  # + torch.max(x, dim=3)[0]
        embedding = torch.mean(x, dim=2)
        # print(f"feature_map: mean-dim3{x.shape}")
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(self.dropout1(x)))
        x = x.transpose(1, 2)
        # print(f"pool1d_map: mean-dim3{x.shape}")
        (clipwise_output, _, segmentwise_output) = self.att_block(self.dropout2(x))
        segmentwise_output = segmentwise_output.transpose(1, 2)

        output_dict = {
            "y_frame": segmentwise_output,  # (B, T', n_class)
            "y_clip": clipwise_output,  # (B, n_class)
            "embedding": embedding,  # (B, feat_dim)
        }
        if self.use_dializer:
            # (B, T', 1)
            output_dict["frame_mask"] = self.dialize_layer(segmentwise_output)

        return output_dict
