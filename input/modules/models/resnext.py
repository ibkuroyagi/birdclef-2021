import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from .cnn import AttBlock
from .utils import init_bn
from .utils import init_layer


class ResNext50(nn.Module):
    def __init__(
        self,
        num_mels=128,
        classes_num=24,
        training=False,
        is_spec_augmenter=False,
        require_prep=False,
        use_dializer=False,
    ):

        super(self.__class__, self).__init__()
        self.bn0 = nn.BatchNorm2d(num_mels)
        self.conv0 = nn.Conv2d(1, 3, 1, 1)
        self.resnext50 = torchvision.models.resnext50_32x4d(pretrained=True)
        self.resnext50.fc = nn.Linear(2048, classes_num, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.att_block = AttBlock(2048, classes_num, activation="linear")

        # self.init_weight()
        self.training = training
        self.is_spec_augmenter = is_spec_augmenter
        self.require_prep = require_prep
        self.use_dializer = use_dializer
        if is_spec_augmenter:
            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=80,
                time_stripes_num=2,
                freq_drop_width=20,
                freq_stripes_num=2,
            )
        if use_dializer:
            self.dialize_layer = nn.Linear(classes_num, 1, bias=True)

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input):
        """Input: (batch_size, mels, T')"""

        x = input.unsqueeze(3)
        x = self.bn0(x)  # (B, mels, T', 1)
        x = x.transpose(1, 3)  # (B, 1, T', mels)
        if self.training and self.is_spec_augmenter:
            x = self.spec_augmenter(x)
        x = self.conv0(x)
        x = self.resnext50.conv1(x)
        x = self.resnext50.bn1(x)
        x = self.resnext50.relu(x)
        x = self.resnext50.maxpool(x)
        x = self.resnext50.layer1(x)
        x = self.resnext50.layer2(x)
        x = self.resnext50.layer3(x)
        x = self.resnext50.layer4(x)
        x = torch.mean(x, dim=3)
        embedding = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(self.dropout1(x)))
        x = x.transpose(1, 2)
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
