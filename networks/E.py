from networks.encoder import ConvEncoder
from networks.classification import Classification
import torch.nn as nn

from networks.ConvLSTM import ConvLSTM


class VP(nn.Module):
    def __init__(self, num_classes=2):
        super(VP, self).__init__()
        self.convenc = ConvEncoder()
        self.convlstm = ConvLSTM(input_size=(14, 14), input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=1)
        self.classification = Classification(in_size=(14, 14), in_channels=512, num_classes=num_classes)

    def forward(self, clips):
        clips_feature_maps = self.convenc(clips)

        clips_feature_maps = self.convlstm(clips_feature_maps)
        last_frames_feature_maps = clips_feature_maps[:, -1, :, :, :].contiguous()

        classification = self.classification(last_frames_feature_maps)
        return {'classification': classification}
