from networks.encoder import ConvEncoder
from networks.classification import Classification
import torch.nn as nn

from networks.BiConvLSTM import BiConvLSTM


class VP(nn.Module):
    def __init__(self, num_classes=2):
        super(VP, self).__init__()
        self.convenc = ConvEncoder()
        self.biconvlstm = BiConvLSTM(input_size=(14, 14), input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=1)
        self.classification = Classification(in_size=(14, 14), in_channels=512, num_classes=num_classes)

    def forward(self, clips):
        clips_feature_maps = self.convenc(clips)

        clips_feature_maps = self.biconvlstm(clips_feature_maps)

        # Max pool :)
        classification = self.classification(clips_feature_maps.max(dim=1)[0])
        return {'classification': classification}
