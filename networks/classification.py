import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self, in_size=(11, 11), in_channels=64, num_classes=2):
        super(Classification, self).__init__()

        in_height, in_width = in_size

        self.conv_bn = nn.BatchNorm2d(in_channels)

        self.main = nn.Sequential(nn.Linear(in_height * in_width * in_channels, 1000),
                                  nn.Tanh(),
                                  nn.Linear(1000, 256),
                                  nn.Tanh(),
                                  nn.Linear(256, 10),
                                  nn.Tanh(),
                                  nn.Linear(10, num_classes))

    def forward(self, x):
        x = self.conv_bn(F.relu(x))
        x = x.view(x.shape[0], -1)
        classification = self.main(x)
        return classification
