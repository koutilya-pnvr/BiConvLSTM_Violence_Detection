import torch
import torch.nn as nn
import torchvision.models as models


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        vggnet = models.__dict__['vgg13_bn'](pretrained=True)
        # Remove last max pooling layer of vggnet
        self.encoder = nn.Sequential(*list(vggnet.features.children())[:-1])

    def forward(self, clips):
        # Permute to run encoder on batch of each frame
        # NOTE: This requires clips to have the same number of frames!!
        frame_ordered_clips = clips.permute(1, 0, 2, 3, 4)
        clips_feature_maps = [self.encoder(frame) for frame in frame_ordered_clips]
        return torch.stack(clips_feature_maps, dim=0).permute(1, 0, 2, 3, 4)
