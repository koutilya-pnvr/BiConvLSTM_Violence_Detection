import numpy as np
import cv2
import torch

class SelectFrames():
    def __init__(self, num_frames=20):

        self.num_frames = num_frames

    def __call__(self,clip):
        
        frame_count = clip.shape[0]
        step_size = frame_count // self.num_frames

        clip = clip[:self.num_frames*step_size,:,:,:]
        clip = clip[::step_size,:,:,:]
        return clip


class FrameDifference():
    def __init__(self, dim=0):

        self.dim = dim

    def __call__(self, clip):

        clip = clip.astype(np.float32)
        clip = np.diff(clip, axis=self.dim)
        return clip


class Downsample():
    def __init__(self, sampling_rate):

        self.sampling_rate = sampling_rate

    def __call__(self, clip):

        return clip[::self.sampling_rate, :, :, :]


class TileVideo():
    def __init__(self, num_frames):

        self.num_frames = num_frames

    def __call__(self, clip):

        num_tile = self.num_frames // clip.shape[0] + 1
        clip = np.tile(clip, (num_tile, 1, 1, 1))
        clip = clip[:self.num_frames, :, :, :]
        return clip


class RandomCrop():
    def __init__(self, size):

        self.size = size

    def __vid_crop(self, height, width):
        rand = np.random.random_integers(0, 4)
        if rand == 0:
            # center
            start_height = height // 2 - self.size // 2
            end_height = height // 2 + self.size // 2
            start_width = width // 2 - self.size // 2
            end_width = width // 2 + self.size // 2
        elif rand == 1:
            start_height = 0
            end_height = self.size
            start_width = 0
            end_width = self.size
        elif rand == 2:
            start_height = 0
            end_height = self.size
            start_width = width - self.size
            end_width = width
        elif rand == 3:
            start_height = height - self.size
            end_height = height
            start_width = 0
            end_width = self.size
        elif rand == 4:
            start_height = height - self.size
            end_height = height
            start_width = width - self.size
            end_width = width

        return start_height, end_height, start_width, end_width

    def __call__(self, clip):

        _, height, width, _ = clip.shape

        if height < self.size and width < self.size:
            return clip
        else:
            start_height, end_height, start_width, end_width = self.__vid_crop(height, width)

        clip = clip[:, start_height:end_height, start_width:end_width, :]

        return clip


class Resize():
    def __init__(self, size):

        self.size = size

    def __call__(self, clip):

        return np.array([cv2.resize(img, (self.size, self.size)) for img in clip])


class RandomHorizontalFlip():
    def __call__(self, clip):

        flip = np.random.choice([-1, 1])
        clip = clip[:, :, ::flip, :]
        return clip


class Normalize():
    def __call__(self, clip):

        return clip / 255.0


class ToTensor(object):
    def __call__(self, clip):

        return torch.from_numpy(clip.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
