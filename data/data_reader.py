import glob
import re
import random

import numpy as np
import cv2
from torch.utils.data import Dataset

from data.data_label_factory import label_factory


def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frames.append(frame)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames, we stop
            break
    cap.release()
    video = np.stack(frames)
    return video


class DatasetReader(Dataset):
    def __init__(self, root_dir, data_name):
        super(DatasetReader, self).__init__()

        self.root_dir = root_dir

        regex = re.compile(r'.*\.avi$')
        data_files = list(filter(regex.search, glob.glob(self.root_dir + '/*')))
        random.shuffle(data_files)

        data_labeler = label_factory(data_name)
        self.labeled_data = data_labeler(data_files)

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, idx):
        data_file, label = self.labeled_data[idx]
        videodata = read_video(data_file)

        return (videodata, label)
