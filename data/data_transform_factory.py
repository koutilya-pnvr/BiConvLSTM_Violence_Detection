import numpy as np
import cv2


def __random_crop(clip):
    _, height, width, _ = clip.shape

    if height < 224 and width < 224:
        return clip
    else:
        start_height, end_height, start_width, end_width = __vid_crop(height, width)

    flip = np.random.choice([-1, 1])
    clip = clip[:, start_height:end_height, start_width:end_width, :]
    clip = clip[:, :, ::flip, :]

    return clip


def __vid_crop(height, width):
    rand = np.random.random_integers(0, 4)
    if rand == 0:
        # center
        start_height = height // 2 - 112
        end_height = height // 2 + 112
        start_width = width // 2 - 112
        end_width = width // 2 + 112
    elif rand == 1:
        start_height = 0
        end_height = 224
        start_width = 0
        end_width = 224
    elif rand == 2:
        start_height = 0
        end_height = 224
        start_width = width - 224
        end_width = width
    elif rand == 3:
        start_height = height - 224
        end_height = height
        start_width = 0
        end_width = 224
    elif rand == 4:
        start_height = height - 224
        end_height = height
        start_width = width - 224
        end_width = width

    return start_height, end_height, start_width, end_width


def __resize224(clip):
    return np.array([cv2.resize(img, (224, 224)) for img in clip])


def __resize256(clip):
    return np.array([cv2.resize(img, (256, 256)) for img in clip])


def transform_factory(transform):
    if transform == 'R': return __resize224
    if transform == 'C': return __random_crop
    if transform == 'RC': return lambda x: __random_crop(__resize256(x))
    if callable(transform): return transform
    assert 0, "Bad transform: " + transform
