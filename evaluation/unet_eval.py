from evaluation.core import *
from data.dataset import FullSizeVideoDataset
from torch.utils.data import DataLoader
import random as rand
import numpy as np
import os
from tqdm import tqdm
import pickle
from experiment_unet import Run
from data.video_data_generation import arr_from_video_cropped_normalised
from multi_slice_viewer.multi_slice_viewer import multi_slice_viewer
import cv2
from datetime import time

video_arr = arr_from_video_cropped_normalised("/media/leon/2tbssd/ULTRAZVOK_COLLAB/ULTRAZVOK_old/TEST_VIDEJO.AVI")
print(video_arr.shape)

output_folder = "/media/leon/2tbssd/ULTRAZVOK_COLLAB/ULTRASOUND_MASTER/files/video_by_batches_output"
i_start = 10  # število frameov..
i_stop = 0
for i in tqdm(range(len(os.listdir(output_folder)))):
    i = i + i_start
    with open(os.path.join(output_folder, str(i)), "rb") as f:
        obj = pickle.load(f)
    i_stop = i
    video_arr[i] = np.abs(video_arr[i] - obj)
video_arr = video_arr[i_start:i_stop]
multi_slice_viewer(video_arr)

"""def to_u(a):
    a -= np.min(a)
    a /= np.max(a)
    # print('Read a new frame: ', success)
    a *= 255
    # u = np.concatenate((u, np.uint8(image)), axis=0)
    return np.uint8(a)


def array_to_video(array, file="../{}.avi".format(time()), to_u_=True):
    out = cv2.VideoWriter(file, apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=18,
                          frameSize=(array[0].shape[1], array[0].shape[0]))
    for frame in array:
        if frame.dtype != np.uint8 and to_u_:
            frame = to_u(frame)
        out.write(frame)"""
