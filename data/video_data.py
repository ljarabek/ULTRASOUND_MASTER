from config import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rand
from tqdm import tqdm


def get_video_list():
    vlist = list()
    for root, dirs, files in os.walk(video_directory):
        for f in files:
            if f.lower().endswith("avi"):
                vlist.append(os.path.join(root, f))
    return vlist


def array_from_video(file, maxframes=2000):
    """
    :param file: string file location
    :param maxframes: int frames to array
    :return:  shape is N,H,W,C !has RGB channels
    """
    vidcap = cv2.VideoCapture(file)
    array = []
    frame = 0
    success, image = vidcap.read()
    while success:
        array.append(image)
        frame += 1
        success, image = vidcap.read()
        # array.append(np.array(image))
    return np.array(array)


def get_mean_std():
    """
    :return: need 15GB ram to run
    """

    """
    mean 41.489833348607036
    std 32.18347933366483
    """
    master_list = list()
    video_path_list = get_video_list()
    for i, d in tqdm(enumerate(video_path_list)):
        vid = array_from_video(d)[:, 15:-25, 165:475, 0]
        master_list.extend(vid)
        if i > 5:
            break

    sample = rand.sample(master_list, 200)
    video_mean = np.mean(sample)
    video_std = np.std(sample)
    master_list = np.array(master_list, dtype=np.float)
    master_list -= video_mean
    master_list /= video_std

    with open("./files/mean_std", "w") as f:
        f.write("mean " + str(video_mean) + "\n")
        f.write("std " + str(video_std) + "\n")

    return video_mean, video_std


def generate_training_example(N: int, video_list: list):
    """
    :param N: number of frames for training
    :param video_list: list of videos to choose from
    :return: N normalised consecutive frames from random time in random video
    """


if __name__ == "__main__":
    vp = get_video_list()
    print(len(vp))
    videos_val = rand.sample(vp, no_val_videos)
    vp = [v for v in vp if v not in videos_val]
    print(len(vp))
    videos_test = rand.sample(vp, no_train_videos)
    vp = [v for v in vp if v not in videos_test]
    videos_train = vp
    print(len(vp))

# video_path_list = get_video_list()
# print(len(video_path_list))  # 37 videos
# vid = array_from_video(video_path_list[0])
# sample_frame = vid[500]
# plt.imshow(sample_frame[15:-25, 165:475, 0])  # cviknemo prvih pa zadnih 25 pixlov, izberemo en kanal
# plt.show()
