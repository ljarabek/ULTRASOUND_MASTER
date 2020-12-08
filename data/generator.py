from config import *
import json
import os
import cv2
import numpy as np
import time
from random import seed
from random import randint
from random import shuffle
import pickle
import pprint
from datetime import datetime
seed(14)

# In order to generate training examples, you need to create a "files" folder in ULTRASOUND_MASTER folder!!!
# reports folder contains train_test_split video names 


class MasterGenerator():

    def __init__(self):
        self.mean = 41.489833348607036
        self.std = 32.18347933366483
        self.no_test = no_test_videos
        self.no_val = no_val_videos
        self.video_directory = video_directory

    def get_video_list(self):
        vlist = list()
        for root, _, files in os.walk(video_directory):
            for f in files:
                if f.lower().endswith("avi"):
                    vlist.append(os.path.join(root, f))
        return vlist

    def array_from_video(self, file, maxframes=2000):
        # Converts RGB to grayscale, crops the video and normalizes the data
        vidcap = cv2.VideoCapture(file)
        array = []
        frame = 0
        success, image = vidcap.read()

        while success:
            # Convert image to grayscale
            array.append(image)
            frame += 1
            success, image = vidcap.read()

        video_arr = np.array(array)
        video_arr = np.array(video_arr[:, 15:-25, 165:475, 0], dtype=np.float)
        video_arr -= self.mean
        video_arr /= self.std

        return video_arr

    def train_val_test_split(self):
        video_list = self.get_video_list()
        shuffle(video_list)

        val_list = video_list[0:5]
        test_list = video_list[5:10]
        train_list = video_list[10:len(video_list)-1]
        full_dict = {"train" : train_list, "val": val_list, "test" : test_list}

        # Current date and time
        dt_string = datetime.now().strftime("%d%m%Y_%H:%M:%S")

        with open("./reports/" + dt_string + ".json", "w") as file:
            json.dump(full_dict, file)

        return train_list, val_list, test_list


class GeneratorSmallCrops(MasterGenerator):
    def __init__(self):
        super(GeneratorSmallCrops, self).__init__()

    def generate_pickles_from_video(self, N: int,num_per_video: int, video, folder: str):
        video_arr = self.array_from_video(video)
        # video dims: 0 - len, 1 - height, 2 - width
        for i in range(num_per_video):
            # Create a temporary placeholder
            final_array = list()

            # Boundaries for top left coordinates
            x_bound = randint(0, video_arr.shape[2] - 32 - 1)
            y_bound = randint(0, video_arr.shape[1] - 32 - 1)
            # Now the boundaries for the frames
            # We have to take the frames and the stride into account

            danger_zone = N + 1
            current_frame = randint(0, video_arr.shape[0] - danger_zone)
            cropped_box = video_arr[:, y_bound:y_bound + 32, x_bound:x_bound + 32]

            metadata = dict()
            metadata["videofile"] = video
            metadata["x_bound"] = x_bound
            metadata["y_bound"] = y_bound
            metadata["starting_frame"] = current_frame

            for j in range(N):
                final_array.append(cropped_box[current_frame])
                current_frame += 1

            filename = str(hash(video + str(i)))
            with open(os.path.join("./files/", folder, filename), "wb") as file:
                pickle.dump(final_array, file)

            with open(os.path.join("./files/", folder, (filename + "_metadata")), "wb") as file:
                pickle.dump(metadata, file)
            

class GeneratorLeon(MasterGenerator):

    def __init__(self):
        super(GeneratorLeon, self).__init__()

    def generate_training_example(self, N: int, num_per_video: int, video, folder: str):
        """
        :param N: number of frames for training
        :param video: list of videos to choose from or specific one
        :return: N normalised consecutive frames from random time in random video
        """
        arr = None

        if type(video) == list:
            video_choice = rand.choice(video)
        else:
            video_choice = video
        video_arr = array_from_video(video_choice)
        video_arr = np.array(video_arr[:, 15:-25, 165:475, 0], dtype=np.float)
        video_len = video_arr.shape[0]
        video_arr -= mean
        video_arr /= std

        for i in range(num_per_video):
            N0 = rand.randint(1, video_len - N - 1)
            arr = video_arr[N0:N0 + N]
            fname = str(hash(video_choice))[1:] + "_" + str(i)
            with open(os.path.join("./files/", folder, fname), "wb") as f:
                pickle.dump(arr, f)

        return arr  # returns last arr

generator = GeneratorSmallCrops()
train, val, test = generator.train_val_test_split()
for i, video in enumerate(train):
    generator.generate_pickles_from_video(30,1000, video, "train/")
    print("Progress: ", (i+1)/len(train)*100, " %")
print("-----------------------")
for i, video in enumerate(val):
    generator.generate_pickles_from_video(30,1000, video, "val/")
    print("Progress: ", (i+1)/len(val)*100, " %")
print("-----------------------")
for i, video in enumerate(test):
    generator.generate_pickles_from_video(30,1000, video, "test/")
    print("Progress: ", (i+1)/len(test)*100, " %")
