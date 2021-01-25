# import tensorflow as tf
import json
import sys
import numpy as np
import cv2
import pickle

class Evaluator():

    def __init__(self, pickle_path, speed):
        with open(pickle_path + "_metadata.pkl", "rb") as file:
            self.metadata = pickle.load(file)
            self.videofile = self.metadata["videofile"]
            print(self.videofile)
            self.sampling_freq = 18
            self.speed = int(speed)
            self.interval = 1000//18//self.speed # in ms

    def draw(self):
        # Converts RGB to grayscale, crops the video and normalizes the data
        vidcap = cv2.VideoCapture(self.videofile)
        array = []
        frame = 0
        ret, image = vidcap.read()

        while ret:
            
            image = np.array(image[15:-25, 165:475, 0])
            if frame >= self.metadata["starting_frame"] and frame < self.metadata["starting_frame"] + 32:
                x = self.metadata["x_bound"]
                y = self.metadata["y_bound"]
                window_size = 32
                # BUG image format is not supported
                image = cv2.rectangle(image, (x,y), (x+window_size, y+window_size), (255,255,255), 2)
                if cv2.waitKey(1000//18) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(self.interval) & 0xFF == ord("q"):
                    break
            cv2.imshow("image", image)
            frame += 1
            ret, image = vidcap.read()

        vidcap.release()
        cv2.destroyAllWindows()
            

        # video_arr = np.array(array)
        # video_arr = np.array(video_arr[:, 15:-25, 165:475, 0], dtype=np.float)
        # video_arr -= self.mean
        # video_arr /= self.std

# evaluator = Evaluator("/media/tim/Elements/data/uz/test/-9184959580254118173.pkl", 9)
# evaluator.draw()