import pickle
import os
import random
import re
import numpy as np
import tensorflow as tf

def generator(filepath):
    batch = []
    for _, _, files in os.walk(filepath):
        for file in files:
            try:
                pattern = r"metadata"
                if re.search(pattern, file):
                    continue
                with open(filepath + file, "rb") as file:
                    arr = tf.convert_to_tensor(pickle.load(file))
                    arr = tf.expand_dims(arr,0)
                    yield arr,arr
            except StopIteration:
                break;
