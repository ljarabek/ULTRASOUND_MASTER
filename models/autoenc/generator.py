import pickle
import os
import random
import re
import numpy as np
import tensorflow as tf
from config import *

def generator(filepath):
    for _, _, files in os.walk(filepath):
        for file in files:
            try:
                pattern = r"metadata"
                if re.search(pattern, file):
                    continue
                with open(filepath + file, "rb") as file:
                    arr = tf.convert_to_tensor(pickle.load(file))
                    yield arr,arr
            except StopIteration:
                break;
            except tf.errors.OutOfRangeError: continue

def numpy_generator(filepath):
    batch = []
    for _, _, files in os.walk(filepath):
        for file in files:
            try:
                pattern = r"metadata"
                if re.search(pattern, file):
                    continue
                with open(filepath + file, "rb") as file:
                    arr = pickle.load(file)
                    yield arr,arr
            except StopIteration:
                break;

def vis_gen(filepath):
    for _, _, files in os.walk(filepath):
        for file in files:
            try:
                pattern = r"metadata"
                if re.search(pattern, file):
                    continue
                p = filepath+file
                with open(p, "rb") as f:
                    arr = np.array(pickle.load(f))
                    yield arr,arr, p
            except StopIteration:
                break;
            except tf.errors.OutOfRangeError: continue

