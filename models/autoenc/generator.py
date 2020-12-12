import pickle
import os
import random
import re
import numpy as np
import tensorflow as tf

train_path = '/media/tim/Elements/ultrazvok/files/train/'
val_path = '/media/tim/Elements/ultrazvok/files/val/'
test_path = '/media/tim/Elements/ultrazvok/files/test/'


def batch_generator(filepath, batch_size=32):
    batch = []
    for _, _, files in os.walk(filepath):
        for file in files:
            try:
                pattern = r"metadata"
                if re.search(pattern, file):
                    continue
              # yield file
                with open(filepath + file, "rb") as file:
                    arr = pickle.load(file)
                    batch.append(np.array(arr))
                    if(len(batch)==batch_size):
                        yield tf.convert_to_tensor(batch),tf.convert_to_tensor(batch)
                        batch.clear()
            except StopIteration:
                break;
