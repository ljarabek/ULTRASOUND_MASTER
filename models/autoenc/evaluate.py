#!/usr/bin/env python

import pickle
import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import logging
import argparse
import tensorflow

from generator import generator
from timeit import default_timer as timer
from tensorflow.keras.models import load_model


# In[2]:
base_path ='/media/tim/Elements/data/uz/'

train_path = base_path+ 'train/'
val_path = base_path+ 'val/'
test_path = base_path+ 'test/'

model = load_model('models')
train_generator = generator(train_path)
val_generator = generator(val_path)
test_generator = generator(test_path)

model.evaluate(test_generator)

#loss ->0.0431
# rmse -> 0.2

n = next(train_generator)
prediction = model.predict_generator(n)