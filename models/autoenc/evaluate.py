#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *

def evaluate(load=True):
    if load:
        model.load_model(model_dir)
    model.evaluate(test_generator)

#loss ->0.0431
# rmse -> 0.2