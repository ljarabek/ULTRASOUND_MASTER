#!/usr/bin/env python
from tensorflow.keras.models import load_model
from config import *

def predict(data,load=True):
    if load:
        model = load_model(model_dir)
    return model.predict(data)
