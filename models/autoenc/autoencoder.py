#!/usr/bin/env python

# LSTM Autoencoder in TensorFlow
import pickle
import os
import random
import sys
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import logging
import argparse

from generator import generator
from timeit import default_timer as timer
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Reshape, TimeDistributed, Conv2DTranspose,Conv2D, ConvLSTM2D,Conv3D,LayerNormalization, Lambda
from tensorflow.keras.models import load_model

from generator import generator
from config import *

# DATA GENERATORS
train_generator = generator(train_path)
val_generator = generator(val_path)
test_generator = generator(test_path)

## PARSER FOR HYPERPARAMETERS
def parseArguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Training Parser")
    parser.add_argument("-epochs", type=int, default=epochs,
                        help="Number of epochs for training.")
    parser.add_argument("-batch_size", type=int, default=batch_size,
                        help="Set the training batch size.")
    parser.add_argument("-filepath", default=base_path,
                        help="Where you keep your data. ")                    
    parser.add_argument("-model_dir", default=model_dir,
                        help="Set where you want your model saved. ")
    args = parser.parse_args(args)
    return args

# LOGS
logging.basicConfig(filename=log_path,level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# GPU REQUIRED TENSORFLOW DISTRIBUTED TRAINING
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


## AUTOENCODER
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = Sequential([
            Input(shape=input_shape),
            Reshape((1,30,32,32)),
            TimeDistributed(Conv2D(32,(3,3),padding='SAME')),
            TimeDistributed(Conv2D(128,(3,3),padding='SAME')),
            LayerNormalization(),
            ConvLSTM2D(64, (3,3), strides = (1,1),padding='SAME', return_sequences=True),
            LayerNormalization(),
            ConvLSTM2D(8, (3,3), strides = (1,1),padding='SAME', return_sequences=True)
        ], name = 'Encoder')
        self.decoder = Sequential([
            TimeDistributed(Conv2DTranspose(32,(3,3),padding='SAME')),
            LayerNormalization(),
            TimeDistributed(Conv2DTranspose(32,(3,3),padding='SAME')),
        ], name = 'Decoder')        
    def call(self,x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

def initAutoencoder():
    model = Autoencoder()
    model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.build((64, 30, 32, 32))
    model.summary()
    return model
    
model = initAutoencoder()

## CALLBACKS
class BatchLogger(Callback):
    def __init__(self,logs={}):
        self.logs=[]
    def on_train_batch_begin(self,batch, logs=None):
        self.starting_time = timer()
    def on_train_batch_end(self,batch,logs=None):
        print("\nEnd of batch: {} \nLoss: {} \nTime needed: {}".format(batch, logs['loss'], timer()-self.starting_time))

batch_logger = BatchLogger()
early_stopper = EarlyStopping(monitor='val_loss',patience=3, min_delta=0.01)
reduce_lr = ReduceLROnPlateau(monitor='loss',patience=2, cooldown=1)
store_best = ModelCheckpoint(filepath = model_dir, 
                             monitor = 'loss', 
                             save_best_only=True,
                             save_freq = 'epoch')
callbacks = [store_best, early_stopper, reduce_lr, batch_logger]

## INIT PARSER
args = parseArguments()
history = None


## TRAIN
def train(restore=False, filename=model_dir, args=args):
    if restore: 
        model = load_model(filename)
    history = model.fit(train_generator, validation_data = val_generator,epochs=args.epochs,callbacks=callbacks,batch_size=args.batch_size, verbose=3)
    model.save_weights(filename)

train(restore=True)
def get_model_history():
    return pd.DataFrame(history.history)
