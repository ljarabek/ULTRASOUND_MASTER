#!/usr/bin/env python
from tensorflow.keras.models import load_model
from autoencoder import get_model_history

# PLOT TRAINING HISTORY
history = get_model_history()

## PLOT MODEL TRAINING HISTORY
def plot(history=history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history['root_mean_squared_error'])
    plt.plot(history['val_root_mean_squared_error'])
    plt.title('Rmse by epoch')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

plot(history)