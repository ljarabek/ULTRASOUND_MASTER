# LSTM Autoencoder in TensorFlow

import pickle
import os
import random
import re
import tensorflow as tf
import numpy as np
from generator import *


# In[2]:


train_path = '/media/tim/Elements/ultrazvok/files/train/'
val_path = '/media/tim/Elements/ultrazvok/files/val/'
test_path = '/media/tim/Elements/ultrazvok/files/test/'


list = os.listdir(train_path)
number_files = len(list)
print (number_files)


# In[4]:


batch_size = 64
epochs = 3
train_steps = round(number_files/batch_size)
train_steps


# In[5]:


# In[6]:


train_generator = batch_generator(train_path, batch_size=batch_size)
val_generator = batch_generator(val_path, batch_size=batch_size)


# In[7]:


print(next(train_generator)[0].shape)
print(next(val_generator)[1].shape)


# In[9]:


from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, TimeDistributed, Conv2DTranspose,Conv2D, ConvLSTM2D,Conv3D,LayerNormalization, Lambda


# In[10]:


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = Sequential([
            Input(shape=(30,32,32)),
            Lambda(lambda x: tf.expand_dims(x,axis=4)),
            TimeDistributed(Conv2D(64,(3,3),padding='SAME')),
            LayerNormalization(),
            ConvLSTM2D(128, (3,3), strides = (1,1), return_sequences=True),
            LayerNormalization(),
            ConvLSTM2D(64, (3,3), strides = (1,1), return_sequences=True)
        ], name = 'Encoder')
        self.decoder = Sequential([
            TimeDistributed(Conv2DTranspose(64,(3,3),padding="same")),
            LayerNormalization(),
            TimeDistributed(Conv2DTranspose(1,(3,3),activation='sigmoid',padding="same"))
        ], name = 'Decoder')        
    def call(self,x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
model = Autoencoder()
model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
model.build((64, 30, 32, 32))
model.summary()


# In[11]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
filepath = 'checkpoints'

early_stopper = EarlyStopping(monitor='val_loss',patience=3, min_delta=0.001)
reduce_lr = ReduceLROnPlateau(monitor='loss',patience=2, cooldown=1)
store_best = ModelCheckpoint(filepath = filepath, 
                             monitor = 'val_accuracy', 
                             save_best_only=True,
                             save_freq = 'epoch')

callbacks = [early_stopper, reduce_lr, store_best]


# In[ ]:


model.fit(train_generator, validation_data = val_generator, steps_per_epoch =train_steps,
callbacks=callbacks, verbose=2)


# In[ ]:



