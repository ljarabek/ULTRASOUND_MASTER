import tensorflow

from tensorflow.keras import Model, Input, Sequential, Layer
from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LayerNormalization


# batch dimensions as inputs, dictionary 
# layer normalization normalizes the activations for each example in a batch 
# instead of across a batch like batchnorm. 
# timedistributed is a wrapper for each slice of the input, 5D
# conv2dlstm, consider dropout, stateful and padding
# namesto time distributed pa conv2D lahko Conv3D

class TransposeofConvLSTM(Layer):
    def __init__(self, units=32, input_dim=(None,500,300)):
        super(TransposeofConvLSTM,self).__init__()
        self.units = units;
    def call(self, inputs, layer_filters):
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(16, (3,3), strides = 1, activation='relu')(inputs)
        return tf.nn.relu(x);

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = Sequential([
            TimeDistributed(Conv2D(64, (7,7),strides = 2, padding="same",batch_input_shape=(None, 10, 500, 300, 1))),
            LayerNormalization(),
            ConvLSTM2D(128, (3,3), strides = (1,1), return_sequences=True),
            ConvLSTM2D(64, (3,3), strides = (1,1), return_sequences=True)
        ])
        self.decoder = Sequential([
            TransposeofConvLSTM(32),
            LayerNormalization(),
            TimeDistributed(Conv2DTranspose(64,(7,7),padding="same")),
            TimeDistributed(Conv2DTranspose(1,(7,7),activation='sigmoid',padding="same"))
        ])        
    def call(self,x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


model = Autoencoder()
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')
model.fit_generator(train_generator, steps_per_epoch=train_steps, verbose=2)