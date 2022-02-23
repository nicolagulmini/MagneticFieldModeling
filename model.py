from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dot
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class neural_network:
    
    def __init__(self):

        position = Input(shape=(3), name='input_position')
        orientation = Input(shape=(3), name='input_orientation')
        
        x = Dense(3, activation='linear', name='magnetic_field_components')(position) 
        x = RBFLayer(3, .5, name='rbf')(x) # 3 is the dimension, .5 means the 2 at denominator of the exponent 
        x = Dot(axes=1, name='output')([x, orientation])
        
        model = Model(inputs=[position, orientation], outputs=x)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mse'])
        
        self.model = model
