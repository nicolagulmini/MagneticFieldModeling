from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class neural_network:
    
    def __init__(self, rbf=False, lr=.001):
        
        self.lr = .001
        
        position = Input(shape=(3), name='input_position')
        orientation = Input(shape=(3), name='input_orientation')
        
        # try to increase the size of the neural network
        x = Dense(32, activation='sigmoid', name='intermediate_layer')(position)
        # only with this layer
        
        x = Dense(3, activation='linear', name='magnetic_field_components')(x) 
        '''
        if rbf:
            x = RBFLayer(3, .5, name='rbf')(x) # 3 is the dimension, .5 means the 2 at denominator of the exponent 
        '''
        x = Dot(axes=1, name='output')([x, orientation])
        
        model = Model(inputs=[position, orientation], outputs=x)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse', metrics=['mse'])
        
        self.model = model
        self.magnetic_field_components_predictor = Model(inputs=model.get_layer('input_position').input, outputs=model.get_layer('magnetic_field_components').output)
        
    def train(self, x_pos, x_or, y, verbose=0, max_epochs=200, patience_for_early_stopping=10):
        # train - validation division at the moment, no dataset are saved inside the class
        x_pos_train = x_pos[:int(.75*x_pos.shape[0])]
        x_orientation_train = x_or[:int(.75*x_or.shape[0])]
        y_train = y[:int(.75*y.shape[0])]

        x_pos_val = x_pos[int(.75*x_pos.shape[0]):]
        x_orientation_val = x_or[int(.75*x_or.shape[0]):]
        y_val = y[int(.75*y.shape[0]):]
        
        history = self.model.fit(x=[x_pos_train, x_orientation_train], 
                       y=y_train,
                       validation_data=([x_pos_val, x_orientation_val], y_val), 
                       verbose=verbose, 
                       epochs=max_epochs, 
                       callbacks=[EarlyStopping(patience=patience_for_early_stopping)])
        
        self.last_history = history
        
    def info(self):
        self.model.summary()
