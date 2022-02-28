from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

class neural_network:
    
    def __init__(self, rbf=False, learning_rate=.001, loss='mse'):
        
        self.lr = learning_rate
        self.loss = loss
        
        position = Input(shape=(3), name='input_position')
        orientation = Input(shape=(3), name='input_orientation')
        
        x = Dense(30, activation='sigmoid',
                    bias_regularizer=regularizers.l2(1e-3),
                    kernel_regularizer=regularizers.l2(1e-3),
                    kernel_initializer="random_normal",
                    use_bias=True,
                    bias_initializer='zeros',
                    name='intermediate_layer')(position)
        
        x = Dropout(.3)(x)
                
        x = Dense(3, activation='linear',
                    kernel_regularizer=regularizers.l2(1e-3),
                    kernel_initializer="random_normal",
                    use_bias=False,
                    name='magnetic_field_components')(x) 
        '''
        if rbf:
            x = RBFLayer(3, .5, name='rbf')(x) # 3 is the dimension, .5 means the 2 at denominator of the exponent 
        '''
        x = Dot(axes=1, name='output')([x, orientation])
        
        model = Model(inputs=[position, orientation], outputs=x)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss, metrics=[self.loss])
        
        self.model = model
        self.magnetic_field_components_predictor = Model(inputs=model.get_layer('input_position').input, outputs=model.get_layer('magnetic_field_components').output)
        
    def train(self, x_pos, x_or, y, verbose=0, max_epochs=200, batch=32, patience_for_early_stopping=10):
        # train - validation division at the moment, no dataset are saved inside the class
        x_pos_train = x_pos[:int(.75*x_pos.shape[0])]
        x_orientation_train = x_or[:int(.75*x_or.shape[0])]
        y_train = y[:int(.75*y.shape[0])]

        x_pos_val = x_pos[int(.75*x_pos.shape[0]):]
        x_orientation_val = x_or[int(.75*x_or.shape[0]):]
        y_val = y[int(.75*y.shape[0]):]
        
        history = self.model.fit(x=[x_pos_train, x_orientation_train], 
                       y=y_train,
                       batch_size=batch,
                       validation_data=([x_pos_val, x_orientation_val], y_val), 
                       verbose=verbose, 
                       epochs=max_epochs, 
                       callbacks=[EarlyStopping(patience=patience_for_early_stopping)])
        
        self.last_history = history
        
    def info(self):
        self.model.summary()
        
    def print_training_performance(self, save=False):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.last_history.history[self.loss])), self.last_history.history[self.loss], ls='--', color='green', label="train "+self.loss)
        plt.plot(range(len(self.last_history.history["val_"+self.loss])), self.last_history.history["val_"+self.loss], ls='--', color='red', label="validation "+self.loss)
        plt.legend(loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel(self.loss)
        plt.grid(linewidth=.5, color='gray')
        plt.title('Training performance')
        if save:
            plt.savefig('training_performance.png')
        else:
            plt.show()
            
class tridim_nn:
    
    def __init__(self):
        
        position = Input(shape=(3), name='input_position')
        
        x = Dense(200, activation='sigmoid')(position)
        x = Dropout(.1)(x)
        x = Dense(200, activation='sigmoid')(x)
        x = Dropout(.2)(x)
        x = Dense(200, activation='sigmoid')(x)
        x = Dropout(.3)(x)
        x = Dense(200, activation='sigmoid')(x)
        x = Dropout(.4)(x)
        x = Dense(200, activation='sigmoid')(x)
        x = Dropout(.5)(x)
        x = Dense(3, activation='linear')(x)
        
        model = Model(inputs=position, outputs=x)
        model.compile(optimizer=Adam(learning_rate=.001), loss='mse', metrics=['mse'])
        
        self.model = model
