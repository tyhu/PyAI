### rnn builder for keras
### Ting-Yao Hu, 2016.05

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import SimpleRNN, LSTM
from keras.layers import TimeDistributedDense
from keras.regularizers import l2, activity_l2, l1

### recurrent neural network mapping feature sequence to one label
class KerasRNN(object):
    
    def __init__(self, idim, hdim, odim, activation='tanh', activation_c='softmax', droprate=0.0, ld=0.01, rtype='rnn', s2s=False):
        model = Sequential()
        if rtype=='rnn':
            model.add(SimpleRNN(hdim, input_dim=idim, activation=activation,
                W_regularizer=l2(ld), U_regularizer=l2(ld), 
                b_regularizer=l2(ld), dropout_W=droprate, 
                dropout_U=droprate, return_sequences=s2s))
        elif rtype=='lstm':
            model.add(LSTM(hdim, input_dim=idim, activation=activation,
                W_regularizer=l2(ld), U_regularizer=l2(ld), 
                b_regularizer=l2(ld), dropout_W=droprate, 
                dropout_U=droprate, return_sequences=s2s))

        if s2s:
            model.add(TimeDistributedDense(odim))
        else:
            model.add(Dense(odim))
        model.add(Activation(activation_c))

        self.model = model

    def compile(self,loss='categorical_crossentropy', optimizer='rmsprop'):
        self.model.compile(loss=loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def get_model(self):
        return self.model

