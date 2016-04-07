### mlp builder for keras
### Ting-Yao Hu, 2016.04

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD

class KerasMLP(object):

    ### constructor: assign network structure, activation functions and dropout rate
    def __init__(self, layer_sizes, activates, droprate=0.5, ld=0.01):
        model = Sequential()

        ### first hidden layers
        model.add(Dense(layer_sizes[1], W_regularizer=l2(ld), input_dim=layer_sizes[0], init='uniform'))
        model.add(Activation(activates[0]))
        model.add(Dropout(droprate))

        ### hidden layers
        for idx in range(len(layer_sizes)-3):
            model.add(Dense(layer_sizes[idx+2], W_regularizer=l2(ld), init='uniform'))
            model.add(Activation(activates[idx+1]))
            model.add(Dropout(droprate))

        ### output layer
        model.add(Dense(layer_sizes[-1], W_regularizer=l2(ld), init='uniform'))
        model.add(Activation(activates[-1]))

        self.model = model

    
    def compile(self,loss='categorical_crossentropy', optimizer='adadelta'):
        self.model.compile(loss=loss,optimizer=optimizer)



if __name__=='__main__':
    mlp = KerasMLP([45,64,64,2],['tanh','tanh','softmax'])
    mlp.compile()
