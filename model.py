import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config

def initial_weights(input_size,output_size,factor=4):
    return np.asarray(
        np.random.uniform(
            low  = -factor * np.sqrt(6. / (input_size + output_size)),
            high =  factor * np.sqrt(6. / (input_size + output_size)),
            size =  (input_size,output_size)
        ),
        dtype=theano.config.floatX
    )

def build_feedforward(P,input_size,layer_sizes,output_size):
    prev_layer_size = input_size
#    factor = 4 if config.hidden_activation == T.nnet.sigmoid else 1
#    factor = 1
    for i,curr_size in enumerate(layer_sizes):
        P["W_hidden_%d"%i] = initial_weights(prev_layer_size,curr_size,factor=4 if i>0 else 0.1)
        P["b_hidden_%d"%i] = np.zeros((curr_size,))
        prev_layer_size = curr_size
    P["W_output"] = np.zeros((layer_sizes[-1],output_size))
    P["b_output"] = np.zeros((output_size,))

    def feedforward(X):
        hidden_layers = [X]
        for i in xrange(len(layer_sizes)):
            layer = T.nnet.sigmoid(
                T.dot(hidden_layers[-1],P["W_hidden_%d"%i]) + P["b_hidden_%d"%i]
            )
            layer.name = "hidden_%d"%i
            hidden_layers.append(layer)
        output = T.nnet.softmax(T.dot(hidden_layers[-1],P["W_output"]) + P["b_output"])
        return hidden_layers,output
    return feedforward
