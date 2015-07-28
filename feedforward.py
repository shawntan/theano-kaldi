import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters


def initial_weights(input_size,output_size,factor=4):
    return np.asarray(
        np.random.uniform(
            low  = -factor * np.sqrt(6. / (input_size + output_size)),
            high =  factor * np.sqrt(6. / (input_size + output_size)),
            size =  (input_size,output_size)
        ),
        dtype=theano.config.floatX
    )

def build(P, name,
          input_size, hidden_sizes, output_size,
          activation=T.tanh):

    hidden_weights = []
    prev_size = input_size
    for i,size in enumerate(hidden_sizes):
        P["W_%s_hidden_%d" % (name, i)] = initial_weights(
				prev_size, size,
			)
        P["b_%s_hidden_%d" % (name, i)] = np.zeros((size,), dtype=np.float32)
        hidden_weights.append((P["W_%s_hidden_%d" % (name, i)], P["b_%s_hidden_%d" % (name, i)]))
        prev_size = size

    P["W_%s_output" % name] = np.zeros((prev_size, output_size), dtype=np.float32)
    P["b_%s_output" % name] = np.zeros((output_size,), dtype=np.float32)
    W_output = P["W_%s_output" % name]
    b_output = P["b_%s_output" % name]

    def feedforward(X):
        hidden = X
        for W, b in hidden_weights:
            hidden = activation(T.dot(hidden, W) + b)
        output = T.dot(hidden, W_output) + b_output
        return output
    return feedforward
