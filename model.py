import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano.printing import Print

def build(P,input_size,layer_sizes,output_size):
    state = {}
    state['training'] = True

    def activation(x):
        z = T.nnet.sigmoid(x)
        if state['training']:
            return T.switch(U.theano_rng.binomial(size=x.shape,p=0.5),0,z)
        else:
            return 0.5 * z

    def initial_weights(input_size,output_size,factor=4):
        return np.asarray(
          np.random.uniform(
             low  = -factor * np.sqrt(6 / (input_size + output_size)),
             high =  factor * np.sqrt(6 / (input_size + output_size)),
             size =  (input_size,output_size)
          ),
          dtype=theano.config.floatX
        )

    classifier = feedforward.build_classifier(
            P, "classifier",
            [input_size], layer_sizes, output_size,
            activation=activation,
            initial_weights=initial_weights
        )

    def predict(X,training=False):
        state['training'] = training
        hiddens, outputs = classifier([X])
        return hiddens, outputs
    return predict


if __name__ == "__main__":
    P = Parameters()
    classify = build(P,10,[5,5,5,5,5],10)
    X = T.matrix('X')
    f = theano.function(
            inputs=[X],
            outputs=[classify(X,training=True),classify(X,training=False)]
        )
    print f(np.eye(10).astype(np.float32))

