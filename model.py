import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def build(P,input_size,layer_sizes,output_size,training=True):
    classifier = feedforward.build_classifier(
            P, "classifier",
            [input_size], layer_sizes, output_size,
            activation=T.nnet.sigmoid
        )
    def predict(X):
        _,outputs = classifier([X])
        return outputs
    return predict

