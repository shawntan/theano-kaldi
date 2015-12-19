import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def gaussian_nll(X, mean, std):
    return - 0.5 * T.sum(
            np.log(2 * np.pi) + 2 * T.log(std) +
            T.sqr(X - mean) / T.sqr(std) , axis=-1
        )


def build(P,input_size,layer_sizes,output_size,training=True):
    classifier = feedforward.build_classifier(
            P, "classifier",
            [input_size], layer_sizes, output_size,
            activation=T.nnet.sigmoid
        )

    P.W_Y_X_mean = np.zeros(output_size,input_size,dtype=np.float32)
    P.W_Y_X_std = np.zeros(output_size,input_size,dtype=np.float32)

    Y_X_mean = P.W_Y_X_mean.dimshuffle('x',0,1)
    Y_X_std = T.nnet.softplus(P.W_Y_X_std).dimshuffle('x',0,1)

    def predict(X):
        _,outputs = classifier([X])

        recon_cost_per_class = gaussian_nll(
                X.dimshuffle(0,'x',1),  # batch_size x 1 x input_size
                Y_X_mean,               # 1 x output_size x input_size
                Y_X_std
            ) # batch_size x output_size

        recon_cost = T.sum(outputs * recon_cost_per_class,axis=1)

        return outputs, recon_cost

    return predict

