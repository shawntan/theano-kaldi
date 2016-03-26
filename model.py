import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def gaussian_ll(X, mean, std):
    return 0.5 * T.sum(
            np.log(2 * np.pi) + 2 * T.log(std) +
            T.sqr(X - mean) / T.sqr(std) , axis=-1
        )


def build(P,input_size,layer_sizes,output_size,training=True):
    classifier = feedforward.build_classifier(
            P, "classifier",
            [input_size], layer_sizes, output_size,
            activation=T.nnet.sigmoid
        )


    P.W_recon = 0.01 * np.random.randn(output_size,64)
    P.b_recon = np.zeros((64,),dtype=np.float32)
    P.W_X_mean = np.zeros((64,input_size),dtype=np.float32)
    P.b_X_mean = np.zeros((input_size,),dtype=np.float32)
    P.W_X_std = np.zeros((64,input_size),dtype=np.float32) + 0.6
    P.b_X_std = np.zeros((input_size,),dtype=np.float32)

    hidden = T.nnet.sigmoid(P.W_recon + P.b_recon)
    Y_X_mean = T.dot(hidden,P.W_X_mean) + P.b_X_mean
    Y_X_std = T.dot(hidden,P.W_X_std) + P.b_X_std


    def predict(X):
        _,outputs = classifier([X])

        recon_cost_per_class = gaussian_ll(
                X.dimshuffle(0,'x',1),  # batch_size x 1 x input_size
                Y_X_mean,               # 1 x output_size x input_size
                Y_X_std
            ) # batch_size x output_size

        regulariser = T.sum(outputs * T.log(outputs),axis=1)
        recon_cost = T.sum(outputs * recon_cost_per_class,axis=1)
        return outputs, recon_cost # - regulariser

    return predict

