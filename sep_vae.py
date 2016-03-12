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

import vae

def build(P,
        x_size,x_layer_sizes,z_size,
        y_size,y_layer_sizes):

    X_encode, X_decode = vae.build(P,
            name="x_model",
            input_size=x_size,
            encoder_hidden_sizes=x_layer_sizes,
            latent_size=z_size,
        )

    Y_encode = vae.build_inferer(P,
            name="y_model_encoder", 
            input_sizes=[y_size],
            hidden_sizes=y_layer_sizes,
            output_size=z_size,
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
            initialise_outputs=True
        )

    Y_decode = feedforward.build_classifier(
            P, "y_model_decoder",
            input_sizes=[z_size],
            hidden_sizes=y_layer_sizes[::-1],
            output_size=y_size,
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
            output_activation=T.nnet.softmax
        )


    def training_cost(X,Y):
        Z, mean_Z, std_Z = X_encode([X])
        prior_Z, prior_mean_Z, prior_std_Z = Y_encode([Y])

        _, recon_mean_X, recon_std_X = X_decode([Z])
        _, recon_Y = Y_decode([prior_Z])

        prior_cost = vae.kl_divergence(prior_mean_Z, prior_std_Z,0,1)
        z_divergence = vae.kl_divergence(mean_Z, std_Z, prior_mean_Z, prior_std_Z)
        x_recon_cost = -vae.gaussian_nll(X, recon_mean_X, recon_std_X)
        y_recon_cost = T.nnet.categorical_crossentropy(recon_Y,Y)
        return T.mean(prior_cost),\
                T.mean(z_divergence),\
                T.mean(x_recon_cost),\
                T.mean(y_recon_cost)

    def classification_cost(X,Y):
        Z, mean_Z, std_Z = X_encode([X])
        _, recon_Y = Y_decode([Z])
        return T.mean(T.nnet.categorical_crossentropy(recon_Y,Y)),\
                T.mean(T.neq(T.argmax(recon_Y,axis=1),Y))

    return training_cost,classification_cost
