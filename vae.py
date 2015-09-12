import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

import feedforward

def build(P, name,
            input_size,
            encoder_hidden_sizes,
            latent_size,
            decoder_hidden_sizes=None,
            activation=T.nnet.softplus,
            initial_weights=feedforward.relu_init):

    if decoder_hidden_sizes == None:
        decoder_hidden_sizes = encoder_hidden_sizes[::-1]

    encode = build_inferer(P,"%s_encoder"%name,
            [input_size],
            encoder_hidden_sizes,
            latent_size,
			activation=activation,
            initial_weights=initial_weights
        )
    decode = build_inferer(P,"%s_decoder"%name,
            [latent_size],
            decoder_hidden_sizes,
            input_size,
			activation=activation,
            initial_weights=initial_weights)

    def recon_error(X,encode=encode,decode=decode):
        Z_latent, Z_mean, Z_logvar = encode([X])
        _, recon_X_mean, recon_X_logvar = decode([Z_latent])
        KL_d = kl_divergence(Z_mean,Z_logvar)
        log_p_X = gaussian_log(recon_X_mean,recon_X_logvar,X)
        cost = -(log_p_X - KL_d)
        return recon_X_mean,T.mean(cost),T.mean(KL_d),T.mean(log_p_X)
    return encode,decode,recon_error

def gaussian_log(mean,logvar,X):
    return - 0.5 * T.sum(
        np.log(2 * np.pi) + logvar +\
                T.sqr(X - mean)/T.exp(logvar),axis=-1)

def kl_divergence(mean, logvar):
    return -0.5 * T.sum(1 + logvar - T.sqr(mean) - T.exp(logvar), axis=-1)

def build_inferer(P,name,input_sizes,hidden_sizes,output_size,
        initial_weights,activation):
    combine_inputs = feedforward.build_combine_transform(
            P,"%s_input"%name,
            input_sizes,hidden_sizes[0],
            initial_weights=initial_weights,
            activation=activation
        )
    transform = feedforward.build_stacked_transforms(
            P,name,hidden_sizes,
            initial_weights=initial_weights,
            activation=activation)
    output = build_encoder_output(
            P,name,
            hidden_sizes[-1],output_size,
        )
    return lambda Xs:output(transform(combine_inputs(Xs))[-1])

def build_encoder_output(P,name,input_size,output_size):
    P["W_%s_mean"%name]   = np.zeros((input_size,output_size))
    P["b_%s_mean"%name]   = np.zeros((output_size,))
    P["W_%s_logvar"%name] = np.zeros((input_size,output_size))
    P["b_%s_logvar"%name] = np.zeros((output_size,))
    def output(X):
        mean   = T.dot(X,P["W_%s_mean"%name]) + P["b_%s_mean"%name]
        logvar = T.dot(X,P["W_%s_logvar"%name]) + P["b_%s_logvar"%name]
        eps = U.theano_rng.normal(size=logvar.shape)
        latent = mean + eps * T.exp(0.5 * logvar)
        return latent, mean, logvar
    return output
