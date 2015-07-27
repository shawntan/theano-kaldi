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
          activation=T.tanh):

    if decoder_hidden_sizes == None:
        decoder_hidden_sizes = encoder_hidden_sizes[::-1]

    encoder = feedforward.build(P, "%s_encoder" % name,
                                [input_size], encoder_hidden_sizes, latent_size * 2,
                                activation=activation
                                )

    decoder = feedforward.build(P, "%s_decoder" % name,
                                [latent_size], decoder_hidden_sizes, input_size,
                                activation=activation
                                )

    def encode_decode(X):
        mean_logvar = encoder([X])
        mean = mean_logvar[:, :latent_size]
        logvar = mean_logvar[:, latent_size:]

        e = U.theano_rng.normal(size=logvar.shape)
        latent = mean + e * T.exp(0.5 * logvar) # 0.5 * log std**2 = log std

        recon_X = decoder([latent])

        cost = -(
            0.5 * T.sum(1 + logvar - mean**2 - T.exp(logvar), axis=1) -
            0.5 * T.sum((recon_X - X)**2, axis=1)
        )

        return mean, logvar, recon_X, T.mean(cost)
    return encode_decode
