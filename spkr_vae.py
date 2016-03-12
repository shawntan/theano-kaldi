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
        x_size=440,
        acoustic_latent_size=2,
        speaker_latent_size=2,
        speaker_layer_sizes=[1024,1024],
        acoustic_layer_sizes=[2048,2048],
        decoder_layer_sizes=[2048,2048],
        speaker_count=83):

    P.speaker_means = np.random.randn(speaker_count,speaker_latent_size).astype(np.float32)
    speaker_means = P.speaker_means

    acoustic_encode = vae.build_inferer(P,
            name="acoustic_encoder", 
            input_sizes=[x_size],
            hidden_sizes=acoustic_layer_sizes,
            output_size=acoustic_latent_size,
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
            initialise_outputs=True
        )

    speaker_encode = vae.build_inferer(P,
            name="speaker_encoder", 
            input_sizes=[x_size],
            hidden_sizes=speaker_layer_sizes,
            output_size=speaker_latent_size,
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
            initialise_outputs=True
        )

    decode = vae.build_inferer(P,
            name="decode",
            input_sizes=[acoustic_latent_size,speaker_latent_size],
            hidden_sizes=decoder_layer_sizes,
            output_size=x_size,
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
            initialise_outputs=False
        )



    def training_cost(X,spkr_id):
        acoustic, acoustic_mean, acoustic_std = acoustic_encode([X])
        frame_speaker, frame_speaker_mean, frame_speaker_std = speaker_encode([X])
        _, recon_X_mean, recon_X_std = decode([acoustic,frame_speaker])

        acoustic_latent_cost = vae.kl_divergence(acoustic_mean, acoustic_std, 0, 1)

        batch_speaker_means = speaker_means[spkr_id]
        speaker_latent_cost = vae.kl_divergence(frame_speaker_mean, frame_speaker_std, batch_speaker_means, 1)
        speaker_prior_cost = vae.gaussian_nll(batch_speaker_means, 0, 1)
        recon_cost = vae.gaussian_nll(X, recon_X_mean, recon_X_std)
        return T.mean(acoustic_latent_cost),\
                T.mean(speaker_latent_cost),\
                T.mean(speaker_prior_cost),\
                T.mean(recon_cost)

    def speaker_vectors(X):
        frame_speaker, frame_speaker_mean, frame_speaker_std = speaker_encode([X])
        return frame_speaker_mean

    def acoustic_vectors(X):
        acoustic, acoustic_mean, acoustic_std = acoustic_encode([X])
        return acoustic_mean 
 
 
    return training_cost,speaker_vectors,acoustic_vectors
