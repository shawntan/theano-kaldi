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

def build_utterance_aware(P,
        x_size=440,
        acoustic_latent_size=64,
        speaker_latent_size=32,
        speaker_layer_sizes=[2048,1024],
        acoustic_layer_sizes=[2048,1024],
        decoder_layer_sizes=[2048,2048],
        speaker_count=83):

    P.speaker_means = 0. * np.random.randn(speaker_count,speaker_latent_size).astype(np.float32)
    P.speaker_stds = 0. * np.random.randn(speaker_count,speaker_latent_size).astype(np.float32) + 1
    speaker_means = P.speaker_means
    speaker_stds  = T.nnet.softplus(P.speaker_stds)

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
            initialise_outputs=False
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
        # X: batch_size, sequence_length, input_size

        # Get latent variables
        acoustic, acoustic_mean, acoustic_std = acoustic_encode([X])
        _, frame_speaker_mean, frame_speaker_std = speaker_encode([X])

        # Combine distributions for utterance
        utterance_speaker_mean = T.mean(frame_speaker_mean,axis=1)
        utterance_speaker_std  = T.sqrt(T.mean(T.sqr(frame_speaker_std),axis=1))
        eps = U.theano_rng.normal(size=utterance_speaker_std.shape)
        utterance_speaker = utterance_speaker_mean + eps * utterance_speaker_std

        # Reconstruct
        _, recon_X_mean, recon_X_std = decode([
                acoustic,
                utterance_speaker.dimshuffle(0,'x',1)
            ])

        batch_speaker_means = speaker_means[spkr_id]
        batch_speaker_stds = speaker_stds[spkr_id]

        acoustic_latent_cost = vae.kl_divergence(acoustic_mean, acoustic_std, 0, 1) # batch_size, sequence_length
        speaker_latent_cost = vae.kl_divergence(
                utterance_speaker_mean,utterance_speaker_std,
                batch_speaker_means, batch_speaker_stds
            ) # batch_size
        speaker_prior_cost = vae.kl_divergence(batch_speaker_stds, batch_speaker_stds, 0, 1)
        recon_cost = vae.gaussian_nll(X, recon_X_mean, recon_X_std)

        return T.mean(T.sum(acoustic_latent_cost,axis=1),axis=0),\
                T.mean(speaker_latent_cost,axis=0),\
                T.mean(speaker_prior_cost,axis=0),\
                T.mean(T.sum(recon_cost,axis=1),axis=0)

    return training_cost

