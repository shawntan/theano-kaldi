import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano.printing import Print
import feedforward
import vae


def build_speaker_inferer(P,method="average",
        x_size=440,
        speaker_latent_size=32,
        speaker_layer_sizes=[1024,1024]):
    if method == "average":
        frame_speaker_encode = vae.build_inferer(P,
                name="speaker_encoder", 
                input_sizes=[x_size],
                hidden_sizes=speaker_layer_sizes,
                output_size=speaker_latent_size,
                initial_weights=feedforward.relu_init,
                activation=T.nnet.softplus,
                initialise_outputs=True
            )

        def speaker_encode(X):
            _, frame_speaker_mean, frame_speaker_std = frame_speaker_encode([X])
            utterance_speaker_mean = T.mean(frame_speaker_mean,axis=1)
            utterance_speaker_std  = T.sqrt(T.mean(T.sqr(frame_speaker_std),axis=1))
            eps = U.theano_rng.normal(size=utterance_speaker_std.shape)
            utterance_speaker = utterance_speaker_mean + eps * utterance_speaker_std
            return utterance_speaker,utterance_speaker_mean, utterance_speaker_std

    else:
        input_transform = feedforward.build_transform(
            P,
            name="speaker_encoder_input",
            input_size=x_size,
            output_size=speaker_layer_sizes[0],
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
        )
        P.W_speaker_encoder_pool = feedforward.relu_init(
                speaker_layer_sizes[0],
                speaker_layer_sizes[0]
            )
        output = vae.build_inferer(P,
                name="speaker_encoder_pooled", 
                input_sizes=[speaker_layer_sizes[0]],
                hidden_sizes=[speaker_layer_sizes[1]],
                output_size=speaker_latent_size,
                initial_weights=feedforward.relu_init,
                activation=T.nnet.softplus,
                initialise_outputs=True
            )

        def speaker_encode(X):
            hidden_1 = input_transform(X)
            pooled = T.max(T.dot(hidden_1,P.W_speaker_encoder_pool),axis=1)
            return output([pooled]) 
             



    return speaker_encode 





def build_encoder(
        P,pooling_method="average",
        x_size=440,
        acoustic_latent_size=64,
        speaker_latent_size=32,
        speaker_layer_sizes=[1024,1024],
        acoustic_layer_sizes=[1024,1024]):

    speaker_encode = build_speaker_inferer(
            P,method=pooling_method,
            x_size=x_size,
            speaker_latent_size=speaker_latent_size,
            speaker_layer_sizes=speaker_layer_sizes
        )

    acoustic_encode = vae.build_inferer(
            P,name="acoustic_encoder", 
            input_sizes=[x_size,speaker_latent_size],
            hidden_sizes=acoustic_layer_sizes,
            output_size=acoustic_latent_size,
            initial_weights=feedforward.relu_init,
            activation=T.nnet.softplus,
            initialise_outputs=True
        )

    return speaker_encode, acoustic_encode


def build(P,
        pooling_method="average",
        x_size=440,
        acoustic_latent_size=64,
        speaker_latent_size=32,
        speaker_layer_sizes=[1024,1024],
        acoustic_layer_sizes=[1024,1024],
        decoder_layer_sizes=[2048,2048],
        speaker_count=83):

    P.speaker_vector = 0. * np.random.randn(speaker_count,speaker_latent_size).astype(np.float32)
    speaker_encode, acoustic_encode = build_encoder(
            P, pooling_method, x_size,
            acoustic_latent_size,
            speaker_latent_size,
            speaker_layer_sizes,
            acoustic_layer_sizes
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


    def unsupervised_training_cost(X):
        # X: batch_size, sequence_length, input_size

        # Get latent variables
        
        utterance_speaker,\
                utterance_speaker_mean,\
                utterance_speaker_std = speaker_encode(X)
        utterance_speaker = utterance_speaker.dimshuffle(0,'x',1)
        acoustic, acoustic_mean, acoustic_std = acoustic_encode([X,utterance_speaker])

        # Combine distributions for utterance

        # Reconstruct
        _, recon_X_mean, recon_X_std = decode([acoustic,utterance_speaker])

        acoustic_latent_cost = vae.kl_divergence(acoustic_mean, acoustic_std, 0, 1) # batch_size, sequence_length
        speaker_latent_cost = vae.kl_divergence(
                utterance_speaker_mean,utterance_speaker_std,
                0, 1 
            ) # batch_size
        recon_cost = vae.gaussian_nll(X, recon_X_mean, recon_X_std)

        batch_speaker_latent_cost  = T.mean(speaker_latent_cost,axis=0)
        batch_acoustic_latent_cost = T.mean(T.sum(acoustic_latent_cost,axis=1),axis=0)
        batch_reconstruction_cost  = T.mean(T.sum(recon_cost,axis=1),axis=0)
        
        return batch_speaker_latent_cost,\
                batch_acoustic_latent_cost,\
                batch_reconstruction_cost

    def supervised_training_cost(X,spkr_id):
        # X: batch_size, sequence_length, input_size

        # Get latent variables
        utterance_speaker_ = P.speaker_vector[spkr_id]
        utterance_speaker = utterance_speaker_.dimshuffle(0,'x',1)
        acoustic, acoustic_mean, acoustic_std = acoustic_encode([X,utterance_speaker])

        # Combine distributions for utterance

        # Reconstruct
        _, recon_X_mean, recon_X_std = decode([acoustic,utterance_speaker])

        acoustic_latent_cost = vae.kl_divergence(acoustic_mean, acoustic_std, 0, 1) # batch_size, sequence_length
        speaker_prior_cost = vae.gaussian_nll(utterance_speaker_, 0, 1)
        recon_cost = vae.gaussian_nll(X, recon_X_mean, recon_X_std)

        batch_acoustic_latent_cost = T.mean(T.sum(acoustic_latent_cost,axis=1),axis=0)
        batch_reconstruction_cost  = T.mean(T.sum(recon_cost,axis=1),axis=0)
        batch_speaker_prior_cost   = T.mean(speaker_prior_cost,axis=0)
        return batch_speaker_prior_cost,\
                batch_acoustic_latent_cost,\
                batch_reconstruction_cost


    return unsupervised_training_cost,supervised_training_cost

