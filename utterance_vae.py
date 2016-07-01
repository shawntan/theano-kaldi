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


activation_map = {
        "softplus" : T.nnet.softplus,
        "sigmoid" : T.nnet.sigmoid,
        "tanh" : T.tanh,
        "relu" : T.nnet.relu
    }

@config.option("activation_function","Activation function.", default="softplus")
def activation(x,activation_function):
    return activation_map[activation](x)

#activation = T.nnet.softplus

weight_init = lambda x,y: 0.1 * np.random.randn(x,y)

input_dimension = config.option("input_dimension","Size of input.",type=config.int)
acoustic_structure = config.option("acoustic_structure","Structure for acoustic portion.",type=config.structure)
speaker_structure = config.option("speaker_structure","Structure for speaker portion.",type=config.structure)
pooling_method = config.option("pooling_method","Method for pooling over utterance.")

@input_dimension
@speaker_structure
@pooling_method
def build_speaker_inferer(P,input_dimension,pooling_method,speaker_structure):
    x_size = input_dimension
    speaker_latent_size = speaker_structure[-1]
    speaker_layer_sizes = speaker_structure[:-1]
    if pooling_method == "average":
        frame_speaker_encode = vae.build_inferer(P,
                name="speaker_encoder", 
                input_sizes=[x_size],
                hidden_sizes=speaker_layer_sizes,
                output_size=speaker_latent_size,
                initial_weights=weight_init,
                activation=activation,
                initialise_outputs=True
            )

        def speaker_encode(X,mask):
            mask = mask.dimshuffle(0,1,'x')
            _, frame_speaker_mean, frame_speaker_std = frame_speaker_encode([X])
            utt_lengths = T.cast(T.sum(mask,axis=1),'float32').dimshuffle(0,'x')
            utterance_speaker_mean = \
                    T.sum(T.switch(mask,frame_speaker_mean,0) ,axis=1) \
                    / utt_lengths 
            utterance_speaker_std  = T.sqrt(
                    T.sum(T.switch(mask,T.sqr(frame_speaker_std),0),axis=1) \
                    / utt_lengths)
            eps = U.theano_rng.normal(size=utterance_speaker_std.shape)
            utterance_speaker = utterance_speaker_mean + eps * utterance_speaker_std
            return utterance_speaker,utterance_speaker_mean, utterance_speaker_std

    else:
        input_transform = feedforward.build_transform(
            P,
            name="speaker_encoder_input",
            input_size=x_size,
            output_size=speaker_layer_sizes[0],
            initial_weights=weight_init,
            activation=activation,
        )

        P.W_speaker_encoder_pool = weight_init(
                speaker_layer_sizes[0],
                speaker_layer_sizes[0]
            )


        if pooling_method == "max":
            output = vae.build_inferer(P,
                    name="speaker_encoder_pooled", 
                    input_sizes=[speaker_layer_sizes[0]],
                    hidden_sizes=[speaker_layer_sizes[1]],
                    output_size=speaker_latent_size,
                    initial_weights=weight_init,
                    activation=activation,
                    initialise_outputs=True
                )

            def speaker_encode(X,mask):
                mask = mask.dimshuffle(0,1,'x')
                hidden_1 = input_transform(X)
                features = T.dot(hidden_1,P.W_speaker_encoder_pool)
                pooled = T.max(T.switch(mask,features,-np.inf),axis=1)
                return output([pooled])

        elif pooling_method == "attention":
            P.w_attention = np.zeros((speaker_layer_sizes[0] + 1,),dtype=np.float32)
            output = vae.build_inferer(P,
                    name="speaker_encoder_pooled", 
                    input_sizes=[speaker_layer_sizes[0]],
                    hidden_sizes=[speaker_layer_sizes[1]],
                    output_size=speaker_latent_size,
                    initial_weights=weight_init,
                    activation=activation,
                    initialise_outputs=True
                )

            def speaker_encode(X,mask):
                hidden_1 = input_transform(X)
                softmask = T.switch(mask,
                    T.nnet.sigmoid(T.dot(hidden_1,P.w_attention[:-1]) + P.w_attention[-1]) + 1e-4,0)

                pooled = T.sum(T.dot(hidden_1,P.W_speaker_encoder_pool) * softmask.dimshuffle(0,1,'x'), axis=1) /\
                            T.sum(softmask,axis=1,keepdims=True)

                return output([pooled])
    
    def _speaker_encode(X,mask=None):
        if mask is None:
            mask = T.ones_like(X[:,:,0])
        return speaker_encode(X,mask)

             
    return _speaker_encode 

@input_dimension
@acoustic_structure
@speaker_structure
def build_encoder(P,input_dimension,acoustic_structure,speaker_structure):
    x_size = input_dimension
    acoustic_latent_size = acoustic_structure[-1]
    acoustic_layer_sizes = acoustic_structure[1:]
    speaker_latent_size = speaker_structure[-1]

    speaker_encode = build_speaker_inferer(P)
    acoustic_encode = vae.build_inferer(
            P,name="acoustic_encoder", 
            input_sizes=[x_size,speaker_latent_size],
            hidden_sizes=acoustic_layer_sizes,
            output_size=acoustic_latent_size,
            initial_weights=weight_init,
            activation=activation,
            initialise_outputs=True
        )

    return speaker_encode, acoustic_encode

@input_dimension
@acoustic_structure
@speaker_structure
@config.option("decoder_structure","Structure for decoder.",type=config.structure)
def build(P,input_dimension,acoustic_structure,speaker_structure,decoder_structure):
    x_size = input_dimension 
    decoder_layer_sizes = decoder_structure
    acoustic_latent_size = acoustic_structure[-1]
    speaker_latent_size = speaker_structure[-1]

#    P.speaker_vector = 0.01 * np.random.randn(speaker_count,speaker_latent_size).astype(np.float32)
    speaker_encode, acoustic_encode = build_encoder(P)

    decode = vae.build_inferer(P,
            name="decode",
            input_sizes=[acoustic_latent_size,speaker_latent_size],
            hidden_sizes=decoder_layer_sizes,
            output_size=x_size,
            initial_weights=weight_init,
            activation=activation,
            initialise_outputs=False
        )


    def unsupervised_training_cost(X,utt_lengths):
        # X: batch_size, sequence_length, input_size
        # utt_lengths: batch_size
        mask = T.arange(X.shape[1]).dimshuffle('x',0) < \
                utt_lengths.dimshuffle(0,'x')


        # Get latent variables
        
        utterance_speaker,\
                utterance_speaker_mean,\
                utterance_speaker_std = speaker_encode(X,mask)
        utterance_speaker = utterance_speaker_mean.dimshuffle(0,'x',1) +\
                            utterance_speaker_std.dimshuffle(0,'x',1) *\
                            U.theano_rng.normal(size=(
                                utterance_speaker_std.shape[0],
                                X.shape[1],
                                utterance_speaker_std.shape[1]
                            ))

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

        batch_speaker_latent_cost  = speaker_latent_cost
        batch_acoustic_latent_cost = T.sum(T.switch(mask,acoustic_latent_cost,0),axis=1)
        batch_reconstruction_cost  = T.sum(T.switch(mask,recon_cost,0),axis=1)
        
        return batch_speaker_latent_cost,\
                batch_acoustic_latent_cost,\
                batch_reconstruction_cost
#
#    def supervised_training_cost(X,spkr_id):
#        # X: batch_size, sequence_length, input_size
#
#        # Get latent variables
#        utterance_speaker_ = P.speaker_vector[spkr_id]
#        utterance_speaker = utterance_speaker_.dimshuffle(0,'x',1)
#        acoustic, acoustic_mean, acoustic_std = acoustic_encode([X,utterance_speaker])
#
#        # Combine distributions for utterance
#
#        # Reconstruct
#        _, recon_X_mean, recon_X_std = decode([acoustic,utterance_speaker])
#
#        acoustic_latent_cost = vae.kl_divergence(acoustic_mean, acoustic_std, 0, 1) # batch_size, sequence_length
#        speaker_prior_cost = vae.gaussian_nll(utterance_speaker_, 0, 1)
#        recon_cost = vae.gaussian_nll(X, recon_X_mean, recon_X_std)
#
#        batch_acoustic_latent_cost = T.mean(T.sum(acoustic_latent_cost,axis=1),axis=0)
#        batch_reconstruction_cost  = T.mean(T.sum(recon_cost,axis=1),axis=0)
#        batch_speaker_prior_cost   = T.mean(speaker_prior_cost,axis=0)
#        return batch_speaker_prior_cost,\
#                batch_acoustic_latent_cost,\
#                batch_reconstruction_cost
#
    return unsupervised_training_cost
