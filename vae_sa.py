import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

import feedforward
def initial_weights(input_size,output_size,factor=4):
	return (
			 0.1 * np.random.randn(input_size,output_size)
		).astype(np.float32)


def build(P, name,
		  input_size,
		  encoder_hidden_sizes,
		  latent_size,
		  speaker_count,
		  speaker_embedding_size,
		  decoder_hidden_sizes=None,
		  activation=T.tanh,
		  initial_weights=initial_weights):

	if decoder_hidden_sizes == None:
		decoder_hidden_sizes = encoder_hidden_sizes[::-1]

	encode = feedforward.build(P, "%s_encoder" % name,
								speaker_embedding_size,
								encoder_hidden_sizes,
								latent_size * 2,
								activation=activation,
								)

	decode = feedforward.build(P, "%s_decoder" % name,
								latent_size,
								decoder_hidden_sizes,
								speaker_embedding_size * 2,
								activation=activation,
								)
	P.W_vae_decoder_output.set_value(
			initial_weights(
				decoder_hidden_sizes[-1],
				speaker_embedding_size * 2
			)
		)

	P.W_input_embedding  = initial_weights(input_size,speaker_embedding_size)
	P.b_input_embedding  = np.zeros((speaker_embedding_size,),dtype=np.float32)

	P.W_embedding_recon_mean   = np.zeros((speaker_embedding_size,input_size))
	P.b_embedding_recon_mean   = np.zeros((input_size,))
	P.W_embedding_recon_logvar = np.zeros((speaker_embedding_size,input_size))
	P.b_embedding_recon_logvar = np.zeros((input_size,))

	P.encode_speaker_embedding = np.ones((speaker_count,speaker_embedding_size),dtype=np.float32)
	P.decode_speaker_embedding = np.ones((speaker_count,speaker_embedding_size),dtype=np.float32)

	encode_speaker_embedding = P.encode_speaker_embedding
	decode_speaker_embedding = P.decode_speaker_embedding
#	encode_speaker_embedding = P.encode_speaker_embedding /\
#			T.sqrt(T.sum(P.encode_speaker_embedding**2,axis=1)).dimshuffle(0,'x')
#	decode_speaker_embedding = P.decode_speaker_embedding /\
#			T.sqrt(T.sum(P.decode_speaker_embedding**2,axis=1)).dimshuffle(0,'x')

	def sample_encode(X,S):
		X_ = T.dot(X,P.W_input_embedding) + P.b_input_embedding
		X_S = X_ * P.encode_speaker_embedding[S]
		mean_logvar = encode(X_S)
		mean = mean_logvar[:, :latent_size]
		logvar = mean_logvar[:, latent_size:]
		e = U.theano_rng.normal(size=logvar.shape)
		latent = mean + e * T.exp(0.5 * logvar) # 0.5 * log std**2 = log std

		return mean, logvar, latent

	def recon_error(X,S,encoder_out=None):
		if encoder_out:
			mean,logvar,latent = encoder_out
		else:
			mean,logvar,latent = sample_encode(X,S)

		spkr_recon = decode(latent)
		spkr_recon_mean   = spkr_recon[:,:speaker_embedding_size] * P.decode_speaker_embedding[S]
		spkr_recon_logvar = spkr_recon[:,speaker_embedding_size:] * P.decode_speaker_embedding[S]
		recon_X_mean   = T.dot(spkr_recon_mean,P.W_embedding_recon_mean) + P.b_embedding_recon_mean
		recon_X_logvar = T.dot(spkr_recon_logvar,P.W_embedding_recon_logvar) + P.b_embedding_recon_logvar

		kl_divergence = -0.5 * T.sum(1 + logvar - mean**2 - T.exp(logvar), axis=1)
		log_p_x_z = - 0.5 * T.sum(recon_X_logvar+(X - recon_X_mean)**2/T.exp(recon_X_logvar),axis=1)
		cost = -(-kl_divergence + log_p_x_z)
		return recon_X_mean, T.mean(cost)
	return sample_encode,recon_error
