import numpy as np
import theano
import theano.tensor as T

def variance_inverse(covariance_tensor):
	inverted,_ = theano.map(
			T.nlinalg.matrix_inverse,
			sequences = [covariance_tensor]
		)
	return inverted

def variance_determinant(covariance_tensor):
	determinants,_ = theano.map(
			T.nlinalg.det,
			sequences = [covariance_tensor]
			)
	return determinants


def build(params,name,means,rows,cols):
	n_hidden = rows * cols
	phoneme_means = theano.shared(means)  									# phonemes x components x 2
	phoneme_var   = theano.shared(np.array(means.shape[0] * [ np.eye(2) ])) # phonemes x components x 2 x 2
	phoneme_precisions = variance_inverse(phoneme_var)
	phoneme_determinants = variance_determinant(phoneme_var) # phonemes x components

	points = theano.shared(np.dstack(np.meshgrid(
	    np.arange(cols),np.arange(rows)
	)).reshape(n_hidden,2))                                            # n_hidden x 2

	phoneme_means_      = phoneme_means.dimshuffle(0,'x',1)         # phonemes x 1 x 2
	phoneme_precisions_ = phoneme_precisions.dimshuffle(0,'x',1,2)  # phonemes x 1 x 2 x 2
	points_ = points.dimshuffle('x',0,1)                            # 1 x n_hidden x 2
	deviations = points_ - phoneme_means_                           # phonemes x n_hidden x 2
	deviations_ = deviations.dimshuffle(0,1,'x',2)                  # phonemes x n_hidden x 1 x 2
	normalised_deviations = T.sum(phoneme_precisions_ * deviations_,axis=2) # phonemes x n_hidden x 2 
	score = T.sum(normalised_deviations**2,axis=2)         # phonemes  x n_hidden
	gaussians = T.exp(- np.float32(0.5) * score)\
			/ (T.sqrt(np.float32(2 * (np.pi**2)) * phoneme_determinants)).dimshuffle(0,'x')

	def constraint(hidden,ctx):
		# hidden - batch_size x n_hidden
		# ctx - batch_size x 3
		hidden = hidden /\
				T.sum(hidden,axis=1).dimshuffle(0,'x')
		frame_gaussians = gaussians[ctx] # batch_size x 3 x n_hidden
		ctx_surface = T.sum(frame_gaussians,axis=1) # batch_size x n_hidden
		ctx_surface = ctx_surface /\
				T.sum(ctx_surface,axis=1).dimshuffle(0,'x') # batch_size x n_hidden
		kl_divergence = -T.sum(ctx_surface * T.log(hidden),axis=1)
		return T.mean(kl_divergence)
	return constraint
