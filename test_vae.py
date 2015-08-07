import config
config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
config.parse_args()

import theano
import theano.tensor as T
import theano_toolkit

import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
from itertools import izip, chain


import theano_toolkit.utils   as U
from theano_toolkit.parameters import Parameters
import vae

from pprint import pprint
if __name__ == "__main__":
	frames_files = config.frames_files
	minibatch_size = config.minibatch

	X = T.matrix('X')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')

	def run_test(layer_count,activation,latent_size,learning_rate,initialisation):	
		P = Parameters()
		_, recon_error = vae.build(P, "vae",
					config.input_size,
					[1024] * layer_count,
					latent_size,
					activation=activation,
					initial_weights=initialisation
				)
		X_recon,cost = recon_error(X)
		X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
		

		parameters = P.values()
		loss = cost + 0.5 * sum(T.sum(w**2) for w in parameters)
		gradients  = T.grad(cost,wrt=parameters)
		#pprint(sorted((p.name,p.get_value().shape) for p in parameters ))
		train = theano.function(
				inputs = [start_idx,end_idx],
				outputs = T.mean(T.sum((X-X_recon)**2,axis=1)),
				updates = updates.adadelta(parameters,gradients,eps=learning_rate),
				givens  = {
					X: X_shared[start_idx:end_idx],
				}
			)

		for epoch in xrange(config.max_epochs):	
			split_streams = [ data_io.stream(f) for f in frames_files ]
			stream = chain(*split_streams)
			total_count = 0
			total_loss  = 0
			for f,size in data_io.randomise(stream):
				X_shared.set_value(f)
				batch_count = int(math.ceil(size/float(minibatch_size)))
				for idx in xrange(batch_count):
					start = idx*minibatch_size
					end = min((idx+1)*minibatch_size,size)
					loss = train(start,end)
					if np.isnan(loss): return loss
	#				print loss
					total_loss  += loss
					total_count += 1
			avg_loss = total_loss/total_count
		return avg_loss

	def uniform_weights(factor):
		def init(input_size,output_size):
			return (
					factor * ( 2 * np.random.rand(input_size,output_size) - 1.0)
				).astype(np.float32)
		return init

	def gaussian_weights(factor):
		def init(input_size,output_size):
			return (
					factor * np.random.randn(input_size,output_size)
				).astype(np.float32)
		return init

	def special_weight(input_size,output_size):
		return np.asarray(
			np.random.uniform(
				low  = - np.sqrt(6. / (input_size + output_size)),
				high =   np.sqrt(6. / (input_size + output_size)),
				size =  (input_size,output_size)
			),
			dtype=theano.config.floatX
		)




	activations = [
			('relu',lambda x: (x > 0) * x),
			('sigmoid',T.nnet.sigmoid),
			('softplus',T.nnet.softplus),
			('tanh',T.tanh)
		]
	initialisations = [
		('uniform_%s'%str(10**-f),uniform_weights(10**-f)) 
		for f in xrange(0,4)
	] + [ 
		('gaussian_%s'%str(10**-f),gaussian_weights(10**-f)) 
		for f in xrange(0,4)
	] + [ ('special_weight',special_weight) ]



	best_loss = np.inf
	best_comb = None
	for act_name,activation in activations:
		for latent_size in [ 2 ** (6 + i) for i in xrange(5)]: 
			for layer_count in xrange(1,4):
				for learning_rate in  [ 10 ** -p for p in xrange(8,10) ]:
					for init_name,initialisation in initialisations:
						comb = layer_count,act_name,learning_rate,latent_size,init_name
						print comb,
						loss = run_test(layer_count,activation,latent_size,learning_rate,initialisation)
						print loss
						if loss < best_loss:
							best_loss = loss
							best_comb = comb
	print "Best comb:",best_comb
