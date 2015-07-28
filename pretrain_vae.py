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
	labels_files = config.labels_files
	
	minibatch_size = config.minibatch

	P = Parameters()
	X = T.matrix('X')
	print config.layer_sizes
	_, recon_error = vae.build(P, "vae",
				config.input_size,
				config.layer_sizes,
				config.output_size,
				activation=T.nnet.sigmoid
			)
	_,cost = recon_error(X)
	
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	

	parameters = P.values()
	gradients  = T.grad(cost,wrt=parameters)
	pprint(sorted((p.name,p.get_value().shape)for p in parameters ))
	print "Compiling function...",
	train = theano.function(
			inputs = [start_idx,end_idx],
			outputs = cost,
			updates = updates.momentum(parameters,gradients),
			givens  = {
				X: X_shared[start_idx:end_idx],
			}
		)
	print "Done."

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
				#print loss
				total_loss  += loss
				total_count += 1
		print total_loss/total_count
	P.save(config.output_file)

	
