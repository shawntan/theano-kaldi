import config
config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
config.file_sequence("frames_files",".pklgz file containing audio frames.")
config.structure("generative_structure","Structure of generative model.")
config.file("validation_frames_file","Validation set.")
config.file("output_file","Output file.")
config.integer("minibatch","Minibatch size.",default=128)
config.integer("max_epochs","Maximum number of epochs to train.",default=20)
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
	frames_files = config.args.frames_files
	
	minibatch_size = config.args.minibatch

	print config.args.generative_structure
	input_size  = config.args.generative_structure[0]
	layer_sizes = config.args.generative_structure[1:-1]
	output_size = config.args.generative_structure[-1]
	
	X = T.matrix('X')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))

	def test_validation(test):
		total_errors = 0
		total_frames = 0
		for f in data_io.stream(config.args.validation_frames_file):
			f = f[0]
			errors = np.array(test(f))
			total_frames += f.shape[0]
			total_errors += f.shape[0] * errors
		return total_errors/total_frames

	def train_epoch(train):
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
#				total_loss  += loss
#				total_count += 1
#		print total_loss/total_count

	prev_P = None
	train = None
	for layer in xrange(len(layer_sizes)):
		P = Parameters()
		_, recon_error = vae.build(P, "vae",
					input_size,
					layer_sizes[:layer+1],
					output_size,
					activation=T.nnet.sigmoid
				)
	
		if layer > 0:
			print "decoder_output to decoder_output"
			P.W_vae_decoder_output.set_value(prev_P.W_vae_decoder_output.get_value())
			P.b_vae_decoder_output.set_value(prev_P.b_vae_decoder_output.get_value())
			for i in xrange(layer):
				print "encoder_%d to encoder_%d"%(i,i)
				P["W_vae_encoder_hidden_%d"%i].set_value(prev_P["W_vae_encoder_hidden_%d"%i].get_value())
				P["b_vae_encoder_hidden_%d"%i].set_value(prev_P["b_vae_encoder_hidden_%d"%i].get_value())
				if i > 0:
					print "decoder_%d to decoder_%d"%(i,i+1)
					P["W_vae_decoder_hidden_%d"%(i+1)].set_value(prev_P["W_vae_decoder_hidden_%d"%i].get_value())
					P["b_vae_decoder_hidden_%d"%(i+1)].set_value(prev_P["b_vae_decoder_hidden_%d"%i].get_value())


		parameters = P.values()
		
		X_recon,cost = recon_error(X)
		loss = cost + 0.5 * sum(T.sum(w**2) for w in parameters)
		gradients  = T.grad(cost,wrt=parameters)
		pprint(sorted((p.name,p.get_value().shape) for p in parameters ))
		print "Compiling function...",
		train = theano.function(
				inputs = [start_idx,end_idx],
				updates = updates.adadelta(parameters,gradients,eps=1e-8),
				givens  = {
					X: X_shared[start_idx:end_idx],
				}
			)
		test = theano.function(
				inputs = [X],
				outputs = [T.mean(T.sum((X-X_recon)**2,axis=1))],
			)
		print "Done."
		for _ in xrange(5):
			train_epoch(train)
			print test_validation(test)
		prev_P = P

	for epoch in xrange(config.args.max_epochs-5):
		train_epoch(train)
		print test_validation(test)
	P.save(config.args.output_file)
