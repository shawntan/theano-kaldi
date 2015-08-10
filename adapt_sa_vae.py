
import config
if __name__ == "__main__":
	config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
	config.file_sequence("frames_files",".pklgz file containing audio frames.")
	config.structure("generative_structure","Structure of generative model.")
	config.file("validation_frames_file","Validation set.")
	config.file("generative_model","Trained generative model file.")
	config.file("output_file","Output file.")
	config.file("spk2utt_file","spk2utt file from Kaldi.")
	config.integer("minibatch","Minibatch size.",default=128)
	config.integer("max_epochs","Maximum number of epochs to train.",default=20)
	config.parse_args()
	
import theano
import theano.tensor as T
import theano_toolkit 
import numpy as np
import math
import sys
import random

import data_io
import model
import updates
import cPickle as pickle
from itertools import izip, chain


import theano_toolkit.utils   as U
from theano_toolkit.parameters import Parameters
import vae_sa
from train_sa_vae import *
from pprint import pprint

if __name__ == "__main__":
	frames_files = config.args.frames_files
	
	minibatch_size = config.args.minibatch

	print config.args.generative_structure
	input_size  = config.args.generative_structure[0]
	layer_sizes = config.args.generative_structure[1:-1]
	output_size = config.args.generative_structure[-1]
	speaker_ids = get_speaker_ids(config.args.spk2utt_file)
	X = T.matrix('X')
	S = T.ivector('S')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))
	S_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	def test_validation(test):
		total_errors = 0
		total_frames = 0
		for f,s in frame_speaker_stream(
				data_io.stream(config.args.validation_frames_file,with_name=True),speaker_ids):
			errors = np.array(test(f,s))
			total_frames += f.shape[0]
			total_errors += f.shape[0] * errors
		return total_errors/total_frames
	
	def train_epoch(train):
		stream = randomised_speaker_groups(
				speaker_grouped_stream(frames_files),
				speaker_ids
			)
		total_count = 0
		total_loss  = 0
		for f,s,size in stream:
			X_shared.set_value(f)
			S_shared.set_value(s)
			batch_count = int(math.ceil(size/float(minibatch_size)))
			seq = range(batch_count)
			random.shuffle(seq)
			for idx in seq:
				start = idx*minibatch_size
				end = min((idx+1)*minibatch_size,size)
				loss = train(start,end)

	train = None
	P = Parameters()
	speaker_count = len(speaker_ids)
	speaker_embedding_size = 100
	_, recon_error = vae_sa.build(P, "vae",
				input_size,
				layer_sizes,
				output_size,
				speaker_count = speaker_count,
				speaker_embedding_size = speaker_embedding_size,
				activation=T.nnet.sigmoid
			)
	P.load(config.args.generative_model)
	P.encode_speaker_embedding = vae_sa.initial_weights(speaker_count,speaker_embedding_size)
	P.decode_speaker_embedding = vae_sa.initial_weights(speaker_count,speaker_embedding_size)


	parameters = [ p for p in P.values() if "embedding" in p.name ]
	X_recon,cost = recon_error(X,S)
	loss = cost
	gradients  = T.grad(cost,wrt=parameters)
	print "Parameters to tune:"
	pprint(sorted((p.name,p.get_value().shape) for p in parameters ))
	print "Compiling function...",
	train = theano.function(
			inputs = [start_idx,end_idx],
			updates = updates.adadelta(parameters,gradients,learning_rate=1e-8),
			#outputs = [T.mean(T.sum((X-X_recon)**2,axis=1)),cost],
			givens  = {
				X: X_shared[start_idx:end_idx],
				S: S_shared[start_idx:end_idx],
			}
		)
	test = theano.function(
			inputs = [X,S],
			outputs = [T.mean(T.sum((X-X_recon)**2,axis=1)),cost],
		)
	print "Done."
	scores = test_validation(test)
	print scores
	score = scores[-1]
	best_score = score
	for _ in xrange(config.args.max_epochs):
		train_epoch(train)
		scores = test_validation(test)
		print scores,
		score = scores[-1]
		if score < best_score:
			best_score = score
			P.save(config.args.output_file)
			print "Saved."
		else:
			print
