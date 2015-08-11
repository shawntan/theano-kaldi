import config
if __name__ == "__main__":
	config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
	config.file_sequence("frames_files",".pklgz file containing audio frames.")
	config.structure("generative_structure","Structure of generative model.")
	config.file("validation_frames_file","Validation set.")
	config.file("output_file","Output file.")
	config.file("spk2utt_file","spk2utt file from Kaldi.")
	config.integer("minibatch","Minibatch size.",default=128)
	config.integer("max_epochs","Maximum number of epochs to train.",default=20)
	config.integer("speaker_embedding_size","Speaker embedding size.",default=128)
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

from pprint import pprint

from sa_io import *
import shutil
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
	lr = T.scalar('lr')
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
	
	def train_epoch(train,stream,learning_rate):
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
				train(start,end,learning_rate)

	P = Parameters()
	_, recon_error = vae_sa.build(P, "vae",
				input_size,
				layer_sizes,
				output_size,
				speaker_count = len(speaker_ids),
				speaker_embedding_size = config.args.speaker_embedding_size,
				activation=T.nnet.sigmoid
			)
	

	parameters = P.values()
	X_recon,cost = recon_error(X,S)
	loss = cost + 0.5 * sum(T.sum(w**2) for w in parameters)
	general_params = [ w for w in parameters if "speaker_embedding" not in w.name ]
	speaker_params = [ w for w in parameters if "speaker_embedding" in w.name ]
	general_grads = T.grad(cost,wrt=general_params)
	speaker_grads = T.grad(cost,wrt=speaker_params)

	print "Compiling function...",
	def create_trainer(params,grads):
		return theano.function(
				inputs = [start_idx,end_idx,lr],
				updates = updates.adadelta(params,grads,learning_rate=lr),
				givens  = {
					X: X_shared[start_idx:end_idx],
					S: S_shared[start_idx:end_idx],
				}
			)

	train_all = create_trainer(general_params + speaker_params,
								general_grads + speaker_grads)
	train_general = create_trainer(general_params,general_grads)
	train_speaker = create_trainer(speaker_params,speaker_grads)
	test = theano.function(
			inputs = [X,S],
			outputs = [T.mean(T.sum((X-X_recon)**2,axis=1)),cost],
		)
	print "Done."
	
	def speaker_stream():
		return randomised_speaker_groups(
				speaker_grouped_stream(frames_files),
				speaker_ids
			)
	def general_stream():
		return randomised_speaker_groups(
				speaker_grouped_stream(frames_files),
				speaker_ids
			)



	learning_rate = 5e-7
	train_epoch(train_all,speaker_stream(),learning_rate)
	scores = test_validation(test)
	print "All training:", scores
	best_score = scores[-1]
	for epoch in xrange(config.args.max_epochs):
		train_epoch(train_speaker,speaker_stream(),learning_rate)
		scores = test_validation(test)
		print "Speaker training:", scores
		train_epoch(train_general,general_stream(),learning_rate)
		scores = test_validation(test)
		print "General training:", scores,

		score = scores[-1]
		if score < best_score:
			best_score = score
			P.save(config.args.output_file + ".tmp")
			print "Saved."
		else:
			print

		learning_rate = max(0.5 * learning_rate,1e-7)
		print "Learning rate:",learning_rate

		if (epoch + 1) % 5 == 0:
			shutil.copyfile(config.args.output_file + ".tmp",
							config.args.output_file + "." + str(epoch+1))
