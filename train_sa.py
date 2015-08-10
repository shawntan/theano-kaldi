import config
config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
config.file_sequence("frames_files",".pklgz file containing audio frames.")
config.file_sequence("labels_files",".pklgz file containing frames labels.")
config.structure("generative_structure","Structure of generative model.")
config.structure("discriminative_structure","Structure of discriminative model.")
config.file("generative_model",".pkl file containing generative model")
config.file("validation_frames_file","Validation set frames file.")
config.file("validation_labels_file","Validation set labels file.")
config.file("output_file","Output file.")
config.file("temporary_file","Temporary file.")
config.file("spk2utt_file","spk2utt file from Kaldi.")
config.integer("minibatch","Minibatch size.",default=128)
config.integer("max_epochs","Maximum number of epochs to train.",default=200)

config.parse_args()



import theano
import theano.tensor as T

import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
import random

from theano_toolkit.parameters import Parameters
import vae_sa,feedforward
from train_sa_vae import get_speaker_ids, frame_speaker_stream
from sa_io import *
if __name__ == "__main__":

	frames_files = config.args.frames_files
	labels_files = config.args.labels_files
	val_frames_file = config.args.validation_frames_file
	val_labels_file = config.args.validation_labels_file
	minibatch_size = config.args.minibatch
	
	gen_input_size  = config.args.generative_structure[0]
	gen_layer_sizes = config.args.generative_structure[1:-1]
	gen_output_size = config.args.generative_structure[-1]
	dis_input_size  = config.args.discriminative_structure[0]
	dis_layer_sizes = config.args.discriminative_structure[1:-1]
	dis_output_size = config.args.discriminative_structure[-1]
	speaker_ids = get_speaker_ids(config.args.spk2utt_file)

	assert(gen_output_size == dis_input_size)

	X = T.matrix('X')
	Y = T.ivector('Y')
	S = T.ivector('S')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	X_shared = theano.shared(np.zeros((1,gen_input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))
	S_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	def run_test(test):
		total_cost = 0
		total_errors = 0
		total_frames = 0
		stream = frame_speaker_stream(
				data_io.stream(val_frames_file,val_labels_file,with_name=True),
				speaker_ids
			)
		for f,l,s in stream:
			test_outputs  = test(f,l,s)
			loss = test_outputs[0]
			errors = np.array(test_outputs[1:])
			total_frames += f.shape[0]
			total_cost   += f.shape[0] * loss
			total_errors += f.shape[0] * errors

		return total_cost/total_frames,total_errors/total_frames

	def run_train(train,learning_rate):
		stream = utterance_random_stream(frames_files,labels_files)
		stream = data_io.buffered_random(stream)
		stream = frame_speaker_stream(stream,speaker_ids)
		total_frames = 0
		for f,l,s,size in data_io.randomise(stream):
			total_frames += f.shape[0]
			X_shared.set_value(f)
			Y_shared.set_value(l)
			S_shared.set_value(s)
			batch_count = int(math.ceil(size/float(minibatch_size)))
			for idx in xrange(batch_count):
				start = idx*minibatch_size
				end = min((idx+1)*minibatch_size,size)
				train(learning_rate,start,end)

	P_vae  = Parameters()
	sample_encode, recon_error = vae_sa.build(P_vae, "vae",
			gen_input_size,  
			gen_layer_sizes, 
			gen_output_size,
			speaker_count = len(speaker_ids),
			speaker_embedding_size = 100,
			activation=T.nnet.sigmoid
		)

	mean, logvar, latent = sample_encode(X,S)
	P_vae.load(config.args.generative_model)

	learning_rate = 1e-5
	best_score = np.inf

	prev_P_disc = None
	for layer in xrange(len(dis_layer_sizes)):
		P_disc = Parameters()
		discriminate = feedforward.build(
			P_disc,
			name = "discriminate",
			input_size   = dis_input_size, 
			hidden_sizes = dis_layer_sizes[:layer+1],
			output_size  = dis_output_size,
			activation=T.nnet.sigmoid
		)
		lin_output = discriminate(latent)
		outputs = T.nnet.softmax(lin_output)
		
		for i in xrange(layer):
			P_disc["W_discriminate_hidden_%d"%i].set_value(
					prev_P_disc["W_discriminate_hidden_%d"%i].get_value())
			P_disc["b_discriminate_hidden_%d"%i].set_value(
					prev_P_disc["b_discriminate_hidden_%d"%i].get_value())

		parameters = P_disc.values()
		cross_entropy = T.mean(T.nnet.categorical_crossentropy(outputs,Y))
		loss = cross_entropy

		print "Parameters to tune:"
		pprint(parameters)

		gradients = T.grad(loss,wrt=parameters)
		train = theano.function(
				inputs  = [lr,start_idx,end_idx],
				outputs = cross_entropy,
				updates = updates.adadelta(parameters,gradients,learning_rate=lr),
				givens  = {
					X: X_shared[start_idx:end_idx],
					Y: Y_shared[start_idx:end_idx],
					S: S_shared[start_idx:end_idx],
				}
			)
		test = theano.function(
				inputs = [X,Y,S],
				outputs = [cross_entropy]  + [ T.mean(T.neq(T.argmax(outputs,axis=1),Y))]
			)


		P_disc.save(config.args.temporary_file)
		best_score = np.inf
		for _ in xrange(5):
			run_train(train,learning_rate)
			cost,errors = run_test(test)
			print cost,errors,

			if cost < best_score:
				best_score = cost
				P_disc.save(config.args.temporary_file)
				print "Saved."
			else:
				print
		P_disc.load(config.args.temporary_file)
		prev_P_disc = P_disc


	for epoch in xrange(config.args.max_epochs-1):
		cost, errors = run_test(test)
		print cost, errors,
		_best_score = best_score

		if cost < _best_score:
			best_score = cost
			P_disc.save(config.args.temporary_file)
			print "Saved."
		else:
			print

		if cost/_best_score > 0.995:
			learning_rate *= 0.5
			P_disc.load(config.args.temporary_file)

		if learning_rate < 1e-10: break

		print "Learning rate is now",learning_rate
		run_train(train,learning_rate)
	P_disc.load(config.args.temporary_file)
	P_disc.save(config.args.output_file)
