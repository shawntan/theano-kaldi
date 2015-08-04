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


from theano_toolkit.parameters import Parameters
import vae,feedforward
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
	
	assert(gen_output_size == dis_input_size)

	X = T.matrix('X')
	Y = T.ivector('Y')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	X_shared = theano.shared(np.zeros((1,gen_input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	def run_test(test):
		total_cost = 0
		total_errors = 0
		total_frames = 0
		for f,l in data_io.stream(val_frames_file,val_labels_file):
			test_outputs  = test(f,l)
			loss = test_outputs[0]
			errors = np.array(test_outputs[1:])
			total_frames += f.shape[0]

			total_cost   += f.shape[0] * loss
			total_errors += f.shape[0] * errors

		return total_cost/total_frames,total_errors/total_frames

	def run_train(train,learning_rate):
		split_streams = [ data_io.stream(f,l) for f,l in izip(frames_files,labels_files) ]
		stream = chain(*split_streams)
		total_frames = 0
		for f,l,size in data_io.randomise(stream):
			total_frames += f.shape[0]
			X_shared.set_value(f)
			Y_shared.set_value(l)
			batch_count = int(math.ceil(size/float(minibatch_size)))
			for idx in xrange(batch_count):
				start = idx*minibatch_size
				end = min((idx+1)*minibatch_size,size)
				train(learning_rate,start,end)


	P_vae  = Parameters()
	sample_encode, recon_error = vae.build(P_vae, "vae",
				gen_input_size,
				gen_layer_sizes,
				gen_output_size,
				activation=T.nnet.sigmoid
			)

	mean, logvar, latent = sample_encode(X)
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

		parameters = P_disc.values() #+ P_vae.values()
		cross_entropy = T.mean(T.nnet.categorical_crossentropy(outputs,Y))
		loss = cross_entropy #+\
#				recon_error(X,encoder_out=(mean,logvar,latent))[1]  +\
#				0.5 * sum(T.sum(w**2) for w in parameters)

		print "Parameters to tune:"
		pprint(parameters)

		gradients = T.grad(loss,wrt=parameters)
		train = theano.function(
				inputs  = [lr,start_idx,end_idx],
				outputs = cross_entropy,
#				updates = updates.momentum(parameters,gradients,mu=0.9,eps=lr),
				updates = updates.adadelta(parameters,gradients,eps=lr),
#			updates = updates.rmsprop(parameters,gradients,learning_rate=lr),
#			updates = [ (p,p - lr * g) for p,g in zip(parameters,gradients) ],
				givens  = {
					X: X_shared[start_idx:end_idx],
					Y: Y_shared[start_idx:end_idx]
				}
			)
		test = theano.function(
				inputs = [X,Y],
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
	P_disc.save(config.args.output_file)
