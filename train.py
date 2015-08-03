import config
config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
config.parser.add_argument(
		'--validation-frames-file',
		dest = 'val_frames_file',
		required = True,
		type = str,
		help = ".pklgz file containing pickled (name,frames) pairs for training"
	)

config.parser.add_argument(
		'--validation-labels-file',
		dest = 'val_labels_file',
		required = True,
		type = str,
		help = ".pklgz file containing pickled (name,frames) pairs for training"
	)

config.parser.add_argument(
		'--pretrain-file',
		dest = 'pretrain_file',
		type = str,
		help = ".pkl file containing pre-trained model"
	)
config.parser.add_argument(
		'--temporary-file',
		dest = 'temporary_file',
		type = str,
		help = "Location to write intermediate models to."
	)
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

	frames_files = config.frames_files
	labels_files = config.labels_files
	val_frames_file = config.args.val_frames_file
	val_labels_file = config.args.val_labels_file
	minibatch_size = config.minibatch

	X = T.matrix('X')
	Y = T.ivector('Y')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
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
				config.input_size,
				config.layer_sizes[:len(config.layer_sizes)/2],
				512,
				activation=T.nnet.sigmoid
			)
	mean, logvar, latent = sample_encode(X)
	if config.args.pretrain_file != None:
		P_vae.load(config.args.pretrain_file)

	learning_rate = 1e-5


	disc_layers = config.layer_sizes[len(config.layer_sizes)/2:]
	prev_P_disc = None

	for layer in xrange(len(disc_layers)):
		P_disc = Parameters()
		discriminate = feedforward.build(
			P_disc,
			name = "discriminate",
			input_size = 512, 
			hidden_sizes = disc_layers[:layer+1],
			output_size = config.output_size,
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
		
		for _ in xrange(5):
			run_train(train,learning_rate)
			cost,errors = run_test(test)
			print cost,errors
			prev_P_disc = P_disc





	best_score = np.inf
	for epoch in xrange(config.max_epochs):
		cost, errors = run_test(test)
		print cost, errors
		_best_score = best_score

		if cost < _best_score:
			best_score = cost
			P_disc.save(config.args.temporary_file)

		if cost/_best_score > 0.995:
			learning_rate *= 0.5
			P_disc.load(config.args.temporary_file)

		if learning_rate < 1e-10: break
		print "Learning rate is now",learning_rate

		run_train(train,learning_rate)


	P_disc.load(config.args.temporary_file)
	P_disc.save(config.output_file)
