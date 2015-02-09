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

if __name__ == "__main__":
	frames_file = config.frames_file
	labels_file = config.labels_file
	val_frames_file = config.args.val_frames_file
	val_labels_file = config.args.val_labels_file
	
	minibatch_size = config.minibatch

	params = {}

	feedforward = model.build_feedforward(params)
	
	X = T.matrix('X')
	Y = T.ivector('Y')

	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	_,probs = feedforward(X)
	loss = T.mean(T.nnet.categorical_crossentropy(probs,Y))

	parameters = params.values()
	gradients = T.grad(loss,wrt=parameters)
	

	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	train = theano.function(
			inputs  = [lr,start_idx,end_idx],
			outputs = loss,
			updates = updates.momentum(parameters,gradients,eps=lr),
			givens  = {
				X: X_shared[start_idx:end_idx],
				Y: Y_shared[start_idx:end_idx]
			}
		)
	test = theano.function(
			inputs = [X,Y],
			outputs = [loss,T.mean(T.neq(T.argmax(probs,axis=1),Y))]
		)
	if config.args.pretrain_file != None:
		model.load(config.args.pretrain_file,params)

	learning_rate = 0.1
	utt_count = sum(1 for _ in data_io.stream(frames_file,labels_file))
	frame_count = sum(f.shape[0] for f,_ in data_io.stream(frames_file,labels_file))
	#print frame_count
	test_utt_count = int(math.ceil( 0.05 * utt_count))
	best_score = np.inf
	for epoch in xrange(config.max_epochs):
		stream = data_io.stream(frames_file,labels_file)
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
		#print total_frames
		total_cost = 0
		total_errors = 0
		total_frames = 0
		for f,l in data_io.stream(val_frames_file,val_labels_file):
			loss, errors = test(f,l)
			total_frames += f.shape[0]

			total_cost   += f.shape[0] * loss
			total_errors += f.shape[0] * errors

		cost = total_cost/total_frames

		print total_errors/total_frames,cost
		if cost < best_score:
			best_score = cost
			model.save(config.args.temporary_file,params)
		else:
			learning_rate *= 0.5
			model.load(config.args.temporary_file,params)
			if learning_rate < 0.00001: break
		print "Learning rate is now",learning_rate

	model.load(config.args.temporary_file,params)
	model.save(config.output_file,params)
