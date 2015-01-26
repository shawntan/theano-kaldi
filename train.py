import config
config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
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

import cPickle as pickle

if __name__ == "__main__":
	frames_file = config.frames_file
	labels_file = config.labels_file
	
	minibatch_size = 128

	params = {}

	feedforward = model.build_feedforward(params)
	
	X = T.matrix('X')
	Y = T.ivector('Y')
	idx = T.iscalar('idx')
	lr = T.scalar('lr')

	_,probs = feedforward(X)
	loss = T.mean(T.nnet.categorical_crossentropy(probs,Y))

	parameters = params.values()
	gradients = T.grad(loss,wrt=parameters)
	updates = [ (p, p - lr * g) for p,g in zip(parameters,gradients) ]
	

	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	train = theano.function(
			inputs  = [lr,idx],
			outputs = loss,
			updates = updates,
			givens  = {
				X: X_shared[idx*minibatch_size:(idx+1)*minibatch_size],
				Y: Y_shared[idx*minibatch_size:(idx+1)*minibatch_size]
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
		for f,l in data_io.randomise(stream,limit=utt_count - test_utt_count):
			total_frames += f.shape[0]
			X_shared.set_value(f)
			Y_shared.set_value(l)
			batch_count = int(math.ceil(f.shape[0]/float(minibatch_size)))
			for idx in xrange(batch_count): train(learning_rate,idx)
		#print total_frames
		total_cost = 0
		total_errors = 0
		total_frames = 0
		for f,l in stream:
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
			if learning_rate < 0.001: break
		print "Learning rate is now",learning_rate

	model.load(config.args.temporary_file,params)
	model.save(config.output_file,params)
