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

	_,output_layers,outputs = feedforward(X)
	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))

#	cross_entropy = [ T.mean(T.nnet.categorical_crossentropy(o,Y)) for o in output_layers ]
#	loss = sum(cross_entropy) #+ 1e-8 * T.sum(params['W_gates']**2)

	loss = cross_entropy = T.mean(T.nnet.categorical_crossentropy(outputs,Y)
	parameters = params.values()

#	parameters = [ p for p in params.values() if "_gate" in p.name ]
#	parameters = [ params["W_output_%d"%i] for i in range(0,5) ] + \
#				 [ params["b_output_%d"%i] for i in range(0,5) ] 

#	parameters = [ p for p in params.values()
#					if p.name not in [
#						"b_output_%d"%i for i in xrange(0,6) 
#					]
#				]

#	parameters = [ params["W_output_%d"%i] for i in range(0,6) ] + \
#				 [ params["b_output_6"] ]

	print "Parameters to tune:"
	pprint(parameters)
	gradients = T.grad(loss,wrt=parameters)

	train = theano.function(
			inputs  = [lr,start_idx,end_idx],
			outputs = cross_entropy,
			updates = updates.momentum(parameters,gradients,eps=lr),
#			updates = [ (p,p - lr * g) for p,g in zip(parameters,gradients) ],
			givens  = {
				X: X_shared[start_idx:end_idx],
				Y: Y_shared[start_idx:end_idx]
			}
		)

	"""
	oracles = []
	for final in xrange(2,7):
		correct = sum(T.eq(T.argmax(o,axis=1),Y) for o in output_layers[-final:])
		correct = correct > 0
		oracle_score = 1 - T.mean(correct)
		oracles.append(oracle_score)
	"""

	test = theano.function(
			inputs = [X,Y],
			outputs = [loss] + [ T.mean(T.neq(T.argmax(o,axis=1),Y)) for o in output_layers ]
		)

	if config.args.pretrain_file != None:
		with open(config.args.pretrain_file,'rb') as f:
			for k,v in pickle.load(f).iteritems():
				if k in params and k != "W_gates":
					params[k].set_value(v)
		model.save(config.args.temporary_file,params)
		#with open(config.args.pretrain_file,'rb') as f:
		#	p = pickle.load(f)
		#	params['W_output_5'].set_value(p['W_output'])
		#	params['b_output_5'].set_value(p['b_output'])

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

	learning_rate = 0.5
	best_score = total_cost/total_frames

	print total_errors/total_frames,best_score

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
			test_outputs  = test(f,l)
			loss = test_outputs[0]
			errors = np.array(test_outputs[1:])
			total_frames += f.shape[0]

			total_cost   += f.shape[0] * loss
			total_errors += f.shape[0] * errors

		cost = total_cost/total_frames

		print total_errors/total_frames,cost
		
		_best_score = best_score

		if cost < _best_score:
			best_score = cost
			model.save(config.args.temporary_file,params)

		if cost/_best_score > 0.995:
			learning_rate *= 0.5
			model.load(config.args.temporary_file,params)

		if learning_rate < 1e-6: break
		print "Learning rate is now",learning_rate

	model.load(config.args.temporary_file,params)
	model.save(config.output_file,params)
