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

config.parser.add_argument(
		'--constraint-layer',
		required = True,
		dest = 'constraint_layer',
		type = int,
		help = "Layer to apply spatial constraint."
	)
config.parser.add_argument(
		'--constraint-coeff',
		required = True,
		dest = 'constraint_coeff',
		type = float,
		help = "Coeffecient of constraint term."
	)


config.parse_args()



import theano
import theano.tensor as T
import constraint
import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain

def normalise_weights(norm_size,weight_pairs):
	result = []
	for W,b in weight_pairs:
		norm = T.sqrt(T.sum(W**2,axis=0) + b**2)
		result.append((W,norm_size * W/norm))
		result.append((b,norm_size * b/norm))
	return result

def norm(W):
	return T.sqrt(T.sum(W**2,axis=1)).dimshuffle('x',0)


if __name__ == "__main__":
	frames_files = config.frames_files
	labels_files = config.labels_files
	print frames_files
	val_frames_file = config.args.val_frames_file
	val_labels_file = config.args.val_labels_file
	print val_frames_file
	print val_labels_file
	minibatch_size = config.minibatch

	params = {}

	feedforward = model.build_feedforward(params)
	X = T.matrix('X')
	Y = T.ivector('Y')

	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	hiddens,outputs = feedforward(X)
	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))
	if config.args.pretrain_file != None:
		with open(config.args.pretrain_file,'rb') as f:
			for k,v in pickle.load(f).iteritems():
				if k in params and k != "W_gates":
					params[k].set_value(v)
		model.save(config.args.temporary_file,params)

		loss = cross_entropy = T.mean(T.nnet.categorical_crossentropy(outputs,Y))
		act_surface = hiddens[1:]
		norms = [ norm(params["W_hidden_%d"%i]) for i in xrange(1,len(hiddens)-1) ]
		act_surface = [ h * n for h,n in zip(act_surface[:-1],norms) ] + [hiddens[-1]]

		constraint_params = {}
		if config.args.constraint_layer != 0:
			if config.args.constraint_layer != -1:
				loss += config.args.constraint_coeff * \
						constraint.gaussian_shape(
								params = constraint_params,
								name   = str(config.args.constraint_layer-1),
								inputs = act_surface[config.args.constraint_layer-1],
								rows = 32, cols = 32,
								components = 4
							)
			else:
				loss += config.args.constraint_coeff * \
						sum(constraint.gaussian_shape(
								params = constraint_params,
								name   = str(i),
								inputs = h,
								rows = 32, cols = 32,
								components = 4
							) for i,h in enumerate(act_surface))

		parameters = params.values() 
		print "Parameters to tune:"
		pprint(parameters)
		gradients = T.grad(loss,wrt=parameters)
		train = theano.function(
				inputs  = [lr,start_idx,end_idx],
				outputs = cross_entropy,
				updates = updates.momentum(parameters,gradients,eps=lr),
		#		updates = [ (p,p - lr * g) for p,g in zip(parameters,gradients) ],
				givens  = {
					X: X_shared[start_idx:end_idx],
					Y: Y_shared[start_idx:end_idx]
				}
			)
		test = theano.function(
				inputs = [X,Y],
				outputs = [loss]  + [ T.mean(T.neq(T.argmax(outputs,axis=1),Y))]
			)

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

		learning_rate = 0.08
		best_score = total_cost/total_frames

		print total_errors/total_frames,best_score

		for epoch in xrange(config.max_epochs):
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

			if cost/_best_score > 0.99995:
				learning_rate *= 0.5
				model.load(config.args.temporary_file,params)

			if learning_rate < 1e-6: break
			print "Learning rate is now",learning_rate

		model.load(config.args.temporary_file,params)
		model.save(config.output_file,params)
