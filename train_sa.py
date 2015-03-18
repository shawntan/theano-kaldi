import config
config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
config.parser.add_argument(
		'--pretrain-file',
		dest = 'pretrain_file',
		type = str,
		help = ".pkl file containing pre-trained model"
	)
config.parser.add_argument(
		'--constraint-layer',
		dest = 'constraint_layer',
		required = True,
		type = int,
		help = "Layer to apply constraint."
	)
config.parser.add_argument(
		'--constraint-weight',
		dest = 'constraint_weight',
		required = True,
		type = float,
		help = "Weight of constraint."
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
import constraint
if __name__ == "__main__":
	frames_file = config.frames_file
	labels_file = config.labels_file
	minibatch_size = config.minibatch
	params = {}
	params_original = {}
	feedforward = model.build_feedforward(params)
	feedforward_original = model.build_feedforward(params_original)
	X = T.matrix('X')
	Y = T.ivector('Y')

	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	layers,probs = feedforward(X)
	_,probs_orig = feedforward_original(X)
	
	con_weight = config.args.constraint_weight
	cross_ent     = - T.mean(T.log(probs[T.arange(Y.shape[0]),Y]))
	kl_divergence = - T.mean(T.sum(probs_orig * T.log(probs) + (1 - probs_orig) * T.log(1 - probs), axis=1))
	loss = cross_ent + 1e-3 * kl_divergence

	"""
	if con_weight and con_weight > 0:
		print "Constraining hidden layer %d"%config.args.constraint_layer
		loss += con_weight *\
				constraint.adjacency(layers[config.args.constraint_layer+1],32,32)
	"""

#	parameters = [ params["W_hidden_%d"%i] for i in xrange(config.args.constraint_layer) ] +\
#				 [ params["b_hidden_%d"%i] for i in xrange(config.args.constraint_layer) ]

#	parameters = [
#			params["W_hidden_%d"%config.args.constraint_layer],
#			params["b_hidden_%d"%config.args.constraint_layer],
#			params["W_hidden_%d"%(config.args.constraint_layer+1)],
#		]
	parameters = params.values()
	print parameters
	gradients = T.grad(loss,wrt=parameters)
	

	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	train = theano.function(
			inputs  = [lr,start_idx,end_idx],
			outputs = cross_ent,
			#updates = updates.momentum(parameters,gradients,eps=lr),
			updates = [ (p, p - lr * g) for p,g in zip(parameters,gradients)],
			givens  = {
				X: X_shared[start_idx:end_idx],
				Y: Y_shared[start_idx:end_idx]
			}
		)
	test = theano.function(
			inputs = [X,Y],
			outputs = [cross_ent,T.mean(T.neq(T.argmax(probs,axis=1),Y))]
		)
	speakers = set(n.split('_')[0] for n,_,_ in 
			data_io.stream(frames_file,labels_file,with_name=True))
	print speakers

	model.load(config.args.pretrain_file,params_original)
	for p in params_original.values():p.name=p.name + "_"
	for spkr in speakers:
		print spkr
		model.load(config.args.pretrain_file,params)
		learning_rate = 1e-8
		best_score = np.inf
		for epoch in xrange(config.max_epochs):
			stream = ((f,l) for n,f,l in data_io.stream(frames_file,labels_file,with_name=True)
							if n.startswith(spkr))
			total_frames = 0
			for f,l,size in data_io.randomise(stream):
				total_frames += f.shape[0]
				X_shared.set_value(f)
				Y_shared.set_value(l)
				batch_count = int(math.ceil(size/float(minibatch_size)))
				total_cost =  0
				for idx in xrange(batch_count):
					start = idx*minibatch_size
					end = min((idx+1)*minibatch_size,size)
					total_cost += train(learning_rate,start,end)
				print total_cost/batch_count
		model.save(config.output_file + "/" + spkr + ".pkl",params)
