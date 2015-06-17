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
		'--validation-phonemes-file',
		dest = 'val_phonemes_file',
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

config.parser.add_argument(
		'--constraint-surface',
		required = True,
		dest = 'constraint_surface',
		type = str,
		help = "Constraint surface."
	)

config.parser.add_argument(
		'--constraint-spread',
		required = True,
		dest = 'constraint_spread',
		type = float,
		help = "Constraint spread."
	)


config.parser.add_argument(
		'--phoneme-files',
		nargs = '+',
		dest = 'phoneme_files',
		required = True,
		type = str,
		help = ".pklgz files containing pickled (name,frames) pairs for training"
	)

config.parser.add_argument(
		'--log-directory',
		dest = 'log_directory',
		required = True,
		type = str,
		help = "Log directory."
	)


config.parse_args()



import theano
import theano.tensor as T
import constraint
import numpy as np
import math
import sys
import os
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
	val_frames_file = config.args.val_frames_file
	val_labels_file = config.args.val_labels_file
	phoneme_files = config.args.phoneme_files
	val_phonemes_file = config.args.val_phonemes_file
	minibatch_size = config.minibatch
	
	


	params = {}

	feedforward = model.build_feedforward(params)
	X = T.matrix('X')
	Y = T.ivector('Y')
	P = T.imatrix('P')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	lr = T.scalar('lr')

	hiddens,predict= feedforward(X)

	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))
	P_shared = theano.shared(np.zeros((1,3),dtype=np.int32))

	if config.args.pretrain_file != None:
		with open(config.args.pretrain_file,'rb') as f:
			for k,v in pickle.load(f).iteritems():
				if k in params and k != "W_gates":
					params[k].set_value(v)
		model.save(config.args.temporary_file,params)

	loss = cross_entropy = T.mean(T.nnet.categorical_crossentropy(predict,Y)) 

	act_surface = hiddens[1:]
	
	if config.args.constraint_surface == "raw":
		pass
	elif config.args.constraint_surface == "norm":
		norms = [ norm(params["W_hidden_%d"%i]) for i in xrange(1,len(hiddens)-1) ]
		act_surface = [ h * n for h,n in zip(act_surface[:-1],norms) ] + [hiddens[-1]]
	elif config.args.constraint_surface == "h0scale":
		hiddens_0,_ = feedforward(X*0)
		act_surface = [ h / h0 for h,h0 in zip(act_surface,hiddens_0[1:]) ]
	elif config.args.constraint_surface == "meanscale":
		hidden_avg = [ T.mean(h,axis=0).dimshuffle('x',0) for h in act_surface ]
		act_surface = [ h / m for h,m in zip(act_surface,hidden_avg) ]
	kl_divergence = None
	parameters = params.values()
	if config.args.constraint_coeff > 0:
		config.args.constraint_coeff
		import ctx_gaussians
		gaussian_ctr = pickle.load(open('gaussian_ctr.pkl'))
		gmm_params = {}
		constraint = ctx_gaussians.build(
				params = gmm_params,
				name = str('all'),
				means = gaussian_ctr,
				rows = 32,cols = 32
			)
		if config.args.constraint_layer == -1:
			kl_divergence = sum(constraint(s,P-1) for s in act_surface)
		else:
			kl_divergence = constraint(act_surface[config.args.constraint_layer-1], P-1)

		loss += config.args.constraint_coeff * kl_divergence
		all_surface = sum(act_surface)
		params.update(gmm_params)

	print "Parameters to tune:"
	pprint(parameters)

	gradients = T.grad(loss,wrt=parameters)
	train = theano.function(
			inputs  = [lr,start_idx,end_idx],
			outputs = loss,
			updates = updates.momentum(parameters,gradients,eps=lr),
			givens  = {
				X: X_shared[start_idx:end_idx],
				Y: Y_shared[start_idx:end_idx],
				P: P_shared[start_idx:end_idx]
			},
			on_unused_input='warn'
		)
	
	outputs = [
			("cross_entropy",cross_entropy),
			("classification_error", T.mean(T.neq(T.argmax(predict,axis=1),Y))),
		]
	if kl_divergence:
			outputs.append(("kl_divergence", kl_divergence))


	test = theano.function(
			inputs = [X,Y,P],
			outputs = [ o[1] for o in outputs ],
			on_unused_input='warn'
		)

	learning_rate = 0.08
	best_score = np.inf

	for epoch in xrange(config.max_epochs):
		total_errors = 0
		total_frames = 0
		for f,p,l in data_io.stream(val_frames_file,val_phonemes_file,val_labels_file):
			test_outputs  = np.array(test(f,l,p))
			total_errors += f.shape[0] * test_outputs 
			total_frames += f.shape[0]
		avg_errors = total_errors / total_frames
		named_errors = { n[0]:v for n,v in zip(outputs,avg_errors) }
		
		score = named_errors["classification_error"]

		for k,v in named_errors.iteritems():
			with open(config.args.log_directory + "/"+k,'a') as f:
				f.write("%0.5f\n"%v)

		
		prev_best_score = best_score
		if score < best_score:
			best_score = score 
			model.save(config.args.temporary_file,params)

		if score/prev_best_score > 0.99995:
			learning_rate *= 0.5
			if score > prev_best_score:
				model.load(config.args.temporary_file,params)

		if learning_rate < 1e-6: break

		pprint(named_errors)		
		print "Learning rate is now",learning_rate


	
	
		split_streams = [ data_io.stream(f,l,p) for f,l,p in izip(frames_files,labels_files,phoneme_files) ]
		stream = chain(*split_streams)
		total_frames = 0
		for f,l,p,size in data_io.randomise(stream):
			total_frames += f.shape[0]
			X_shared.set_value(f)
			Y_shared.set_value(l)
			P_shared.set_value(p)
			batch_count = int(math.ceil(size/float(minibatch_size)))
			for idx in xrange(batch_count):
				start = idx*minibatch_size
				end = min((idx+1)*minibatch_size,size)
				train(learning_rate,start,end)

		split_streams = [ data_io.stream(f,p) for f,p in izip(frames_files,phoneme_files) ]
		stream = chain(*split_streams)
		count = 0
#		for f,p in stream:
#			print update_stats(f,p)
#			count += 1
#		update_params()

	model.load(config.args.temporary_file,params)
	model.save(config.output_file,params)
