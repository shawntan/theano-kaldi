import config
config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
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
		'--phoneme-files',
		nargs = '+',
		dest = 'phoneme_files',
		required = True,
		type = str,
		help = ".pklgz files containing pickled (name,frames) pairs for training"
	)

config.parser.add_argument(
		'--gmm-param-dir',
		dest = 'gmm_param_dir',
		required = True,
		type = str,
		help = "directory to put parameters."
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
from itertools import izip, chain
import constraint
import em_test as em

theano_rng = T.shared_randomstreams.RandomStreams(np.random.RandomState(1234).randint(2**30))
def norm(W):
	return T.sqrt(T.sum(W**2,axis=1)).dimshuffle('x',0)



def corrupt(x,corr=0.2):
	corr_x = theano_rng.binomial(size=x.shape,n=1,p=1-corr,dtype=theano.config.floatX) * x
	corr_x.name = "corr_" + x.name
	return corr_x

def reconstruct(corr_x,W,b,b_rec,input_layer):
#	if input_layer:
#		hidden = T.tanh(T.dot(corr_x,W) + b)
#	else:
	hidden = config.hidden_activation(T.dot(corr_x,W) + b)

	recon  = T.dot(hidden,W.T) + b_rec
	if not input_layer:
		recon = config.hidden_activation(recon)
	return recon

def cost(x,recon_x,kl_divergence):
	if not kl_divergence:
		return T.mean(T.sum((x - recon_x)**2,axis=1))
	else:
		return -T.mean(T.sum(x * T.log(recon_x) + (1 - x) * T.log(1 - recon_x), axis=1))

if __name__ == "__main__":
	frames_files = config.frames_files
	labels_files = config.labels_files
	phoneme_files = config.args.phoneme_files
	minibatch_size = config.minibatch

	params = {}

	feedforward = model.build_feedforward(params)
	
	X = T.matrix('X')
	P = T.ivector('P')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')

	layers,_ = feedforward(X)
	act_surface = layers[1:]
	norms = [ norm(params["W_hidden_%d"%i]) for i in xrange(1,len(layers)-1) ]
	act_surface = [ h * n for h,n in zip(act_surface[:-1],norms) ] + [layers[-1]]


	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))
	P_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	layer_sizes = [config.input_size] + config.layer_sizes[:-1]

	pretrain_functions = []
	constraint_functions = [None] * len(layers[:-1])
	for i,layer in enumerate(layers[:-1]):
		W = params["W_hidden_%d"%i]
		b = params["b_hidden_%d"%i]
		b_rec = theano.shared(np.zeros((layer_sizes[i],),dtype=theano.config.floatX))


		loss = cost(
				layer,
				reconstruct(
					corrupt(layer),
					W,b,b_rec,
					input_layer = (layer.name == 'X')
				),
				kl_divergence = ((layer.name != 'X') and (config.hidden_activation == T.nnet.sigmoid))
			)
		if i + 1 == config.args.constraint_layer or config.args.constraint_layer==-1:
			print "Constraint this layer."
			gmm_params = {}
			gmm_constraint,em_updates = em.build(
				params = gmm_params,
				name = str(config.args.constraint_layer),
				phonemes = 48,
				components = 4,
				rows = 32,cols = 32
			)
			loss += gmm_constraint(act_surface[i],P-1)
			stats_updates,param_updates = em_updates(act_surface[i],P-1)
			update_stats  = theano.function(inputs=[X,P], updates = stats_updates)
			update_params = theano.function(inputs=[],    updates = param_updates) 
			constraint_functions[i] = (update_stats,update_params)
			params.update(gmm_params)
		else:
			stats_updates = []
			
		lr = 0.01 if i > 0 else 0.003
		parameters = [W,b,b_rec]
		gradients  = T.grad(loss,wrt=parameters)
		train = theano.function(
				inputs = [start_idx,end_idx],
				outputs = loss,
				updates = updates.momentum(parameters,gradients,eps=lr),
				givens  = {
					X: X_shared[start_idx:end_idx],
					P: P_shared[start_idx:end_idx]
				},
				on_unused_input='warn'
			)
		pretrain_functions.append(train)

	for layer_idx,train in enumerate(pretrain_functions):	
		print "Pretraining layer",layer_idx,"..."
		for epoch in xrange(config.max_epochs):	
			split_streams = [ data_io.stream(f,l,p) for f,l,p in izip(frames_files,labels_files,phoneme_files) ]
			stream = chain(*split_streams)
			total_count = 0
			total_loss  = 0
			for f,_,p,size in data_io.randomise(stream):
				X_shared.set_value(f)
				P_shared.set_value(p)
				batch_count = int(math.ceil(size/float(minibatch_size)))
				for idx in xrange(batch_count):
					start = idx*minibatch_size
					end = min((idx+1)*minibatch_size,size)
#					print "Training:",(start,end)
					total_loss += train(start,end)
					total_count += 1
			print total_loss/total_count
			if constraint_functions[layer_idx]:
				split_streams = [ data_io.stream(f,l,p) for f,l,p in izip(frames_files,labels_files,phoneme_files) ]
				stream = chain(*split_streams)
				update_stats,update_params = constraint_functions[layer_idx]
				for f,_,p in stream: update_stats(f,p)
				update_params()
				model.save("%s/layer-%d-epoch-%d.pkl"%(config.args.gmm_param_dir,layer_idx,epoch),gmm_params)
	model.save(config.output_file,params)
