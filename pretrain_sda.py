import config
config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
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

theano_rng = T.shared_randomstreams.RandomStreams(np.random.RandomState(1234).randint(2**30))

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
	frames_file = config.frames_file
	labels_file = config.labels_file
	
	minibatch_size = config.minibatch

	params = {}

	feedforward = model.build_feedforward(params)
	
	X = T.matrix('X')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')

	layers,_ = feedforward(X)
	
	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))

	layer_sizes = [config.input_size] + config.layer_sizes[:-1]

	pretrain_functions = []
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
		lr = 0.01 if i > 0 else 0.003
		parameters = [W,b,b_rec]
		gradients  = T.grad(loss,wrt=parameters)
		train = theano.function(
				inputs = [start_idx,end_idx],
				outputs = loss,
				updates = updates.momentum(parameters,gradients,eps=lr),
			#	 [
			#		(p, p - lr * g) for p,g in zip(parameters,gradients)
			#	],
				givens  = {
					X: X_shared[start_idx:end_idx],
				}
			)
		pretrain_functions.append(train)

	for layer_idx,train in enumerate(pretrain_functions):	
		print "Pretraining layer",layer_idx,"..."
		for epoch in xrange(config.max_epochs):	
			stream = data_io.stream(frames_file,labels_file)
			total_count = 0
			total_loss  = 0
			for f,_,size in data_io.randomise(stream):
				X_shared.set_value(f)
				batch_count = int(math.ceil(size/float(minibatch_size)))
				for idx in xrange(batch_count):
					start = idx*minibatch_size
					end = min((idx+1)*minibatch_size,size)
#					print "Training:",(start,end)
					total_loss += train(start,end)
					total_count += 1
			print total_loss/total_count
	model.save(config.output_file,params)

	
