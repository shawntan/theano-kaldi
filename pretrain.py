import theano
import theano.tensor as T

import numpy as np
import math
import sys

import data_io
import model
import cPickle as pickle

theano_rng = T.shared_randomstreams.RandomStreams(np.random.RandomState(1234).randint(2**30))

def corrupt(x,corr=0.5):
	corr_x = theano_rng.binomial(size=x.shape,n=1,p=1-corr,dtype=theano.config.floatX) * x
	corr_x.name = "corr_" + x.name
	return corr_x

def reconstruct(corr_x,W,b,b_rec,input_layer):
	hidden = model.hidden_activation(T.dot(corr_x,W) + b)
	recon  = T.dot(hidden,W.T) + b_rec
	if not input_layer:
		recon = model.hidden_activation(recon)
	return recon

def cost(x,recon_x,kl_divergence):
	print kl_divergence
	if not kl_divergence:
		return T.mean(T.sum((x - recon_x)**2,axis=1))
	else:
		return - T.mean(T.sum(x * T.log(recon_x) + (1 - x) * T.log(1 - recon_x), axis=1))

if __name__ == "__main__":
	frames_file = sys.argv[1]
	labels_file = sys.argv[2]
	
	minibatch_size = 128

	params = {}

	feedforward = model.build_feedforward(params)
	
	X = T.matrix('X')
	idx = T.iscalar('idx')

	layers,_ = feedforward(X)
	
	X_shared = theano.shared(np.zeros((1,model.input_size),dtype=theano.config.floatX))

	layer_sizes = [model.input_size] + model.layer_sizes[:-1]
	pretrain_functions = []
	for i,layer in enumerate(layers[:-1]):
		W = params["W_hidden_%d"%i]
		b = params["b_hidden_%d"%i]
		b_rec = theano.shared(np.zeros((layer_sizes[i],),dtype=theano.config.floatX))
		
		print layer.name != 'X',model.hidden_activation == T.nnet.sigmoid

		loss = cost(
				layer,
				reconstruct(
					corrupt(layer),
					W,b,b_rec,
					input_layer = (layer.name == 'X')
				),
				kl_divergence = ((layer.name != 'X') and (model.hidden_activation == T.nnet.sigmoid))
			)
		parameters = [W,b,b_rec]
		gradients  = T.grad(loss,wrt=parameters)
	
		train = theano.function(
				inputs = [idx],
				outputs = loss,
				updates = [
					(p, p - 0.001 * g) for p,g in zip(parameters,gradients)
				],
				givens  = {
					X: X_shared[idx*minibatch_size:(idx+1)*minibatch_size],
				}
			)
		pretrain_functions.append(train)

	for layer_idx,train in enumerate(pretrain_functions):	
		print "Pretraining layer",layer_idx,"..."
		for epoch in xrange(5):	
			stream = data_io.stream(frames_file,labels_file)
			
			total_count = 0
			total_loss  = 0
			for f,_ in data_io.randomise(stream):
				X_shared.set_value(f)
				batch_count = int(math.ceil(f.shape[0]/float(minibatch_size)))
				for idx in xrange(batch_count):
					total_loss += train(idx)
					total_count += 1
			print total_loss/total_count
	model.save('pretrain.pkl',params)

	
