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
import cPickle as pickle
import updates
theano_rng = T.shared_randomstreams.RandomStreams(np.random.RandomState(1234).randint(2**30))

def reconstruct(vis,W,b,b_rec,act_vis):
	hidden_mean  = config.hidden_activation(T.dot(vis,W) + b)
	hidden_sample = theano_rng.binomial(
			size = hidden_mean.shape,
			n = 1, p = hidden_mean,
			dtype = theano.config.floatX
		)
	
	visible_mean = act_vis(T.dot(hidden_sample,W.T) + b_rec)
	if act_vis == T.nnet.sigmoid:
		visible_sample = theano_rng.binomial(
				size = visible_mean.shape,
				n = 1, p = visible_mean,
				dtype = theano.config.floatX
			)
	else:
		visible_sample = theano_rng.normal(
				size = visible_mean.shape,
				avg = 0.0, std = 1.0,
				dtype = theano.config.floatX
			) + visible_mean

	return visible_sample,visible_mean

def free_energy(vis,W,b,b_rec,act_vis):
	if act_vis == T.nnet.sigmoid:
		visible_term = T.dot(vis,b_rec)
	else:
		visible_term = 0.5 * T.sum((vis - b_rec)**2,axis=1)

	hidden_lin = T.dot(vis,W) + b
	hidden_term = T.sum(T.log(1 + T.exp(hidden_lin)), axis=1)

	return - hidden_term - visible_term

def cost(vis,vis_rec,W,b,b_rec,act_vis):
	return T.mean(free_energy(vis,W,b,b_rec,act_vis)) \
		   - T.mean(free_energy(vis_rec,W,b,b_rec,act_vis))


if __name__ == "__main__":
	frames_file = config.frames_file
	labels_file = config.labels_file
	
	minibatch_size = config.minibatch

	params = {}

	feedforward = model.build_feedforward(params)
	
	X = T.matrix('X')
	idx = T.iscalar('idx')

	layers,_ = feedforward(X)
	
	X_shared = theano.shared(np.zeros((1,config.input_size),dtype=theano.config.floatX))

	layer_sizes = [config.input_size] + config.layer_sizes[:-1]

	pretrain_functions = []
	print "Compiling functions..."

	lr = T.scalar('lr')
	for i,layer in enumerate(layers[:-1]):
		W = params["W_hidden_%d"%i]
		b = params["b_hidden_%d"%i]
		b_rec = theano.shared(np.zeros((layer_sizes[i],),dtype=theano.config.floatX))
		
		act_vis = T.nnet.sigmoid if i > 0 else lambda x:x
		layer_rec,layer_mean = reconstruct(layer,W,b,b_rec,act_vis)
		loss = cost(
				layer,layer_rec,
				W,b,b_rec,act_vis
			) + 0.0002 * T.sum(W**2)
		if i == 0:
			rec_cost = T.mean(T.sum((layer - layer_mean)**2,axis=1))
		else:
			rec_cost = -T.mean(
				T.sum(
					layer * T.log(layer_mean) +\
					(1 - layer) * T.log(1 - layer_mean),
					axis=1
				)
			)

		parameters = [W,b,b_rec]
		gradients  = T.grad(loss,wrt=parameters,consider_constant=[layer_rec])
		
		train = theano.function(
				inputs = [idx,lr],
				outputs = rec_cost,
				updates = updates.momentum(parameters,gradients,mu=0.9,eps=lr),
				#[
				#	(p, p - lr * g) for p,g in zip(parameters,gradients)
				#],
				
				givens  = {
					X: X_shared[idx*minibatch_size:(idx+1)*minibatch_size],
				}
			)

		pretrain_functions.append(train)

	lrates = [ 0.001 ] + [0.15] * (len(layers) - 2)
	layer_epochs = [ 40 ] + [20] * (len(layers) - 2)
	best_loss = np.inf
	for layer_idx,train in enumerate(pretrain_functions):
		learning_rate = lrates[layer_idx]
		print "Pretraining layer",layer_idx,"..."
		for epoch in xrange(layer_epochs[layer_idx]):	
			stream = data_io.stream(frames_file,labels_file)
			total_count = 0
			total_loss  = 0
			for f,_ in data_io.randomise(stream):
				
				X_shared.set_value(f)
				batch_count = int(math.ceil(f.shape[0]/float(minibatch_size)))
				for idx in xrange(batch_count):
					loss = train(idx,learning_rate)
					#print loss
					total_loss += loss
					total_count += 1
			print total_loss / total_count

			if total_loss / total_count < best_loss:
				best_loss = total_loss / total_count
				model.save(config.output_file,params)
			else:
				learning_rate = max(learning_rate - 0.0001,0.001)
				
		model.load(config.output_file,params)


	model.save(config.output_file,params)

	
