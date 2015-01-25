import theano
import theano.tensor as T
import numpy as np
import math

import cPickle as pickle

hidden_activation = T.nnet.sigmoid
input_size = 360
layer_sizes = [1024]*5
output_size = 1874

def initial_weights(*argv):
	return np.asarray(
		np.random.uniform(
			low  = -np.sqrt(6. / sum(argv)),
			high =  np.sqrt(6. / sum(argv)),
			size =  argv
		),
		dtype=theano.config.floatX
	)

def build_feedforward(params,input_size=input_size,layer_sizes=layer_sizes,output_size=output_size):
	

	prev_layer_size = input_size

	for i,curr_size in enumerate(layer_sizes):
		W_name = "W_hidden_%d"%i 
		b_name = "b_hidden_%d"%i
		params[W_name] = theano.shared(initial_weights(prev_layer_size,curr_size),name=W_name)
		params[b_name] = theano.shared(np.zeros((curr_size,),dtype=theano.config.floatX),name=b_name)
		prev_layer_size = curr_size
	W_name = "W_output"
	b_name = "b_output"
	params[W_name] = theano.shared(np.zeros((layer_sizes[-1],output_size),dtype=theano.config.floatX),name=W_name)
	params[b_name] = theano.shared(np.zeros((output_size,),dtype=theano.config.floatX),name=b_name)

	def feedforward(X):
		hidden_layers = [X]
		for i in xrange(len(layer_sizes)):
			layer = hidden_activation(
				T.dot(hidden_layers[-1],params["W_hidden_%d"%i]) +\
				params["b_hidden_%d"%i]
			)
			layer.name = "hidden_%d"%i
			hidden_layers.append(layer)
		print hidden_layers
		output = T.nnet.softmax(T.dot(hidden_layers[-1],params["W_output"]) + params["b_output"])
		return hidden_layers,output
	return feedforward

def load(filename,params):
	with open(filename,'rb') as f:
		for k,v in pickle.load(f).iteritems():
			params[k].set_value(v)

def save(filename,params):
	with open(filename,'wb') as f:
		pickle.dump({k:v.get_value() for k,v in params.iteritems()},f,2)
	


