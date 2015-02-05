import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config

def initial_weights(input_size,output_size,factor=4):
	return np.asarray(
		np.random.uniform(
			low  = -factor * np.sqrt(6. / (input_size + output_size)),
			high =  factor * np.sqrt(6. / (input_size + output_size)),
			size =  (input_size,output_size)
		),
		dtype=theano.config.floatX
	)

def build_feedforward(params,input_size=None,layer_sizes=None,output_size=None):
	input_size = input_size or config.input_size
	layer_sizes = layer_sizes or config.layer_sizes
	output_size = output_size or config.output_size

	prev_layer_size = input_size
#	factor = 4 if config.hidden_activation == T.nnet.sigmoid else 1
#	factor = 1
	for i,curr_size in enumerate(layer_sizes):
		W_name = "W_hidden_%d"%i 
		b_name = "b_hidden_%d"%i
		params[W_name] = theano.shared(initial_weights(prev_layer_size,curr_size,factor=4 if i>0 else 0.1),name=W_name)
		params[b_name] = theano.shared(np.zeros((curr_size,),dtype=theano.config.floatX),name=b_name)
		prev_layer_size = curr_size
	W_name = "W_output"
	b_name = "b_output"
	params[W_name] = theano.shared(np.zeros((layer_sizes[-1],output_size),dtype=theano.config.floatX),name=W_name)
	params[b_name] = theano.shared(np.zeros((output_size,),dtype=theano.config.floatX),name=b_name)

	def feedforward(X):
		hidden_layers = [X]
		for i in xrange(len(layer_sizes)):
			layer = config.hidden_activation(
				T.dot(hidden_layers[-1],params["W_hidden_%d"%i]) +\
				params["b_hidden_%d"%i]
			)
			layer.name = "hidden_%d"%i
			hidden_layers.append(layer)
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
	


