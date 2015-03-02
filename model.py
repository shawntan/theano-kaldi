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
	input_size  = input_size  or config.input_size
	layer_sizes = layer_sizes or config.layer_sizes
	output_size = output_size or config.output_size

	prev_layer_size = input_size
	params["W_gates"] = theano.shared(np.zeros((output_size,len(layer_sizes)+1),dtype=theano.config.floatX),name="W_gates")
	for i,curr_size in enumerate(layer_sizes):
		W_name        = "W_hidden_%d"%i 
		b_name        = "b_hidden_%d"%i
		W_output_name = "W_output_%d"%i 
		b_output_name = "b_output_%d"%i
		params[W_name]        = theano.shared(initial_weights(prev_layer_size,curr_size,factor=4 if i>0 else 0.1),name=W_name)
		params[b_name]        = theano.shared(np.zeros((curr_size,),dtype=theano.config.floatX),name=b_name)
		params[W_output_name] = theano.shared(np.zeros((prev_layer_size,output_size),dtype=theano.config.floatX),name=W_output_name)
		params[b_output_name] = theano.shared(np.zeros((output_size,),dtype=theano.config.floatX),name=b_output_name)
		prev_layer_size = curr_size

	params[W_output_name] = theano.shared(np.zeros((prev_layer_size,output_size),dtype=theano.config.floatX),name=W_output_name)
	params[b_output_name] = theano.shared(np.zeros((output_size,),dtype=theano.config.floatX),name=b_output_name)
	def feedforward(X):
		gates = T.nnet.softmax(params["W_gates"]).T
		hidden_layers = [X]
		output_layers = []

		for i in xrange(len(layer_sizes)+1):
			if len(output_layers) > 0:
				output = output_layers[-1]
			else:
				output = 0
			output += (T.dot(hidden_layers[-1],params["W_output_%d"%i]) + params["b_output_%d"%i]) * gates[i].dimshuffle('x',0)
			output.name = "lin_output_%d"%i
			output_layers.append(output)

			if i < len(layer_sizes):
				hidden = config.hidden_activation(
						T.dot(hidden_layers[-1],params["W_hidden_%d"%i]) +\
						params["b_hidden_%d"%i]
					)
				hidden.name = "hidden_%d"%i
				hidden_layers.append(hidden)

		for i in xrange(len(output_layers)): output_layers[i] = T.nnet.softmax(output_layers[i])

		return hidden_layers,output_layers

	return feedforward

def load(filename,params):
	with open(filename,'rb') as f:
		for k,v in pickle.load(f).iteritems():
			if k in params:
				params[k].set_value(v)

def save(filename,params):
	with open(filename,'wb') as f:
		pickle.dump({k:v.get_value() for k,v in params.iteritems()},f,2)
	


