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
		W_hidden_name = "W_hidden_%d"%i 
		b_hidden_name = "b_hidden_%d"%i
		W_output_name = "W_output_%d"%i
		b_output_name = "b_output_%d"%i
		W_gate_name   = "W_gate_%d"%i

		params[W_hidden_name] = theano.shared(initial_weights(prev_layer_size,curr_size,factor=4 if i>0 else 0.1),name=W_hidden_name)
		params[b_hidden_name] = theano.shared(np.zeros((curr_size,),dtype=theano.config.floatX),                  name=b_hidden_name)
		params[W_output_name] = theano.shared(np.zeros((layer_sizes[-1],output_size),dtype=theano.config.floatX), name=W_output_name)
		params[b_output_name] = theano.shared(np.zeros((output_size,),dtype=theano.config.floatX),                name=b_output_name)
		if i < len(layer_sizes) - 1:
			gate_weights = np.zeros((curr_size + 1,),dtype=theano.config.floatX)
			gate_weights[-1] = -1
			params[W_gate_name] = theano.shared(gate_weights,name=W_gate_name) 
		prev_layer_size = curr_size


	def feedforward(X):

		prev_hidden = X
		hidden_layers,output_layers,gates = [],[],[]
		for i in xrange(len(layer_sizes)):

			W_hidden_name = "W_hidden_%d"%i 
			b_hidden_name = "b_hidden_%d"%i
			W_output_name = "W_output_%d"%i
			b_output_name = "b_output_%d"%i
			W_h = params[W_hidden_name]
			b_h = params[b_hidden_name]
			W_o = params[W_output_name]
			b_o = params[b_output_name]

			hidden = config.hidden_activation(T.dot(prev_hidden,W_h) + b_h)
			output = T.nnet.softmax(T.dot(hidden,W_o) + b_o)
			hidden_layers.append(hidden)
			output_layers.append(output)
			hidden.name = "hidden_%d"%i
			output.name = "output_%d"%i

			if i < len(layer_sizes) - 1:
				W_gate_name = "W_gate_%d"%i
				W_g = params[W_gate_name][:-1]
				b_g = params[W_gate_name][-1]
				gate = T.nnet.sigmoid(T.dot(hidden,W_g) + b_g)
				gate = gate.dimshuffle(0,'x')
				gates.append(gate)

			prev_hidden = hidden

		output = output_layers[-1]
		for i in xrange(len(layer_sizes)-2,-1,-1):
			output = gates[i] * output_layers[i]  + (1 - gates[i]) * output

		return hidden_layers,output_layers,output
	return feedforward

def load(filename,params):
	with open(filename,'rb') as f:
		for k,v in pickle.load(f).iteritems():
			params[k].set_value(v)

def save(filename,params):
	with open(filename,'wb') as f:
		pickle.dump({k:v.get_value() for k,v in params.iteritems()},f,2)
	


