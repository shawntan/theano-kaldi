import theano
import theano.tensor as T
import ark_io
import numpy as np

import cPickle as pickle

import sys

def ark_stream():
	return ark_io.parse(sys.stdin)

def create_model(params,counts):
	X = T.matrix('X')
	prev_layer = X
	
	layer_idx = 0
	while "W_hidden_%d"%layer_idx in params:
		W = theano.shared(params["W_hidden_%d"%layer_idx])
		b = theano.shared(params["b_hidden_%d"%layer_idx])
		prev_layer = T.nnet.sigmoid(T.dot(prev_layer,W) + b)
		layer_idx += 1
	
	W = theano.shared(params["W_output"])
	b = theano.shared(params["b_output"])
	output = T.nnet.softmax(T.dot(prev_layer,W) + b)

	f = theano.function(
			inputs = [X],
			outputs = T.log(output) - T.log(counts/T.sum(counts))
		)
	return f

def print_ark(name,array):
	print name,"["
	for i,row in enumerate(array):
		print " ",
		for cell in row:
			print "%0.6f"%cell,
		if i == array.shape[0]-1:
			print "]"
		else:
			print
	
if __name__ == "__main__":
	model_file = sys.argv[1]
	counts_file = sys.argv[2]
	counts = None
	predict = None

	with open(counts_file) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])

	with open(model_file) as f:
		params = pickle.load(f)
		predict = create_model(params,counts)
		
	if predict != None:
		for name,frames in ark_stream():
			print_ark(name,predict(frames))

