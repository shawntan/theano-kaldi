import theano
import theano.tensor as T
import ark_io
import numpy as np
import model

import cPickle as pickle

import sys

def ark_stream():
	return ark_io.parse(sys.stdin)

def create_model(filename,counts,input_size,layer_sizes,output_size,output_layer=-1):
	X = T.matrix('X')
	params = {}
	predict = model.build_feedforward(params,input_size,layer_sizes,output_size)
	model.load(filename,params)
	_,outputs = predict(X)
	f = theano.function(
			inputs = [X],
			outputs = T.log(outputs) - T.log(counts/T.sum(counts))
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
	structure = map(int,sys.argv[1].split(':'))
	model_file = sys.argv[2]
	counts_file = sys.argv[3]
	output = int(sys.argv[4])
	counts = None
	predict = None

	input_size = structure[0]
	output_size = structure[-1]
	layer_sizes = structure[1:-1]

	with open(counts_file) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])
	predict = create_model(model_file,counts,input_size,layer_sizes,output_size,output_layer=output)
		
	if predict != None:
		for name,frames in ark_stream():
			print_ark(name,predict(frames))

