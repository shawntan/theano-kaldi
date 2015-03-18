import theano
import theano.tensor as T
import ark_io
import numpy as np
import model

import cPickle as pickle

import sys
from nnet_forward import *
def create_model(counts,input_size,layer_sizes,output_size):
	X = T.matrix('X')
	params = {}
	predict = model.build_feedforward(params,input_size,layer_sizes,output_size)
	_,output = predict(X)
	f = theano.function(
			inputs = [X],
			outputs = T.log(output) - T.log(counts/T.sum(counts))
		)
	return f,params

if __name__ == "__main__":
	structure = map(int,sys.argv[1].split(':'))
	model_dir = sys.argv[2]
	counts_file = sys.argv[3]
	counts = None
	predict = None

	input_size = structure[0]
	output_size = structure[-1]
	layer_sizes = structure[1:-1]

	with open(counts_file) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])
	predict,params = create_model(counts,input_size,layer_sizes,output_size)

	if predict != None:
		prev_spkr = None
		for name,frames in ark_stream():
			spkr_id = name.split("_")[0]
			model_file = model_dir+"/"+spkr_id+".pkl"
			if spkr_id == prev_spkr:
				print >> sys.stderr, "Using " + model_file
			else:
				model.load(model_file,params)
				print >> sys.stderr, "Changing to " + model_file

			print_ark(name,predict(frames))
