import theano
import theano.tensor as T
import ark_io
import numpy as np
import model

import cPickle as pickle
import itertools
import sys

def ark_stream():
	return ark_io.parse(sys.stdin)


def build_feedforward(params):
	parameters = []
	for i in itertools.count():
		if 'W_hidden_%d'%i in params[0]:
			weights = [ p['W_hidden_%d'%i] for p in params ]
			biases  = [ p['b_hidden_%d'%i] for p in params ]
			parameters.append((weights,biases))
		else:
			break
	output_parameters = (
		[ p['W_output'] for p in params ],
		[ p['b_output'] for p in params ]
	)

	def feedforward(X,cmb):
		prev = X
		for weights,biases in parameters:
			W = 0
			b = 0
			for i,(weight,bias) in enumerate(zip(weights,biases)):
				W += cmb[i] * weight
				b += cmb[i] * bias
			#	print >> sys.stderr, weight.shape
			#	print >> sys.stderr, bias.shape

			prev = T.nnet.sigmoid(T.dot(prev,W) + b)
		
		W = 0
		b = 0
		for i,(weight,bias) in enumerate(zip(*output_parameters)):
			W += cmb[i] * weight
			b += cmb[i] * bias
			#print >> sys.stderr, weight.shape
			#print >> sys.stderr, bias.shape
		output = T.nnet.softmax(T.dot(prev,W) + b)
		
		return output
	return feedforward


				





def create_model(filenames,counts):
	X = T.matrix('X')
	cmb = T.vector('cmb')
	params = {}

	predict = build_feedforward([
		pickle.load(open(f)) for f in filenames
	])

	output = predict(X,cmb)
	f = theano.function(
			inputs = [X,cmb],
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
	structure = map(int,sys.argv[1].split(':'))
	counts_file = sys.argv[2]
	model_files = sys.argv[3:]
	counts = None
	predict = None

	input_size = structure[0]
	output_size = structure[-1]
	layer_sizes = structure[1:-1]
	
	with open(counts_file) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])
	predict = create_model(model_files,counts)
	
	if predict != None:
		for name,frames in ark_stream():
			if name.startswith('M'):
				cmb = [ 1, 0 ]
			elif name.startswith('F'):
				cmb = [ 0, 1 ]
			print_ark(name,predict(frames,cmb))
