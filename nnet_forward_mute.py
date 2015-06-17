import theano
import theano.tensor as T
import ark_io
import numpy as np
import model_mute as model
import em_test as em
import cPickle as pickle

import sys

def ark_stream():
	return ark_io.parse(sys.stdin)

def create_model(filename,counts,input_size,layer_sizes,output_size):
	X = T.matrix('X')
	phoneme = T.iscalar('phoneme')
	params = {}
	gmm_params = {}
	predict = model.build_feedforward(params,input_size,layer_sizes,output_size)
	mask = em.mask(
		params = gmm_params,
		name = "-1",
		phonemes = 48,
		components = 1,
		rows = 32,cols = 32
	)
	_,output = predict(X,mask(phoneme))
	f = theano.function(
			inputs = [X,phoneme],
			outputs = T.log(output) - T.log(counts/T.sum(counts))
		)

	gaussian_ctr = pickle.load(open('gaussian_ctr.pkl'))
	model.load(filename,params)
	gmm_params["phoneme_means_%d"%-1].set_value(
			gaussian_ctr.reshape(48,1,2).astype(np.float32)
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
	phoneme_id = int(sys.argv[4])
	counts = None
	predict = None

	input_size = structure[0]
	output_size = structure[-1]
	layer_sizes = structure[1:-1]

	with open(counts_file) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])
	predict = create_model(model_file,counts,input_size,layer_sizes,output_size)
		
	if predict != None:
		for name,frames in ark_stream():
			print_ark(name,predict(frames,phoneme_id))

