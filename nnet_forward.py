import theano
import theano.tensor as T
import ark_io
import numpy as np
import model

import cPickle as pickle

import sys
import vae
import feedforward
from theano_toolkit.parameters import Parameters
def ark_stream():
	return ark_io.parse(sys.stdin)

def create_model(filename_vae,filename_disc,
		counts,input_size,layer_sizes,output_size):

	X = T.matrix('X')
	
	P_vae  = Parameters()
	P_disc = Parameters()

	sample_encode, recon_error = vae.build(P_vae, "vae",
				input_size,
				layer_sizes[:len(layer_sizes)/2],
				512,
				activation=T.nnet.sigmoid
			)

	discriminate = feedforward.build(
		P_disc,
		name = "discriminate",
		input_size = 512, 
		hidden_sizes = layer_sizes[len(layer_sizes)/2:],
		output_size = output_size,
		activation=T.nnet.sigmoid
	)
	mean, logvar, latent = sample_encode(X)
	lin_output = discriminate(latent)
	output = T.nnet.softmax(lin_output)

	f = theano.function(
			inputs = [X],
			outputs = T.log(output) - T.log(counts/T.sum(counts))
		)

	P_vae.load(filename_vae)
	P_disc.load(filename_disc)

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
	model_file = sys.argv[2].split(',')
	counts_file = sys.argv[3]
	counts = None
	predict = None

	input_size = structure[0]
	output_size = structure[-1]
	layer_sizes = structure[1:-1]
	with open(counts_file) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])
	predict = create_model(model_file[0],model_file[1],counts,input_size,layer_sizes,output_size)

	if predict != None:
		for name,frames in ark_stream():
			print_ark(name,predict(frames))

