import sys
import gzip
import cPickle as pickle

import numpy as np

import theano
import theano.tensor as T

from theano_toolkit.parameters import Parameters
from theano.tensor.shared_randomstreams import RandomStreams

def data_stream(data_file):
	with gzip.open(data_file,'rb') as f:
		data_stream = iter(lambda:pickle.load(f),'')
		try:
			for name,data in data_stream:
				yield data
		except:
			pass
		
def build_model(P):
	P.W_hidden_0 = np.zeros((360,1024))
	P.W_hidden_1 = np.zeros((1024,1024))
	P.W_hidden_2 = np.zeros((1024,1024))
	P.W_hidden_3 = np.zeros((1024,1024))
	P.b_hidden_0 = np.zeros((1024,))
	P.b_hidden_1 = np.zeros((1024,))
	P.b_hidden_2 = np.zeros((1024,))
	P.b_hidden_3 = np.zeros((1024,))
	def predict(X):
		hidden_0 = T.nnet.sigmoid(T.dot(X,P.W_hidden_0) + P.b_hidden_0)
		hidden_1 = T.nnet.sigmoid(T.dot(hidden_0,P.W_hidden_1) + P.b_hidden_1)
		hidden_2 = T.nnet.sigmoid(T.dot(hidden_1,P.W_hidden_2) + P.b_hidden_2)
		hidden_3 = T.nnet.sigmoid(T.dot(hidden_2,P.W_hidden_3) + P.b_hidden_3)
		return hidden_3
	return predict

	

if __name__ == "__main__":
	data_file  = sys.argv[1]
	model_file = sys.argv[2]

	P = Parameters()
	X = T.matrix('X')
	model = build_model(P)

	numpy_rng = np.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	corr_X = theano_rng.binomial(size=X.shape,n=1,p=0.5,dtype=theano.config.floatX) * X
	
	corr_layer = model(corr_X)
	layer = model(X)
	
	kl_divergence = -T.sum(T.sum(layer * T.log(corr_layer) + (1 - layer) * T.log(1 - corr_layer), axis=1))
	f = theano.function(
			inputs = [X],
			outputs = kl_divergence
		)
	P.load(model_file)
	total_diff = 0
	total_frames = 0
	for data in data_stream(data_file):
		length = data.shape[0]
		total_diff += f(data)
		total_frames += length

	print total_diff / total_frames



	


