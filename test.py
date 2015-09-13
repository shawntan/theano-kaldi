import config
config.parser.add_argument(
		'--model-file',
		dest = 'model_file',
		required = True,
		type = str,
		help = ".pkl containing model parameters."
	)
config.parser.add_argument(
		'--mute-phoneme',
		dest = 'mute_phoneme',
		required = True,
		type = int,
		help = "Phoneme muting",
		default = -1
	)

config.parse_args()

import theano
import theano.tensor as T
import constraint
import numpy as np
import math
import sys
import em_test as em
import data_io
import updates
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain

if __name__ == "__main__":
	frames_files = config.frames_files
	labels_files = config.labels_files

	X = T.matrix('X')
	Y = T.ivector('Y')
	params = {}
	gmm_params = {}
	if config.args.mute_phoneme == -1:
		import model_mute as model
		feedforward = model.build_feedforward(params)
		mask = em.mask(
			params = gmm_params,
			name = "-1",
			phonemes = 48,
			components = 1,
			rows = 32,cols = 32
		)
		_,outputs = feedforward(X,mask(config.args.mute_phoneme))
	else:
		import model
		feedforward = model.build_feedforward(params)
		_,outputs = feedforward(X)
	gaussian_ctr = pickle.load(open('gaussian_ctr.pkl'))
	model.load(config.args.model_file,params)
	gmm_params["phoneme_means_%d"%-1].set_value(
			gaussian_ctr.reshape(48,1,2).astype(np.float32)
		)


	test = theano.function(
			inputs = [X,Y],
			outputs = T.eq(T.argmax(outputs,axis=1),Y)
		)

	total_errors = np.zeros((config.output_size,),dtype=np.int32)
	total_counts = np.zeros((config.output_size,),dtype=np.int32)
	split_streams = [ data_io.stream(f,l) for f,l in izip(frames_files,labels_files) ]
	stream = chain(*split_streams)
	for f,l in stream:
		test_outputs = test(f,l)
		np.add.at(total_errors,l,test_outputs)
		np.add.at(total_counts,l,1)
	
	pickle.dump({
			'errors' : total_errors,
			'counts' : total_counts,
		},open(config.output_file,'wb'))

