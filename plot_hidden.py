import config
config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
config.parser.add_argument(
		'--model-file',
		required = True,
		dest = 'model_file',
		type = str,
		help = ".pkl file containing trained model"
	)
config.parse_args()
import theano
import theano.tensor as T

import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import constraint
if __name__ == "__main__":

	frames_files = config.frames_files
	labels_files = config.labels_files
	params = {}
	feedforward = model.build_feedforward(params)
	X = T.matrix('X')
	hiddens,outputs = feedforward(X)
	
	prop = theano.function(
			inputs = [X],
			outputs = hiddens[1:]
		)
	print "Compiled functions."
	model.load(config.args.model_file,params)
	
	split_streams = [ data_io.stream(f,l) for f,l in izip(frames_files,labels_files) ]
	stream = chain(*split_streams)
	phoneme_frame = {}
	for f,l in stream:
		hiddens = np.dstack(prop(f))
		for i in xrange(f.shape[0]):
			phoneme_frame[l[i]] = phoneme_frame.get(l[i],0) + hiddens[i]

	for key,value in phoneme_frame.iteritems():
		value = value.T
		for layer,activation in enumerate(value):
			activation_grid = activation.reshape(32,32)
			plt.imshow(activation_grid,cmap=cm.Reds,interpolation="nearest")
			plt.savefig(config.output_file + "/layer-%d-phn-%d.png"%(layer,key))

