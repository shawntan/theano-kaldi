import config
config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
config.parser.add_argument(
		'--model-location',
		dest = 'model_location',
		required = True,
		type = str,
		help = ".pkl file for DNN model."
	)
config.parser.add_argument(
		'--constraint-surface',
		required = True,
		dest = 'constraint_surface',
		type = str,
		help = "Constraint surface."
	)
config.parser.add_argument(
		'--spk2utt',
		required = True,
		dest = 'spk2utt',
		type = str,
		help = "spk2utt file."
	)

config.parse_args()


config.parse_args()

import theano
import theano.tensor as T

from itertools import izip
import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
import em_test as em
from pprint import pprint
from itertools import izip, chain

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import constraint

def norm(W):
	return T.sqrt(T.sum(W**2,axis=1)).dimshuffle('x',0)


if __name__ == "__main__":
	frames_files = config.frames_files
	labels_files = config.labels_files
	params = {}
	feedforward = model.build_feedforward(params)
	X = T.matrix('X')
	hiddens,outputs = feedforward(X)

	act_surface = hiddens[1:]
	if config.args.constraint_surface == "raw":
		pass
	elif config.args.constraint_surface == "norm":
		norms = [ norm(params["W_hidden_%d"%i]) for i in xrange(1,len(hiddens)-1) ]
		act_surface = [ h * n for h,n in zip(act_surface[:-1],norms) ] + [hiddens[-1]]
	elif config.args.constraint_surface == "h0scale":
		hiddens_0,_ = feedforward(X*0)
		act_surface = [ h / h0 for h,h0 in zip(act_surface,hiddens_0[1:]) ]
	elif config.args.constraint_surface == "meanscale":
		hidden_avg = [ T.mean(h,axis=0).dimshuffle('x',0) for h in act_surface ]
		act_surface = [ h / m for h,m in zip(act_surface,hidden_avg) ]
	prop = theano.function(
			inputs = [X],
			outputs = act_surface
		)
	print "Compiled functions for context stats."
	model.load(config.args.model_location,params)

	split_streams = [ data_io.stream(f,l,with_name=True) for f,l in izip(frames_files,labels_files) ]
	stream = chain(*split_streams)
	seen = 0
	
	activations = {}
	counts = {}
	for name,f,l in stream:
		hidden_act = np.array(prop(f))
		for t in xrange(len(l)):
			lbl = tuple(l[t])
			activations[lbl] = activations.get(lbl,0) + hidden_act[:,t]
			counts[lbl] = counts.get(lbl,0) + 1
		seen += 1
		if seen % 100 == 0:
			print "Processed",seen
		
		pickle.dump(
				{ 
					"activations":activations,
					"counts":counts
				},open(config.output_file,'wb'),2
			)

