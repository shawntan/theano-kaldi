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
config.parse_args()


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
			outputs = hiddens[1:] #hidden_contri
		)
	print "Compiled functions."
	model.load(config.args.model_location,params)
	
	split_streams = [ data_io.stream(f,l,with_name=True) for f,l in izip(frames_files,labels_files) ]
	stream = chain(*split_streams)
	spkr_phoneme = {}
	for name,f,l in stream:
		spkr_id,_ = name.split('_')
		hiddens = np.dstack(prop(f))
		for i in xrange(f.shape[0]):
			phoneme_frame = spkr_phoneme[spkr_id] = spkr_phoneme.get(spkr_id,{})
			phoneme_frame[l[i]] = phoneme_frame.get(l[i],0) + hiddens[i]

	gaussian_ctr = pickle.load(open('gaussian_ctr.pkl'))
	phns = { int(line.strip().split()[-1]) : line.strip().split()[0]
				for line in open('exp/tri3/graph/phones.txt') }
	for spkr in spkr_phoneme:
		phoneme_frame = spkr_phoneme[spkr]
		for key,value in phoneme_frame.iteritems():
			value = value.T
			for layer,activation in enumerate(value):
				activation_grid = activation.reshape(32,32)

				plt.imshow(activation_grid,cmap=cm.Reds,interpolation="nearest")
				plt.scatter(gaussian_ctr[:,0],gaussian_ctr[:,1])

				for i in xrange(gaussian_ctr.shape[0]):
					plt.annotate(phns[i+1],(gaussian_ctr[i,0],gaussian_ctr[i,1]))

				plt.savefig(config.output_file + "/spkr-%s-layer-%d-phn-%d.png"%(spkr,layer,key))
				plt.clf()
	
