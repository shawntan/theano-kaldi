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
config.parser.add_argument(
		'--phonemes-files',
		nargs = '+',
		dest = 'phonemes_files',
		required = True,
		type = str,
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
	phonemes_files = config.args.phonemes_files
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
	gmm_params = {}
	gmm_constraint,em_updates = em.build(
		params = gmm_params,
		name = "constraint",
		phonemes = 48,
		components = 1,
		rows = 32,cols = 32
	)
#	kl_divergences = [ gmm_constraint(s,P-1) for s in act_surface ]

	prop = theano.function(
			inputs = [X],
			outputs = act_surface
		)
	print "Compiled functions."
	model.load(config.args.model_location,params)

#	gaussian_ctr = pickle.load(open('gaussian_ctr.pkl'))
#	gmm_params["phoneme_means_constraint"].set_value(gaussian_ctr.reshape(48,1,2))
	spkr_ids = [ l.split()[0] for l in open(config.args.spk2utt,'r') ]
	spkr2idx = { s:i for i,s in enumerate(spkr_ids) }	
	split_streams = [ data_io.stream(f,l,p,with_name=True) for f,l,p in izip(frames_files,labels_files,phonemes_files) ]
	stream = chain(*split_streams)
	
	stats = np.zeros((
			len(config.layer_sizes),
			len(spkr_ids),
			config.output_size,
			config.layer_sizes[0]
		),
		dtype=np.float32)
	counts = np.zeros((
			len(spkr_ids),
			config.output_size,
		),dtype=np.int32)

	pdfphoneme = {}
	seen = 0
	for name,f,l,p in stream:
		hidden_act = prop(f)
		spkr_id = spkr2idx[name.split('_')[0]]
#		print spkr_id
#		print l
		for layer in xrange(len(config.layer_sizes)):
			np.add.at(stats,(layer,spkr_id,l),hidden_act[layer])
		np.add.at(counts,(spkr_id,l),1)

		for pair in izip(l,p):pdfphoneme[pair] = pdfphoneme.get(pair,0) + 1
			
		seen += 1
		if seen % 100 == 0:
			print "Processed",seen

		
	

	for layer in xrange(len(config.layer_sizes)):
		print "Writing",config.output_file+".%d"%layer
		np.save(open(config.output_file+".%d"%layer,'wb'),stats[layer])
	np.save(open(config.output_file+".counts",'wb'),counts)
	pickle.dump(pdfphoneme,open(config.output_file+".pairs",'wb'))


