import data_io
import glob
from itertools import chain
import model
import theano
import theano.tensor as T

import numpy as np
import cPickle as pickle

def norm(W):
	return T.sqrt(T.sum(W**2,axis=1)).dimshuffle('x',0)



if __name__ == "__main__":
	files = glob.glob('../exp/dnn_fmllr_tk_feedforward/pkl/train_ctx.*.pklgz')
	streams = [ data_io.stream(f) for f in files ]
	stream = chain(*streams)
	unique = set()
	for (frames,) in stream:
		for i in xrange(frames.shape[0]):
			unique.add(tuple(frames[i]))

	params = {}
	feedforward = model.build_feedforward(params,layer_sizes=[1024]*6)
	X = T.matrix('X')
	P = T.imatrix('P')
	hiddens,outputs = feedforward(X)

	act_surface = hiddens[1:]
	norms = [ norm(params["W_hidden_%d"%i]) for i in xrange(1,len(hiddens)-1) ]
	act_surface = [ h * n for h,n in zip(act_surface[:-1],norms) ] + [hiddens[-1]]

	import ctx_gaussians
	gaussian_ctr = pickle.load(open('../gaussian_ctr.pkl'))
	gmm_params = {}
	constraint = ctx_gaussians.build(
			params = gmm_params,
			name = str('all'),
			means = gaussian_ctr,
			rows = 32,cols = 32
		)

	kl_divergence = [ constraint(s,P-1) for s in act_surface ]
	score = theano.function(
			inputs = [X,P],
			outputs = [ T.mean(d) for d in kl_divergence ],
		)
	prop = theano.function(
			inputs = [X],
			outputs = act_surface
		)
	print "Compiled functions."
	model.load("../exp/dnn_fmllr_tk_feedforward/ctx_phoneme_gaussian_tsne_layer--1_coeff-0.5-init_norm_spread-1.5/dnn.pkl",params)

	ctx_files = glob.glob('../exp/dnn_fmllr_tk_feedforward/pkl/dev_ctx.*.pklgz')
	frm_files = glob.glob('../exp/dnn_fmllr_tk_feedforward/pkl/dev.*.pklgz')
	ctx_files.sort()
	frm_files.sort()
	streams = [ data_io.stream(c,f) for c,f in zip(ctx_files,frm_files) ]
	stream = chain(*streams)
	frame_bins = {}
	for ctxs, frames in stream:
		for i in xrange(ctxs.shape[0]):
			context = tuple(ctxs[i])
			val = frame_bins.get(context,[])
			val.append(frames[i])
			frame_bins[context] = val
	
	print frame_bins.keys()
	average_frames = {}
	cost_of_frames = {}
	for ctx,frames in frame_bins.iteritems():
		frames = np.array(frames)           # n x 440
		output = np.array(prop(frames)) # 6 x n x 1024
		average_frames[ctx] = np.mean(output,axis=1)
		cost_of_frames[ctx] = np.array(score(frames,np.array(frames.shape[0]*[ctx])))
	
	pickle.dump(average_frames,open('average_frames.pkl','wb'))
	pickle.dump(cost_of_frames,open('cost_of_frames.pkl','wb'))
	pickle.dump(unique,open('seen_context.pkl','wb'))
