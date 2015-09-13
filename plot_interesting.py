import config
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

import sys
import model
import theano
import theano.tensor as T
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gzip
import cPickle as pickle
import itertools
import scipy.io.wavfile
import glob

def norm(W):
	return T.sqrt(T.sum(W**2,axis=1)).dimshuffle('x',0)


if __name__ == "__main__":
	params = {}
	feedforward = model.build_feedforward(params,input_size=1,layer_sizes=[1]*6,output_size=1)
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
		inputs=[X],
		outputs=act_surface
	)

	with gzip.open(config.frames_files[0]) as f:
		name_f,frames = pickle.load(f)
	with gzip.open(config.labels_files[0]) as f:
	    name_l,phns= pickle.load(f)
	assert(name_f == name_l)
	spkr_id,utt_id = name_l.split("_")
	wav_file = glob.glob("/home/shawn/timit_wav/TRAIN/*/%s/%s.WAV"%(spkr_id,utt_id))[0]
	_,data = scipy.io.wavfile.read(wav_file)


	print wav_file
	print name_f,name_l

	# Load mapping
	id2phn = {}
	with open('data/lang/phones.txt') as phone_file:
		for line in phone_file:
			phone,id_str = line.strip().split()
			id2phn[int(id_str)] = phone
	phn_seq      = [id2phn[id] for id in phns]
	grp_phn_seq  = [ (phn,len(list(grp))) for phn,grp in itertools.groupby(phn_seq) ]

	cum_grp_phn_seq = []
	prev_t = 0
	for phn, frame_count in grp_phn_seq:
		cum_grp_phn_seq.append(
				(phn,prev_t,prev_t + frame_count)
			)
		prev_t += frame_count
	sorted_grp_phn_seq = sorted(cum_grp_phn_seq,key=lambda x: x[2]-x[1])
	snapshot_grp_phn_seq = sorted(sorted_grp_phn_seq[-5:],key=lambda x:x[1])

	print snapshot_grp_phn_seq
	model.load(config.args.model_location,params)
	hiddens = np.dstack(prop(frames)) 		# frames x hiddens x layers

	#hidden_range = np.max(hiddens,axis=0) - np.min(hiddens,axis=0)
	#hiddens = (hiddens - np.min(hiddens,axis=0))/hidden_range
	#hiddens = hiddens - 0.5

	hidden_max = np.max(hiddens,axis=(0,1))	# layers
	hidden_min = np.min(hiddens,axis=(0,1))	# layers

	print hidden_max
	print hidden_min
	print grp_phn_seq
	
	snapshots = [ (t + et)/2 for _,t,et in snapshot_grp_phn_seq ]
	snapshots = [6, 40, 59, 89, 128]
	snapshots_phn = [ "sil", "ow", "v", "s", "iy" ]
	print snapshots
	plot_grid_width = 3
	plot_grid_height = 3
	grid_cols = plot_grid_width * len(snapshots)
	grid_rows = 1 + 6 * plot_grid_height

	fig = plt.figure(figsize=(grid_cols,grid_rows))

	wv_ax = plt.subplot2grid((grid_rows,grid_cols),(0,0),colspan=grid_cols)
	wv_ax.plot(np.arange(data.shape[0]),data)
	cum_loc = 0
	min_y, max_y = min_max_y = [ np.min(data), np.max(data) ]
	for i in xrange(len(grp_phn_seq)):
		phn,duration = grp_phn_seq[i]
		wv_ax.annotate(
				phn,
				xy=(
					int(data.shape[0] * ( cum_loc + duration * 0.5 )/\
								float(hiddens.shape[0])),
					max_y  * 1.25
				),
				xytext=(
					int(data.shape[0] * ( cum_loc + duration * 0.5 )/\
								float(hiddens.shape[0])),
					max_y * 1.3
				),
				ha="center"
			)
		if i % 2 == 0:
			wv_ax.axvspan(
					int(data.shape[0] * cum_loc/float(hiddens.shape[0])), 
					int(data.shape[0] * (cum_loc+duration)/float(hiddens.shape[0])), 
					facecolor='grey', alpha=0.5
				)
		cum_loc += duration

	for col, t in enumerate(snapshots):
		t_loc = int(data.shape[0] * t/float(hiddens.shape[0]))
		wv_ax.set_xlim(0,data.shape[0])
		wv_ax.axvline(x=t_loc,linewidth=4, color='r')
		wv_ax.axes.get_yaxis().set_visible(False)
		wv_ax.axes.get_xaxis().set_visible(False)

		activations = hiddens[t].T
		for layer in xrange(6):
			ax = plt.subplot2grid(
					(grid_rows,grid_cols),
					(
						1 + layer * plot_grid_height,
						col * plot_grid_width
					),
					colspan=plot_grid_width,
					rowspan=plot_grid_height
				)

			activation = activations[layer]
			activation_grid = activation.reshape(32,32)
			ax.axes.get_yaxis().set_ticks([])
			ax.axes.get_xaxis().set_ticks([])
			if col == 0:
				ax.set_ylabel('Layer %d'% (1+layer))
			if layer == 5:
				ax.set_xlabel(snapshots_phn[col])


			ax.imshow(
					activation_grid,
					cmap=cm.Reds,
					interpolation="nearest",
					vmax=hidden_max[layer],
					vmin=hidden_min[layer]
				)

	plt.savefig(config.output_file,bbox_inches='tight')

