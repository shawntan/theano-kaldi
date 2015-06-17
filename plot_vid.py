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



	fig = plt.figure(figsize=(10,7))

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
	def run(t):
		ax = plt.subplot2grid((7,9),(0,0),colspan=9)
		ax.plot(np.arange(data.shape[0]),data)
		
		cum_loc = 0
		for phn,duration in grp_phn_seq:
			ax.annotate(
					phn,
					xy=(int(data.shape[0] * cum_loc/float(hiddens.shape[0])),np.max(data)*0.1)
				)
			cum_loc += duration

		t_loc = int(data.shape[0] * t/float(hiddens.shape[0]))
		ax.set_xlim(0,data.shape[0])
		ax.axvline(x=t_loc,linewidth=4, color='r')
		activations = hiddens[t].T
		for i in xrange(6):
			ax = plt.subplot2grid((7,9),(1 + (i//3)*3,(i%3)*3),colspan=3,rowspan=3)
			activation = activations[i]
			activation_grid = activation.reshape(32,32)
			ax.imshow(activation_grid,cmap=cm.Reds,interpolation="nearest",vmax=hidden_max[i],vmin=hidden_min[i])
			#axs.append(ax)
	ani = animation.FuncAnimation(fig, run, frames.shape[0], repeat=False, blit=True)
	ani.save(config.output_file, fps=10,bitrate=1024,dpi=200)#, extra_args=['-vcodec','libx264'])

