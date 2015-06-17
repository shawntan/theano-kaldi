import numpy as np
import theano
import theano.tensor as T

def variance_inverse(covariance_tensor):
	inverted,_ = theano.map(
			T.nlinalg.matrix_inverse,
			sequences = [covariance_tensor]
		)
	return inverted

def build(params,name,phonemes,components,rows,cols):
	n_hidden = rows * cols
	mean_arr = np.zeros((phonemes,components,2))
	mean_arr[:,0,:] = [ 7.5, 7.5]
	mean_arr[:,1,:] = [ 7.5,22.5] mean_arr[:,2,:] = [22.5, 7.5]
	mean_arr[:,3,:] = [22.5,22.5]

	phoneme_mixtures = theano.shared(
	                        np.ones((phonemes,components))/components
	                    )                       # phonemes x components
	phoneme_means    = theano.shared(mean_arr)  # phonemes x components x 2
	phoneme_var      = theano.shared(
	                        np.array(phonemes * [
	                            components * [ np.eye(2) ]
	                        ])
	                    )                       # phonemes x components x 2 x 2

	phoneme_precisions = variance_inverse(
	        phoneme_var.reshape((phonemes * components,2,2))
	    ).reshape((phonemes,components,2,2))

	points = theano.shared(np.dstack(np.meshgrid(
	    np.arange(cols),np.arange(rows)
	)).reshape(n_hidden,2))                                            # n_hidden x 2

	phoneme_means_      = phoneme_means.dimshuffle(0,1,'x',2)          # phonemes x components x 1 x 2
	phoneme_precisions_ = phoneme_precisions.dimshuffle(0,1,'x',2,3)   # phonemes x components x 1 x 2 x 2
	phoneme_mixtures_   = phoneme_mixtures.dimshuffle(0,1,'x')
	points_ = points.dimshuffle('x','x',0,1)                           # 1 x 1 x n_hidden x 2
	deviations = points_ - phoneme_means_                              # phonemes x components x n_hidden x 2
	deviations_ = deviations.dimshuffle(0,1,2,'x',3)                   # phonemes x components x n_hidden x 1 x 2
	normalised_deviations = T.sum(phoneme_precisions_ * deviations_,axis=3) # phonemes x components x n_hidden x 2 
	gaussians = T.exp(-T.sum(normalised_deviations**2,axis=3))         # phonemes x components x n_hidden
	norm_gaussians = gaussians / T.sum(gaussians,axis=2).dimshuffle(0,1,'x')
	weighted_norm_gaussians = norm_gaussians * phoneme_mixtures_



	stats_w_acc    = theano.shared(np.zeros((phonemes,components)))		# phonemes x components
	stats_mean_acc = theano.shared(np.zeros((phonemes,components,2)))	# phonemes x components x 2
	stats_var_acc  = theano.shared(np.zeros((phonemes,components,2,2)))	# phonemes x components x 2 x 2

	def updates(hidden,phonemes):
	    frame_gaussians = weighted_norm_gaussians[phonemes] 	# N x components x n_hidden
	    hidden_ = hidden.dimshuffle(0,'x',1)		# N x n_hidden x 1
	    gamma   = frame_gaussians / T.sum(frame_gaussians,axis=1).dimshuffle(0,'x',1) # N x components x n_hidden
	    contribution   = gamma * hidden_                        # N x components x n_hidden
	    contribution_  = contribution.dimshuffle(0,1,2,'x')     # N x components x n_hidden x 1
	    contribution__ = contribution.dimshuffle(0,1,2,'x','x') # N x components x n_hidden x 1 x 1
	    point_contri = contribution_ *  points_                 # N x components x n_hidden x 2
	    vari_contri = point_contri.dimshuffle(0,1,2,3,'x') \
	                * point_contri.dimshuffle(0,1,2,'x',3)

	    stats_w    = T.sum(contribution,axis=2) # N x components
	    stats_mean = T.sum(point_contri,axis=2) # N x components x 2
	    stats_var  = T.sum(vari_contri,axis=2)  # N x components x 2 x 2

	    stats_updates = [
	        (stats_w_acc,    T.inc_subtensor(stats_w_acc[phonemes],stats_w)),
	        (stats_mean_acc, T.inc_subtensor(stats_mean_acc[phonemes],stats_mean)),
	        (stats_var_acc , T.inc_subtensor(stats_var_acc[phonemes],stats_var)),
	    ]
	    
	    eps = 1e-8
	    stats_w_acc_ = stats_w_acc.dimshuffle(0,1,'x')
	    stats_w_acc__ = stats_w_acc.dimshuffle(0,1,'x','x')
	    phoneme_mixture_update = stats_w_acc  / T.sum(stats_w_acc,axis=1).dimshuffle(0,'x')
	    phoneme_mean_update = stats_mean_acc / stats_w_acc_
	    phoneme_mean_sq = phoneme_mean_update.dimshuffle(0,1,'x',2) \
	                    * phoneme_mean_update.dimshuffle(0,1,2,'x') 
	    phoneme_var_update =  stats_var_acc / stats_w_acc__ - phoneme_mean_sq
	    param_updates = [
	        (phoneme_mixtures, phoneme_mixture_update),
	        (phoneme_means,    phoneme_mean_update),
	        (phoneme_var,      phoneme_var_update),
	        (stats_w_acc,      0. * stats_w_acc),
	        (stats_mean_acc,   0. * stats_mean_acc),
	        (stats_var_acc ,   0. * stats_var_acc),
	    ]
	    
	    return stats_updates,param_updates
	


	def constraint(hidden,phonemes):
	    
	    mog = T.sum(weighted_norm_gaussians,axis=1)
	    surface = mog[phonemes]  # N x n_hidden
	    norm_hidden = hidden / T.sum(hidden,axis=1).dimshuffle(0,'x')
	    return -mog[phonemes] * T.log(hidden)
	
	params["phoneme_mixtures_%s"%name] = phoneme_mixtures
	params["phoneme_means_%s"%name]    = phoneme_means
	params["phoneme_var_%s"%name]      = phoneme_var
	phoneme_mixtures.name = "phoneme_mixtures_%s"%name
	phoneme_means.name = "phoneme_means_%s"%name
	phoneme_var.name = "phoneme_var_%s"%name

	return constraint,updates

def test():
	from pprint import pprint
	params = {}
	constraint,updates,mog = build(
			params = params,
			name = "test",
			phonemes=1,
			components=4,
			rows=5,
			cols=5
		)

	X = T.matrix('X')
	Z = T.ivector('Z')
	hidden = T.nnet.sigmoid(X)
	stats_updates,param_updates = updates(hidden,Z)
	
	update_stats = theano.function(
			inputs=[X,Z],
			outputs=[],
			updates = stats_updates
		)
	update_params = theano.function(
			inputs=[],
			updates = param_updates
		)

#	label_mean = np.random.randint(low=0,high=32,size=(48,4,2))
	
	idxs = np.arange(128)
	for _ in xrange(10):
		for _ in xrange(100): 
			#labels = np.random.randint(48,size=128).astype(np.int32)
			labels = np.zeros((128,),dtype=np.int32)
			#rand_coords = label_mean[labels] + np.random.randint(low=-1,high=1,size=(labels.shape[0],1,2))  #128 x  4 x 2
			data = np.random.randn(128,5,5).astype(np.float32)
			data[:,0,0] = 10
			data[:,0,4] = 10
			data[:,2,2] = 10
			data[:,4,4] = 10
#			data[idxs,rand_coords[:,0,0],rand_coords[:,0,1]] += 10
#			data[idxs,rand_coords[:,1,0],rand_coords[:,1,1]] += 10
#			data[idxs,rand_coords[:,2,0],rand_coords[:,2,1]] += 10
##			data[idxs,rand_coords[:,3,0],rand_coords[:,3,1]] += 10
			print data[0]
			data = data.reshape(128,25)
			update_stats(data,labels)
		update_params()

	surface = mog.eval()
	print surface.shape
	surface = surface.reshape(surface.shape[0],5,5)
	import matplotlib.cm as cm
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	for i in xrange(surface.shape[0]):
		plt.imshow(surface[i],cmap=cm.Reds,interpolation="nearest")
		plt.savefig("mixgauss-%d.png"%i)
	
	print params["phoneme_means_test"].get_value()


if __name__ == "__main__":
	#test()
	covs = np.array(5 * [ [ np.eye(2)/(f+1) for f in range(4) ] ])
	print covs
	
	
	print variance_inverse(covs.reshape(20,2,2)).reshape((5,4,2,2)).eval()

	
