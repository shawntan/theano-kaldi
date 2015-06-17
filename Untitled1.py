import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

def variance_inverse(covariance_tensor):
	inverted,_ = theano.map(
			T.nlinalg.matrix_inverse,
			sequences = [covariance_tensor]
			)
	return inverted

def variance_determinant(covariance_tensor):
	determinants,_ = theano.map(
			T.nlinalg.det,
			sequences = [covariance_tensor]
			)
	return determinants

def build(params,name,phonemes,components,rows,cols):
	n_hidden = rows * cols
	mean_arr = np.zeros((phonemes,components,2))
	mean_arr[:,0,:] = [ 7.5, 7.5]
	mean_arr[:,1,:] = [ 7.5,22.5]
	mean_arr[:,2,:] = [22.5, 7.5]
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
	var_matrices = phoneme_var.reshape((phonemes * components,2,2))
	phoneme_precisions = variance_inverse(var_matrices).reshape((phonemes,components,2,2))
	phoneme_determinants = variance_determinant(var_matrices).reshape((phonemes,components)) # phonemes x components

	points = theano.shared(np.dstack(np.meshgrid(
		np.arange(cols),np.arange(rows)
		)).reshape(n_hidden,2))                                            # n_hidden x 2

	phoneme_means_      = phoneme_means.dimshuffle(0,1,'x',2)          # phonemes x components x 1 x 2
	phoneme_precisions_ = phoneme_precisions.dimshuffle(0,1,'x',2,3)   # phonemes x components x 1 x 2 x 2
	phoneme_mixtures_   = phoneme_mixtures.dimshuffle(0,1,'x')
	points_ = points.dimshuffle('x','x',0,1)                           # 1 x 1 x n_hidden x 2
	points_sq_ = points_.dimshuffle(0,1,2,3,'x') \
			   * points_.dimshuffle(0,1,2,'x',3)                       # 1 x 1 x n_hidden x 2 x 2

	deviations = points_ - phoneme_means_                              #  phonemes x components x n_hidden x 2
	deviations_ = deviations.dimshuffle(0,1,2,'x',3)                   # phonemes x components x n_hidden x 1 x 2
	sigma_deviations = T.sum(phoneme_precisions_ * deviations_,axis=3) # phonemes x components x n_hidden x 2 
	score = T.sum(deviations * sigma_deviations,axis=3)                # phonemes x components x n_hidden
	gaussians = T.exp(- 0.5 * score)\
			/ (T.sqrt((2 * np.pi)**2 * phoneme_determinants)).dimshuffle(0,1,'x')
																	   # phonemes x components x n_hidden
	weighted_gaussians = gaussians * phoneme_mixtures_



	stats_w_acc    = theano.shared(np.zeros((phonemes,components)))		# phonemes x components
	stats_mean_acc = theano.shared(np.zeros((phonemes,components,2)))	# phonemes x components x 2
	stats_var_acc  = theano.shared(np.zeros((phonemes,components,2,2)))	# phonemes x components x 2 x 2

	def updates(hidden,phonemes):
		frame_gaussians = weighted_gaussians[phonemes]          # N x components x n_hidden
		hidden_ = hidden.dimshuffle(0,'x',1)                    # N x 1 x n_hidden
		gamma   = frame_gaussians / T.sum(frame_gaussians,axis=1).dimshuffle(0,'x',1) # N x components x n_hidden
		contribution   = gamma * hidden_                        # N x components x n_hidden
		contribution_  = contribution.dimshuffle(0,1,2,'x')     # N x components x n_hidden x 1
		contribution__ = contribution.dimshuffle(0,1,2,'x','x') # N x components x n_hidden x 1 x 1
		point_contri = contribution_ *  points_                 # N x components x n_hidden x 2
		vari_contri = contribution__ * points_sq_

		stats_w    = T.sum(contribution,axis=2) # N x components
		stats_mean = T.sum(point_contri,axis=2) # N x components x 2
		stats_var  = T.sum(vari_contri,axis=2)  # N x components x 2 x 2

		stats_updates = [
				(stats_w_acc,    T.inc_subtensor(stats_w_acc[phonemes],stats_w)),
				(stats_mean_acc, T.inc_subtensor(stats_mean_acc[phonemes],stats_mean)),
				(stats_var_acc , T.inc_subtensor(stats_var_acc[phonemes],stats_var)),
				]

		stats_w_acc_ = stats_w_acc.dimshuffle(0,1,'x')
		stats_w_acc__ = stats_w_acc.dimshuffle(0,1,'x','x')
		phoneme_mixture_update = stats_w_acc  / T.sum(stats_w_acc,axis=1).dimshuffle(0,'x')
		phoneme_mean_update = stats_mean_acc / stats_w_acc_
		phoneme_mean_sq = phoneme_mean_update.dimshuffle(0,1,'x',2)                         * phoneme_mean_update.dimshuffle(0,1,2,'x') 
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
		mog_ = T.sum(weighted_gaussians,axis=1)
		mog = mog_ / T.sum(mog_,axis=1).dimshuffle(0,'x') 
		surface = mog[phonemes]  # N x n_hidden
		norm_hidden = hidden / T.sum(hidden,axis=1).dimshuffle(0,'x')
		return -T.mean(T.sum(hidden * T.log(surface),axis=1))

	params["phoneme_mixtures_%s"%name] = phoneme_mixtures
	params["phoneme_means_%s"%name]    = phoneme_means
	params["phoneme_var_%s"%name]      = phoneme_var
	phoneme_mixtures.name = "phoneme_mixtures_%s"%name
	phoneme_means.name = "phoneme_means_%s"%name
	phoneme_var.name = "phoneme_var_%s"%name

	return constraint,updates

if __name__ == "__main__":
	# In[307]:

	params = {}
	constraints,updates = build(params,"test",1,4,32,32)

	X = T.matrix('X')
	Y = T.ivector('Y')

	stats_updates,param_updates = updates(X,Y)
	update_stats = theano.function(
			inputs=[X,Y],
			updates = stats_updates,
			outputs = constraints(X,Y)
			)
	update_params = theano.function(
			inputs = [],
			updates = param_updates
			)


	# In[ ]:




	# In[303]:

	data = np.zeros((128,1024),dtype=np.float32)
	data_view = data.reshape(128,32,32)
	points = np.rint(np.random.randn(128,2) + [5,5]).astype(np.int32)
	data_view[np.arange(128),points[:,0],points[:,1]] = 10

	points = np.rint(np.random.randn(128,2) + [21,5]).astype(np.int32)
	data_view[np.arange(128),points[:,0],points[:,1]] = 10

	points = np.rint(np.random.randn(128,2) + [7,21]).astype(np.int32)
	data_view[np.arange(128),points[:,0],points[:,1]] = 10
	plt.imshow(data.mean(axis=0).reshape(32,32),interpolation='nearest')

	data = data.mean(axis=0).reshape(1,1024) + np.random.rand(1,1024)


	# In[348]:

	l = np.zeros((1,),dtype=np.int32)
	for _ in xrange(100):
		for _ in xrange(100):update_stats(data,l)
		update_stats(data,l)
		update_params()


	# In[ ]:




	# In[346]:

	print params["phoneme_means_test"].get_value()
	print params["phoneme_var_test"].get_value()
	print params["phoneme_mixtures_test"].get_value()


	# In[ ]:

	# Check negative log likelihood goes down
	# Try using gaussian data

