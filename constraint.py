import numpy as np
import theano
import theano.tensor as T
def adjacency(inputs,rows,cols):
	n_hidden = rows*cols
	idxs = np.arange(n_hidden).reshape(rows,cols)
	top = idxs[:-1].reshape(n_hidden - cols)
	btm = idxs[1:].reshape(n_hidden - cols)
	lft = idxs[:,:-1].reshape(n_hidden - rows)
	rgt = idxs[:,1:].reshape(n_hidden - rows)
	lft_edge = idxs[:,0].reshape(rows)
	rgt_edge = idxs[:,-1].reshape(rows)
	top_edge = idxs[0].reshape(cols)
	btm_edge = idxs[-1].reshape(cols)
	vert_const = T.sum((inputs[:,top] - inputs[:,btm])**2,axis=1)
	horz_const = T.sum((inputs[:,lft] - inputs[:,rgt])**2,axis=1)
	wrap_const = T.sum((inputs[:,lft_edge] - inputs[:,rgt_edge])**2,axis=1)\
			   + T.sum((inputs[:,top_edge] - inputs[:,btm_edge])**2,axis=1)

	return T.mean(T.sqrt(vert_const + horz_const + wrap_const))

def mute_irrelevant(inputs,labels,class_count,dim_per_class):
	log_loss = - T.log(1 - inputs)
	total_loss = T.sum(log_loss,axis=1)
	log_loss_by_class = log_loss.reshape((inputs.shape[0],class_count,dim_per_class))
	log_unloss = T.sum(log_loss_by_class[T.arange(inputs.shape[0]),labels],axis=1)
	return T.mean(log_loss - log_unloss)

def norm_penalty(W,centre,bias):
	print W,bias
	if not bias:
		return T.sum((T.sqrt(T.sum(W**2,axis=1)) - centre)**2)
	else:
		return (T.sqrt(T.sum(W**2)) - centre)**2

def covariances2precisions(covs):
	# N x components x 2 x 2
	covs_flat = covs.reshape((covs.shape[0],covs.shape[1],4)) # N x components x 4
	a = covs_flat[:,:,0] # N x components
	b = covs_flat[:,:,1] # N x components
	d = covs_flat[:,:,3] # N x components
	factor = a*d - b**2 # N x components
	factor_ = factor.dimshuffle(0,'x',1)
	covs_rev = covs_flat[:,:,::-1] # N x components x 4
	covs_rev = covs_rev * np.array([1,-1,-1,1])
	prec_flat = covs_rev / factor_
	
	prec = prec_flat.reshape((covs.shape[0],covs.shape[1],2,2))
	return prec


def gaussian_field(mixture,precs,deviations):
	"""
	mixture:    N x components
	precs:      N x components x 2 x 2
	deviations: N x n_hidden  x components x 2
	"""	
	precs_ = precs.dimshuffle(0,'x',1,2,3) # N x 1 x  components x 2 x 2
	deviations_ = deviations.dimshuffle(0,1,2,'x',3)
	norm_deviations = T.sum(
		precs_ * deviations_,   # N x n_hidden x components x 2 x 2
		axis = 3				# N x n_hidden x components x 2 
	)
	
	gaussians = T.exp(-T.sum(norm_deviations * deviations, # N x n_hidden x components x 2 
							axis=3))		    # N x n_hidden x components
	mixture_ = mixture.dimshuffle(0,'x',1)		# N x 1        x components
	field = T.sum(gaussians * mixture_,axis=2)  # N x n_hidden
	norm_field = field / T.sum(field,axis=1).dimshuffle(0,'x')
	return norm_field

	
def fixed_gaussian(params,name,inputs,rows,cols,components):
	"""
	inputs - N x n_hidden
	"""
	n_hidden = rows*cols
	contrib_weights = np.zeros((n_hidden,components),dtype=np.float32) - 2
	idxs = np.arange(n_hidden)
	points = theano.shared(np.dstack(np.meshgrid(
		np.arange(cols),np.arange(rows)
	)).reshape(n_hidden,2))							                # n_hidden x 2
	
	covariance_arr = np.zeros((components,2,2),dtype=np.float32)    # components x 2 x 2
	covariance_arr[:,0,0] = 0.25
	covariance_arr[:,1,1] = 0.25
	mean_arr = np.zeros((components,2),dtype=np.float32)			# components x 2
	mean_arr[0,:] = [ 7.5, 7.5]
	mean_arr[1,:] = [22.5, 7.5]
	mean_arr[2,:] = [ 7.5,22.5]
	mean_arr[3,:] = [22.5,22.5]
	covariance = theano.shared(covariance_arr,name="covariance_%s"%name)
	mean       = theano.shared(mean_arr,name="mean_%s"%name)

	points_     = points.dimshuffle(0,'x',1)		# n_hidden x 1 x 2
	covariance_ = covariance.dimshuffle('x',0,1,2)	# 1 x components x 2 x 2
	mean_       = mean.dimshuffle('x',0,1)          # 1 x components x 2
	
	deviations = points_ - mean_					# n_hidden x components x 2
	deviations_ = deviations.dimshuffle(0,1,'x',2)  # n_hidden x components x 1 x 2
	normalised_deviations = T.sum(covariance_ * deviations_,axis=2)		# n_hidden x components x 2
	_gaussians = T.exp(-T.sum(normalised_deviations**2,axis=2))			# n_hidden x components
	gaussians = _gaussians / T.sum(_gaussians,axis=0)					# n_hidden x components
	membership = gaussians / T.sum(gaussians,axis=1).dimshuffle(0,'x')	# n_hidden x components

	membership_ = membership.dimshuffle('x',0,1)						# 1 x n_hidden x components	
	inputs_ = inputs.dimshuffle(0,1,'x')								# N x n_hidden x 1
	
	_mixture = T.sum(membership_ * inputs_,axis=1) 				# N x components
	mixture = _mixture / T.sum(_mixture,axis=1).dimshuffle(0,'x')	# N x components
	mixture_ = mixture.dimshuffle(0,'x',1)					# N x 1        x components
	gaussians_ = gaussians.dimshuffle('x',0,1)				# 1 x n_hidden x components
	surfaces = T.sum(mixture_ * gaussians_,axis=2)			# N x n_hidden

	norm_inputs = inputs / T.sum(inputs_,axis=1)
	kl_divergence = T.sum(
			surfaces * (T.log(surfaces) - T.log(norm_inputs)),
			axis = 1
		)
	return T.mean(kl_divergence)



	
def gaussian_shape(params,name,inputs,rows,cols,components,return_mixtures=False):
	n_hidden = rows*cols
	contrib_weights = np.zeros((n_hidden,components),dtype=np.float32) - 2
	idxs = np.arange(n_hidden)
	contrib_weights[(idxs%32 <  16) * (idxs//32 <  16),0] = 1
	contrib_weights[(idxs%32 >= 16) * (idxs//32 <  16),1] = 1
	contrib_weights[(idxs%32 <  16) * (idxs//32 >= 16),2] = 1
	contrib_weights[(idxs%32 >= 16) * (idxs//32 >= 16),3] = 1
	params["gamma_%s"%name] = theano.shared(contrib_weights)

	mixtures = T.nnet.softmax(params["gamma_%s"%name])		# n_hidden x components
	points = theano.shared(np.dstack(np.meshgrid(
		np.arange(cols),np.arange(rows)
	)).reshape(n_hidden,2))							# n_hidden x 2
	
	mixtures_ = mixtures.dimshuffle('x',0,1,'x')	# 1	x n_hidden	x components	x 1
	points_   = points.dimshuffle('x',0,'x',1)		# 1 x n_hidden	x 1				x 2
	inputs_   = inputs.dimshuffle(0,1,'x','x')		# N x n_hidden	x 1				x 1
	weight	  = mixtures_ * inputs_					# N x n_hidden	x components	x 1
	denom     = T.sum(weight,axis=1)				# N x components x 1

	means = T.sum(
			weight * points_,						# N x n_hidden	x components	x 2
			axis=1									
		)/ T.sum(mixtures_ * inputs_,axis=1)		# N x components x 1
													# N x components x 2

	means_ = means.dimshuffle(0,'x',1,2)			# N x 1 x components x 2
	deviations = points_ - means_					# N x n_hidden  x components x 2
	deviations_1 = deviations.dimshuffle(0,1,2,3,'x') # N x n_hidden  x components x 2 x 1
	deviations_2 = deviations.dimshuffle(0,1,2,'x',3) # N x n_hidden  x components x 1 x 2
	weight_ = weight.dimshuffle(0,1,2,3,'x')				# N x n_hidden	x components x 1 x 1
	covariances = T.sum(
			weight_ * deviations_1 * deviations_2,			# N x n_hidden  x components x 2 x 2
			axis = 1
		) / denom.dimshuffle(0,1,2,'x')						# N x components x 2 x 2
	precisions = covariances2precisions(covariances)		# N x components x 4
	
	g_mixture = T.sum(weight,axis=1) / T.sum(inputs_,axis=1) # N x components	x 1
	g_mixture = g_mixture.dimshuffle(0,1)
	fields = gaussian_field(g_mixture,precisions,deviations)
	normalised_inputs = inputs / T.sum(inputs,axis=1).dimshuffle(0,'x')
	
	kl_divergence = T.sum(
			fields * (T.log(fields) - T.log(normalised_inputs)),
			axis = 1
		)
	if return_mixtures:
		return g_mixture
	else:
		return T.mean(kl_divergence) 

if __name__ == "__main__":
	data = (20 * np.random.random((128,1024))) - 10
	data = theano.shared(data)
	
	surfaces = fixed_gaussian({},"test",T.nnet.sigmoid(data),32,32,4)
	print surfaces.eval()
#	dist = surfaces.eval()[0]
#	dist = dist.reshape(32,32)
#	import matplotlib.cm as cm
#	import matplotlib as mpl
#	mpl.use('Agg')
#	import matplotlib.pyplot as plt
#	plt.imshow(dist,cmap=cm.Reds,interpolation="nearest")
#	plt.savefig("mixgauss.png")





