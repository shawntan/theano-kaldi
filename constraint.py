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

	return T.mean(T.sqrt(vert_const + horz_const))


