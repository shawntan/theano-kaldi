import gzip
import cPickle as pickle
import sys
import numpy as np

def stream(frame_file,label_file,with_name=False):
	with gzip.open(frame_file,'rb') as feat_file,\
		 gzip.open(label_file,'rb') as lbls_file:
		try:
			while True:
				name1,feats = pickle.load(feat_file)
				name2,lbls  = pickle.load(lbls_file)
				assert(name1==name2)
				assert(feats.shape[0] == len(lbls))
				if with_name:
					yield name1,feats,lbls
				else:
					yield feats,lbls
		except EOFError:
			pass


def randomise(stream,buffer_size=2**10,limit=-1):
	buf_feats = []
	buf_labels = []
	buf_instances = 0
	count = 0
	for feats,lbls in stream:
		if buf_instances + feats.shape[0] > buffer_size:
			feat_batch,lbl_batch = np.vstack(buf_feats),np.hstack(buf_labels).astype(np.int32)
			assert(feat_batch.shape[0] == lbl_batch.shape[0])
			idxs = np.arange(feat_batch.shape[0])
			np.random.shuffle(idxs)
			yield feat_batch[idxs],lbl_batch[idxs]
			buf_feats = []
			buf_labels = []
			buf_instances = 0
		buf_feats.append(feats)
		buf_labels.append(lbls)
		buf_instances += feats.shape[0]
		count += 1
		if count == limit:
			break
	if len(buf_feats) > 0:
		feat_batch,lbl_batch = np.vstack(buf_feats),np.hstack(buf_labels).astype(np.int32)
		assert(feat_batch.shape[0] == lbl_batch.shape[0])
		idxs = np.arange(feat_batch.shape[0])
		np.random.shuffle(idxs)
		yield feat_batch[idxs],lbl_batch[idxs]



if __name__ == "__main__":
	for f,l in randomise(sys.argv[1],sys.argv[2]):
		print f.shape
		print l.shape





