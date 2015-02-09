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


def randomise(stream,buffer_size=2**20,limit=-1):
	buf_feats = None
	buf_labels = None
	buf_instances = 0
	count = 0

	for feats,lbls in stream:

		if buf_feats == None:
#			print "Initialise buffer",(buffer_size,feats.shape[1])
			buf_feats  = np.zeros((buffer_size,feats.shape[1]),dtype=np.float32)
			buf_labels = np.zeros((buffer_size,),dtype=np.int32)

		if buf_instances + feats.shape[0] > buffer_size:
#			print "Buffer size reached: ",buf_instances
#			print "Shuffling...",
			idxs = np.arange(buf_instances)
			np.random.shuffle(idxs)
			buf_feats[:buf_instances]  = buf_feats[idxs]
			buf_labels[:buf_instances] = buf_labels[idxs]
			yield buf_feats,buf_labels,buf_instances
#			print "dispatched."
			buf_instances = 0
		else:
			lbls = np.array(lbls,dtype=np.int32)
			assert(feats.shape[0] == lbls.shape[0])
#			print "Copying to buffer", (buf_instances,buf_instances+feats.shape[0])
			buf_feats[buf_instances:buf_instances+feats.shape[0]]  = feats
			buf_labels[buf_instances:buf_instances+feats.shape[0]] = lbls
			buf_instances += feats.shape[0]

			count += 1
			if count == limit: break
		
	if len(buf_feats) > 0:
		idxs = np.arange(buf_instances)
		np.random.shuffle(idxs)
		buf_feats[:buf_instances]  = buf_feats[idxs]
		buf_labels[:buf_instances] = buf_labels[idxs]
		yield buf_feats,buf_labels,buf_instances


if __name__ == "__main__":
	for f,l in randomise(sys.argv[1],sys.argv[2]):
		print f.shape
		print l.shape





