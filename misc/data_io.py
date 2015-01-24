import gzip
import cPickle as pickle
import sys

with gzip.open(sys.argv[1],'rb') as feat_file, gzip.open(sys.argv[2],'rb') as lbls_file:
	try:
		while True:
			name1,feats = pickle.load(feat_file)
			name2,lbls  = pickle.load(lbls_file)
			assert(name1==name2)
			assert(feats.shape[0] == len(lbls))
			print name1
	except EOFError:
		pass



