import subprocess
import numpy as np
import os
from itertools import izip
import sys
import gzip
import cPickle as pickle

import ark_io

def ark_stream():
	return ark_io.parse(sys.stdin)

if __name__ == "__main__":
	output_file = sys.argv[1]
	if len(sys.argv) > 2:
		dim_file = sys.argv[2]
		dim_size = None
	features = ark_stream()
	with gzip.open(output_file,'wb') as f:
		count = 0
		for name,features in features:
			if dim_size == None:
				dim_size = features.shape[1]
				print dim_size
				with open(dim_file,'w') as dimf:
					dimf.write('%d\n'%dim_size)

			pickle.dump((name,features),f,protocol=2)
			count += 1
			if count % 100 == 0:
				print "Wrote %d utterances to %s"%(count,output_file)
	print "Wrote %d utterances to %s"%(count,output_file)

