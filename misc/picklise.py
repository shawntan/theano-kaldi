import subprocess
import numpy as np
import os
from itertools import izip
import sys
import gzip
import cPickle as pickle

def parse_matrix(stream):
	result = []
	line = stream.next().strip()
	while not line.endswith(']'):
		result.append(map(float,line.split()))
		line = stream.next().strip()
	result.append(map(float,line.split()[:-1]))
	return np.array(result,dtype=np.float32)
	
def parse_ark(stream):
	for line in stream:
		if line.endswith('['):
			name = line.strip().split()[0]
			yield name,parse_matrix(stream)

def feature_stream(dataset_dir,context):
	proc1 = subprocess.Popen([
		"apply-cmvn",
		"--norm-vars=true", 
		"--utt2spk=ark:"+ dataset_dir + "/utt2spk",
		"scp:"          + dataset_dir + "/cmvn.scp",
		"scp:"          + dataset_dir + "/feats.scp",
		"ark,t:-"
	],
	stdout=subprocess.PIPE)#,stderr=open(os.devnull,'w')
	proc2 = subprocess.Popen([
		"splice-feats",
		"--left-context=" + context + " --right-context=" + context,
		"ark:-",
		"ark,t:-"
	],
	stdin  = proc1.stdout,
	stdout = subprocess.PIPE)
	proc = proc2
	return parse_ark(l.rstrip() for l in iter(proc.stdout.readline,''))

if __name__ == "__main__":
	data_dir = sys.argv[1]
	DATASET  = sys.argv[2]
	context  = sys.argv[3]
	output_file = sys.argv[4]
	features = feature_stream(data_dir + "/" + DATASET,context)

	with gzip.open(output_file,'wb') as f:
		count = 0
		for name,features in features:
			pickle.dump((name,features),f,protocol=2)
			count += 1
			if count % 100 == 0:
				print "Wrote %d utterances to %s"%(count,output_file)
	print "Wrote %d utterances to %s"%(count,output_file)

