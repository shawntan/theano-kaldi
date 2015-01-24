import subprocess
import numpy as np
import os
import cPickle as pickle
import gzip
from itertools import izip
import sys

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

def unique_phones(filename):
	phones = set()
	for line in open(filename):
		tokens = line.strip().split()
		for t in tokens[1:]: phones.add(t)
	return list(phones)


def parse_phones(filename,phone2id):
	for line in open(filename):
		tokens = line.strip().split()
		name = tokens[0]
		seq  = map(phone2id.get,tokens[1:])
		yield name,seq
		
if __name__ == "__main__":
	data_dir = sys.argv[1]
	DATASET  = sys.argv[2]
	transcript_file = data_dir + "/" + DATASET + "/text"
	output_file = DATASET + ".pklgz"
	proc1 = subprocess.Popen([
		"apply-cmvn",
		"--norm-vars=true", 
		"--utt2spk=ark:"+ data_dir + "/" + DATASET + "/utt2spk",
		"scp:"          + data_dir + "/" + DATASET + "/cmvn.scp",
		"scp:"          + data_dir + "/" + DATASET + "/feats.scp",
		"ark,t:-"
	],
	stdout=subprocess.PIPE)#,stderr=open(os.devnull,'w')
	"""
	proc2 = subprocess.Popen([
		"splice-feats",
		"--left-context=" + context + " --right-context=" + context,
		"ark:-",
		"ark,t:-"
	],
	stdin  = proc1.stdout,
	stdout = subprocess.PIPE)
	"""

	proc = proc1
	
	if DATASET=="train":
		phonemes = unique_phones(transcript_file)
		phone2id = { phn:idx for idx,phn in enumerate(phonemes) }
		with open("phonemes.pkl",'wb') as f:
			pickle.dump(phone2id,f,2)
		print len(phonemes), "phonemes found."
	else:
		phone2id = pickle.load(open("phonemes.pkl","rb"))


	feature_stream = parse_ark(l.rstrip() for l in iter(proc.stdout.readline,''))
	phoneme_stream = parse_phones(transcript_file,phone2id)

	with gzip.open(output_file,'wb') as f:
		count = 0
		for (name1,features),(name2,sequence) in izip(feature_stream,phoneme_stream):
			assert(name1==name2)
			pickle.dump((name1,features,sequence),f,protocol=2)
			count += 1
			if count % 100 == 0:
				print "Wrote %d utterances to %s"%(count,output_file)
	print "Wrote %d utterances to %s"%(count,output_file)

	
