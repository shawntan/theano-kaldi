import subprocess
import numpy as np
import os
from itertools import izip
import sys
import gzip
import cPickle as pickle
import glob

def label_stream(ali_dir):
	part_count = len(glob.glob(ali_dir + "/ali.*.gz"))

	proc1 = subprocess.Popen([
		"gunzip",
		"-c"
	] + [
		ali_dir + "/ali.%d.gz"%i for i in range(1,part_count+1)
	],
	stdout=subprocess.PIPE)
	proc2 = subprocess.Popen([
		"ali-to-pdf",
		ali_dir + "/final.mdl",
		"ark:-",
		"ark,t:-"
	],
	stdin = proc1.stdout,
	stdout = subprocess.PIPE)

	for l in iter(proc2.stdout.readline,''):
		l = l.rstrip().split()
		name = l[0]
		pdfs = map(int,l[1:])
		yield name,pdfs

def stdin_label_stream():
	for l in sys.stdin:
		l = l.rstrip().split()
		name = l[0]
		pdfs = map(int,l[1:])
		yield name,pdfs




if __name__ == "__main__":
	ali_dir = sys.argv[1]
	output_file = sys.argv[2]
	if ali_dir == "-":
		labels = stdin_label_stream()
	else:
		labels = label_stream(ali_dir)
	with gzip.open(output_file,'wb') as f:
		count = 0
		for name,lbls in labels:
			pickle.dump((name,lbls),f,protocol=2)
			count += 1
			if count % 100 == 0:
				print "Wrote %d utterances to %s"%(count,output_file)
	print "Wrote %d utterances to %s"%(count,output_file)


