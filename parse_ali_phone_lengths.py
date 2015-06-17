import numpy as np
import os
from itertools import izip
import sys
import gzip
import cPickle as pickle

def ctx_phn_seq(segments,consecutive = {}):
	prev_phn = None
	next_phn = None
	centre = None
	for segment in segments:
		next_phn = segment[0]
		if centre:
			phn,length = centre
			for _ in xrange(length):
				yield (prev_phn,phn,next_phn)
			if phn == prev_phn: consecutive[phn] = consecutive.get(phn,0) + 1
			prev_phn = phn
		else:
			# First time
			prev_phn = -1
		centre = segment

	phn,length = centre
	for _ in xrange(length):
		yield (prev_phn,phn,-2)


def label_stream():
	consecutive = {}
	for l in sys.stdin:
		utt_id, segment_string = l.rstrip().split(' ',1)
		ctx_phns = list(ctx_phn_seq(
				(
					tuple(int(x) for x in segment.split())
					for segment in segment_string.split(" ; ")
				),consecutive))
		yield utt_id,ctx_phns


if __name__ == "__main__":
	output_file = sys.argv[1]

	labels = label_stream()
	with gzip.open(output_file,'wb') as f:
		count = 0
		for name,lbls in labels:
			pickle.dump((name,lbls),f,protocol=2)
			count += 1
			if count % 100 == 0:
				print "Wrote %d utterances to %s"%(count,output_file)
	print "Wrote %d utterances to %s"%(count,output_file)


