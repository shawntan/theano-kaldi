import tsne
import numpy as np
def get_2d_points(stream):
	phn_acc = {}
	phn_count = {}
	for f,l in stream:
		for i in xrange(l.shape[0]):
			phn_acc[l[i]]   = phn_acc.get(l[i],0)   + f[i]
			phn_count[l[i]] = phn_count.get(l[i],0) + 1

	keys = phn_acc.keys()
	keys.sort()
	points = tsne.tsne(np.vstack([
		phn_acc[k]/phn_count[k]
		for k in keys
	]).astype(np.float64))
	return points

def get_spkr_2d_points(stream):
	spkr_acc = {}
	spkr_count = {}
	for name,f,l in stream:
		for i in xrange(l.shape[0]):
			if l[i] == 1: 
				key = "~"
			else:
				key = name[:5]
			spkr_acc[key]   = spkr_acc.get(key,0)   + f[i]
			spkr_count[key] = spkr_count.get(key,0) + 1

	keys = spkr_acc.keys()
	keys.sort()
	points = tsne.tsne(np.vstack([
		spkr_acc[k]/spkr_count[k]
		for k in keys
	]).astype(np.float64))
	return keys,points



if __name__ == "__main__":
	import sys
	from itertools import izip,chain
	import data_io
	import pickle

	files = sys.argv[1:]
	frames_files  = files[:len(files)/2]
	phoneme_files = files[len(files)/2:]
	split_streams = [ data_io.stream(f,p,with_name=True)
						for f,p in izip(frames_files,phoneme_files) ]
	stream = chain(*split_streams)

	keys,points = get_spkr_2d_points(stream)
	print keys
	points = points - np.mean(points,axis=0)
	points = points / np.max(np.abs(points),axis=0)
	points = (0.9 * 16) * points + [ 16, 16 ]
	pickle.dump(points,open('spkr_gaussian_ctr.pkl','wb'),2)
	pickle.dump(keys,open('spkr_order.pkl','wb'),2)


	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.cm as cm
	import matplotlib.pyplot as plt

	female = sum(1 for k in keys if k.startswith('F'))
	plt.scatter(points[:female,0],points[:female,1],color="red")
	plt.scatter(points[female:-1,0],points[female:-1,1],color="blue")
	plt.scatter(points[-1,0],points[-1,1],color="green")
#	phns = { int(line.strip().split()[-1]) : line.strip().split()[0]
#			for line in open('exp/tri3/graph/phones.txt') }


#	for k,p in zip(keys,points):
#		plt.annotate(k,(p[0],p[1]))
	plt.savefig("spkr_tsne.png")
	plt.clf()
	
