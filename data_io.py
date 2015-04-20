import gzip
import cPickle as pickle
import sys
import numpy as np
from itertools import izip
def stream(*filenames,**kwargs):	
	with_name = kwargs.get('with_name',False)
	fds = [ gzip.open(f,'rb') for f in filenames ]
	try:
		while True:
			items = [ pickle.load(fd) for fd in fds ]
			assert(all(x[0]==items[0][0] for x in items))
			# HACK. To remove.
			result = tuple(
					x[1] if hasattr(x[1],'shape') else np.array(x[1],dtype=np.int32)
					for x in items
				)
			if with_name:
				yield (items[0][0],) + result
			else:
				yield result
	except EOFError:
		pass
	for fd in fds: fd.close()


def randomise(stream,buffer_size=2**16):
	buf = None
	buf_instances = 0
	for item in stream:
		if buf == None:
			buf = [
					np.zeros((buffer_size,) + x.shape[1:],dtype=x.dtype)
					for x in item 
				]
			def randomise_buffers():
				idxs = np.arange(buf_instances)
				np.random.shuffle(idxs)
				for i in xrange(len(buf)): buf[i][:buf_instances]  = buf[i][idxs]

		if buf_instances + item[0].shape[0] > buffer_size:
#			print "Buffer size reached: ",buf_instances
#			print "Shuffling...",
			randomise_buffers()
			yield tuple(buf) + (buf_instances,)
#			print "dispatched."
			buf_instances = 0
		else:
#			print "Copying to buffer", (buf_instances,buf_instances+feats.shape[0])
			for i in xrange(len(buf)):
				buf[i][buf_instances:buf_instances+item[0].shape[0]] = item[i]
			buf_instances += item[0].shape[0]
		
	if len(buf[0]) > 0:
		randomise_buffers()
		yield tuple(buf) + (buf_instances,)

import threading
def randomise_threaded(stream,buffer_size=2**10):
	class RunScope:
		def __init__(self):
			self.buf  = None
			self.buf_tmp = None
			self.done = False
			self.buf_instances = 0
		def loader_shuffler(self):
			try:
				self.buf_instances = 0
				print "Copying data."
				while True:
					item = stream.next()
					if self.buf == None:
						self.buf = [ np.zeros((buffer_size,) + x.shape[1:],dtype=x.dtype)
								for x in item ]
						self.buf_tmp = [ np.zeros((buffer_size,) + x.shape[1:],dtype=x.dtype)
								for x in item ]

					if self.buf_instances + item[0].shape[0] < buffer_size:
						for i in xrange(len(self.buf)):
							self.buf[i][self.buf_instances:self.buf_instances+item[0].shape[0]] = item[i]
						self.buf_instances += item[0].shape[0]
					else:
						print "Done copying."
						break
			except StopIteration:
				print "End of stream."
				self.done = True
			idxs = np.arange(self.buf_instances)
			np.random.shuffle(idxs)
			for i in xrange(len(self.buf)): self.buf[i][:self.buf_instances]  = self.buf[i][idxs]
	t = RunScope()
	worker = threading.Thread(target=t.loader_shuffler)
	worker.start()
	while not t.done:
		worker.join()
		tmp_instance_count = t.buf_instances
		t.buf,t.buf_tmp = t.buf_tmp,t.buf
		if not t.done:
			worker = threading.Thread(target=t.loader_shuffler)
			worker.start()
		yield tuple(t.buf_tmp) + (tmp_instance_count,)

if __name__ == "__main__":
	import time
	data_stream = stream(sys.argv[1],sys.argv[2])
	randomised_stream = randomise_threaded(data_stream)
	value = randomised_stream.next()
	print value
	time.sleep(1)
	print value
	value = randomised_stream.next()
	print value
	time.sleep(1)
	print value




