import gzip
import cPickle as pickle
import sys
import numpy as np
from itertools import izip
import random

def context(stream,left=5,right=5):
    left_buf = right_buf = None
    idxs = np.arange(1000).reshape(1000,1) + np.arange(left + 1 + right)
    for name,frames in stream:
        dim = frames.shape[1]
        if left_buf is None:
            left_buf = np.zeros((left,dim),dtype=np.float32)
            right_buf = np.zeros((right,dim),dtype=np.float32)
        length = frames.shape[0]
        if length > idxs.shape[0]:
            idxs = np.arange(length).reshape(length,1) + np.arange(left + 1 + right)
        frames = np.concatenate([left_buf,frames,right_buf])
        frames = frames[idxs[:length]]
        frames = frames.reshape(length, (left + 1 + right) * dim)
        yield name,frames

def splice(stream,left=5,right=5):
    left_buf = right_buf = None
    idxs = np.arange(1000).reshape(1000,1) + np.arange(left + 1 + right)
    for frames in stream:
        dim = frames.shape[1]
        if left_buf is None:
            left_buf = np.zeros((left,dim),dtype=np.float32)
            right_buf = np.zeros((right,dim),dtype=np.float32)
        length = frames.shape[0]
        if length > idxs.shape[0]:
            idxs = np.arange(length).reshape(length,1) + np.arange(left + 1 + right)
        frames = np.concatenate([left_buf,frames,right_buf])
        frames = frames[idxs[:length]]
        frames = frames.reshape(length, (left + 1 + right) * dim)
        yield frames


def stream_file(filename,open_method=gzip.open):
    with open_method(filename,'rb') as fd:
        try:
            while True:
                x = pickle.load(fd)
                yield x
        except EOFError: pass

def stream(*filenames,**kwargs):
    gens = [ stream_file(f) for f in filenames ]
    return zip_streams(*gens,**kwargs)

def random_select_stream(*streams):
    while len(streams) > 0:
        stream_idx = random.randint(0,len(streams)-1)
        try:
            yield streams[stream_idx].next()
        except StopIteration:
            streams = streams[:stream_idx] + streams[stream_idx+1:]

def zip_streams(*streams,**kwargs):
    with_name = kwargs.get('with_name',False)
    while True:
        items = [ s.next() for s in streams ]
        assert(all(x[0]==items[0][0] for x in items))
        result = tuple(x[1] for x in items)

        if with_name:
            result = (items[0][0],) + result
        if len(result) == 1:
            yield result[0]
        else:
            yield result

def buffered_random(stream,buffer_items=20 * 8,leak_percent=0.9):
    item_buffer = [None] * buffer_items
    leak_count = int(buffer_items * leak_percent)
    item_count = 0
    for item in stream:
        item_buffer[item_count] = item
        item_count += 1
        if buffer_items == item_count:
            random.shuffle(item_buffer)
            for item in item_buffer[leak_count:]: yield item
            item_count = leak_count
    if item_count > 0:
        item_buffer = item_buffer[:item_count]
        random.shuffle(item_buffer)
        for item in item_buffer: yield item

def randomise(stream,buffer_size=2**17):
    buf = None
    buf_instances = 0
    for item in stream:
        if type(item) is not tuple:
            item = (item,)
        if buf == None:
            buf = [
                    np.zeros((buffer_size,) + x.shape[1:],dtype=x.dtype)
                    for x in item
                ]
            def randomise_buffers():
                idxs = np.arange(buf_instances)
                np.random.shuffle(idxs)
                rng_state = np.random.get_state()
                for i in xrange(len(buf)):
                    np.random.set_state(rng_state)
                    np.random.shuffle(buf[i][:buf_instances])

        if buf_instances + item[0].shape[0] > buffer_size:
#            print "Buffer size reached: ",buf_instances
#            print "Shuffling...",
            randomise_buffers()
            yield tuple(buf) + (buf_instances,)
#            print "dispatched."
            buf_instances = 0
        else:
#            print "Copying to buffer", (buf_instances,buf_instances+feats.shape[0])
            for i in xrange(len(buf)):
                buf[i][buf_instances:buf_instances+item[0].shape[0]] = item[i]
            buf_instances += item[0].shape[0]
    if len(buf[0]) > 0:
        randomise_buffers()
        yield tuple(buf) + (buf_instances,)

if __name__ == "__main__":
    import time
    from itertools import chain
    data_streams = [ stream("/home/shawn/kaldi-trunk-2/egs/timit/s5/exp/dnn_fbank_tk_feedforward/pkl/train.0%d.pklgz"%i) 
            for i in xrange(10) ]
    
    randomised_stream = randomise_threaded(chain(*data_streams))
    for value in randomised_stream:
        print "begin sleeping"
        time.sleep(1)
        print "end sleeping"


