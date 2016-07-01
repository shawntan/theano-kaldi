import gzip
import cPickle as pickle
import sys
import numpy as np
from itertools import izip
import random
def context(stream,left=5,right=5):
    for name, frames in stream:
        frames_pad = np.lib.pad(frames,((left,right),(0,0)),'constant')
        windowed_frames = np.lib.stride_tricks.as_strided(
                frames_pad,strides=frames_pad.strides,
                shape=(frames.shape[0],
                        frames.shape[1] * (left + right + 1))
            )
        yield name, windowed_frames

def stream_file(filename,open_method=gzip.open):
    with open_method(filename,'rb') as fd:
        try:
            while True:
                x = pickle.load(fd)
                yield x
        except EOFError: pass

def async(stream,queue_size):
    import threading
    import Queue
    queue = Queue.Queue(maxsize=queue_size)
    end_marker = object()
    def producer():
        for item in stream:
            queue.put(item)
        queue.put(end_marker)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()
    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()

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
        if not all(x[0]==items[0][0] for x in items):
            print ','.join(x[0] for x in items)
            assert(all(x[0]==items[0][0] for x in items))
#        while not all(x[0]==items[0][0] for x in items):
#            for i in xrange(len(items)-1):
#                items[i] = streams[i].next()
        result = tuple(x[1] for x in items)

        if with_name:
            result = (items[0][0],) + result
        yield result

def buffered_random(stream,buffer_items=100,leak_percent=0.9):
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

def chop(stream,piece_size=32):
    import math
    for item in stream:
        pieces = int(math.ceil(item[0].shape[0]/float(piece_size)))
        for i in xrange(pieces):
            yield tuple(
                    x[i*piece_size:(i+1)*piece_size]  for x in item) 
