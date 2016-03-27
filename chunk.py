import config
import math
from itertools import izip
import numpy as np
import theano
import theano.tensor as T

batch_size = config.option("batch_size","Size of the batches.",type=config.int)

def create_shared_variable(data_variable):
    shared_var = theano.shared(np.zeros((1,) * data_variable.ndim,
                                               dtype=data_variable.dtype)) 
    shared_var.name = "%s_shared"%data_variable
    return shared_var

def create_shared_variables(inputs):
    return { var:create_shared_variable(var)
                for var in inputs }
@batch_size
def build_trainer(inputs,updates,outputs=None,batch_size=256,mapping=None):
    """
    Creates a shared variables and a function to load chunk into shared variables and train
    """
    if mapping is None:
        mapping = create_shared_variables(inputs)

    idx = T.iscalar('idx')
    train = theano.function(
            inputs  = [idx],
            outputs = outputs,
            updates = updates,
            givens  = { var:shared_var[idx*batch_size:(idx+1)*batch_size]
                            for var,shared_var in mapping.iteritems() },
        )
    def chunk_train(chunk):
        batch_count = int(math.ceil(chunk[0].shape[0]/float(batch_size)))
        for in_var,data in izip(inputs,chunk):
            mapping[in_var].set_value(data)
        for i in xrange(batch_count):
            if outputs is None:
                train(i)
            else:
                print train(i)

    return chunk_train


@batch_size
def stream(stream,batch_size,batch_per_chunk=256):
    buffer_size = batch_per_chunk * batch_size

    def initialise_buffers(items):
        buffers = tuple(np.empty((buffer_size,) + item.shape[1:],dtype=item.dtype) 
                    for item in items )
        return buffers, 0

    def fill_buffers(buffers,items,buffer_ptr):
        for item, buf in izip(items,buffers):
            buf[buffer_ptr:buffer_ptr+item.shape[0]] = item
        return buffer_ptr + items[0].shape[0]
    
    def shuffle(buffers):
        rng_state = np.random.get_state()
        for i in xrange(len(buffers)):
            np.random.set_state(rng_state)
            np.random.shuffle(buffers[i])
        return buffers

    items = stream.next()
    buffers, buffer_ptr = initialise_buffers(items)
    buffer_ptr = fill_buffers(buffers,items,buffer_ptr)
    for items in stream:
        remaining_size = buffer_size - buffer_ptr
        if items[0].shape[0] >= remaining_size:
            # If next item is bigger than remaining size, chop it up
            buffer_ptr = fill_buffers(buffers,[item[:remaining_size] for item in items],buffer_ptr)
            yield shuffle(buffers)
            buffers, buffer_ptr = initialise_buffers(items)
            buffer_ptr = fill_buffers(buffers,[item[remaining_size:] for item in items],buffer_ptr)
        else:
            buffer_ptr = fill_buffers(buffers,items,buffer_ptr)

    if buffer_ptr > 0:
        yield shuffle(tuple(buf[:buffer_ptr] for buf in buffers))

