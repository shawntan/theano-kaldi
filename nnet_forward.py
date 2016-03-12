if __name__ == "__main__":
    import config
    config.structure("structure","Structure of discriminative model.")
    config.file("model",".pkl file containing discriminative model.")
    config.file("class_counts",".counts file giving counts of all the pdfs.")
    config.parse_args()

import theano
import theano.tensor as T
import ark_io
import numpy as np
import model
import data_io
import cPickle as pickle

import sys
import feedforward
from theano_toolkit.parameters import Parameters
def ark_stream():
    return ark_io.parse(sys.stdin)

def create_model(counts,input_size,layer_sizes,output_size):
    X = T.matrix('X')
    P = Parameters()

    classify = model.build(P,input_size,layer_sizes,output_size)
    _,output = classify(X)
    log_output = T.log(output)
    P.load(config.args.model)
    f = theano.function(
            inputs = [X],
            outputs = log_output - T.log(counts/T.sum(counts))
        )
    return f

def print_ark(name,array):
    print name,"["
    for i,row in enumerate(array):
        print " ",
        for cell in row:
            print "%0.6f"%cell,
        if i == array.shape[0]-1:
            print "]"
        else:
            print

if __name__ == "__main__":
    with open(config.args.class_counts) as f:
        row = f.next().strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])

    predict = create_model(
            counts = counts,
            input_size  = config.args.structure[0],
            layer_sizes = config.args.structure[1:-1],
            output_size = config.args.structure[-1],
        )

    if predict != None:
        stream = data_io.context(ark_stream(),left=5,right=5)
        for name,frames in stream:
            print_ark(name,predict(frames))

