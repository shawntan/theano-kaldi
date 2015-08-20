if __name__ == "__main__":
    import config
    config.structure("structure","Structure of model.")
    config.file("model",".pkl file containing discriminative model.")
    config.file("class_counts",".counts file giving counts of all the pdfs.")
    config.parse_args()
import theano
import theano.tensor as T
import ark_io
import numpy as np
import model

import cPickle as pickle

import sys

def ark_stream():
    return ark_io.parse(sys.stdin)

def create_model(filename,counts,input_size,layer_sizes,output_size):
    X = T.matrix('X')
    params = {}
    predict = model.build_feedforward(params,input_size,layer_sizes,output_size)
    model.load(filename,params)
    _,output = predict(X)
    f = theano.function(
            inputs = [X],
            outputs = T.log(output) - T.log(counts/T.sum(counts))
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
            filename= config.args.model,
            counts = counts,
            input_size  = config.args.structure[0],
            layer_sizes = config.args.structure[1:-1],
            output_size = config.args.structure[1]
        )

    if predict != None:
        for name,frames in ark_stream():
            print_ark(name,predict(frames))

