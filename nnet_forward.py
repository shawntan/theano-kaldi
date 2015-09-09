if __name__ == "__main__":
    import config
    config.structure("structure_z1","Structure of M1.")
    config.structure("structure","Structure of discriminative model.")
    config.file("z1_file","Z1 params file.")
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
import feedforward
from theano_toolkit.parameters import Parameters
def ark_stream():
    return ark_io.parse(sys.stdin)

def create_model(counts,input_size,z1_layer_sizes,z1_output_size,layer_sizes,output_size):
    z1_input_size = input_size
    X = T.matrix('X')
    P = Parameters()
    P_z1_x = Parameters()
    encode_Z1,_,_ = model.build_unsupervised(P_z1_x,z1_input_size,z1_layer_sizes,z1_output_size)
    classify = feedforward.build_classifier(
        P, "classifier",
        [z1_output_size], layer_sizes, output_size,
        activation=T.nnet.sigmoid
    )
    _,Z1,_ = encode_Z1([X])
    output = classify([Z1])
    P_z1_x.load(config.args.z1_file)
    P.load(config.args.model)
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
            counts = counts,
            input_size = config.args.structure_z1[0],
            z1_layer_sizes = config.args.structure_z1[1:-1],
            z1_output_size = config.args.structure_z1[-1],
            layer_sizes    = config.args.structure[:-1],
            output_size    = config.args.structure[-1],
        )

    if predict != None:
        for name,frames in ark_stream():
            print_ark(name,predict(frames))

