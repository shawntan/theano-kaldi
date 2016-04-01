import sys
import theano
import theano.tensor as T
import ark_io
import data_io
import numpy as np
from theano_toolkit.parameters import Parameters

import config
import model

@config.option("class_counts_file","Files for counts of each class.",type=config.file)
def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = f.next().strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts

if __name__ == "__main__":
    config.parse_args()
    
    X = T.matrix('X')
    P = Parameters()
    predict = model.build(P)
    _,outputs = predict(X)
    counts = load_counts()
    predict = theano.function(
            inputs = [X],
            outputs = T.log(outputs) - T.log(counts/T.sum(counts))
        )
    

    if predict != None:
        stream = data_io.context(ark_io.parse(sys.stdin),left=5,right=5)
        for name,frames in stream:
            ark_io.print_ark(name,predict(frames))

