import sys
import theano
import theano.tensor as T
import ark_io
import data_io
import numpy as np
from theano_toolkit.parameters import Parameters

import config
import model
import frame_label_data

if __name__ == "__main__":
    config.parse_args()
    X = T.matrix('X')
    P = Parameters()
    predict = model.build(P)
    _, outputs = predict(X)
    for n in theano.gof.graph.list_of_nodes([X], [outputs]):
        if hasattr(n.out, 'name') and n.out.name is not None:
            if n.out.name.endswith('_bn_mean'):
                print n.out

