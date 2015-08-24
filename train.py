if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("frames_files",".pklgz file containing audio frames.")
    config.file_sequence("labels_files",".pklgz file containing frames labels.")
    config.structure("structure","Structure of discriminative model.")
    config.file("validation_frames_file","Validation set frames file.")
    config.file("validation_labels_file","Validation set labels file.")
    config.file("output_file","Output file.")
    config.file("pretrain_file","Pretrain file.")
    config.file("temporary_file","Temporary file.")
    config.integer("minibatch","Minibatch size.",default=128)
    config.integer("max_epochs","Maximum number of epochs to train.",default=200)
    config.parse_args()

import theano
import theano.tensor as T

import numpy as np
import math
import sys

import data_io
import model
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
if __name__ == "__main__":
    frames_files    = config.args.frames_files
    labels_files    = config.args.labels_files
    val_frames_file = config.args.validation_frames_file
    val_labels_file = config.args.validation_labels_file
    minibatch_size  = config.args.minibatch
    input_size      = config.args.structure[0]
    layer_sizes     = config.args.structure[1:-1]
    output_size     = config.args.structure[-1]

    X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))
    Y_shared = theano.shared(np.zeros((1,),dtype=np.int32))

    P = Parameters()
    feedforward = model.build_feedforward(
            P,
            input_size = input_size,
            layer_sizes = layer_sizes,
            output_size = output_size
        )

    X = T.matrix('X')
    Y = T.ivector('Y')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')
    lr = T.scalar('lr')
    _,outputs = feedforward(X)

    if config.args.pretrain_file != None:
        P.load(config.args.pretrain_file)
        P.save(config.args.temporary_file)

    loss = cross_entropy = T.mean(T.nnet.categorical_crossentropy(outputs,Y))
    parameters = P.values() 
    print "Parameters to tune:"
    pprint(parameters)
    
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)
    train = theano.function(
            inputs  = [lr,start_idx,end_idx],
            outputs = cross_entropy,
            updates = updates.momentum(parameters,gradients,learning_rate=lr,P=update_vars),
            givens  = {
                X: X_shared[start_idx:end_idx],
                Y: Y_shared[start_idx:end_idx]
            }
        )
    test = theano.function(
            inputs = [X,Y],
            outputs = [loss]  + [ T.mean(T.neq(T.argmax(outputs,axis=1),Y))]
        )

    def run_test():
        total_errors = None
        total_frames = 0
        for f,l in data_io.stream(val_frames_file,val_labels_file):
            if total_errors is None:
                total_errors = np.array(test(f,l),dtype=np.float32)
            else:
                total_errors += [f.shape[0] * v for v in test(f,l)]
            total_frames += f.shape[0]
        return total_errors/total_frames
    def run_train():
        split_streams = [ data_io.stream(f,l) for f,l in izip(frames_files,labels_files) ]
        stream = chain(*split_streams)
        total_frames = 0
        for f,l,size in data_io.randomise(stream):
            total_frames += f.shape[0]
            X_shared.set_value(f)
            Y_shared.set_value(l)
            batch_count = int(math.ceil(size/float(minibatch_size)))
            for idx in xrange(batch_count):
                start = idx*minibatch_size
                end = min((idx+1)*minibatch_size,size)
                train(learning_rate,start,end)

    
    learning_rate = 0.08
    best_score = np.inf
    
    for epoch in xrange(config.args.max_epochs):
        scores = run_test()
        score = scores[0]
        print scores
        _best_score = best_score
        if score < _best_score:
            best_score = score
            P.save(config.args.temporary_file)
            update_vars.save("update_vars.tmp")

        if score/_best_score > 0.995 and epoch > 0:
            learning_rate *= 0.5
            print "Learning rate is now",learning_rate
            P.load(config.args.temporary_file)
            update_vars.load("update_vars.tmp")

        if learning_rate < 1e-6: break
        run_train()
    P.load(config.args.temporary_file)
    P.save(config.args.output_file)
