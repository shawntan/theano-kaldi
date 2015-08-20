
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("frames_files",".pklgz file containing audio frames.")
    config.file_sequence("labels_files",".pklgz file containing frames labels.")
    config.structure("structure","Structure of discriminative model.")
    config.file("validation_frames_file","Validation set frames file.")
    config.file("validation_labels_file","Validation set labels file.")
    config.file("output_file","Output file.")
    config.file("temporary_file","Temporary file.")
    config.file("spk2utt_file","spk2utt file from Kaldi.")
    config.integer("minibatch","Minibatch size.",default=128)
    config.integer("speaker_embedding_size","Speaker embedding size.",default=128)
    config.integer("max_epochs","Maximum number of epochs to train.",default=200)
    config.parse_args()

import theano
import theano.tensor as T

import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain

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

    params = {}
    feedforward = model.build_feedforward(params)

    X = T.matrix('X')
    Y = T.ivector('Y')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')
    lr = T.scalar('lr')
    _,outputs = feedforward(X)

    if config.args.pretrain_file != None:
                model.
        model.save(config.args.temporary_file,params)

        loss = cross_entropy = T.mean(T.nnet.categorical_crossentropy(outputs,Y))
        parameters = params.values() 
        print "Parameters to tune:"
        pprint(parameters)
        gradients = T.grad(loss,wrt=parameters)
        train = theano.function(
                inputs  = [lr,start_idx,end_idx],
                outputs = cross_entropy,
                updates = updates.momentum(parameters,gradients,eps=lr),
                givens  = {
                    X: X_shared[start_idx:end_idx],
                    Y: Y_shared[start_idx:end_idx]
                }
            )
        test = theano.function(
                inputs = [X,Y],
                outputs = [loss]  + [ T.mean(T.neq(T.argmax(outputs,axis=1),Y))]
            )
        total_cost = 0
        total_errors = 0
        total_frames = 0
        for f,l in data_io.stream(val_frames_file,val_labels_file):
            test_outputs  = test(f,l)
            loss = test_outputs[0]
            errors = np.array(test_outputs[1:])
            total_frames += f.shape[0]

            total_cost   += f.shape[0] * loss
            total_errors += f.shape[0] * errors

        learning_rate = 0.08
        best_score = total_cost/total_frames

        print total_errors/total_frames,best_score

        for epoch in xrange(config.args.max_epochs):
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
            total_cost = 0
            total_errors = 0
            total_frames = 0

            for f,l in data_io.stream(val_frames_file,val_labels_file):
                test_outputs  = test(f,l)
                loss = test_outputs[0]
                errors = np.array(test_outputs[1:])
                total_frames += f.shape[0]

                total_cost   += f.shape[0] * loss
                total_errors += f.shape[0] * errors

            cost = total_cost/total_frames
            print total_errors/total_frames,cost
            _best_score = best_score

            if cost < _best_score:
                best_score = cost
                model.save(config.args.temporary_file,params)

            if cost/_best_score > 0.99995:
                learning_rate *= 0.5
                model.load(config.args.temporary_file,params)

            if learning_rate < 1e-6: break
            print "Learning rate is now",learning_rate

        model.load(config.args.temporary_file,params)
        model.save(config.args.output_file,params)
