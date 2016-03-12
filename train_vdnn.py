import sys
import logging,json

import theano
import theano.tensor as T
import trainer
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("validation_frames_files","Validation set frames file.")
    config.file_sequence("validation_labels_files","Validation set labels file.")
    config.structure("structure","Structure of discriminative model.")
    config.file("pretrain_file","Pretrain file.",default="")
    config.real("prior_alpha","Parameter for dirichlet prior",default=0.15)

    X = T.matrix('X')
    Y = T.ivector('Y')
    compile_train_epoch = trainer.build_train_epoch([X,Y])
    train_loop = trainer.build_train_loop()

    config.parse_args()
import numpy as np
import math

import data_io
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import vdnn

def make_split_stream(frames_files,labels_files):
    return [ data_io.zip_streams(
                data_io.context(
                    data_io.stream_file(frames_file),
                    left=5,right=5
                ),
                data_io.stream_file(labels_file)
            ) for frames_file,labels_file in izip(frames_files,labels_files) ]


def build_data_stream(context=5):
    def data_stream(file_sequences):
        frames_files = file_sequences[0]
        labels_files = file_sequences[1]
        split_streams = make_split_stream(frames_files,labels_files) 
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)
        stream = data_io.randomise(stream)
        return stream
    return data_stream


def count_frames(frames_files):
    split_streams = [ data_io.stream(f) for f in frames_files ]
    return sum(f.shape[0] for f in chain(*split_streams))

def crossentropy(output,Y):
    if output.owner.op == T.nnet.softmax_op:
        x = output.owner.inputs[0]
        k = T.max(x,axis=1,keepdims=True)
        sum_x = T.log(T.sum(T.exp(x - k),axis=1)) + k
        return - x[T.arange(x.shape[0]),Y] + sum_x
    else:
        return T.nnet.categorical_crossentropy(outputs,Y)
if __name__ == "__main__":
    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]
 
    training_frame_count = count_frames(config.args.X_files)
    logging.debug("Created shared variables")


    P = Parameters()
    _, model_cost,_ = vdnn.build(P,input_size,layer_sizes,output_size)
    cross_entropy, prior_cost, crossentropies = model_cost(X,Y)

    if config.args.pretrain_file != "":
        P.load(config.args.pretrain_file)

    parameters = P.values() 
    logging.info("Parameters to tune:" + ','.join(w.name for w in parameters)) 
    loss = cross_entropy \
            + config.args.prior_alpha * prior_cost \
            + (0.5/training_frame_count) * sum(T.sum(T.sqr(w)) for w in parameters)
    logging.debug("Built model expression.")

    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)


    monitored_values = {
            "cross_entropy": cross_entropy,
        }

    monitored_keys = monitored_values.keys()
    test = theano.function(
            inputs = [X,Y],
            outputs = [ monitored_values[k] for k in monitored_keys ]
        )

    logging.debug("Done.")

    def strat(parameters, gradients, learning_rate=1e-3, P=None):
        return [ (p, p - learning_rate * g) 
                    for p,g in zip(parameters,gradients) ]

    run_train = compile_train_epoch(
            parameters,gradients,update_vars,
            data_stream=build_data_stream(context=5),
            update_strategy=strat,
#            outputs=crossentropies + [cross_entropy]
        )
    def run_test():
        total_errors = None
        total_frames = 0
        split_streams = make_split_stream(config.args.validation_frames_files,
                                            config.args.validation_labels_files) 
        for f,l in chain(*split_streams):
            if total_errors is None:
                total_errors = np.array(test(f,l),dtype=np.float32)
            else:
                total_errors += [f.shape[0] * v for v in test(f,l)]
            total_frames += f.shape[0]
        values = total_errors/total_frames
        return { k:float(v) for k,v in zip(monitored_keys,values) }

    train_loop(logging,run_test,run_train,P,update_vars,monitor_score="cross_entropy")