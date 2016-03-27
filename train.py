import sys
import logging,json

import theano
import theano.tensor as T
import numpy as np
import math

import data_io
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import model

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
    logging.debug("Created shared variables")


    P = Parameters()
    classify = model.build(P,input_size,layer_sizes,output_size)
    outputs = classify(X)


    P.load(config.args.pretrain_file)

    parameters = P.values() 
    logging.info("Parameters to tune:" + ','.join(w.name for w in parameters))

    cross_entropy = T.mean(crossentropy(outputs,Y))
    loss = cross_entropy #+ (0.5/training_frame_count)  * sum(T.sum(T.sqr(w)) for w in parameters)
    logging.debug("Built model expression.")

    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)


    monitored_values = {
            "cross_entropy": loss,
            "classification_error":T.mean(T.neq(T.argmax(outputs,axis=1),Y))
        }

    monitored_keys = monitored_values.keys()
    test = theano.function(
            inputs = [X,Y],
            outputs = [ monitored_values[k] for k in monitored_keys ]
        )

    logging.debug("Done.")

    run_train = compile_train_epoch(
            parameters,gradients,update_vars,
            data_stream=build_data_stream(context=5)
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
