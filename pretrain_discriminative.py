import sys
import logging,json

import theano
import theano.tensor as T
import trainer
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.structure("structure","Structure of discriminative model.")
    config.file("output_file","Output file.")
    config.file_sequence("validation_frames_files","Validation set frames file.")
    config.file_sequence("validation_labels_files","Validation set labels file.")

    X = T.matrix('X')
    Y = T.ivector('Y')
    compile_train_epoch = trainer.build_train_epoch([X,Y])

    config.parse_args()
import numpy as np
import math

import data_io
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import model
from train import *


def build_intermediate_outputs(P,layer_sizes,output_size):
    transforms = []
    for i,layer_size in enumerate(layer_sizes):
        P['W_pretrain_output_%d'%i] = \
                np.zeros((layer_size,output_size),dtype=np.float32)
        P['b_pretrain_output_%d'%i] = \
                np.zeros((output_size,),dtype=np.float32)
        transforms.append(
                (P['W_pretrain_output_%d'%i],P['b_pretrain_output_%d'%i]))

    def intermediate_outputs(hiddens):
        output = []
        for h,(W,b) in zip(hiddens,transforms):
            output.append(
                    T.nnet.softmax(T.dot(h,W) + b))
        return output
    return intermediate_outputs


if __name__ == "__main__":
    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]

    training_frame_count, training_label_count = count_frames(config.args.Y_files,output_size)
    logging.debug("Created shared variables")

    P = Parameters()
    classify = model.build(P,input_size,layer_sizes,output_size)
    P_tmp = Parameters()
    intermediate_outputs = build_intermediate_outputs(P_tmp,layer_sizes,output_size)

    hiddens, _ = classify(X,training=True)
    all_outputs = intermediate_outputs(hiddens)

    hiddens_test, _ = classify(X,training=False)
    all_outputs_test = intermediate_outputs(hiddens_test)

    def strat(parameters, gradients, learning_rate=1e-3, P=None):
        return [ (p, p - learning_rate * g) 
                    for p,g in zip(parameters,gradients) ]

    P.save(config.args.output_file)

    trainers = []
    testers = []
    for i,(output,output_test) in enumerate(zip(all_outputs,all_outputs_test)):
        parameters = [ P['W_classifier_input_0'], P['b_classifier_input'] ] + \
                [ P['W_classifier_%d'%(x+1)] for x in xrange(i) ] + \
                [ P['b_classifier_%d'%(x+1)] for x in xrange(i) ] + \
                [ P_tmp['W_pretrain_output_%d'%i], P_tmp['b_pretrain_output_%d'%i] ]
        logging.info("Parameters to tune:" + ','.join(w.name for w in parameters))

        cross_entropy = T.mean(crossentropy(output,Y))
        loss = cross_entropy + (0.5/training_frame_count)  * sum(T.sum(T.sqr(w)) for w in parameters)
        logging.debug("Compiling functions...")
        update_vars = Parameters()
        gradients = T.grad(loss,wrt=parameters)
        run_train = compile_train_epoch(
                parameters,gradients,update_vars,
                data_stream=build_data_stream(context=5),
#                update_strategy=strat
            )
        def make_run_test():
            monitored_values = {
                    "cross_entropy": T.mean(crossentropy(output_test,Y)),
                    "classification_error":T.mean(T.neq(T.argmax(output_test,axis=1),Y))
                }
            monitored_keys = monitored_values.keys()
            test = theano.function(
                    inputs = [X,Y],
                    outputs = [ monitored_values[k] for k in monitored_keys ]
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
            return run_test

        trainers.append(run_train)
        testers.append(make_run_test())
    

    best_ce = np.inf
    for run_train,run_test in zip(trainers,testers):
        while True:
            run_train(0.08)
            results = run_test()
            logging.info("Results: " + json.dumps(results))
            if results["cross_entropy"] < best_ce:
                best_ce = results["cross_entropy"]
                break

        logging.debug("Next layer.")
    P.save(config.args.output_file)
