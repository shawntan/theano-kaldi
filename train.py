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

import config
import chunk
import frame_label_data
import validator
import model
import epoch_train_loop

learning_file = config.option("learning_file","Parameters used during updates (e.g. momentum..).")
temporary_file = config.option("temporary_file","File to save temporary parameters while training.")
@learning_file
@temporary_file
def save_state(P,update_vars,temporary_file,learning_file):
    logging.info("Saving model and state.")
    P.save(temporary_file)
    update_vars.save(learning_file)

@learning_file
@temporary_file
def load_state(P,update_vars,learning_file,temporary_file):
    logging.info("Loading previous model and state.")
    P.load(temporary_file)
    update_vars.load(learning_file)

@config.option("output_file","Final file to save parameters after training.")
def final_save(P,output_file):
    P.save(output_file)


@config.option("initial_learning_rate","Learning rate for gradient step.",type=config.float)
@config.option("momentum","Momentum to use for gradient step.",type=config.float)
def build_updates(parameters,gradients,update_vars,initial_learning_rate,momentum):
    update_vars._learning_rate = initial_learning_rate
    return updates.momentum(parameters,gradients,P=P,
                            learning_rate=update_vars._learning_rate,
                            mu=momentum)

@config.option("learning_rate_decay", "Factor to multiply when no improvement.",
                                    default=0.5,type=config.float)
@config.option("improvement_threshold", "Improvement threshold",
                default=0.99,type=config.float)
def build_validation_callback(P,update_vars,learning_rate_decay,improvement_threshold):
    def validation_callback(prev_score,curr_score):
        current_learning_rate = update_vars._learning_rate.get_value()
        if curr_score < prev_score:
            save_state(P,update_vars)

        if curr_score > prev_score * improvement_threshold:
            load_state(P,update_vars)
            logging.info("Decaying learning rate: %0.5f -> %0.5f"%(current_learning_rate,
                            current_learning_rate * learning_rate_decay))
            update_vars._learning_rate.set_value(
                            current_learning_rate * learning_rate_decay)

    return validation_callback

@config.option("minimum_learning_rate", "Decay until this number.",
                default=1e-6,type=config.float)
def build_epoch_callback(minimum_learning_rate):
    def epoch_callback(epoch):
        logging.info("Epoch %d validation."%epoch)
        report = validate()
        logging.info(report)
        current_learning_rate = update_vars._learning_rate.get_value()
        return current_learning_rate < minimum_learning_rate
    return epoch_callback


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
    config.parse_args()
    total_frames = sum(x.shape[0] for x,_ in frame_label_data.training_stream())
    logging.info("Total frames: %d"%total_frames)
    P = Parameters()
    predict = model.build(P)

    X = T.matrix('X')
    Y = T.ivector('Y')
    _,outputs = predict(X)
    cross_entropy = T.mean(crossentropy(outputs,Y))
    parameters = P.values() 
    loss = cross_entropy + \
            (0.5/total_frames) * sum(T.sum(T.sqr(w)) for w in parameters)

    gradients = T.grad(loss,wrt=parameters)
    logging.info("Parameters to tune:" + ', '.join(sorted(w.name for w in parameters)))

    update_vars = Parameters()
    logging.debug("Compiling functions...")    
    chunk_trainer = chunk.build_trainer(
            inputs=[X,Y],
            updates = build_updates(parameters,gradients,update_vars)
        )

    validate = validator.build(
            inputs=[X,Y],
            outputs={
                "cross_entropy": cross_entropy,
                "classification_error":T.mean(T.neq(T.argmax(outputs,axis=1),Y))
            },
            monitored_var="cross_entropy",
            validation_stream=frame_label_data.validation_stream,
            callback=build_validation_callback(P,update_vars),
        )
    logging.debug("Done.")


    epoch_train_loop.loop(
            get_data_stream=lambda:data_io.async(
                    chunk.stream(frame_label_data.training_stream()),
                    queue_size=5
                ),
            item_action=chunk_trainer,
            epoch_callback=build_epoch_callback()
        )
    final_save(P)
