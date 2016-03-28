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
if __name__ == "__main__":
    config.parse_args()
    
    P = Parameters()
    predict = model.build(P)

    X = T.matrix('X')
    Y = T.ivector('Y')
    _,outputs = predict(X)
    cross_entropy = T.mean(crossentropy(outputs,Y))
    loss = cross_entropy 

    parameters = P.values() 
    gradients = T.grad(loss,wrt=parameters)
    logging.info("Parameters to tune:" + ','.join(w.name for w in parameters))

    update_vars = Parameters()
    logging.debug("Compiling functions...")    
    chunk_trainer = chunk.build_trainer(
            inputs=[X,Y],
            updates = updates.momentum(parameters,gradients,P=update_vars)
        )

    validate = validator.build(
            inputs=[X,Y],
            outputs={
                "cross_entropy": loss,
                "classification_error":T.mean(T.neq(T.argmax(outputs,axis=1),Y))
            },
            monitored_var="cross_entropy",
            validation_stream=frame_label_data.validation_stream,
            best_score_callback=lambda:save_state(P,update_vars),
            no_improvement_callback=lambda:load_state(P,update_vars),
        )
    logging.debug("Done.")

    def epoch_callback(epoch):
        logging.info("Epoch %d validation."%epoch)
        report = validate()
        logging.info(report)
        return False

    save_state(P,update_vars)
    epoch_train_loop.loop(
            get_data_stream=lambda:data_io.async(
                    chunk.stream(frame_label_data.training_stream()),
                    queue_size=2
                ),
            item_action=chunk_trainer,
            epoch_callback=epoch_callback
        )
