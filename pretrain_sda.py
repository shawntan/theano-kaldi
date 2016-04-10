import sys
import logging,json
import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
theano_rng = T.shared_randomstreams.RandomStreams(np.random.RandomState(1234).randint(2**30))

import config
import frame_data
import model
import chunk
import epoch_train_loop
import validator
import data_io
def corrupt(x,corr=0.2):
    corr_x = theano_rng.binomial(size=x.shape,n=1,p=1-corr,dtype=theano.config.floatX) * x
    corr_x.name = "corr_" + x.name
    return corr_x

def reconstruct(corr_x,W,b,b_rec,input_layer):
    hidden = T.nnet.sigmoid(T.dot(corr_x,W) + b)
    recon  = T.dot(hidden,W.T) + b_rec
    if not input_layer:
        recon = T.nnet.sigmoid(recon)
    return recon

def cost(x,recon_x,kl_divergence):
    if not kl_divergence:
        return T.mean(T.sum((x - recon_x)**2,axis=1))
    else:
        return -T.mean(T.sum(x * T.log(recon_x) + (1 - x) * T.log(1 - recon_x), axis=1))

output_file = config.option("output_file","Output of pretraining.")
@output_file
def save(P,output_file):
    logging.info("Saving model.")
    P.save(output_file)
@output_file
def load(P,output_file):
    logging.info("Loading model.")
    P.load(output_file)

def build_validation_callback(P):
    def validation_callback(prev_score,curr_score):
        if curr_score < prev_score: save(P)
    return validation_callback

if __name__ == "__main__":
    config.parse_args()
 
    P = Parameters()
    predict = model.build(P)
    X = T.matrix('X')
    layers,_ = predict(X)
    

    inputs = [X] + layers[:-1]
    # Make smaller.
    W = P["W_classifier_input_0"]
    W.set_value(W.get_value()/4)

    Ws = [P["W_classifier_input_0"]] + [P["W_classifier_%d"%i] for i in xrange(1,len(layers))] 
    bs = [P["b_classifier_input"]] + [P["b_classifier_%d"%i] for i in xrange(1,len(layers))]
    sizes = [ W.get_value().shape[0] for W in Ws ]
    
    train_fns = []
    validation_fns = []
    shared_variables_mapping = chunk.create_shared_variables([X])
    for layer,W,b,size in zip(inputs,Ws,bs,sizes):
        logging.debug("Compiling functions for layer %s"%layer)
        b_rec = theano.shared(
                np.zeros((size,),dtype=theano.config.floatX),
                name="b_rec_%d"
            )
        loss = cost(
                layer,
                reconstruct(
                    corrupt(layer),
                    W,b,b_rec,
                    input_layer = (layer.name == 'X')
                ),
                kl_divergence = (layer.name != 'X')
            )
        lr = 0.003 if layer.name == 'X'  else 0.01
        parameters = [W,b,b_rec]
        gradients  = T.grad(loss,wrt=parameters)
        train_fns.append(
            chunk.build_trainer(
                inputs=[X],
#                outputs=loss,
                updates=updates.momentum(parameters,gradients,learning_rate=lr),
                mapping=shared_variables_mapping
            )
        )
        

        validation_fns.append(
            validator.build(
                inputs=[X],
                outputs={"loss":loss},
                monitored_var="loss",
                validation_stream=frame_data.validation_stream,
                callback=build_validation_callback(P)
            )
        )
        logging.info("Done compiling for layer %s"%layer)

    save(P)
    for i,(train_fn,validation_fn) in enumerate(zip(train_fns,validation_fns)):
        logging.info("Starting pre-training for epoch %d"%i)
        load(P)
        def epoch_callback(epoch):
            logging.info("Epoch %d validation."%epoch)
            report = validation_fn()
            logging.info(report)
            return False
        epoch_train_loop.loop(
                get_data_stream=lambda:data_io.async(
                    chunk.stream(frame_data.training_stream()),
                    queue_size=2
                ),
                item_action=train_fn,
                epoch_callback=epoch_callback
            )
